#!/usr/bin/env python3
"""Build a reproducible learner-course-calendar-day panel for the KU Leuven data.

The script deliberately keeps observation and outcome information separate:

* clickstream features are calculated only inside the official course interval;
* learner-course registrations come only from the official participation table;
* PASSED and PASSED_FIRST_ATTEMPT are copied from that table and are never
  manufactured as terminal states;
* days without a timestamped click or forum contribution are explicitly present
  and flagged as ``platform_inactive``;
* forum-reading totals have no timestamp in the source and therefore remain a
  registration-level covariate instead of being allocated to days.

Large CSV files are scanned with Polars one cohort at a time.  The largest
in-memory object is consequently one academic year's selected, in-window event
table rather than all three raw logs.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
from datetime import date
from pathlib import Path
from typing import Any, Iterable

import pandas as pd
import polars as pl


COHORTS = ("1819", "1920", "2021")
SESSION_GAP_SECONDS = 30 * 60
TOD_WINDOW_DAYS = 7
MIN_SESSIONS_FOR_TOD_CONCENTRATION = 3

KEYS = ["cohort", "COURSE_ID", "USER_ID"]
DAY_KEYS = [*KEYS, "date"]

CATEGORY_MAP = {
    "Course Material": "CourseMaterial",
    "Course Main Page": "CourseMainPage",
    "Assessments": "Assessment",
    "Forum": "Forum",
    "Grades": "Grades",
    "Other": "Other",
}
CATEGORY_COLUMNS = {
    "CourseMaterial": "course_material_event_count",
    "CourseMainPage": "course_main_page_event_count",
    "Assessment": "assessment_event_count",
    "Forum": "forum_event_count",
    "Grades": "grades_event_count",
    "Other": "other_event_count",
    "Unknown": "unknown_content_event_count",
}


def parse_args() -> argparse.Namespace:
    project_root = Path(__file__).resolve().parents[3]
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--input-root",
        type=Path,
        default=project_root / "crawled_data" / "raw" / "ku_leuven",
        help="Directory containing course_info.json and extracted/.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=project_root / "paper_output" / "data_cleaned",
    )
    parser.add_argument(
        "--cohorts",
        nargs="+",
        choices=COHORTS,
        default=list(COHORTS),
        help="Academic-year folders/files to process.",
    )
    parser.add_argument(
        "--limit-log-rows",
        type=int,
        default=None,
        help="Development-only row cap per log. Omit for the publishable full panel.",
    )
    return parser.parse_args()


def sha256_file(path: Path, chunk_size: int = 4 * 1024 * 1024) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(chunk_size), b""):
            digest.update(chunk)
    return digest.hexdigest()


def scalar(frame: pl.DataFrame, name: str) -> int | float | None:
    value = frame.item(0, name)
    if value is None:
        return None
    if isinstance(value, float):
        return float(value)
    return int(value)


def jsonable(value: Any) -> Any:
    """Convert Polars/Pandas/numpy scalar-like values to JSON primitives."""
    if value is None or isinstance(value, (str, int, float, bool)):
        return value
    if hasattr(value, "item"):
        return value.item()
    if isinstance(value, (date,)):
        return value.isoformat()
    return str(value)


def read_participation(path: Path, cohort: str) -> tuple[pl.DataFrame, dict[str, Any]]:
    raw = pd.read_excel(path)
    required = {
        "COURSE_ID",
        "USER_ID",
        "PASSED_FIRST_ATTEMPT",
        "PASSED",
        "SCORE_CATEGORY_FINAL",
    }
    missing = sorted(required.difference(raw.columns))
    if missing:
        raise ValueError(f"{path.name}: missing participation columns {missing}")

    raw_rows = len(raw)
    exact_duplicates_removed = int(raw.duplicated().sum())
    raw = raw.drop_duplicates().copy()
    missing_key_rows = int(raw[["COURSE_ID", "USER_ID"]].isna().any(axis=1).sum())
    raw = raw.dropna(subset=["COURSE_ID", "USER_ID"])
    duplicate_keys = raw.duplicated(["COURSE_ID", "USER_ID"], keep=False)
    if duplicate_keys.any():
        sample = raw.loc[duplicate_keys, ["COURSE_ID", "USER_ID"]].head(10)
        raise ValueError(
            f"{path.name}: non-identical duplicate registration keys remain: "
            f"{sample.to_dict('records')}"
        )

    keep = [
        "COURSE_ID",
        "USER_ID",
        "SCORE_CATEGORY_FINAL",
        "PASSED_FIRST_ATTEMPT",
        "PASSED",
    ]
    clean = raw[keep].copy()
    clean["COURSE_ID"] = clean["COURSE_ID"].astype(str)
    clean["USER_ID"] = clean["USER_ID"].astype("int64")
    clean["PASSED_FIRST_ATTEMPT"] = clean["PASSED_FIRST_ATTEMPT"].astype("int8")
    clean["PASSED"] = clean["PASSED"].astype("int8")
    frame = pl.from_pandas(clean).with_columns(pl.lit(cohort).alias("cohort"))
    frame = frame.select(
        pl.col("cohort").cast(pl.String),
        pl.col("COURSE_ID").cast(pl.String),
        pl.col("USER_ID").cast(pl.Int64),
        pl.col("SCORE_CATEGORY_FINAL").cast(pl.String),
        pl.col("PASSED_FIRST_ATTEMPT").cast(pl.Int8),
        pl.col("PASSED").cast(pl.Int8),
    )
    info = {
        "raw_rows": raw_rows,
        "exact_duplicate_rows_removed": exact_duplicates_removed,
        "missing_key_rows_removed": missing_key_rows,
        "official_registrations": frame.height,
        "unique_users_within_cohort": frame.get_column("USER_ID").n_unique(),
        "courses": sorted(frame.get_column("COURSE_ID").unique().to_list()),
    }
    return frame, info


def resolve_course_intervals(
    course_info: dict[str, Any], cohort: str, course_names: Iterable[str]
) -> pl.DataFrame:
    configured = course_info[cohort]
    rows: list[dict[str, Any]] = []
    for course_name in sorted(set(course_names)):
        if course_name in configured:
            source_name = course_name
        else:
            candidates = [name for name in configured if course_name.startswith(name)]
            if len(candidates) != 1:
                raise KeyError(
                    f"Cannot map {cohort}/{course_name!r} to a unique course_info entry; "
                    f"candidates={candidates}"
                )
            source_name = candidates[0]
        spec = configured[source_name]
        rows.append(
            {
                "cohort": cohort,
                "COURSE_ID": course_name,
                "course_start": date(*spec["course_start"]),
                "course_finish": date(*spec["course_finish"]),
                "course_info_source_name": source_name,
            }
        )
    return pl.DataFrame(rows).with_columns(
        pl.col("cohort").cast(pl.String),
        pl.col("COURSE_ID").cast(pl.String),
        pl.col("course_start").cast(pl.Date),
        pl.col("course_finish").cast(pl.Date),
    )


def read_content_lookup(path: Path, cohort: str) -> tuple[pl.DataFrame, dict[str, Any]]:
    raw = pd.read_excel(path)
    required = {"COURSE_ID", "CONTENT_ID", "CONTENT_TYPE"}
    missing = sorted(required.difference(raw.columns))
    if missing:
        raise ValueError(f"{path.name}: missing content columns {missing}")
    raw_rows = len(raw)
    exact_duplicates_removed = int(raw.duplicated().sum())
    raw = raw.drop_duplicates().copy()
    missing_key_rows = int(raw[["COURSE_ID", "CONTENT_ID"]].isna().any(axis=1).sum())
    raw = raw.dropna(subset=["COURSE_ID", "CONTENT_ID"])
    duplicate_keys = raw.duplicated(["COURSE_ID", "CONTENT_ID"], keep=False)
    if duplicate_keys.any():
        conflicted = raw.loc[duplicate_keys, ["COURSE_ID", "CONTENT_ID", "CONTENT_TYPE"]]
        if (conflicted.groupby(["COURSE_ID", "CONTENT_ID"])["CONTENT_TYPE"].nunique() > 1).any():
            raise ValueError(f"{path.name}: conflicting CONTENT_TYPE for a content key")
        raw = raw.drop_duplicates(["COURSE_ID", "CONTENT_ID"])

    content_type = raw["CONTENT_TYPE"].map(CATEGORY_MAP).fillna("Other")
    lookup = pl.DataFrame(
        {
            "cohort": [cohort] * len(raw),
            "COURSE_ID": raw["COURSE_ID"].astype(str).tolist(),
            "CONTENT_ID": raw["CONTENT_ID"].astype("int64").tolist(),
            "activity_category": content_type.tolist(),
            "_content_matched": [1] * len(raw),
        }
    ).with_columns(
        pl.col("cohort").cast(pl.String),
        pl.col("COURSE_ID").cast(pl.String),
        pl.col("CONTENT_ID").cast(pl.Int64),
        pl.col("activity_category").cast(pl.String),
        pl.col("_content_matched").cast(pl.Int8),
    )
    info = {
        "raw_rows": raw_rows,
        "exact_duplicate_rows_removed": exact_duplicates_removed,
        "missing_key_rows_removed": missing_key_rows,
        "lookup_rows": lookup.height,
    }
    return lookup, info


def scan_log(path: Path, cohort: str, row_limit: int | None) -> pl.LazyFrame:
    return (
        pl.scan_csv(
            path,
            n_rows=row_limit,
            infer_schema_length=10_000,
            schema_overrides={
                "ACTION_ID": pl.Int64,
                "COURSE_ID": pl.String,
                "USER_ID": pl.Int64,
                "CONTENT_ID": pl.Int64,
                "SESSION_ID": pl.Int64,
                "TIMESTAMP": pl.String,
            },
            low_memory=True,
            rechunk=False,
        )
        .with_columns(
            pl.lit(cohort).alias("cohort"),
            pl.col("TIMESTAMP")
            .str.strptime(pl.Datetime("us"), format="%Y-%m-%d %H:%M:%S", strict=False)
            .alias("event_ts"),
        )
        .with_columns(pl.col("event_ts").dt.date().alias("date"))
    )


def load_events(
    path: Path,
    cohort: str,
    participation: pl.DataFrame,
    intervals: pl.DataFrame,
    content: pl.DataFrame,
    row_limit: int | None,
) -> tuple[pl.DataFrame, dict[str, Any]]:
    participant_marker = participation.select(*KEYS).with_columns(
        pl.lit(1, dtype=pl.Int8).alias("_official_participant")
    )
    enriched = (
        scan_log(path, cohort, row_limit)
        .join(participant_marker.lazy(), on=KEYS, how="left")
        .join(intervals.lazy(), on=["cohort", "COURSE_ID"], how="left")
        .join(content.lazy(), on=["cohort", "COURSE_ID", "CONTENT_ID"], how="left")
    )
    is_official = pl.col("_official_participant").is_not_null()
    parsed = pl.col("event_ts").is_not_null()
    interval_known = pl.col("course_start").is_not_null() & pl.col("course_finish").is_not_null()
    in_window = (
        is_official
        & parsed
        & interval_known
        & (pl.col("date") >= pl.col("course_start"))
        & (pl.col("date") <= pl.col("course_finish"))
    )
    metrics = enriched.select(
        pl.len().alias("raw_log_rows"),
        pl.col("event_ts").is_null().sum().alias("timestamp_parse_failures"),
        is_official.sum().alias("events_matching_official_registration"),
        (~is_official).sum().alias("events_without_official_registration"),
        (is_official & ~interval_known).sum().alias("official_events_without_course_interval"),
        (
            is_official
            & parsed
            & interval_known
            & ((pl.col("date") < pl.col("course_start")) | (pl.col("date") > pl.col("course_finish")))
        )
        .sum()
        .alias("official_events_outside_course_interval"),
        in_window.sum().alias("main_sequence_log_events"),
        (in_window & pl.col("_content_matched").is_null())
        .sum()
        .alias("unmatched_content_events_in_main_sequence"),
        (in_window & pl.col("CONTENT_ID").is_null())
        .sum()
        .alias("null_content_id_events_in_main_sequence"),
    ).collect(engine="streaming")
    event_stats = {name: scalar(metrics, name) for name in metrics.columns}

    selected = (
        enriched.filter(in_window)
        .with_columns(
            pl.when(pl.col("_content_matched").is_null())
            .then(pl.lit("Unknown"))
            .otherwise(pl.col("activity_category"))
            .alias("activity_category")
        )
        .select(
            *KEYS,
            "ACTION_ID",
            "CONTENT_ID",
            "event_ts",
            "date",
            "activity_category",
        )
        .collect(engine="streaming")
        .sort([*KEYS, "event_ts", "ACTION_ID"])
    )
    event_stats["unmatched_content_rate_in_main_sequence"] = (
        event_stats["unmatched_content_events_in_main_sequence"]
        / event_stats["main_sequence_log_events"]
        if event_stats["main_sequence_log_events"]
        else None
    )
    return selected, event_stats


def make_daily_event_features(
    events: pl.DataFrame,
) -> tuple[pl.DataFrame, pl.DataFrame, pl.DataFrame]:
    if events.is_empty():
        raise ValueError("No eligible in-window clickstream events were found")

    events = events.with_columns(
        pl.col("event_ts").diff().over(KEYS).dt.total_seconds().alias("_gap_seconds")
    ).with_columns(
        (
            pl.col("_gap_seconds").is_null()
            | (pl.col("_gap_seconds") > SESSION_GAP_SECONDS)
        )
        .cast(pl.Int64)
        .alias("_new_session")
    ).with_columns(
        pl.col("_new_session").cum_sum().over(KEYS).alias("_session_seq"),
        pl.col("event_ts")
        .diff()
        .over(DAY_KEYS)
        .dt.total_seconds()
        .alias("_within_day_gap_seconds"),
    ).with_columns(
        pl.when(
            pl.col("_within_day_gap_seconds").is_not_null()
            & (pl.col("_within_day_gap_seconds") >= 0)
            & (pl.col("_within_day_gap_seconds") <= SESSION_GAP_SECONDS)
        )
        .then(pl.col("_within_day_gap_seconds") / 60.0)
        .otherwise(0.0)
        .alias("_engaged_gap_minutes")
    )

    category_exprs = [
        (pl.col("activity_category") == category).sum().alias(column)
        for category, column in CATEGORY_COLUMNS.items()
    ]
    daily = (
        events.group_by(DAY_KEYS)
        .agg(
            pl.len().alias("log_event_count"),
            pl.col("_session_seq").n_unique().alias("session_count_30m"),
            pl.col("event_ts").min().alias("first_event_ts"),
            pl.col("event_ts").max().alias("last_event_ts"),
            (
                (pl.col("event_ts").max() - pl.col("event_ts").min()).dt.total_seconds()
                / 60.0
            ).alias("active_span_minutes"),
            pl.col("_engaged_gap_minutes").sum().alias("active_minutes_30m"),
            pl.col("activity_category").n_unique().alias("activity_category_diversity"),
            pl.col("CONTENT_ID").drop_nulls().n_unique().alias("content_diversity"),
            *category_exprs,
        )
        .sort(DAY_KEYS)
    )

    resource_days = (
        events.select(*DAY_KEYS, "CONTENT_ID")
        .drop_nulls("CONTENT_ID")
        .unique()
        .sort([*KEYS, "CONTENT_ID", "date"])
        .with_columns(
            pl.col("date")
            .shift(1)
            .over([*KEYS, "CONTENT_ID"])
            .alias("_previous_resource_date")
        )
        .with_columns(
            (pl.col("date") - pl.col("_previous_resource_date"))
            .dt.total_days()
            .alias("_days_since_resource")
        )
    )
    revisits = (
        resource_days.filter(pl.col("_days_since_resource").is_between(1, 7, closed="both"))
        .group_by(DAY_KEYS)
        .agg(pl.len().alias("resources_revisited_1_7d"))
    )

    session_starts = (
        events.group_by([*KEYS, "_session_seq"])
        .agg(pl.col("event_ts").min().alias("session_start_ts"))
        .with_columns(
            pl.col("session_start_ts").dt.date().alias("date"),
            (
                pl.col("session_start_ts").dt.hour().cast(pl.Float64)
                + pl.col("session_start_ts").dt.minute().cast(pl.Float64) / 60.0
                + pl.col("session_start_ts").dt.second().cast(pl.Float64) / 3600.0
            ).alias("_session_start_hour")
        )
        .with_columns(
            (pl.col("_session_start_hour") * (2.0 * math.pi / 24.0)).cos().alias("_tod_cos"),
            (pl.col("_session_start_hour") * (2.0 * math.pi / 24.0)).sin().alias("_tod_sin"),
        )
        .group_by(DAY_KEYS)
        .agg(
            pl.len().alias("session_start_count"),
            pl.col("_tod_cos").sum().alias("_tod_cos_sum"),
            pl.col("_tod_sin").sum().alias("_tod_sin_sum"),
        )
    )
    return daily, revisits, session_starts


def read_forum_features(
    extracted: Path,
    cohort: str,
    participation: pl.DataFrame,
    intervals: pl.DataFrame,
) -> tuple[pl.DataFrame, pl.DataFrame, dict[str, Any]]:
    pkeys = participation.select(*KEYS)

    consumption_pd = pd.read_excel(extracted / f"{cohort}_df_consumption.xlsx")
    consumption_raw_rows = len(consumption_pd)
    consumption_pd = consumption_pd.drop_duplicates().copy()
    consumption = pl.from_pandas(consumption_pd).with_columns(
        pl.lit(cohort).alias("cohort"),
        pl.col("COURSE_ID").cast(pl.String),
        pl.col("USER_ID").cast(pl.Int64),
        pl.col("NUM_READ_POSTS").cast(pl.Int64, strict=False),
    )
    consumption_joined = consumption.join(pkeys, on=KEYS, how="inner")
    read_totals = consumption_joined.group_by(KEYS).agg(
        pl.col("NUM_READ_POSTS").fill_null(0).sum().alias("forum_posts_read_total_official")
    )

    contribution_pd = pd.read_excel(extracted / f"{cohort}_df_contribution.xlsx")
    contribution_raw_rows = len(contribution_pd)
    contribution_pd = contribution_pd.drop_duplicates().copy()
    contribution_pd["_post_ts"] = pd.to_datetime(contribution_pd["DTCREATED"], errors="coerce")
    contributions = pl.from_pandas(contribution_pd).with_columns(
        pl.lit(cohort).alias("cohort"),
        pl.col("COURSE_ID").cast(pl.String),
        pl.col("USER_ID").cast(pl.Int64),
        pl.col("POST_ID").cast(pl.Int64),
        pl.col("_post_ts").cast(pl.Datetime("us")),
    )
    marker = pkeys.with_columns(pl.lit(1, dtype=pl.Int8).alias("_official_participant"))
    contribution_enriched = (
        contributions.join(marker, on=KEYS, how="left")
        .join(intervals, on=["cohort", "COURSE_ID"], how="left")
        .with_columns(pl.col("_post_ts").dt.date().alias("date"))
    )
    official = pl.col("_official_participant").is_not_null()
    parsed = pl.col("_post_ts").is_not_null()
    in_window = (
        official
        & parsed
        & (pl.col("date") >= pl.col("course_start"))
        & (pl.col("date") <= pl.col("course_finish"))
    )
    post_daily = (
        contribution_enriched.filter(in_window)
        .group_by(DAY_KEYS)
        .agg(pl.col("POST_ID").n_unique().alias("forum_posts_created"))
    )
    info = {
        "consumption_raw_rows": consumption_raw_rows,
        "consumption_exact_duplicate_rows_removed": int(
            consumption_raw_rows - len(consumption_pd)
        ),
        "consumption_rows_matching_official_registration": consumption_joined.height,
        "consumption_rows_without_official_registration": (
            consumption.height - consumption_joined.height
        ),
        "official_forum_posts_read_total": int(
            read_totals.get_column("forum_posts_read_total_official").sum()
            if read_totals.height
            else 0
        ),
        "contribution_raw_rows": contribution_raw_rows,
        "contribution_exact_duplicate_rows_removed": int(
            contribution_raw_rows - len(contribution_pd)
        ),
        "contribution_timestamp_parse_failures": int(
            contribution_enriched.get_column("_post_ts").null_count()
        ),
        "contribution_rows_without_official_registration": int(
            contribution_enriched.filter(~official).height
        ),
        "official_contributions_outside_course_interval": int(
            contribution_enriched.filter(
                official
                & parsed
                & (
                    (pl.col("date") < pl.col("course_start"))
                    | (pl.col("date") > pl.col("course_finish"))
                )
            ).height
        ),
        "forum_posts_in_main_sequence": int(post_daily.get_column("forum_posts_created").sum()),
    }
    return read_totals, post_daily, info


def make_panel(
    participation: pl.DataFrame,
    intervals: pl.DataFrame,
    daily_events: pl.DataFrame,
    revisits: pl.DataFrame,
    session_starts: pl.DataFrame,
    read_totals: pl.DataFrame,
    post_daily: pl.DataFrame,
) -> pl.DataFrame:
    registrations = participation.join(intervals, on=["cohort", "COURSE_ID"], how="inner")
    panel = (
        registrations.with_columns(
            pl.date_ranges(
                pl.col("course_start"),
                pl.col("course_finish"),
                interval="1d",
                closed="both",
            ).alias("date")
        )
        .explode("date", empty_as_null=True)
        .join(daily_events, on=DAY_KEYS, how="left")
        .join(revisits, on=DAY_KEYS, how="left")
        .join(session_starts, on=DAY_KEYS, how="left")
        .join(post_daily, on=DAY_KEYS, how="left")
        .join(read_totals, on=KEYS, how="left")
    )

    count_columns = [
        "log_event_count",
        "session_count_30m",
        "activity_category_diversity",
        "content_diversity",
        *CATEGORY_COLUMNS.values(),
        "resources_revisited_1_7d",
        "session_start_count",
        "forum_posts_created",
        "forum_posts_read_total_official",
    ]
    float_zero_columns = [
        "active_span_minutes",
        "active_minutes_30m",
        "_tod_cos_sum",
        "_tod_sin_sum",
    ]
    panel = panel.with_columns(
        *[pl.col(name).fill_null(0).cast(pl.Int64) for name in count_columns],
        *[pl.col(name).fill_null(0.0).cast(pl.Float64) for name in float_zero_columns],
    ).with_columns(
        (pl.col("assessment_event_count") + pl.col("forum_posts_created")).alias(
            "productive_event_count"
        ),
        ((pl.col("log_event_count") == 0) & (pl.col("forum_posts_created") == 0))
        .cast(pl.Int8)
        .alias("platform_inactive"),
        (pl.col("date") - pl.col("course_start")).dt.total_days().alias("course_day_index"),
        pl.col("date").dt.weekday().alias("weekday_iso"),
        pl.when(pl.col("content_diversity") > 0)
        .then(pl.col("resources_revisited_1_7d") / pl.col("content_diversity"))
        .otherwise(None)
        .alias("resource_revisit_rate_1_7d"),
        pl.concat_str(
            [
                pl.col("cohort"),
                pl.col("COURSE_ID"),
                pl.col("USER_ID").cast(pl.String),
            ],
            separator="::",
        ).alias("registration_id"),
    )

    panel = panel.sort(DAY_KEYS).with_columns(
        pl.col("_tod_cos_sum")
        .rolling_sum(window_size=TOD_WINDOW_DAYS, min_samples=1)
        .over(KEYS)
        .alias("_tod_cos_7d"),
        pl.col("_tod_sin_sum")
        .rolling_sum(window_size=TOD_WINDOW_DAYS, min_samples=1)
        .over(KEYS)
        .alias("_tod_sin_7d"),
        pl.col("session_start_count")
        .rolling_sum(window_size=TOD_WINDOW_DAYS, min_samples=1)
        .over(KEYS)
        .alias("tod_sessions_7d"),
    ).with_columns(
        pl.when(pl.col("tod_sessions_7d") >= MIN_SESSIONS_FOR_TOD_CONCENTRATION)
        .then(
            (
                pl.col("_tod_cos_7d").pow(2) + pl.col("_tod_sin_7d").pow(2)
            ).sqrt()
            / pl.col("tod_sessions_7d")
        )
        .otherwise(None)
        .alias("tod_circular_concentration_7d")
    )

    ordered = [
        "registration_id",
        *KEYS,
        "date",
        "course_start",
        "course_finish",
        "course_day_index",
        "weekday_iso",
        "course_info_source_name",
        "PASSED",
        "PASSED_FIRST_ATTEMPT",
        "SCORE_CATEGORY_FINAL",
        "platform_inactive",
        "log_event_count",
        "session_count_30m",
        "first_event_ts",
        "last_event_ts",
        "active_span_minutes",
        "active_minutes_30m",
        *CATEGORY_COLUMNS.values(),
        "activity_category_diversity",
        "content_diversity",
        "productive_event_count",
        "forum_posts_created",
        "forum_posts_read_total_official",
        "resources_revisited_1_7d",
        "resource_revisit_rate_1_7d",
        "tod_sessions_7d",
        "tod_circular_concentration_7d",
    ]
    return panel.select(ordered).sort(DAY_KEYS)


def cohort_summary(panel: pl.DataFrame) -> dict[str, Any]:
    grouped = (
        panel.group_by(["cohort", "COURSE_ID"])
        .agg(
            pl.col("registration_id").n_unique().alias("registrations"),
            pl.len().alias("panel_days"),
            pl.col("platform_inactive").sum().alias("platform_inactive_days"),
            pl.col("log_event_count").sum().alias("log_events"),
            pl.col("forum_posts_created").sum().alias("forum_posts"),
            pl.col("PASSED").first().alias("_unused"),
        )
        .drop("_unused")
        .sort(["cohort", "COURSE_ID"])
    )
    outcomes = (
        panel.unique("registration_id")
        .group_by(["cohort", "COURSE_ID"])
        .agg(
            pl.col("PASSED").sum().alias("passed"),
            pl.col("PASSED_FIRST_ATTEMPT").sum().alias("passed_first_attempt"),
        )
    )
    grouped = grouped.join(outcomes, on=["cohort", "COURSE_ID"], how="left")
    return {
        f"{row['cohort']}::{row['COURSE_ID']}": {
            key: jsonable(value)
            for key, value in row.items()
            if key not in {"cohort", "COURSE_ID"}
        }
        for row in grouped.to_dicts()
    }


def validate_panel(panel: pl.DataFrame, participation: pl.DataFrame, intervals: pl.DataFrame) -> dict[str, Any]:
    registrations = participation.join(intervals, on=["cohort", "COURSE_ID"], how="inner")
    expected_rows = registrations.select(
        (
            (pl.col("course_finish") - pl.col("course_start")).dt.total_days() + 1
        ).sum().alias("expected")
    ).item()
    unique_day_keys = panel.select(DAY_KEYS).unique().height
    out_of_window = panel.filter(
        (pl.col("date") < pl.col("course_start"))
        | (pl.col("date") > pl.col("course_finish"))
    ).height
    inactive_inconsistency = panel.filter(
        (pl.col("platform_inactive") == 1)
        & ((pl.col("log_event_count") != 0) | (pl.col("forum_posts_created") != 0))
    ).height
    revisit_inconsistency = panel.filter(
        pl.col("resources_revisited_1_7d") > pl.col("content_diversity")
    ).height
    result = {
        "expected_panel_rows": int(expected_rows),
        "actual_panel_rows": panel.height,
        "unique_student_course_day_keys": unique_day_keys,
        "duplicate_student_course_day_rows": panel.height - unique_day_keys,
        "rows_outside_official_course_interval": out_of_window,
        "platform_inactive_flag_inconsistencies": inactive_inconsistency,
        "revisit_count_exceeds_daily_content_diversity": revisit_inconsistency,
        "null_passed_outcomes": panel.get_column("PASSED").null_count(),
        "null_passed_first_attempt_outcomes": panel.get_column(
            "PASSED_FIRST_ATTEMPT"
        ).null_count(),
    }
    result["all_hard_checks_pass"] = all(
        [
            result["expected_panel_rows"] == result["actual_panel_rows"],
            result["duplicate_student_course_day_rows"] == 0,
            result["rows_outside_official_course_interval"] == 0,
            result["platform_inactive_flag_inconsistencies"] == 0,
            result["revisit_count_exceeds_daily_content_diversity"] == 0,
            result["null_passed_outcomes"] == 0,
            result["null_passed_first_attempt_outcomes"] == 0,
        ]
    )
    return result


def field_dictionary_text() -> str:
    return f"""# KU Leuven 日级面板字段字典

粒度：每行是一名官方参与者在一门课程官方起止日期内的一个自然日。时间戳按数据源记录的本地墙钟时间解释；源文件没有时区，脚本未做时区推断或转换。

| 字段 | 定义 |
|---|---|
| `registration_id` | 固定拼接的学年、课程、匿名用户标识。 |
| `cohort`, `COURSE_ID`, `USER_ID`, `date` | 学年、课程、匿名用户、自然日联合主键。 |
| `course_start`, `course_finish`, `course_day_index` | 官方课程区间及从 0 开始的课程日序号。 |
| `PASSED`, `PASSED_FIRST_ATTEMPT` | 官方参与表结局；仅复制，不追加人为终止状态。 |
| `platform_inactive` | 当日既无 clickstream 事件又无带时间戳的论坛发帖时为 1。不能解释为“没有学习”。 |
| `log_event_count` | 当日官方课程区间内的原始 clickstream 行数。 |
| `session_count_30m` | 将相邻事件间隔大于 {SESSION_GAP_SECONDS // 60} 分钟视为新会话后，当日涉及的会话数。忽略源表自带 SESSION_ID。 |
| `active_span_minutes` | 当日首末 clickstream 时间戳跨度；它是跨度而非有效学习时长。 |
| `active_minutes_30m` | 当日相邻事件间隔不超过 {SESSION_GAP_SECONDS // 60} 分钟的间隔和；单次点击会话贡献 0，属于保守代理量。 |
| `*_event_count` | CourseMaterial、CourseMainPage、Assessment、Forum、Grades、Other、Unknown 七类事件计数。Unknown 保留内容表未匹配事件。 |
| `activity_category_diversity` | 当日出现的上述活动大类数。 |
| `content_diversity` | 当日不同 CONTENT_ID 数（包括内容表未匹配但 ID 可识别的资源）。 |
| `productive_event_count` | Assessment 类点击数加当日官方论坛发帖数；是可复算的观测代理，不是学习效果。 |
| `forum_event_count` | 日志中 Forum 内容的点击数，是阅读行为代理，不等于帖子阅读篇数。 |
| `forum_posts_created` | contribution 表中按 DTCREATED 归日、位于课程区间内的不同 POST_ID 数。 |
| `forum_posts_read_total_official` | consumption 表中的官方累计已读帖子数。源表无日期，故只作为注册层协变量重复在各日，禁止当作日级阅读量。 |
| `resources_revisited_1_7d` | 当日访问且同一学生在此前 1–7 个自然日也访问过的不同 CONTENT_ID 数。 |
| `resource_revisit_rate_1_7d` | `resources_revisited_1_7d / content_diversity`；无资源访问日置缺。 |
| `tod_sessions_7d` | 含当日在内最近 7 个自然日的 30 分钟会话开始数。 |
| `tod_circular_concentration_7d` | 最近 7 日会话开始时刻的圆形平均向量长度，范围 0–1；至少 {MIN_SESSIONS_FOR_TOD_CONCENTRATION} 次会话才计算，否则置缺。越高表示学习开始时段越集中，不自动等同于更优学习。 |

## 关键边界

- 面板只含参与表可识别的学生—课程记录，并严格限制在 `course_info.json` 的官方日期内。
- 2020–2021 学年的 `Global economics 1/2` 按名称前缀映射到官方 `Global economics` 课程区间；映射来源保存在 `course_info_source_name`。
- `platform_inactive=1` 只能称为平台无活动。代码学习、线下阅读等平台外活动在该数据中不可观测。
- 论坛累计阅读量没有时间戳，脚本没有按日均摊、回填或随机分配。
"""


def main() -> None:
    args = parse_args()
    input_root = args.input_root.resolve()
    extracted = input_root / "extracted"
    output_dir = args.output_dir.resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    course_info_path = input_root / "course_info.json"
    course_info = json.loads(course_info_path.read_text(encoding="utf-8"))

    all_panels: list[pl.DataFrame] = []
    all_participation: list[pl.DataFrame] = []
    all_intervals: list[pl.DataFrame] = []
    per_cohort: dict[str, Any] = {}
    input_paths: list[Path] = [course_info_path]

    for cohort in args.cohorts:
        participation_path = extracted / f"{cohort}_course_participation.xlsx"
        content_path = extracted / f"{cohort}_course_content.xlsx"
        log_path = extracted / f"{cohort}_log_activity.csv"
        consumption_path = extracted / f"{cohort}_df_consumption.xlsx"
        contribution_path = extracted / f"{cohort}_df_contribution.xlsx"
        input_paths.extend(
            [
                participation_path,
                content_path,
                log_path,
                consumption_path,
                contribution_path,
            ]
        )
        for path in input_paths[-5:]:
            if not path.exists():
                raise FileNotFoundError(path)

        participation, participation_info = read_participation(participation_path, cohort)
        intervals = resolve_course_intervals(
            course_info, cohort, participation.get_column("COURSE_ID").to_list()
        )
        content, content_info = read_content_lookup(content_path, cohort)
        events, log_info = load_events(
            log_path,
            cohort,
            participation,
            intervals,
            content,
            args.limit_log_rows,
        )
        daily, revisits, session_starts = make_daily_event_features(events)
        read_totals, post_daily, forum_info = read_forum_features(
            extracted, cohort, participation, intervals
        )
        panel = make_panel(
            participation,
            intervals,
            daily,
            revisits,
            session_starts,
            read_totals,
            post_daily,
        )
        all_panels.append(panel)
        all_participation.append(participation)
        all_intervals.append(intervals)
        per_cohort[cohort] = {
            "participation": participation_info,
            "content_lookup": content_info,
            "clickstream": log_info,
            "forum": forum_info,
            "panel_rows": panel.height,
            "platform_inactive_days": int(panel.get_column("platform_inactive").sum()),
            "days_with_observed_platform_activity": int(
                panel.height - panel.get_column("platform_inactive").sum()
            ),
        }
        print(
            f"[{cohort}] registrations={participation.height:,} "
            f"events={events.height:,} panel_days={panel.height:,}",
            flush=True,
        )

    full_panel = pl.concat(all_panels, how="vertical_relaxed").sort(DAY_KEYS)
    full_participation = pl.concat(all_participation, how="vertical_relaxed")
    full_intervals = pl.concat(all_intervals, how="vertical_relaxed")
    validation = validate_panel(full_panel, full_participation, full_intervals)
    if not validation["all_hard_checks_pass"]:
        raise RuntimeError(f"Panel validation failed: {validation}")

    panel_path = output_dir / "ku_daily_panel.parquet"
    quality_path = output_dir / "ku_quality_report.json"
    dictionary_path = output_dir / "ku_field_dictionary.md"
    full_panel.write_parquet(
        panel_path,
        compression="zstd",
        compression_level=6,
        statistics=True,
    )

    manifest = [
        {
            "path": str(path.relative_to(input_root.parent.parent.parent)),
            "bytes": path.stat().st_size,
            "sha256": sha256_file(path),
        }
        for path in sorted(set(input_paths))
    ]
    quality = {
        "dataset": "KU Leuven detailed clickstream daily modeling panel",
        "configuration": {
            "cohorts": list(args.cohorts),
            "limit_log_rows": args.limit_log_rows,
            "session_gap_minutes": SESSION_GAP_SECONDS // 60,
            "time_of_day_window_days": TOD_WINDOW_DAYS,
            "minimum_sessions_for_time_of_day_concentration": (
                MIN_SESSIONS_FOR_TOD_CONCENTRATION
            ),
            "calendar": "source-local natural day; source timezone unspecified",
        },
        "input_manifest": manifest,
        "per_cohort": per_cohort,
        "overall": {
            "official_registrations": full_participation.height,
            "unique_anonymous_users_across_cohorts_not_assumed_linkable": int(
                full_participation.select(["cohort", "USER_ID"]).unique().height
            ),
            "courses_by_cohort": int(full_intervals.height),
            "panel_rows": full_panel.height,
            "platform_inactive_days": int(full_panel.get_column("platform_inactive").sum()),
            "days_with_observed_platform_activity": int(
                full_panel.height - full_panel.get_column("platform_inactive").sum()
            ),
            "log_events_in_main_sequence": int(full_panel.get_column("log_event_count").sum()),
            "sessions_30m_summed_over_days": int(
                full_panel.get_column("session_count_30m").sum()
            ),
            "forum_posts_in_main_sequence": int(
                full_panel.get_column("forum_posts_created").sum()
            ),
            "passed_registrations": int(
                full_panel.unique("registration_id").get_column("PASSED").sum()
            ),
            "passed_first_attempt_registrations": int(
                full_panel.unique("registration_id")
                .get_column("PASSED_FIRST_ATTEMPT")
                .sum()
            ),
            "rows_with_defined_tod_concentration": int(
                full_panel.get_column("tod_circular_concentration_7d").is_not_null().sum()
            ),
        },
        "by_course": cohort_summary(full_panel),
        "validation": validation,
        "known_measurement_limits": [
            "Platform-Inactive is an observation category, not evidence of no learning.",
            "Forum consumption has no timestamp; forum_posts_read_total_official is registration-level.",
            "Forum log clicks are a reading proxy and are not equated with official posts-read counts.",
            "active_span_minutes is a clock span; active_minutes_30m is a conservative click-gap proxy.",
            "Timestamps lack an explicit timezone; no timezone conversion is performed.",
            "Anonymous USER_ID values are not linked across cohorts.",
        ],
    }
    quality_path.write_text(
        json.dumps(quality, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    dictionary_path.write_text(field_dictionary_text(), encoding="utf-8")
    print(f"Wrote {panel_path} ({full_panel.height:,} rows)")
    print(f"Wrote {quality_path}")
    print(f"Wrote {dictionary_path}")


if __name__ == "__main__":
    main()
