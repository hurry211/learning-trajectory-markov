#!/usr/bin/env python3
"""Build a reproducible day-level OULAD modelling table.

The script deliberately treats ``Platform-Inactive`` as an observation rather
than a latent learning-state label.  It also keeps pre-course and
post-unregistration activity outside the main sequence, while exposing those
records in dedicated audit/sensitivity outputs.

The large ``studentVle.csv`` table is processed with Polars' streaming engine.
Rows sharing student/course/site/day keys are *summed*, never deduplicated, so
all recorded clicks are conserved.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import platform
import sys
import time
from pathlib import Path
from typing import Any, Iterable

import polars as pl


KEYS = ["code_module", "code_presentation", "id_student"]
SITE_DAY_KEYS = KEYS + ["id_site", "date"]
DAY_KEYS = KEYS + ["date"]

EVENT_CATEGORY_MAP = {
    # Course materials and pages.
    "resource": "content_access",
    "oucontent": "content_access",
    "subpage": "content_access",
    "page": "content_access",
    "folder": "content_access",
    "htmlactivity": "content_access",
    "sharedsubpage": "content_access",
    "dualpane": "content_access",
    # Navigation and outbound links.
    "homepage": "navigation_link",
    "url": "navigation_link",
    "repeatactivity": "navigation_link",
    # Formative testing and questionnaires.
    "quiz": "assessment_practice",
    "externalquiz": "assessment_practice",
    "questionnaire": "assessment_practice",
    # Social and collaborative tools.
    "forumng": "communication_collaboration",
    "oucollaborate": "communication_collaboration",
    "ouwiki": "communication_collaboration",
    "ouelluminate": "communication_collaboration",
    "dataplus": "communication_collaboration",
    # Lookup/reference tool.
    "glossary": "reference_tool",
}

EVENT_CATEGORIES = [
    "content_access",
    "navigation_link",
    "assessment_practice",
    "communication_collaboration",
    "reference_tool",
    "other",
]

VLE_SCHEMA = {
    "code_module": pl.String,
    "code_presentation": pl.String,
    "id_student": pl.Int64,
    "id_site": pl.Int64,
    "date": pl.Int32,
    "sum_click": pl.Int64,
}


def log(message: str) -> None:
    stamp = time.strftime("%H:%M:%S")
    print(f"[{stamp}] {message}", flush=True)


def sha256_file(path: Path, chunk_size: int = 8 * 1024 * 1024) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(chunk_size), b""):
            digest.update(chunk)
    return digest.hexdigest()


def atomic_write_text(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(text, encoding="utf-8", newline="\n")
    os.replace(tmp, path)


def collect(lf: pl.LazyFrame) -> pl.DataFrame:
    return lf.collect(engine="streaming")


def scalar(lf: pl.LazyFrame, column: str) -> Any:
    return collect(lf).item(0, column)


def remove_owned_outputs(paths: Iterable[Path]) -> None:
    for path in paths:
        if path.exists() and path.is_file():
            path.unlink()


def sink(lf: pl.LazyFrame, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    lf.sink_parquet(
        path,
        compression="zstd",
        compression_level=6,
        statistics=True,
        row_group_size=250_000,
        engine="streaming",
        mkdir=True,
    )


def read_small_tables(raw_dir: Path) -> dict[str, pl.DataFrame]:
    courses = pl.read_csv(
        raw_dir / "courses.csv",
        null_values="?",
        schema_overrides={
            "code_module": pl.String,
            "code_presentation": pl.String,
            "module_presentation_length": pl.Int32,
        },
    )
    assessments = pl.read_csv(
        raw_dir / "assessments.csv",
        null_values="?",
        schema_overrides={
            "code_module": pl.String,
            "code_presentation": pl.String,
            "id_assessment": pl.Int64,
            "assessment_type": pl.String,
            "date": pl.Int32,
            "weight": pl.Float64,
        },
    )
    vle = pl.read_csv(
        raw_dir / "vle.csv",
        null_values="?",
        schema_overrides={
            "id_site": pl.Int64,
            "code_module": pl.String,
            "code_presentation": pl.String,
            "activity_type": pl.String,
            "week_from": pl.Int32,
            "week_to": pl.Int32,
        },
    )
    info = pl.read_csv(
        raw_dir / "studentInfo.csv",
        null_values="?",
        schema_overrides={
            "code_module": pl.String,
            "code_presentation": pl.String,
            "id_student": pl.Int64,
            "gender": pl.String,
            "region": pl.String,
            "highest_education": pl.String,
            "imd_band": pl.String,
            "age_band": pl.String,
            "num_of_prev_attempts": pl.Int32,
            "studied_credits": pl.Int32,
            "disability": pl.String,
            "final_result": pl.String,
        },
    )
    registration = pl.read_csv(
        raw_dir / "studentRegistration.csv",
        null_values="?",
        schema_overrides={
            "code_module": pl.String,
            "code_presentation": pl.String,
            "id_student": pl.Int64,
            "date_registration": pl.Int32,
            "date_unregistration": pl.Int32,
        },
    )
    student_assessment = pl.read_csv(
        raw_dir / "studentAssessment.csv",
        null_values="?",
        schema_overrides={
            "id_assessment": pl.Int64,
            "id_student": pl.Int64,
            "date_submitted": pl.Int32,
            "is_banked": pl.Int8,
            "score": pl.Float64,
        },
    )
    return {
        "courses": courses,
        "assessments": assessments,
        "vle": vle,
        "studentInfo": info,
        "studentRegistration": registration,
        "studentAssessment": student_assessment,
    }


def assert_unique(df: pl.DataFrame, keys: list[str], label: str) -> None:
    duplicates = df.group_by(keys).len().filter(pl.col("len") > 1)
    if duplicates.height:
        raise ValueError(
            f"{label} is not unique on {keys}: {duplicates.height} duplicate key groups"
        )


def build_registration_base(tables: dict[str, pl.DataFrame]) -> pl.DataFrame:
    info = tables["studentInfo"]
    registration = tables["studentRegistration"]
    courses = tables["courses"]
    assert_unique(info, KEYS, "studentInfo")
    assert_unique(registration, KEYS, "studentRegistration")
    assert_unique(courses, ["code_module", "code_presentation"], "courses")

    base = registration.join(info, on=KEYS, how="left", validate="1:1").join(
        courses,
        on=["code_module", "code_presentation"],
        how="left",
        validate="m:1",
    )
    if base.height != registration.height:
        raise AssertionError("Registration join changed the number of registrations")
    if base.select(pl.col("final_result").is_null().sum()).item() != 0:
        raise AssertionError("Some registrations have no official final_result")
    if base.select(pl.col("module_presentation_length").is_null().sum()).item() != 0:
        raise AssertionError("Some registrations have no course length")

    base = base.with_columns(
        pl.when(pl.col("date_unregistration").is_null())
        .then(pl.col("module_presentation_length") - 1)
        .otherwise(
            pl.min_horizontal(
                pl.col("date_unregistration"),
                pl.col("module_presentation_length") - 1,
            )
        )
        .cast(pl.Int32)
        .alias("main_window_end_day")
    ).with_columns(
        pl.when(pl.col("main_window_end_day") >= 0)
        .then(pl.col("main_window_end_day") + 1)
        .otherwise(0)
        .cast(pl.Int32)
        .alias("main_sequence_days"),
        pl.concat_str(
            [
                pl.col("code_module"),
                pl.col("code_presentation"),
                pl.col("id_student").cast(pl.String),
            ],
            separator="::",
        ).alias("sequence_id"),
        pl.when(pl.col("date_unregistration").is_null())
        .then(pl.lit("not_unregistered"))
        .when(pl.col("date_unregistration") < 0)
        .then(pl.lit("pre_course"))
        .when(pl.col("date_unregistration") < pl.col("module_presentation_length"))
        .then(pl.lit("during_course"))
        .otherwise(pl.lit("at_or_after_course_end"))
        .alias("unregistration_timing"),
    )
    return base


def prepare_vle_metadata(vle: pl.DataFrame) -> tuple[pl.DataFrame, list[str]]:
    meta_keys = ["code_module", "code_presentation", "id_site"]
    conflicts = (
        vle.group_by(meta_keys)
        .agg(pl.col("activity_type").drop_nulls().n_unique().alias("n_activity_types"))
        .filter(pl.col("n_activity_types") > 1)
    )
    if conflicts.height:
        raise ValueError(f"vle metadata has {conflicts.height} conflicting site definitions")

    meta = (
        vle.select(meta_keys + ["activity_type"])
        .unique(subset=meta_keys, keep="first")
        .with_columns(
            pl.col("activity_type")
            .replace_strict(EVENT_CATEGORY_MAP, default="other")
            .alias("event_category")
        )
    )
    observed_types = sorted(x for x in meta["activity_type"].drop_nulls().unique().to_list())
    unmapped = sorted(set(observed_types) - set(EVENT_CATEGORY_MAP))
    return meta, unmapped


def build_site_day_intermediate(
    raw_dir: Path, meta: pl.DataFrame, intermediate_path: Path
) -> None:
    log("Streaming studentVle.csv and summing all rows at student-course-site-day level")
    raw_lf = pl.scan_csv(
        raw_dir / "studentVle.csv",
        null_values="?",
        schema_overrides=VLE_SCHEMA,
        infer_schema_length=0,
        low_memory=True,
    )
    site_day = (
        raw_lf.group_by(SITE_DAY_KEYS)
        .agg(
            pl.col("sum_click").sum().cast(pl.Int64).alias("sum_click"),
            pl.len().cast(pl.Int32).alias("source_row_count"),
        )
        .join(
            meta.lazy(),
            on=["code_module", "code_presentation", "id_site"],
            how="left",
            validate="m:1",
        )
        .with_columns(
            pl.col("activity_type").fill_null("unknown"),
            pl.col("event_category").fill_null("other"),
        )
        .sort(SITE_DAY_KEYS)
    )
    sink(site_day, intermediate_path)


def aggregate_vle_daily(site_day_lf: pl.LazyFrame) -> pl.LazyFrame:
    expressions: list[pl.Expr] = [
        pl.col("sum_click").sum().cast(pl.Int64).alias("click_count"),
        pl.col("event_category")
        .n_unique()
        .cast(pl.Int16)
        .alias("event_category_diversity"),
        pl.col("activity_type")
        .n_unique()
        .cast(pl.Int16)
        .alias("activity_type_diversity"),
        pl.col("id_site")
        .n_unique()
        .cast(pl.Int32)
        .alias("resource_diversity_n_sites"),
        pl.col("source_row_count").sum().cast(pl.Int32).alias("source_vle_rows"),
    ]
    for category in EVENT_CATEGORIES:
        expressions.append(
            pl.when(pl.col("event_category") == category)
            .then(pl.col("sum_click"))
            .otherwise(0)
            .sum()
            .cast(pl.Int64)
            .alias(f"clicks_{category}")
        )
    return site_day_lf.group_by(DAY_KEYS).agg(expressions)


def aggregate_revisits(site_day_lf: pl.LazyFrame) -> pl.LazyFrame:
    site_keys = KEYS + ["id_site"]
    return (
        site_day_lf.select(SITE_DAY_KEYS)
        .sort(site_keys + ["date"])
        .with_columns(
            pl.col("date").shift(1).over(site_keys).alias("previous_visit_day")
        )
        .with_columns(
            (pl.col("date") - pl.col("previous_visit_day")).alias(
                "days_since_previous_site_visit"
            )
        )
        .with_columns(
            pl.col("days_since_previous_site_visit")
            .is_between(1, 7, closed="both")
            .cast(pl.Int8)
            .alias("site_revisited_1_7d")
        )
        .group_by(DAY_KEYS)
        .agg(
            pl.col("site_revisited_1_7d")
            .sum()
            .cast(pl.Int32)
            .alias("revisit_site_count_1_7d")
        )
    )


def build_assessment_daily(
    tables: dict[str, pl.DataFrame], base: pl.DataFrame
) -> tuple[pl.LazyFrame, pl.LazyFrame]:
    assessments = tables["assessments"]
    student_assessment = tables["studentAssessment"]
    assert_unique(assessments, ["id_assessment"], "assessments")

    assessment_events = (
        student_assessment.join(
            assessments.select(
                [
                    "id_assessment",
                    "code_module",
                    "code_presentation",
                    "assessment_type",
                    pl.col("date").alias("assessment_due_day"),
                    "weight",
                ]
            ),
            on="id_assessment",
            how="left",
            validate="m:1",
        )
        .rename({"date_submitted": "date"})
        .with_columns(pl.col("date").cast(pl.Int32))
    )
    if assessment_events.select(pl.col("code_module").is_null().sum()).item() != 0:
        raise AssertionError("Some studentAssessment rows could not be linked to assessments")

    daily = assessment_events.lazy().group_by(DAY_KEYS).agg(
        pl.len().cast(pl.Int16).alias("assessment_submission_records_n"),
        pl.col("id_assessment")
        .n_unique()
        .cast(pl.Int16)
        .alias("assessment_items_n"),
        (pl.col("is_banked") == 0)
        .sum()
        .cast(pl.Int16)
        .alias("assessment_active_submissions_n"),
        (pl.col("is_banked") == 1)
        .sum()
        .cast(pl.Int16)
        .alias("assessment_banked_n"),
        pl.col("score").count().cast(pl.Int16).alias("assessment_scores_observed_n"),
        pl.col("score").mean().alias("assessment_score_mean"),
        pl.col("score").min().alias("assessment_score_min"),
        pl.col("score").max().alias("assessment_score_max"),
        pl.col("weight").filter(pl.col("score").is_not_null()).sum().alias(
            "assessment_weight_observed_sum"
        ),
        (pl.col("assessment_type") == "TMA")
        .sum()
        .cast(pl.Int16)
        .alias("assessment_tma_n"),
        (pl.col("assessment_type") == "CMA")
        .sum()
        .cast(pl.Int16)
        .alias("assessment_cma_n"),
        (pl.col("assessment_type") == "Exam")
        .sum()
        .cast(pl.Int16)
        .alias("assessment_exam_n"),
    )

    event_with_window = assessment_events.lazy().join(
        base.lazy().select(
            KEYS
            + [
                "date_unregistration",
                "module_presentation_length",
                "main_window_end_day",
            ]
        ),
        on=KEYS,
        how="left",
        validate="m:1",
    )
    return daily, event_with_window


def build_main_panel(
    base: pl.DataFrame,
    vle_daily: pl.LazyFrame,
    revisits: pl.LazyFrame,
    assessment_daily: pl.LazyFrame,
) -> pl.LazyFrame:
    panel = (
        base.lazy()
        .with_columns(
            pl.int_ranges(
                pl.lit(0),
                pl.max_horizontal(pl.lit(0), pl.col("main_window_end_day") + 1),
                dtype=pl.Int32,
            ).alias("date")
        )
        .explode("date", empty_as_null=True)
        .filter(pl.col("date").is_not_null())
        .join(vle_daily, on=DAY_KEYS, how="left", validate="1:1")
        .join(revisits, on=DAY_KEYS, how="left", validate="1:1")
        .join(assessment_daily, on=DAY_KEYS, how="left", validate="1:1")
    )

    zero_columns = [
        "click_count",
        "event_category_diversity",
        "activity_type_diversity",
        "resource_diversity_n_sites",
        "source_vle_rows",
        *[f"clicks_{category}" for category in EVENT_CATEGORIES],
        "revisit_site_count_1_7d",
        "assessment_submission_records_n",
        "assessment_items_n",
        "assessment_active_submissions_n",
        "assessment_banked_n",
        "assessment_scores_observed_n",
        "assessment_weight_observed_sum",
        "assessment_tma_n",
        "assessment_cma_n",
        "assessment_exam_n",
    ]
    panel = panel.with_columns([pl.col(column).fill_null(0) for column in zero_columns])
    panel = panel.with_columns(
        (pl.col("date") // 7 + 1).cast(pl.Int16).alias("course_week"),
        pl.when(pl.col("module_presentation_length") > 1)
        .then(pl.col("date") / (pl.col("module_presentation_length") - 1))
        .otherwise(0.0)
        .alias("course_progress_fraction"),
        (pl.col("click_count") > 0).alias("vle_active_flag"),
        (pl.col("assessment_submission_records_n") > 0).alias(
            "assessment_record_flag"
        ),
        (pl.col("assessment_active_submissions_n") > 0).alias(
            "assessment_output_flag"
        ),
        (
            (pl.col("click_count") == 0)
            & (pl.col("assessment_submission_records_n") == 0)
        ).alias("platform_inactive_flag"),
        pl.when(
            (pl.col("click_count") == 0)
            & (pl.col("assessment_submission_records_n") == 0)
        )
        .then(pl.lit("Platform-Inactive"))
        .otherwise(pl.lit("Platform-Active"))
        .alias("platform_observation"),
        pl.when(pl.col("resource_diversity_n_sites") > 0)
        .then(
            pl.col("revisit_site_count_1_7d")
            / pl.col("resource_diversity_n_sites")
        )
        .otherwise(0.0)
        .alias("revisit_site_share_1_7d"),
        (pl.col("revisit_site_count_1_7d") > 0).alias("revisit_flag_1_7d"),
        # OULAD records only integer course-day offsets, not within-day timestamps.
        pl.lit(None, dtype=pl.Float64).alias("time_regularity"),
        pl.lit(False).alias("time_regularity_applicable"),
    )
    return panel.select(
        [
            "sequence_id",
            *KEYS,
            "date",
            "course_week",
            "course_progress_fraction",
            "date_registration",
            "date_unregistration",
            "module_presentation_length",
            "main_window_end_day",
            "main_sequence_days",
            "unregistration_timing",
            "final_result",
            "gender",
            "region",
            "highest_education",
            "imd_band",
            "age_band",
            "num_of_prev_attempts",
            "studied_credits",
            "disability",
            "click_count",
            "event_category_diversity",
            "activity_type_diversity",
            "resource_diversity_n_sites",
            "source_vle_rows",
            *[f"clicks_{category}" for category in EVENT_CATEGORIES],
            "revisit_site_count_1_7d",
            "revisit_site_share_1_7d",
            "revisit_flag_1_7d",
            "assessment_submission_records_n",
            "assessment_items_n",
            "assessment_active_submissions_n",
            "assessment_banked_n",
            "assessment_scores_observed_n",
            "assessment_score_mean",
            "assessment_score_min",
            "assessment_score_max",
            "assessment_weight_observed_sum",
            "assessment_tma_n",
            "assessment_cma_n",
            "assessment_exam_n",
            "vle_active_flag",
            "assessment_record_flag",
            "assessment_output_flag",
            "platform_inactive_flag",
            "platform_observation",
            "time_regularity",
            "time_regularity_applicable",
        ]
    ).sort(KEYS + ["date"])


def build_audit_daily(site_day_with_base: pl.LazyFrame, condition: pl.Expr) -> pl.LazyFrame:
    return (
        aggregate_vle_daily(site_day_with_base.filter(condition))
        .join(
            site_day_with_base.select(
                KEYS
                + [
                    "date_unregistration",
                    "module_presentation_length",
                    "main_window_end_day",
                    "final_result",
                ]
            ).unique(subset=KEYS),
            on=KEYS,
            how="left",
            validate="m:1",
        )
        .sort(KEYS + ["date"])
    )


def registration_period_summary(
    events: pl.LazyFrame,
    condition: pl.Expr,
    prefix: str,
    click_column: str | None = None,
) -> pl.LazyFrame:
    filtered = events.filter(condition)
    expressions: list[pl.Expr] = [
        pl.col("date").n_unique().cast(pl.Int32).alias(f"{prefix}_active_days"),
        pl.len().cast(pl.Int32).alias(f"{prefix}_records"),
    ]
    if click_column:
        expressions.append(
            pl.col(click_column).sum().cast(pl.Int64).alias(f"{prefix}_clicks")
        )
    return filtered.group_by(KEYS).agg(expressions)


def build_sensitivity_table(
    base: pl.DataFrame,
    site_day_with_base: pl.LazyFrame,
    assessment_with_window: pl.LazyFrame,
) -> pl.LazyFrame:
    pre = registration_period_summary(
        site_day_with_base, pl.col("date") < 0, "pre_course_vle", "sum_click"
    )
    post_unregistration = registration_period_summary(
        site_day_with_base,
        pl.col("date_unregistration").is_not_null()
        & (pl.col("date") > pl.col("date_unregistration")),
        "post_unregistration_vle",
        "sum_click",
    )
    post_course = registration_period_summary(
        site_day_with_base,
        pl.col("date") >= pl.col("module_presentation_length"),
        "post_course_vle",
        "sum_click",
    )
    pre_assessment = registration_period_summary(
        assessment_with_window, pl.col("date") < 0, "pre_course_assessment"
    )
    post_unreg_assessment = registration_period_summary(
        assessment_with_window,
        pl.col("date_unregistration").is_not_null()
        & (pl.col("date") > pl.col("date_unregistration")),
        "post_unregistration_assessment",
    )
    post_course_assessment = registration_period_summary(
        assessment_with_window,
        pl.col("date") >= pl.col("module_presentation_length"),
        "post_course_assessment",
    )

    sensitivity = base.lazy().select(
        [
            "sequence_id",
            *KEYS,
            "date_registration",
            "date_unregistration",
            "module_presentation_length",
            "main_window_end_day",
            "main_sequence_days",
            "unregistration_timing",
            "final_result",
        ]
    )
    for frame in [
        pre,
        post_unregistration,
        post_course,
        pre_assessment,
        post_unreg_assessment,
        post_course_assessment,
    ]:
        sensitivity = sensitivity.join(frame, on=KEYS, how="left", validate="1:1")

    numeric_audit = [
        "pre_course_vle_active_days",
        "pre_course_vle_records",
        "pre_course_vle_clicks",
        "post_unregistration_vle_active_days",
        "post_unregistration_vle_records",
        "post_unregistration_vle_clicks",
        "post_course_vle_active_days",
        "post_course_vle_records",
        "post_course_vle_clicks",
        "pre_course_assessment_active_days",
        "pre_course_assessment_records",
        "post_unregistration_assessment_active_days",
        "post_unregistration_assessment_records",
        "post_course_assessment_active_days",
        "post_course_assessment_records",
    ]
    return (
        sensitivity.with_columns([pl.col(column).fill_null(0) for column in numeric_audit])
        .with_columns(
            (pl.col("main_sequence_days") > 0).alias("main_sequence_available"),
            (pl.col("pre_course_vle_active_days") > 0).alias(
                "has_pre_course_vle_activity"
            ),
            (pl.col("post_unregistration_vle_active_days") > 0).alias(
                "has_post_unregistration_vle_activity"
            ),
            (pl.col("post_course_vle_active_days") > 0).alias(
                "has_post_course_vle_activity"
            ),
        )
        .sort(KEYS)
    )


def dataframe_counts(df: pl.DataFrame, column: str) -> dict[str, int]:
    return {
        str(row[column]): int(row["len"])
        for row in df.group_by(column).len().sort(column).iter_rows(named=True)
    }


def write_dictionary(path: Path) -> None:
    text = """# OULAD 日级建模表字段字典

## 口径

- 主表：`oulad_daily.parquet`。
- 分析单位：学生—课程模块—开课批次—课程日。
- 主日窗：课程日 `0` 至 `min(date_unregistration, module_presentation_length-1)`；未注销者至课程末。注销日在 0 日以前的注册没有合法主日窗，保留在敏感性表而不虚构第 0 日。
- `Platform-Inactive` 是“该日未记录 VLE 点击且未记录 assessment 提交”的**平台观测**，不是“没有学习”或任何隐状态真值。
- OULAD 只有整数课程日偏移，没有日内时刻。因此 `time_regularity` 全部为缺失，`time_regularity_applicable=false`；不得由该数据估计“固定学习时段”。
- 同一学生、课程、站点、日期出现多行时，对 `sum_click` 求和；不删除所谓“重复行”。

## 主表字段

| 字段 | 类型/单位 | 定义 |
|---|---|---|
| sequence_id | string | `module::presentation::student`，学生—课程注册序列标识。 |
| code_module, code_presentation, id_student | key | 课程、开课批次、匿名学生键。 |
| date | int/day | 相对课程开始日；主表仅含非负日且不超过该注册的主窗终点。 |
| course_week | int | `floor(date/7)+1`。 |
| course_progress_fraction | float | `date/(course_length-1)`。 |
| date_registration, date_unregistration | int/day | 官方注册/注销相对日；`date_unregistration` 可缺失。 |
| module_presentation_length | int/day | 官方开课长度。 |
| main_window_end_day | int/day | `min(注销日, 课程长度-1)`；无注销时为课程长度减一。 |
| main_sequence_days | int/day | 主序列日数；注销日在开课前时为 0。 |
| unregistration_timing | category | 未注销、课前注销、课中注销、课程末及以后注销。 |
| final_result | category | OULAD 官方 `Fail/Pass/Distinction/Withdrawn`，没有重编码为吸收状态。 |
| gender...disability | category/numeric | `studentInfo` 的官方匿名背景字段，用于分层/异质性分析。 |
| click_count | clicks/day | 当日所有站点 `sum_click` 的总和。 |
| event_category_diversity | count/day | 当日统一事件大类的去重数量。 |
| activity_type_diversity | count/day | 当日原始 OULAD `activity_type` 的去重数量。 |
| resource_diversity_n_sites | count/day | 当日去重 `id_site` 数量，作为资源多样性。 |
| source_vle_rows | count/day | 聚合前贡献到该日的 `studentVle` 原始记录数。 |
| clicks_content_access | clicks/day | 内容/资源访问点击。 |
| clicks_navigation_link | clicks/day | 首页、链接、重复活动等导航点击。 |
| clicks_assessment_practice | clicks/day | quiz/externalquiz/questionnaire 点击。 |
| clicks_communication_collaboration | clicks/day | 论坛、协作、wiki 等点击。 |
| clicks_reference_tool | clicks/day | glossary 点击。 |
| clicks_other | clicks/day | 元数据缺失或未映射类型的点击；正常应为 0 或极少。 |
| revisit_site_count_1_7d | count/day | 当日站点中，最近一次访问距当日 1–7 日的站点数。 |
| revisit_site_share_1_7d | [0,1] | 上述复访站点数/当日站点数。 |
| revisit_flag_1_7d | bool | 当日至少有一个 1–7 日复访站点。 |
| assessment_submission_records_n | count/day | `studentAssessment` 当日记录数，含 banked。 |
| assessment_items_n | count/day | 当日去重 assessment 数。 |
| assessment_active_submissions_n | count/day | `is_banked=0` 的提交记录数。 |
| assessment_banked_n | count/day | `is_banked=1` 的记录数，单列避免误作新产出。 |
| assessment_scores_observed_n | count/day | 非缺失 score 数。 |
| assessment_score_mean/min/max | score 0–100 | 当日已观测成绩统计；无提交时保持缺失。 |
| assessment_weight_observed_sum | weight | 有成绩 assessment 的官方权重合计。 |
| assessment_tma_n/cma_n/exam_n | count/day | 各 assessment 类型记录数。 |
| vle_active_flag | bool | `click_count>0`。 |
| assessment_record_flag | bool | 当日存在任意 assessment 记录。 |
| assessment_output_flag | bool | 当日存在非 banked assessment 提交。 |
| platform_inactive_flag | bool | 点击与 assessment 记录同时为 0。 |
| platform_observation | category | `Platform-Active` 或 `Platform-Inactive`。 |
| time_regularity | null | 不适用于 OULAD；有意全缺失。 |
| time_regularity_applicable | bool | 恒为 false。 |

## 统一事件大类映射

| 大类 | 原始 activity_type |
|---|---|
| content_access | resource, oucontent, subpage, page, folder, htmlactivity, sharedsubpage, dualpane |
| navigation_link | homepage, url, repeatactivity |
| assessment_practice | quiz, externalquiz, questionnaire |
| communication_collaboration | forumng, oucollaborate, ouwiki, ouelluminate, dataplus |
| reference_tool | glossary |
| other | 未知或元数据缺失（审计兜底，不作实质解释） |

## 窗外与敏感性表

- `oulad_pre_course_activity.parquet`：所有 `date<0` 的 VLE 活跃日，只作课前行为审计/敏感性分析。
- `oulad_post_unregistration_activity.parquet`：所有有注销日期且 `date>date_unregistration` 的 VLE 活跃日；不并入主序列。
- `oulad_out_of_window_sensitivity.parquet`：每个注册一行，汇总课前、注销后、课程结束后的 VLE 与 assessment 记录；也保留课前已注销、因而主序列日数为 0 的注册。
"""
    atomic_write_text(path, text)


def build_quality_report(
    raw_dir: Path,
    out_paths: dict[str, Path],
    tables: dict[str, pl.DataFrame],
    base: pl.DataFrame,
    unmapped_activity_types: list[str],
    started_at: float,
) -> dict[str, Any]:
    raw_vle = pl.scan_csv(
        raw_dir / "studentVle.csv",
        null_values="?",
        schema_overrides=VLE_SCHEMA,
        infer_schema_length=0,
        low_memory=True,
    )
    raw_vle_stats = collect(
        raw_vle.select(
            pl.len().alias("rows"),
            pl.col("sum_click").sum().alias("click_sum"),
            pl.col("date").min().alias("min_date"),
            pl.col("date").max().alias("max_date"),
        )
    ).row(0, named=True)

    site_day = pl.scan_parquet(out_paths["intermediate"])
    site_day_stats = collect(
        site_day.select(
            pl.len().alias("rows"),
            pl.col("sum_click").sum().alias("click_sum"),
            (pl.col("source_row_count") > 1).sum().alias("multirow_key_groups"),
            pl.col("source_row_count").max().alias("max_rows_per_key"),
            pl.col("activity_type").eq("unknown").sum().alias("missing_site_metadata_rows"),
        )
    ).row(0, named=True)

    daily = pl.scan_parquet(out_paths["daily"])
    daily_stats = collect(
        daily.select(
            pl.len().alias("rows"),
            pl.struct(KEYS + ["date"]).n_unique().alias("unique_keys"),
            pl.col("sequence_id").n_unique().alias("sequences"),
            pl.col("date").min().alias("min_date"),
            pl.col("date").max().alias("max_date"),
            (pl.col("date") > pl.col("main_window_end_day")).sum().alias(
                "rows_after_registration_window"
            ),
            pl.col("click_count").sum().alias("main_window_click_sum"),
            pl.col("platform_inactive_flag").sum().alias("platform_inactive_days"),
            pl.col("vle_active_flag").sum().alias("vle_active_days"),
            pl.col("assessment_record_flag").sum().alias("assessment_record_days"),
            pl.col("assessment_output_flag").sum().alias("assessment_output_days"),
            pl.col("assessment_submission_records_n").sum().alias(
                "assessment_records_in_main_window"
            ),
            pl.col("time_regularity").is_null().sum().alias(
                "time_regularity_null_rows"
            ),
            pl.col("assessment_score_mean").min().alias("min_daily_score_mean"),
            pl.col("assessment_score_mean").max().alias("max_daily_score_mean"),
        )
    ).row(0, named=True)

    pre = pl.scan_parquet(out_paths["pre"])
    post = pl.scan_parquet(out_paths["post"])
    sensitivity = pl.scan_parquet(out_paths["sensitivity"])
    pre_stats = collect(
        pre.select(
            pl.len().alias("active_day_rows"),
            pl.col("click_count").sum().alias("click_sum"),
            pl.struct(KEYS).n_unique().alias("registrations"),
        )
    ).row(0, named=True)
    post_stats = collect(
        post.select(
            pl.len().alias("active_day_rows"),
            pl.col("click_count").sum().alias("click_sum"),
            pl.struct(KEYS).n_unique().alias("registrations"),
        )
    ).row(0, named=True)
    sensitivity_stats = collect(
        sensitivity.select(
            pl.len().alias("registrations"),
            pl.col("main_sequence_available").sum().alias(
                "registrations_with_main_sequence"
            ),
            (~pl.col("main_sequence_available")).sum().alias(
                "zero_length_pre_start_registrations"
            ),
            pl.col("has_pre_course_vle_activity").sum().alias(
                "registrations_with_pre_course_vle"
            ),
            pl.col("has_post_unregistration_vle_activity").sum().alias(
                "registrations_with_post_unregistration_vle"
            ),
            pl.col("has_post_course_vle_activity").sum().alias(
                "registrations_with_post_course_vle"
            ),
            pl.col("pre_course_assessment_records").sum().alias(
                "pre_course_assessment_records"
            ),
            pl.col("post_unregistration_assessment_records").sum().alias(
                "post_unregistration_assessment_records"
            ),
            pl.col("post_course_assessment_records").sum().alias(
                "post_course_assessment_records"
            ),
        )
    ).row(0, named=True)

    expected_panel_rows = int(base["main_sequence_days"].sum())
    final_result_counts = dataframe_counts(base, "final_result")
    course_counts = dataframe_counts(base, "code_module")
    presentation_counts = dataframe_counts(base, "code_presentation")
    early_during_course = base.filter(
        pl.col("date_unregistration").is_not_null()
        & (pl.col("date_unregistration") >= 0)
        & (pl.col("date_unregistration") < pl.col("module_presentation_length"))
    ).height
    source_files = {}
    for name in [
        "courses.csv",
        "assessments.csv",
        "vle.csv",
        "studentInfo.csv",
        "studentRegistration.csv",
        "studentAssessment.csv",
        "studentVle.csv",
    ]:
        path = raw_dir / name
        source_files[name] = {
            "bytes": path.stat().st_size,
            "sha256": sha256_file(path),
        }

    checks = {
        "all_official_final_results_allowed": set(final_result_counts)
        == {"Fail", "Pass", "Distinction", "Withdrawn"},
        "vle_clicks_conserved_by_site_day_aggregation": int(raw_vle_stats["click_sum"])
        == int(site_day_stats["click_sum"]),
        "panel_row_count_matches_registration_windows": int(daily_stats["rows"])
        == expected_panel_rows,
        "panel_student_course_day_key_unique": int(daily_stats["rows"])
        == int(daily_stats["unique_keys"]),
        "main_panel_has_no_negative_days": int(daily_stats["min_date"]) >= 0,
        "main_panel_respects_each_registration_window": int(
            daily_stats["rows_after_registration_window"]
        )
        == 0,
        "time_regularity_all_missing_by_design": int(
            daily_stats["time_regularity_null_rows"]
        )
        == int(daily_stats["rows"]),
        "all_activity_types_mapped": len(unmapped_activity_types) == 0,
        "sensitivity_table_retains_all_registrations": int(
            sensitivity_stats["registrations"]
        )
        == base.height,
    }
    if not all(checks.values()):
        failed = [name for name, passed in checks.items() if not passed]
        raise AssertionError(f"Data-quality checks failed: {failed}")

    report: dict[str, Any] = {
        "schema_version": "1.0",
        "generated_at_local": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
        "runtime": {
            "python": sys.version.split()[0],
            "polars": pl.__version__,
            "platform": platform.platform(),
            "elapsed_seconds": round(time.time() - started_at, 3),
        },
        "scope": {
            "unit": "student-module-presentation-course_day",
            "main_window": "date 0 through min(date_unregistration, module_presentation_length-1); missing unregistration uses course end",
            "pre_course_rule": "date < 0; audit only",
            "post_unregistration_rule": "non-null date_unregistration and date > date_unregistration; audit only",
            "platform_inactive_definition": "click_count == 0 and assessment_submission_records_n == 0; observation, not latent-state truth",
            "time_regularity": "not applicable: OULAD lacks within-day timestamps",
        },
        "source_files": source_files,
        "input_rows": {
            "courses": tables["courses"].height,
            "assessments": tables["assessments"].height,
            "vle": tables["vle"].height,
            "studentInfo": tables["studentInfo"].height,
            "studentRegistration": tables["studentRegistration"].height,
            "studentAssessment": tables["studentAssessment"].height,
            "studentVle": int(raw_vle_stats["rows"]),
        },
        "vle_aggregation": {
            "raw_rows": int(raw_vle_stats["rows"]),
            "aggregated_student_course_site_day_rows": int(site_day_stats["rows"]),
            "rows_collapsed_by_key_sum": int(raw_vle_stats["rows"])
            - int(site_day_stats["rows"]),
            "site_day_keys_with_multiple_source_rows": int(
                site_day_stats["multirow_key_groups"]
            ),
            "max_source_rows_for_one_site_day_key": int(
                site_day_stats["max_rows_per_key"]
            ),
            "raw_click_sum": int(raw_vle_stats["click_sum"]),
            "aggregated_click_sum": int(site_day_stats["click_sum"]),
            "raw_date_range": [
                int(raw_vle_stats["min_date"]),
                int(raw_vle_stats["max_date"]),
            ],
            "missing_site_metadata_rows_after_join": int(
                site_day_stats["missing_site_metadata_rows"]
            ),
            "method_note": "All sum_click contributions were summed by student/module/presentation/site/date; no duplicate row was discarded.",
        },
        "event_mapping": {
            "mapping": EVENT_CATEGORY_MAP,
            "unmapped_observed_activity_types": unmapped_activity_types,
        },
        "registrations": {
            "total": base.height,
            "by_final_result": final_result_counts,
            "by_module": course_counts,
            "by_presentation": presentation_counts,
            "with_main_sequence": int(
                sensitivity_stats["registrations_with_main_sequence"]
            ),
            "zero_length_because_unregistered_before_day_0": int(
                sensitivity_stats["zero_length_pre_start_registrations"]
            ),
            "early_unregistration_day_0_through_course_end_minus_1": early_during_course,
        },
        "main_daily_panel": {
            "rows": int(daily_stats["rows"]),
            "expected_rows_from_windows": expected_panel_rows,
            "sequences": int(daily_stats["sequences"]),
            "date_range": [int(daily_stats["min_date"]), int(daily_stats["max_date"])],
            "main_window_click_sum": int(daily_stats["main_window_click_sum"]),
            "vle_active_days": int(daily_stats["vle_active_days"]),
            "platform_inactive_days": int(daily_stats["platform_inactive_days"]),
            "platform_inactive_share": round(
                int(daily_stats["platform_inactive_days"]) / int(daily_stats["rows"]),
                8,
            ),
            "assessment_record_days": int(daily_stats["assessment_record_days"]),
            "assessment_output_days": int(daily_stats["assessment_output_days"]),
            "assessment_records": int(
                daily_stats["assessment_records_in_main_window"]
            ),
            "daily_assessment_score_mean_range": [
                float(daily_stats["min_daily_score_mean"]),
                float(daily_stats["max_daily_score_mean"]),
            ],
        },
        "out_of_window_audit": {
            "pre_course_vle": {
                key: int(value) for key, value in pre_stats.items()
            },
            "post_unregistration_vle": {
                key: int(value) for key, value in post_stats.items()
            },
            "registration_level": {
                key: int(value) for key, value in sensitivity_stats.items()
            },
            "overlap_note": "For a registration unregistered before day 0, an event can be both pre-course and post-unregistration; audit definitions are intentionally independent.",
        },
        "checks": checks,
        "outputs": {
            key: {
                "path": str(path.resolve()),
                "bytes": path.stat().st_size,
                "sha256": sha256_file(path),
            }
            for key, path in out_paths.items()
            if key != "intermediate" and path.exists()
        },
        "interpretation_limits": [
            "Platform-Inactive cannot distinguish off-platform learning (for example coding) from disengagement.",
            "OULAD daily offsets cannot identify a stable within-day study-time strategy.",
            "final_result is an observed course outcome and must not be manufactured as a daily absorbing state at the last active day.",
            "Scenario interventions on transition/emission parameters are conditional simulations, not causal effects.",
        ],
    }
    return report


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--raw-dir",
        type=Path,
        default=Path("crawled_data/raw/oulad"),
        help="Directory containing the seven OULAD CSV files.",
    )
    parser.add_argument(
        "--out-dir",
        type=Path,
        default=Path("paper_output/data_cleaned"),
        help="Output directory.",
    )
    parser.add_argument(
        "--keep-intermediate",
        action="store_true",
        help="Keep the site-day intermediate parquet for diagnostics.",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    raw_dir = args.raw_dir.resolve()
    out_dir = args.out_dir.resolve()
    required = [
        "courses.csv",
        "assessments.csv",
        "vle.csv",
        "studentInfo.csv",
        "studentRegistration.csv",
        "studentAssessment.csv",
        "studentVle.csv",
    ]
    missing = [name for name in required if not (raw_dir / name).is_file()]
    if missing:
        raise FileNotFoundError(f"Missing OULAD files under {raw_dir}: {missing}")

    out_dir.mkdir(parents=True, exist_ok=True)
    out_paths = {
        "daily": out_dir / "oulad_daily.parquet",
        "pre": out_dir / "oulad_pre_course_activity.parquet",
        "post": out_dir / "oulad_post_unregistration_activity.parquet",
        "sensitivity": out_dir / "oulad_out_of_window_sensitivity.parquet",
        "quality": out_dir / "oulad_data_quality.json",
        "dictionary": out_dir / "oulad_field_dictionary.md",
        "intermediate": out_dir / "oulad_site_day_intermediate.parquet",
    }
    remove_owned_outputs(out_paths.values())
    started_at = time.time()
    log(f"Raw directory: {raw_dir}")
    log(f"Output directory: {out_dir}")

    try:
        tables = read_small_tables(raw_dir)
        base = build_registration_base(tables)
        log(f"Loaded {base.height:,} registrations with official final_result")

        vle_meta, unmapped = prepare_vle_metadata(tables["vle"])
        build_site_day_intermediate(raw_dir, vle_meta, out_paths["intermediate"])
        site_day = pl.scan_parquet(out_paths["intermediate"])

        vle_daily = aggregate_vle_daily(site_day)
        revisits = aggregate_revisits(site_day)
        assessment_daily, assessment_with_window = build_assessment_daily(tables, base)

        log("Building complete student-course-day panel with explicit inactive days")
        main_panel = build_main_panel(base, vle_daily, revisits, assessment_daily)
        sink(main_panel, out_paths["daily"])

        site_day_with_base = site_day.join(
            base.lazy().select(
                KEYS
                + [
                    "date_unregistration",
                    "module_presentation_length",
                    "main_window_end_day",
                    "final_result",
                ]
            ),
            on=KEYS,
            how="left",
            validate="m:1",
        )
        log("Writing pre-course and post-unregistration VLE audit tables")
        pre_daily = build_audit_daily(site_day_with_base, pl.col("date") < 0)
        post_daily = build_audit_daily(
            site_day_with_base,
            pl.col("date_unregistration").is_not_null()
            & (pl.col("date") > pl.col("date_unregistration")),
        ).with_columns(
            (pl.col("date") - pl.col("date_unregistration"))
            .cast(pl.Int32)
            .alias("days_after_unregistration")
        )
        sink(pre_daily, out_paths["pre"])
        sink(post_daily, out_paths["post"])

        log("Writing registration-level window sensitivity table")
        sensitivity = build_sensitivity_table(
            base, site_day_with_base, assessment_with_window
        )
        sink(sensitivity, out_paths["sensitivity"])

        write_dictionary(out_paths["dictionary"])
        log("Running conservation, uniqueness, window, and coverage checks")
        report = build_quality_report(
            raw_dir, out_paths, tables, base, unmapped, started_at
        )
        atomic_write_text(
            out_paths["quality"],
            json.dumps(report, ensure_ascii=False, indent=2) + "\n",
        )
        log(
            "PASS: "
            f"{report['main_daily_panel']['rows']:,} daily rows, "
            f"{report['main_daily_panel']['sequences']:,} non-empty sequences, "
            f"{report['registrations']['zero_length_because_unregistered_before_day_0']:,} "
            "pre-start withdrawals retained in sensitivity output"
        )
        return 0
    finally:
        if not args.keep_intermediate and out_paths["intermediate"].exists():
            out_paths["intermediate"].unlink()


if __name__ == "__main__":
    raise SystemExit(main())
