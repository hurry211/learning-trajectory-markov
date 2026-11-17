"""Q2: leakage-resistant OULAD replication and external-validity analysis.

This script deliberately keeps hidden states numbered and treats
``Platform-Inactive`` as a platform observation.  It implements:

* an outcome-blind capacity sample for categorical-HMM estimation;
* nested presentation-based model selection with an untouched 2014J test;
* observable Markov-chain (MC) baselines that are allowed to win;
* history-conditioned prediction of subsequent observations;
* fixed landmark outcome analyses (28+28 days; 14/56 sensitivities);
* student-cluster bootstrap uncertainty for AUC/Brier scores;
* leave-module-out external validation with fold-internal K selection; and
* an auditable discrete-time administrative-unregistration risk model.

No model output is interpreted as a cognitive-state truth or a causal strategy
effect.  Pre-course unregistrations are retained in population bounds and are
never manufactured into a day-zero state.
"""

from __future__ import annotations

import argparse
import gc
import hashlib
import json
import math
import platform
import sys
import time
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable, Sequence

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.special import expit
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import brier_score_loss, log_loss, roc_auc_score
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler


SCRIPT_PATH = Path(__file__).resolve()
PROJECT_ROOT = SCRIPT_PATH.parents[3]
PANEL_PATH = PROJECT_ROOT / "paper_output" / "data_cleaned" / "oulad_daily.parquet"
SENSITIVITY_PATH = (
    PROJECT_ROOT
    / "paper_output"
    / "data_cleaned"
    / "oulad_out_of_window_sensitivity.parquet"
)
POST_ACTIVITY_PATH = (
    PROJECT_ROOT
    / "paper_output"
    / "data_cleaned"
    / "oulad_post_unregistration_activity.parquet"
)
QUALITY_PATH = (
    PROJECT_ROOT / "paper_output" / "data_cleaned" / "oulad_data_quality.json"
)
RESULTS_DIR = PROJECT_ROOT / "paper_output" / "results" / "q2"
TABLES_DIR = PROJECT_ROOT / "paper_output" / "tables" / "q2"
FIGURES_DIR = PROJECT_ROOT / "paper_output" / "figures"

sys.path.insert(0, str(SCRIPT_PATH.parent))
from sequence_models import (  # noqa: E402
    ObservationEncodingConfig,
    TransparentObservationEncoder,
    align_hmm_to_reference,
    emission_profile_frame,
    fit_categorical_hmm_grid,
    fit_dirichlet_markov_chain,
    fit_mc_hmm_comparison,
    hungarian_state_alignment,
    make_sequence_bundle,
)


SCHEMA_VERSION = "1.0"
OBS_LABELS = [
    "Platform-Inactive",
    "Low-Diversity-Content-Consumption",
    "Diverse-Interaction",
    "Assessment-or-Production",
]
MAIN_LANDMARK = 28
MAIN_HORIZON = 28
MAIN_PRESENTATION_TRAIN = ("2013B", "2013J")
MAIN_PRESENTATION_INNER_VALIDATION = "2014B"
MAIN_PRESENTATION_OUTER_TEST = "2014J"
MC_IMPROVEMENT_MARGIN = 0.001


@dataclass
class FittedRepresentation:
    encoder: TransparentObservationEncoder
    mc_model: Any
    hmm_model: Any | None
    chosen_kind: str
    selected_k: int | None
    inner_delta_vs_mc: float | None
    selection_status: str
    refit_status: str


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def artifact_record(path: Path) -> dict[str, Any]:
    return {
        "path": str(path.relative_to(PROJECT_ROOT)).replace("\\", "/"),
        "bytes": int(path.stat().st_size),
        "sha256": sha256_file(path),
    }


def _json_ready(value: Any) -> Any:
    if isinstance(value, dict):
        return {str(key): _json_ready(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_ready(item) for item in value]
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, (np.integer,)):
        return int(value)
    if isinstance(value, (np.floating, float)):
        result = float(value)
        return result if math.isfinite(result) else None
    if isinstance(value, (np.bool_, bool)):
        return bool(value)
    if pd.isna(value) if not isinstance(value, (str, bytes)) else False:
        return None
    return value


def write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        json.dump(_json_ready(payload), handle, ensure_ascii=False, indent=2, allow_nan=False)
        handle.write("\n")


def write_table(name: str, frame: pd.DataFrame, output_paths: list[Path]) -> Path:
    path = TABLES_DIR / name
    path.parent.mkdir(parents=True, exist_ok=True)
    frame.to_csv(path, index=False, encoding="utf-8-sig")
    output_paths.append(path)
    return path


def stable_hash(text: str, seed: int) -> str:
    return hashlib.sha256(f"{seed}|{text}".encode("utf-8")).hexdigest()


def hash_bucket(text: str, seed: int, modulus: int) -> int:
    return int(stable_hash(text, seed)[:16], 16) % modulus


def load_registry() -> pd.DataFrame:
    registry = pd.read_parquet(SENSITIVITY_PATH)
    registry = registry.copy()
    registry["date_registration"] = pd.to_numeric(
        registry["date_registration"], errors="coerce"
    )
    registry["date_unregistration"] = pd.to_numeric(
        registry["date_unregistration"], errors="coerce"
    )
    registry["adverse_final"] = registry["final_result"].isin(["Fail", "Withdrawn"]).astype(int)
    if registry["sequence_id"].duplicated().any():
        raise ValueError("OULAD registration sensitivity table has duplicate sequence_id")
    return registry


def assert_official_termination_contract(registry: pd.DataFrame) -> dict[str, int]:
    """Fail fast if the four official registration/termination counts drift."""

    by_label_pre = int(registry["unregistration_timing"].eq("pre_course").sum())
    by_date_pre = int(registry["date_unregistration"].lt(0).fillna(False).sum())
    counts = {
        "official_registrations": int(len(registry)),
        "pre_course_unregistrations": by_label_pre,
        "pre_course_unregistrations_by_date": by_date_pre,
        "during_course_unregistrations": int(
            registry["unregistration_timing"].eq("during_course").sum()
        ),
        "post_unregistration_activity_registrations": int(
            registry["has_post_unregistration_vle_activity"].fillna(False).astype(bool).sum()
        ),
    }
    expected = {
        "official_registrations": 32593,
        "pre_course_unregistrations": 2678,
        "pre_course_unregistrations_by_date": 2678,
        "during_course_unregistrations": 7393,
        "post_unregistration_activity_registrations": 1279,
    }
    if counts != expected:
        raise AssertionError(f"official termination contract drifted: observed={counts}, expected={expected}")
    return counts


def load_history_panel(max_day_exclusive: int) -> pd.DataFrame:
    columns = [
        "sequence_id",
        "code_module",
        "code_presentation",
        "id_student",
        "date",
        "date_registration",
        "date_unregistration",
        "module_presentation_length",
        "final_result",
        "platform_inactive_flag",
        "activity_type_diversity",
        "assessment_output_flag",
        "clicks_communication_collaboration",
        "click_count",
    ]
    frame = pd.read_parquet(
        PANEL_PATH,
        columns=columns,
        filters=[("date", "<", int(max_day_exclusive))],
    )
    frame = frame.sort_values(["sequence_id", "date"], kind="mergesort").reset_index(drop=True)
    if frame.duplicated(["sequence_id", "date"]).any():
        raise ValueError("history panel contains duplicate student-course-days")
    return frame


def encoder_config() -> ObservationEncodingConfig:
    return ObservationEncodingConfig(
        inactive_col="platform_inactive_flag",
        diversity_col="activity_type_diversity",
        productive_col="assessment_output_flag",
        interaction_col="clicks_communication_collaboration",
        diversity_quantile=0.67,
        missing_policy="raise",
    )


def risk_set(
    registry: pd.DataFrame, landmark: int, horizon: int = MAIN_HORIZON
) -> pd.DataFrame:
    mask = (
        registry["main_sequence_available"].fillna(False).astype(bool)
        & registry["date_registration"].notna()
        & registry["date_registration"].le(0)
        & (
            registry["date_unregistration"].isna()
            | registry["date_unregistration"].ge(landmark)
        )
    )
    result = registry.loc[mask].copy()
    result[f"withdrawal_next_{horizon}d_at_lm{landmark}"] = (
        result["date_unregistration"].notna()
        & result["date_unregistration"].ge(landmark)
        & result["date_unregistration"].lt(landmark + horizon)
        & result["date_unregistration"].lt(result["module_presentation_length"])
    ).astype(int)
    return result


def main_partition(registry: pd.DataFrame, landmark: int) -> tuple[dict[str, pd.DataFrame], dict[str, Any]]:
    eligible = risk_set(registry, landmark, MAIN_HORIZON)
    train = eligible.loc[eligible["code_presentation"].isin(MAIN_PRESENTATION_TRAIN)].copy()
    train_students = set(train["id_student"].tolist())
    inner_validation = eligible.loc[
        eligible["code_presentation"].eq(MAIN_PRESENTATION_INNER_VALIDATION)
        & ~eligible["id_student"].isin(train_students)
    ].copy()
    development_students = train_students | set(inner_validation["id_student"].tolist())
    outer_test = eligible.loc[
        eligible["code_presentation"].eq(MAIN_PRESENTATION_OUTER_TEST)
        & ~eligible["id_student"].isin(development_students)
    ].copy()
    overlap_train_val = len(set(train["id_student"]) & set(inner_validation["id_student"]))
    overlap_dev_outer = len(development_students & set(outer_test["id_student"]))
    metadata = {
        "landmark": landmark,
        "risk_set_definition": (
            "registered by day 0 and not administratively unregistered before landmark"
        ),
        "inner_training_presentations": list(MAIN_PRESENTATION_TRAIN),
        "inner_validation_presentation": MAIN_PRESENTATION_INNER_VALIDATION,
        "untouched_outer_presentation": MAIN_PRESENTATION_OUTER_TEST,
        "student_overlap_inner_train_validation": overlap_train_val,
        "student_overlap_development_outer": overlap_dev_outer,
        "excluded_late_or_missing_registration": int(
            (
                registry["main_sequence_available"].fillna(False).astype(bool)
                & (registry["date_registration"].isna() | registry["date_registration"].gt(0))
            ).sum()
        ),
    }
    return {"inner_train": train, "inner_validation": inner_validation, "outer_test": outer_test}, metadata


def stable_capacity_sample(
    registrations: pd.DataFrame,
    *,
    cap_per_stratum: int,
    seed: int,
    partition: str,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    values = registrations.copy()
    values["selection_hash"] = values["sequence_id"].astype(str).map(
        lambda text: stable_hash(text, seed)
    )
    values = values.sort_values(
        ["code_module", "code_presentation", "selection_hash", "sequence_id"],
        kind="mergesort",
    )
    values["sampling_rank"] = (
        values.groupby(["code_module", "code_presentation"], sort=True).cumcount() + 1
    )
    values["selected"] = values["sampling_rank"].le(cap_per_stratum)
    audit = (
        values.groupby(["code_module", "code_presentation"], sort=True)
        .agg(
            available_sequences=("sequence_id", "size"),
            selected_sequences=("selected", "sum"),
        )
        .reset_index()
    )
    audit.insert(0, "partition", partition)
    audit["cap_per_stratum"] = int(cap_per_stratum)
    audit["sampling_rule"] = "outcome-blind SHA-256(seed|sequence_id) ordering"
    audit["outcome_columns_used"] = False
    return values.loc[values["selected"]].drop(columns=["selected"]), audit


def history_frame(panel: pd.DataFrame, registrations: pd.DataFrame, landmark: int) -> pd.DataFrame:
    identifiers = set(registrations["sequence_id"].tolist())
    result = panel.loc[
        panel["sequence_id"].isin(identifiers) & panel["date"].between(0, landmark - 1)
    ].copy()
    counts = result.groupby("sequence_id", sort=False)["date"].agg(["size", "min", "max"])
    bad = counts.loc[
        counts["size"].ne(landmark) | counts["min"].ne(0) | counts["max"].ne(landmark - 1)
    ]
    missing = identifiers - set(counts.index)
    if len(bad) or missing:
        raise ValueError(
            f"landmark={landmark}: incomplete histories: bad={len(bad)}, missing={len(missing)}"
        )
    return result.sort_values(["sequence_id", "date"], kind="mergesort")


def bundle(frame: pd.DataFrame, encoder: TransparentObservationEncoder, name: str) -> Any:
    return make_sequence_bundle(
        frame,
        encoder,
        group_cols="sequence_id",
        time_col="date",
        min_sequence_length=2,
        expected_step=1,
        gap_policy="raise",
        duplicate_time_policy="raise",
        dataset_name=name,
    )


def choose_hmm_against_mc(comparison: Any) -> tuple[str, int | None, float | None]:
    selection = comparison.hmm_selection
    if selection.best_model is None or selection.best_k is None:
        return "Observable-MC", None, None
    mc_row = comparison.table.loc[comparison.table["model"].eq("Observable-MC")].iloc[0]
    hmm_row = comparison.table.loc[
        comparison.table["model"].eq("CategoricalHMM")
        & comparison.table["k"].eq(selection.best_k)
    ].iloc[0]
    delta = float(
        hmm_row["validation_loglik_per_observation"]
        - mc_row["validation_loglik_per_observation"]
    )
    if delta >= MC_IMPROVEMENT_MARGIN:
        return "CategoricalHMM", int(selection.best_k), delta
    return "Observable-MC", int(selection.best_k), delta


def fit_nested_representation(
    panel: pd.DataFrame,
    partitions: dict[str, pd.DataFrame],
    *,
    cap_per_stratum: int,
    seed: int,
    k_values: Sequence[int],
    n_starts: int,
    n_iter: int,
    tol: float,
    scope: str,
) -> tuple[FittedRepresentation, pd.DataFrame, pd.DataFrame, pd.DataFrame, list[str]]:
    sampling_parts: list[pd.DataFrame] = []
    samples: dict[str, pd.DataFrame] = {}
    for offset, part in enumerate(["inner_train", "inner_validation", "outer_test"]):
        selected, audit = stable_capacity_sample(
            partitions[part],
            cap_per_stratum=cap_per_stratum,
            seed=seed + offset,
            partition=f"{scope}:{part}",
        )
        samples[part] = selected
        sampling_parts.append(audit)

    train_full = history_frame(panel, partitions["inner_train"], MAIN_LANDMARK)
    validation_full = history_frame(panel, partitions["inner_validation"], MAIN_LANDMARK)
    train_sample = history_frame(panel, samples["inner_train"], MAIN_LANDMARK)
    validation_sample = history_frame(panel, samples["inner_validation"], MAIN_LANDMARK)
    inner_encoder = TransparentObservationEncoder(encoder_config()).fit(train_full)
    train_bundle = bundle(train_sample, inner_encoder, f"{scope}:inner_train")
    validation_bundle = bundle(
        validation_sample, inner_encoder, f"{scope}:inner_validation"
    )
    comparison = fit_mc_hmm_comparison(
        train_bundle,
        validation_bundle,
        mc_alpha=0.5,
        k_values=k_values,
        n_starts=n_starts,
        random_state=seed,
        n_iter=n_iter,
        tol=tol,
        hmm_prior_strength=0.1,
        min_state_occupancy=0.01,
        selection_metric="validation_loglik",
    )
    comparison_table = comparison.table.copy()
    comparison_table.insert(0, "scope", scope)
    starts_table = comparison.hmm_selection.starts.copy()
    starts_table.insert(0, "scope", scope)
    chosen_kind, selected_k, delta = choose_hmm_against_mc(comparison)

    development_regs = pd.concat(
        [partitions["inner_train"], partitions["inner_validation"]],
        ignore_index=True,
    ).drop_duplicates("sequence_id")
    development_sample = pd.concat(
        [samples["inner_train"], samples["inner_validation"]],
        ignore_index=True,
    ).drop_duplicates("sequence_id")
    development_full = history_frame(panel, development_regs, MAIN_LANDMARK)
    development_sample_frame = history_frame(panel, development_sample, MAIN_LANDMARK)
    final_encoder = TransparentObservationEncoder(encoder_config()).fit(development_full)
    development_bundle = bundle(
        development_sample_frame, final_encoder, f"{scope}:development_refit"
    )
    final_mc = fit_dirichlet_markov_chain(development_bundle, alpha=0.5)
    final_hmm = None
    refit_status = "not_requested_mc_selected"
    refit_rows: list[pd.DataFrame] = []
    warnings_out: list[str] = []
    if chosen_kind == "CategoricalHMM" and selected_k is not None:
        refit = fit_categorical_hmm_grid(
            development_bundle,
            development_bundle,
            k_values=[selected_k],
            n_starts=n_starts,
            random_state=seed + 10_000,
            n_iter=n_iter,
            tol=tol,
            prior_strength=0.1,
            min_state_occupancy=0.01,
            selection_metric="validation_loglik",
        )
        refit_summary = refit.summary.copy()
        refit_summary.insert(0, "scope", f"{scope}:fixed_k_refit")
        refit_rows.append(refit_summary)
        refit_starts = refit.starts.copy()
        refit_starts.insert(0, "scope", f"{scope}:fixed_k_refit")
        starts_table = pd.concat([starts_table, refit_starts], ignore_index=True)
        if refit.best_model is None:
            chosen_kind = "Observable-MC"
            final_hmm = None
            refit_status = "strict_refit_failed_mc_fallback"
            warnings_out.append(
                f"{scope}: inner-selected HMM K={selected_k} did not pass strict refit convergence; MC used."
            )
        else:
            final_hmm = refit.best_model
            refit_status = "strict_fixed_k_refit_passed"
    if refit_rows:
        comparison_table = pd.concat([comparison_table, *refit_rows], ignore_index=True)
    representation = FittedRepresentation(
        encoder=final_encoder,
        mc_model=final_mc,
        hmm_model=final_hmm,
        chosen_kind=chosen_kind,
        selected_k=selected_k,
        inner_delta_vs_mc=delta,
        selection_status=comparison.hmm_selection.status,
        refit_status=refit_status,
    )
    return (
        representation,
        comparison_table,
        starts_table,
        pd.concat(sampling_parts, ignore_index=True),
        warnings_out,
    )


def forward_filter(model: Any, observations: np.ndarray) -> np.ndarray:
    emissions = np.asarray(model.emissionprob_, dtype=float)
    transition = np.asarray(model.transmat_, dtype=float)
    alpha = np.asarray(model.startprob_, dtype=float) * emissions[:, int(observations[0])]
    alpha /= alpha.sum()
    rows = [alpha.copy()]
    for observation in observations[1:]:
        alpha = (alpha @ transition) * emissions[:, int(observation)]
        alpha /= alpha.sum()
        rows.append(alpha.copy())
    return np.vstack(rows)


def conditional_hmm_loglik(model: Any, history: np.ndarray, future: np.ndarray) -> float:
    alpha = forward_filter(model, history)[-1]
    transition = np.asarray(model.transmat_, dtype=float)
    emissions = np.asarray(model.emissionprob_, dtype=float)
    total = 0.0
    for observation in future:
        predictive = alpha @ transition
        probability = float(predictive @ emissions[:, int(observation)])
        probability = max(probability, np.finfo(float).tiny)
        total += math.log(probability)
        alpha = predictive * emissions[:, int(observation)]
        alpha /= alpha.sum()
    return total


def conditional_mc_loglik(model: Any, history: np.ndarray, future: np.ndarray) -> float:
    previous = int(history[-1])
    total = 0.0
    transition = np.asarray(model.transmat, dtype=float)
    for observation in future:
        current = int(observation)
        total += math.log(max(float(transition[previous, current]), np.finfo(float).tiny))
        previous = current
    return total


def conditional_future_scores(
    panel: pd.DataFrame,
    outer_registry: pd.DataFrame,
    representation: FittedRepresentation,
    *,
    landmark: int,
    horizon: int,
) -> pd.DataFrame:
    end_day = landmark + horizon - 1
    complete = outer_registry.loc[
        outer_registry["date_registration"].notna()
        & outer_registry["date_registration"].le(0)
        & (
            outer_registry["date_unregistration"].isna()
            | outer_registry["date_unregistration"].ge(end_day)
        )
    ].copy()
    identifiers = set(complete["sequence_id"])
    rows = panel.loc[
        panel["sequence_id"].isin(identifiers)
        & panel["date"].between(0, end_day)
    ].copy()
    rows["observation_code"] = representation.encoder.transform(rows)
    records: list[dict[str, Any]] = []
    metadata = complete.set_index("sequence_id")[["code_module", "code_presentation", "id_student"]]
    for sequence_id, group in rows.groupby("sequence_id", sort=True):
        group = group.sort_values("date")
        history = group.loc[group["date"].lt(landmark), "observation_code"].to_numpy(dtype=int)
        future = group.loc[group["date"].ge(landmark), "observation_code"].to_numpy(dtype=int)
        if len(history) != landmark or len(future) != horizon:
            continue
        meta = metadata.loc[sequence_id]
        record = {
            "sequence_id": sequence_id,
            "code_module": meta["code_module"],
            "code_presentation": meta["code_presentation"],
            "id_student": int(meta["id_student"]),
            "history_days": landmark,
            "future_days": horizon,
            "followup_rule": "complete observed platform window; termination handled separately",
            "mc_conditional_loglik": conditional_mc_loglik(
                representation.mc_model, history, future
            ),
        }
        if representation.hmm_model is not None:
            record["hmm_conditional_loglik"] = conditional_hmm_loglik(
                representation.hmm_model, history, future
            )
        else:
            record["hmm_conditional_loglik"] = np.nan
        records.append(record)
    return pd.DataFrame(records)


def history_features(
    frame: pd.DataFrame,
    encoder: TransparentObservationEncoder,
    hmm_model: Any | None,
) -> pd.DataFrame:
    work = frame.loc[:, ["sequence_id", "date"]].copy()
    work["observation_code"] = encoder.transform(frame)
    records: list[dict[str, Any]] = []
    for sequence_id, group in work.groupby("sequence_id", sort=True):
        group = group.sort_values("date")
        codes = group["observation_code"].to_numpy(dtype=int)
        counts = np.bincount(codes, minlength=4).astype(float)
        shares = counts / len(codes)
        transitions = np.zeros((4, 4), dtype=float)
        if len(codes) > 1:
            np.add.at(transitions, (codes[:-1], codes[1:]), 1)
            transitions /= len(codes) - 1
        nonzero = shares[shares > 0]
        row: dict[str, Any] = {
            "sequence_id": sequence_id,
            "history_days": int(len(codes)),
            "obs_persistence": float(np.trace(transitions)),
            "obs_entropy": float(-(nonzero * np.log(nonzero)).sum()),
        }
        for index in range(4):
            row[f"obs_share_{index}"] = float(shares[index])
            row[f"obs_last_{index}"] = float(codes[-1] == index)
        for source in range(4):
            for target in range(4):
                row[f"trans_{source}_{target}"] = float(transitions[source, target])
        if hmm_model is not None:
            filtered = forward_filter(hmm_model, codes)
            for state in range(int(hmm_model.n_components)):
                row[f"state_mean_{state}"] = float(filtered[:, state].mean())
                row[f"state_last_{state}"] = float(filtered[-1, state])
        records.append(row)
    return pd.DataFrame.from_records(records)


def observed_feature_columns(frame: pd.DataFrame) -> list[str]:
    prefixes = ("obs_share_", "obs_last_", "trans_")
    fixed = ["obs_persistence", "obs_entropy"]
    return fixed + sorted([column for column in frame.columns if column.startswith(prefixes)])


def hmm_feature_columns(frame: pd.DataFrame) -> list[str]:
    return observed_feature_columns(frame) + sorted(
        [column for column in frame.columns if column.startswith(("state_mean_", "state_last_"))]
    )


def calibration_summary(y: np.ndarray, probabilities: np.ndarray) -> tuple[float, float, float]:
    clipped = np.clip(probabilities, 1e-6, 1 - 1e-6)
    logits = np.log(clipped / (1 - clipped)).reshape(-1, 1)
    if len(np.unique(y)) < 2 or np.std(logits) == 0:
        return float("nan"), float("nan"), float("nan")
    model = LogisticRegression(C=1e6, solver="lbfgs", max_iter=2000)
    model.fit(logits, y)
    intercept = float(model.intercept_[0])
    slope = float(model.coef_[0, 0])
    bins = pd.qcut(pd.Series(probabilities), q=10, duplicates="drop")
    table = pd.DataFrame({"bin": bins, "y": y, "p": probabilities}).groupby(
        "bin", observed=True
    ).agg(observed=("y", "mean"), predicted=("p", "mean"), n=("y", "size"))
    ece = float(((table["observed"] - table["predicted"]).abs() * table["n"]).sum() / table["n"].sum())
    return intercept, slope, ece


def cluster_bootstrap_metrics(
    y: np.ndarray,
    probabilities: np.ndarray,
    cluster: np.ndarray,
    *,
    seed: int,
    replications: int,
) -> dict[str, float]:
    unique = np.unique(cluster)
    indices = {value: np.flatnonzero(cluster == value) for value in unique}
    rng = np.random.default_rng(seed)
    auc_values: list[float] = []
    brier_values: list[float] = []
    for _ in range(replications):
        drawn = rng.choice(unique, size=len(unique), replace=True)
        selected = np.concatenate([indices[value] for value in drawn])
        sampled_y = y[selected]
        sampled_p = probabilities[selected]
        if len(np.unique(sampled_y)) < 2:
            continue
        auc_values.append(float(roc_auc_score(sampled_y, sampled_p)))
        brier_values.append(float(brier_score_loss(sampled_y, sampled_p)))
    if not auc_values:
        return {
            "auc_ci_low": float("nan"),
            "auc_ci_high": float("nan"),
            "brier_ci_low": float("nan"),
            "brier_ci_high": float("nan"),
            "valid_bootstrap_replications": 0,
        }
    return {
        "auc_ci_low": float(np.quantile(auc_values, 0.025)),
        "auc_ci_high": float(np.quantile(auc_values, 0.975)),
        "brier_ci_low": float(np.quantile(brier_values, 0.025)),
        "brier_ci_high": float(np.quantile(brier_values, 0.975)),
        "valid_bootstrap_replications": int(len(auc_values)),
    }


def fit_outcome_model(
    train_features: pd.DataFrame,
    test_features: pd.DataFrame,
    train_registry: pd.DataFrame,
    test_registry: pd.DataFrame,
    *,
    target: str,
    representation_name: str,
    feature_columns: list[str],
    bootstrap_replications: int,
    seed: int,
    validation_scope: str,
) -> tuple[dict[str, Any], pd.DataFrame, np.ndarray]:
    train = train_registry[["sequence_id", "id_student", target]].merge(
        train_features, on="sequence_id", how="inner", validate="one_to_one"
    )
    test = test_registry[["sequence_id", "id_student", target]].merge(
        test_features, on="sequence_id", how="inner", validate="one_to_one"
    )
    if len(np.unique(train[target])) < 2 or len(np.unique(test[target])) < 2:
        raise ValueError(f"{validation_scope}/{target}: target has fewer than two classes")
    pipeline = make_pipeline(
        StandardScaler(),
        LogisticRegression(C=1.0, solver="lbfgs", max_iter=2000),
    )
    pipeline.fit(train[feature_columns].to_numpy(dtype=float), train[target].to_numpy(dtype=int))
    probabilities = pipeline.predict_proba(test[feature_columns].to_numpy(dtype=float))[:, 1]
    y = test[target].to_numpy(dtype=int)
    clusters = test["id_student"].to_numpy()
    auc = float(roc_auc_score(y, probabilities))
    brier = float(brier_score_loss(y, probabilities))
    loss = float(log_loss(y, probabilities, labels=[0, 1]))
    intercept, slope, ece = calibration_summary(y, probabilities)
    uncertainty = cluster_bootstrap_metrics(
        y,
        probabilities,
        clusters,
        seed=seed,
        replications=bootstrap_replications,
    )
    result = {
        "validation_scope": validation_scope,
        "target": target,
        "representation": representation_name,
        "train_n": int(len(train)),
        "test_n": int(len(test)),
        "test_students": int(test["id_student"].nunique()),
        "test_events": int(y.sum()),
        "test_event_rate": float(y.mean()),
        "auc": auc,
        "brier": brier,
        "log_loss": loss,
        "calibration_intercept": intercept,
        "calibration_slope": slope,
        "ece_10bin": ece,
        "bootstrap_cluster": "id_student",
        "bootstrap_requested_replications": bootstrap_replications,
        **uncertainty,
    }
    prediction = test[["sequence_id", "id_student", target]].copy()
    prediction["predicted_probability"] = probabilities
    prediction["representation"] = representation_name
    prediction["validation_scope"] = validation_scope
    prediction["target"] = target
    return result, prediction, probabilities


def calibration_bins(predictions: pd.DataFrame) -> pd.DataFrame:
    records: list[pd.DataFrame] = []
    for keys, group in predictions.groupby(
        ["validation_scope", "target", "representation"], sort=True
    ):
        target_column = keys[1]
        bins = pd.qcut(group["predicted_probability"], q=10, duplicates="drop")
        result = group.assign(calibration_bin=bins).groupby(
            "calibration_bin", observed=True
        ).agg(
            n=(target_column, "size"),
            observed_rate=(target_column, "mean"),
            mean_predicted_probability=("predicted_probability", "mean"),
            probability_min=("predicted_probability", "min"),
            probability_max=("predicted_probability", "max"),
        ).reset_index()
        result.insert(0, "representation", keys[2])
        result.insert(0, "target", keys[1])
        result.insert(0, "validation_scope", keys[0])
        result["calibration_bin"] = result["calibration_bin"].astype(str)
        records.append(result)
    return pd.concat(records, ignore_index=True) if records else pd.DataFrame()


def registration_flow_and_bounds(
    registry: pd.DataFrame, panel: pd.DataFrame, landmarks: Sequence[int]
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    total = int(len(registry))
    prestart = registry["unregistration_timing"].eq("pre_course")
    during = registry["unregistration_timing"].eq("during_course")
    post = registry["has_post_unregistration_vle_activity"].fillna(False).astype(bool)
    flow = pd.DataFrame(
        [
            {"stage": "all_official_registrations", "n": total, "share": 1.0},
            {
                "stage": "pre_course_unregistration_zero_length",
                "n": int(prestart.sum()),
                "share": float(prestart.mean()),
            },
            {
                "stage": "entered_main_daily_panel",
                "n": int(registry["main_sequence_available"].sum()),
                "share": float(registry["main_sequence_available"].mean()),
            },
            {
                "stage": "during_course_unregistration",
                "n": int(during.sum()),
                "share": float(during.mean()),
            },
            {
                "stage": "registrations_with_post_unregistration_vle_activity",
                "n": int(post.sum()),
                "share": float(post.mean()),
            },
        ]
    )
    bound_rows: list[dict[str, Any]] = []
    for landmark in landmarks:
        observed = panel.loc[panel["date"].lt(landmark)]
        active_days = int((~observed["platform_inactive_flag"].astype(bool)).sum())
        observed_days = int(len(observed))
        all_possible_days = total * landmark
        missing_days = all_possible_days - observed_days
        bound_rows.append(
            {
                "landmark_days": landmark,
                "all_registrations": total,
                "observed_legal_student_days": observed_days,
                "unobserved_days_due_to_zero_length_or_termination": missing_days,
                "observed_platform_active_days": active_days,
                "population_active_share_lower_bound": active_days / all_possible_days,
                "population_active_share_upper_bound": (active_days + missing_days) / all_possible_days,
                "bound_rule": (
                    "unobserved days assigned inactive for lower bound and active for upper bound; no latent state imputed"
                ),
            }
        )
    bounds = pd.DataFrame(bound_rows)
    post_by_module = (
        registry.groupby("code_module", sort=True)
        .agg(
            registrations=("sequence_id", "size"),
            during_course_unregistrations=("unregistration_timing", lambda x: int((x == "during_course").sum())),
            registrations_with_post_unregistration_activity=("has_post_unregistration_vle_activity", "sum"),
            post_unregistration_clicks=("post_unregistration_vle_clicks", "sum"),
            post_unregistration_active_days=("post_unregistration_vle_active_days", "sum"),
        )
        .reset_index()
    )
    post_by_module["post_activity_share_among_during_course_unregistrations"] = (
        post_by_module["registrations_with_post_unregistration_activity"]
        / post_by_module["during_course_unregistrations"].replace(0, np.nan)
    )
    return flow, bounds, post_by_module


def discrete_time_termination_model(
    *,
    control_days_per_sequence: int,
) -> tuple[pd.DataFrame, dict[str, Any]]:
    columns = [
        "sequence_id",
        "code_module",
        "code_presentation",
        "id_student",
        "date",
        "date_unregistration",
        "module_presentation_length",
        "platform_inactive_flag",
        "assessment_output_flag",
        "clicks_communication_collaboration",
        "activity_type_diversity",
    ]
    frame = pd.read_parquet(PANEL_PATH, columns=columns)
    frame = frame.sort_values(["sequence_id", "date"], kind="mergesort").reset_index(drop=True)
    early = (
        frame["date_unregistration"].notna()
        & frame["date_unregistration"].ge(0)
        & frame["date_unregistration"].lt(frame["module_presentation_length"] - 1)
    )
    frame["termination_event"] = (
        early & frame["date"].eq(frame["date_unregistration"])
    ).astype(np.int8)
    group = frame.groupby("sequence_id", sort=False)
    position = group.cumcount().to_numpy(dtype=float)
    count_prior = np.minimum(position, 7.0)
    source_columns = {
        "prior7_inactive_share": frame["platform_inactive_flag"].astype(float),
        "prior7_assessment_output_share": frame["assessment_output_flag"].astype(float),
        "prior7_communication_share": frame["clicks_communication_collaboration"].gt(0).astype(float),
        "prior7_activity_diversity_mean": pd.to_numeric(
            frame["activity_type_diversity"], errors="coerce"
        ).fillna(0.0),
    }
    for target, source in source_columns.items():
        cumulative = source.groupby(frame["sequence_id"], sort=False).cumsum()
        previous = cumulative.groupby(frame["sequence_id"], sort=False).shift(1).fillna(0.0)
        previous_eight = cumulative.groupby(frame["sequence_id"], sort=False).shift(8).fillna(0.0)
        frame[target] = (previous - previous_eight).to_numpy(dtype=float) / np.maximum(count_prior, 1.0)
    eligible = frame.loc[frame["date"].ge(7)].copy()
    event_rows = eligible.loc[eligible["termination_event"].eq(1)].copy()
    controls = eligible.loc[eligible["termination_event"].eq(0)].copy()
    controls["control_rank"] = controls.groupby("sequence_id", sort=False).cumcount()
    controls["control_n"] = controls.groupby("sequence_id", sort=False)["date"].transform("size")
    controls["control_step"] = np.ceil(
        controls["control_n"] / float(control_days_per_sequence)
    ).clip(lower=1).astype(int)
    controls = controls.loc[(controls["control_rank"] % controls["control_step"]).eq(0)].copy()
    controls["selected_control_n"] = controls.groupby("sequence_id", sort=False)["date"].transform("size")
    controls["sampling_weight"] = controls["control_n"] / controls["selected_control_n"]
    event_rows["sampling_weight"] = 1.0
    sampled = pd.concat([event_rows, controls], ignore_index=True)
    sampled["inactive_prior_per10pp"] = sampled["prior7_inactive_share"] * 10.0
    sampled["assessment_prior_per10pp"] = sampled["prior7_assessment_output_share"] * 10.0
    sampled["communication_prior_per10pp"] = sampled["prior7_communication_share"] * 10.0
    sampled["diversity_prior_per_unit"] = sampled["prior7_activity_diversity_mean"]
    sampled["course_day_per28"] = sampled["date"] / 28.0
    sampled["course_day_per28_sq"] = sampled["course_day_per28"] ** 2
    numeric_features = [
        "inactive_prior_per10pp",
        "assessment_prior_per10pp",
        "communication_prior_per10pp",
        "diversity_prior_per_unit",
        "course_day_per28",
        "course_day_per28_sq",
    ]
    categorical = pd.get_dummies(
        sampled[["code_module", "code_presentation"]],
        drop_first=True,
        dtype=float,
    )
    design_frame = pd.concat(
        [sampled[numeric_features].reset_index(drop=True), categorical.reset_index(drop=True)],
        axis=1,
    )
    X = design_frame.to_numpy(dtype=float)
    y = sampled["termination_event"].to_numpy(dtype=int)
    weights = sampled["sampling_weight"].to_numpy(dtype=float)
    model = LogisticRegression(C=1e6, solver="lbfgs", max_iter=2000)
    model.fit(X, y, sample_weight=weights)
    p = model.predict_proba(X)[:, 1]
    X_augmented = np.column_stack([np.ones(len(X)), X])
    beta = np.concatenate([model.intercept_, model.coef_.ravel()])
    hessian_weight = weights * p * (1 - p)
    bread = X_augmented.T @ (hessian_weight[:, None] * X_augmented)
    bread_inv = np.linalg.pinv(bread)
    score = X_augmented * (weights * (y - p))[:, None]
    cluster_codes, _ = pd.factorize(sampled["id_student"], sort=True)
    cluster_score = np.zeros((cluster_codes.max() + 1, score.shape[1]), dtype=float)
    np.add.at(cluster_score, cluster_codes, score)
    meat = cluster_score.T @ cluster_score
    covariance = bread_inv @ meat @ bread_inv
    standard_errors = np.sqrt(np.clip(np.diag(covariance), 0, None))
    names = ["intercept", *design_frame.columns.tolist()]
    coefficient_table = pd.DataFrame(
        {
            "term": names,
            "coefficient_log_odds": beta,
            "cluster_robust_se": standard_errors,
            "odds_ratio": np.exp(beta),
            "or_ci_low": np.exp(beta - 1.96 * standard_errors),
            "or_ci_high": np.exp(beta + 1.96 * standard_errors),
        }
    )
    coefficient_table["cluster_unit"] = "id_student"
    coefficient_table["control_sampling"] = (
        f"deterministic approximately-even spacing, <= {control_days_per_sequence} controls/sequence; sampling-fraction/frequency weighted"
    )
    metadata = {
        "all_main_panel_rows": int(len(frame)),
        "all_during_course_termination_events": int(frame["termination_event"].sum()),
        "events_with_at_least_7_prior_days": int(len(event_rows)),
        "events_before_day_7_excluded_from_regression": int(
            frame.loc[frame["date"].lt(7), "termination_event"].sum()
        ),
        "sampled_control_rows": int(len(controls)),
        "weighted_control_person_days": float(controls["sampling_weight"].sum()),
        "eligible_non_event_person_days": int(
            eligible["termination_event"].eq(0).sum()
        ),
        "clusters": int(sampled["id_student"].nunique()),
        "model": "sampling-fraction/frequency-weighted pooled logistic discrete-time risk",
        "time_at_risk": "course day 7 through administrative termination or prepared course window end",
        "interpretation": "conditional association; not a causal effect",
    }
    del frame, eligible, event_rows, controls, sampled, X, X_augmented, score, cluster_score
    gc.collect()
    return coefficient_table, metadata


def evaluate_landmark(
    panel: pd.DataFrame,
    registry: pd.DataFrame,
    representation: FittedRepresentation,
    *,
    landmark: int,
    bootstrap_replications: int,
    seed: int,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    partitions, _ = main_partition(registry, landmark)
    development = pd.concat(
        [partitions["inner_train"], partitions["inner_validation"]], ignore_index=True
    ).drop_duplicates("sequence_id")
    outer = partitions["outer_test"]
    development_history = history_frame(panel, development, landmark)
    outer_history = history_frame(panel, outer, landmark)
    development_features = history_features(
        development_history, representation.encoder, representation.hmm_model
    )
    outer_features = history_features(
        outer_history, representation.encoder, representation.hmm_model
    )
    target = f"withdrawal_next_{MAIN_HORIZON}d_at_lm{landmark}"
    performance_rows: list[dict[str, Any]] = []
    predictions: list[pd.DataFrame] = []
    representations = [
        ("Observable-history", observed_feature_columns(development_features))
    ]
    if representation.hmm_model is not None:
        representations.append(
            ("Observable+numbered-HMM-history", hmm_feature_columns(development_features))
        )
    for outcome in ["adverse_final", target]:
        for index, (name, columns) in enumerate(representations):
            result, prediction, _ = fit_outcome_model(
                development_features,
                outer_features,
                development,
                outer,
                target=outcome,
                representation_name=name,
                feature_columns=columns,
                bootstrap_replications=bootstrap_replications,
                seed=seed + landmark * 100 + index,
                validation_scope=f"untouched_{MAIN_PRESENTATION_OUTER_TEST}_landmark_{landmark}",
            )
            result["landmark_days"] = landmark
            if outcome == "adverse_final":
                result["outcome_time_scope"] = "course_final_outcome"
                result["forecast_horizon_days"] = np.nan
            else:
                result["outcome_time_scope"] = (
                    f"[{landmark},{landmark + MAIN_HORIZON}) course days"
                )
                result["forecast_horizon_days"] = MAIN_HORIZON
            result["student_disjoint_development_test"] = True
            performance_rows.append(result)
            predictions.append(prediction)
    return pd.DataFrame(performance_rows), pd.concat(predictions, ignore_index=True)


def leave_module_validation(
    panel: pd.DataFrame,
    registry: pd.DataFrame,
    reference: FittedRepresentation,
    *,
    seed: int,
    cap_per_stratum: int,
    k_values: Sequence[int],
    n_starts: int,
    n_iter: int,
    tol: float,
    bootstrap_replications: int,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame, list[str]]:
    eligible = risk_set(registry, MAIN_LANDMARK, MAIN_HORIZON)
    modules = sorted(eligible["code_module"].unique().tolist())
    validation_rows: list[dict[str, Any]] = []
    alignment_rows: list[pd.DataFrame] = []
    selection_rows: list[pd.DataFrame] = []
    starts_rows: list[pd.DataFrame] = []
    warnings_out: list[str] = []
    for fold_index, held_module in enumerate(modules):
        print(
            f"[Q2] leave-module fold {fold_index + 1}/{len(modules)}: hold out {held_module}",
            flush=True,
        )
        outer = eligible.loc[eligible["code_module"].eq(held_module)].copy()
        outer_students = set(outer["id_student"].tolist())
        development = eligible.loc[
            eligible["code_module"].ne(held_module)
            & ~eligible["id_student"].isin(outer_students)
        ].copy()
        student_bucket = development["id_student"].astype(str).map(
            lambda text: hash_bucket(text, seed + fold_index, 5)
        )
        inner_validation = development.loc[student_bucket.eq(0)].copy()
        inner_train = development.loc[~student_bucket.eq(0)].copy()
        partitions = {
            "inner_train": inner_train,
            "inner_validation": inner_validation,
            "outer_test": outer,
        }
        fold_scope = f"leave_module_{held_module}"
        fold_representation, comparison, starts, _, fold_warnings = fit_nested_representation(
            panel,
            partitions,
            cap_per_stratum=cap_per_stratum,
            seed=seed + 1000 * (fold_index + 1),
            k_values=k_values,
            n_starts=n_starts,
            n_iter=n_iter,
            tol=tol,
            scope=fold_scope,
        )
        selection_rows.append(comparison)
        starts_rows.append(starts)
        warnings_out.extend(fold_warnings)
        train_history = history_frame(panel, development, MAIN_LANDMARK)
        outer_history = history_frame(panel, outer, MAIN_LANDMARK)
        outer_bundle = bundle(outer_history, fold_representation.encoder, f"{fold_scope}:outer")
        mc_score = fold_representation.mc_model.score(outer_bundle).per_observation
        hmm_score = np.nan
        if fold_representation.hmm_model is not None:
            hmm_score = float(
                fold_representation.hmm_model.score(
                    outer_bundle.X, lengths=outer_bundle.lengths
                )
                / outer_bundle.n_observations
            )
            if reference.hmm_model is not None:
                alignment = hungarian_state_alignment(
                    reference.hmm_model.emissionprob_,
                    fold_representation.hmm_model.emissionprob_,
                )
                alignment.insert(0, "held_out_module", held_module)
                alignment["reference_k"] = int(reference.hmm_model.n_components)
                alignment["fold_k"] = int(fold_representation.hmm_model.n_components)
                alignment_rows.append(alignment)
        train_features = history_features(
            train_history, fold_representation.encoder, fold_representation.hmm_model
        )
        outer_features = history_features(
            outer_history, fold_representation.encoder, fold_representation.hmm_model
        )
        feature_sets = [
            ("Observable-history", observed_feature_columns(train_features))
        ]
        if fold_representation.hmm_model is not None:
            feature_sets.append(
                ("Observable+numbered-HMM-history", hmm_feature_columns(train_features))
            )
        for feature_index, (feature_name, feature_columns) in enumerate(feature_sets):
            result, _, _ = fit_outcome_model(
                train_features,
                outer_features,
                development,
                outer,
                target="adverse_final",
                representation_name=feature_name,
                feature_columns=feature_columns,
                bootstrap_replications=bootstrap_replications,
                seed=seed + 50_000 + fold_index * 100 + feature_index,
                validation_scope=fold_scope,
            )
            result.update(
                {
                    "held_out_module": held_module,
                    "student_overlap_train_test": int(
                        len(set(development["id_student"]) & set(outer["id_student"]))
                    ),
                    "nested_selected_kind": fold_representation.chosen_kind,
                    "nested_selected_k": fold_representation.selected_k,
                    "inner_hmm_delta_vs_mc": fold_representation.inner_delta_vs_mc,
                    "mc_outer_loglik_per_observation": mc_score,
                    "hmm_outer_loglik_per_observation": hmm_score,
                    "hmm_refit_status": fold_representation.refit_status,
                }
            )
            validation_rows.append(result)
        del train_history, outer_history, train_features, outer_features, outer_bundle
        gc.collect()
        print(
            f"[Q2] leave-module fold {fold_index + 1}/{len(modules)} complete: "
            f"held={held_module}, selected={fold_representation.chosen_kind}, "
            f"K={fold_representation.selected_k}",
            flush=True,
        )
    return (
        pd.DataFrame(validation_rows),
        pd.concat(alignment_rows, ignore_index=True) if alignment_rows else pd.DataFrame(),
        pd.concat(selection_rows, ignore_index=True),
        pd.concat(starts_rows, ignore_index=True),
        warnings_out,
    )


def save_figures(
    comparison: pd.DataFrame,
    calibration: pd.DataFrame,
    leave_module: pd.DataFrame,
    flow: pd.DataFrame,
    output_paths: list[Path],
) -> None:
    FIGURES_DIR.mkdir(parents=True, exist_ok=True)
    plt.style.use("seaborn-v0_8-whitegrid")

    main = comparison.loc[comparison["scope"].eq("main_presentation_nested")].copy()
    fig, ax = plt.subplots(figsize=(7.2, 4.5))
    mc = main.loc[main["model"].eq("Observable-MC")]
    if not mc.empty:
        ax.axhline(
            mc.iloc[0]["validation_loglik_per_observation"],
            color="#4C78A8",
            linestyle="--",
            label="Observable MC",
        )
    hmm = main.loc[main["model"].eq("CategoricalHMM")]
    ax.plot(
        hmm["k"],
        hmm["validation_loglik_per_observation"],
        marker="o",
        color="#E45756",
        label="Categorical HMM",
    )
    for row in hmm.itertuples(index=False):
        if not bool(getattr(row, "feasible", False)):
            ax.annotate("provisional", (row.k, row.validation_loglik_per_observation), fontsize=7)
    ax.set_xlabel("Number of hidden states K")
    ax.set_ylabel("Inner-validation log likelihood / day")
    ax.set_title("Nested model selection (outer 2014J untouched)")
    ax.legend(frameon=False)
    fig.tight_layout()
    path = FIGURES_DIR / "fig_q2_nested_model_validation.png"
    fig.savefig(path, dpi=300, bbox_inches="tight")
    plt.close(fig)
    output_paths.append(path)

    main_cal = calibration.loc[
        calibration["validation_scope"].eq("untouched_2014J_landmark_28")
        & calibration["target"].eq("adverse_final")
    ]
    fig, ax = plt.subplots(figsize=(5.4, 5.0))
    ax.plot([0, 1], [0, 1], color="black", linewidth=1, linestyle="--", label="Ideal")
    colors = ["#4C78A8", "#E45756"]
    for color, (name, group) in zip(colors, main_cal.groupby("representation", sort=True)):
        ax.plot(
            group["mean_predicted_probability"],
            group["observed_rate"],
            marker="o",
            label=name,
            color=color,
        )
    ax.set_xlabel("Mean predicted probability")
    ax.set_ylabel("Observed adverse-outcome rate")
    ax.set_title("Untouched 2014J calibration (28-day history)")
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.legend(frameon=False, fontsize=8)
    fig.tight_layout()
    path = FIGURES_DIR / "fig_q2_outcome_calibration.png"
    fig.savefig(path, dpi=300, bbox_inches="tight")
    plt.close(fig)
    output_paths.append(path)

    fig, ax = plt.subplots(figsize=(8.0, 4.8))
    observed = leave_module.loc[leave_module["representation"].eq("Observable-history")]
    x = np.arange(len(observed))
    ax.errorbar(
        x,
        observed["auc"],
        yerr=np.vstack(
            [observed["auc"] - observed["auc_ci_low"], observed["auc_ci_high"] - observed["auc"]]
        ),
        fmt="o",
        color="#4C78A8",
        capsize=3,
        label="Observable history",
    )
    hmm = leave_module.loc[
        leave_module["representation"].eq("Observable+numbered-HMM-history")
    ].set_index("held_out_module")
    for index, module in enumerate(observed["held_out_module"]):
        if module in hmm.index:
            row = hmm.loc[module]
            ax.errorbar(
                index + 0.12,
                row["auc"],
                yerr=np.array([[row["auc"] - row["auc_ci_low"]], [row["auc_ci_high"] - row["auc"]]]),
                fmt="s",
                color="#E45756",
                capsize=3,
            )
    ax.axhline(0.5, color="black", linestyle="--", linewidth=1)
    ax.set_xticks(x, observed["held_out_module"])
    ax.set_ylabel("AUC (student-cluster bootstrap 95% CI)")
    ax.set_xlabel("Held-out module")
    ax.set_title("Leave-module-out adverse-outcome association")
    ax.legend(frameon=False)
    fig.tight_layout()
    path = FIGURES_DIR / "fig_q2_leave_module_external.png"
    fig.savefig(path, dpi=300, bbox_inches="tight")
    plt.close(fig)
    output_paths.append(path)

    fig, ax = plt.subplots(figsize=(8.0, 4.5))
    plot_flow = flow.copy()
    labels = [
        "All registrations",
        "Pre-course unregister",
        "Entered panel",
        "During-course unregister",
        "Post-unregister activity",
    ]
    ax.bar(labels, plot_flow["n"], color=["#4C78A8", "#F2CF5B", "#72B7B2", "#E45756", "#B279A2"])
    for index, value in enumerate(plot_flow["n"]):
        ax.text(index, value, f"{int(value):,}", ha="center", va="bottom", fontsize=8)
    ax.set_ylabel("Registrations")
    ax.set_title("Administrative termination is retained as a separate process")
    ax.tick_params(axis="x", rotation=18)
    fig.tight_layout()
    path = FIGURES_DIR / "fig_q2_termination_flow.png"
    fig.savefig(path, dpi=300, bbox_inches="tight")
    plt.close(fig)
    output_paths.append(path)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--seed", type=int, default=20260822)
    parser.add_argument("--main-cap-per-stratum", type=int, default=30)
    parser.add_argument("--main-k", type=int, nargs="+", default=[2, 3, 4, 5])
    parser.add_argument("--main-starts", type=int, default=10)
    parser.add_argument("--leave-cap-per-stratum", type=int, default=15)
    parser.add_argument("--leave-k", type=int, nargs="+", default=[2, 3, 4])
    parser.add_argument("--leave-starts", type=int, default=3)
    parser.add_argument("--n-iter", type=int, default=300)
    parser.add_argument("--tol", type=float, default=1e-3)
    parser.add_argument("--bootstrap-replications", type=int, default=500)
    parser.add_argument("--leave-bootstrap-replications", type=int, default=200)
    parser.add_argument("--hazard-control-days", type=int, default=20)
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    if args.main_starts < 2 or args.leave_starts < 2:
        raise ValueError("at least two random starts are required")
    if any(value not in range(2, 7) for value in [*args.main_k, *args.leave_k]):
        raise ValueError("HMM K values must be in 2..6")
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    TABLES_DIR.mkdir(parents=True, exist_ok=True)
    FIGURES_DIR.mkdir(parents=True, exist_ok=True)
    output_paths: list[Path] = []
    warnings_out: list[str] = []
    started_at = utc_now()
    started_clock = time.perf_counter()
    print("[Q2] loading OULAD registry and first 56 course days", flush=True)
    registry = load_registry()
    official_contract_counts = assert_official_termination_contract(registry)
    panel = load_history_panel(56)

    partitions, partition_metadata = main_partition(registry, MAIN_LANDMARK)
    print("[Q2] nested presentation selection: inner 2013 -> 2014B; untouched 2014J", flush=True)
    representation, comparison, starts, sampling_audit, fit_warnings = fit_nested_representation(
        panel,
        partitions,
        cap_per_stratum=args.main_cap_per_stratum,
        seed=args.seed,
        k_values=args.main_k,
        n_starts=args.main_starts,
        n_iter=args.n_iter,
        tol=args.tol,
        scope="main_presentation_nested",
    )
    warnings_out.extend(fit_warnings)

    outer_history = history_frame(panel, partitions["outer_test"], MAIN_LANDMARK)
    outer_bundle = bundle(outer_history, representation.encoder, "main:untouched_2014J")
    outer_mc_ll = representation.mc_model.score(outer_bundle).per_observation
    outer_hmm_ll = None
    if representation.hmm_model is not None:
        outer_hmm_ll = float(
            representation.hmm_model.score(outer_bundle.X, lengths=outer_bundle.lengths)
            / outer_bundle.n_observations
        )

    print("[Q2] history-conditioned future score and landmark outcome validation", flush=True)
    conditional_scores = conditional_future_scores(
        panel,
        partitions["outer_test"],
        representation,
        landmark=MAIN_LANDMARK,
        horizon=MAIN_HORIZON,
    )
    performance_parts: list[pd.DataFrame] = []
    prediction_parts: list[pd.DataFrame] = []
    for landmark in [14, 28, 56]:
        performance, predictions = evaluate_landmark(
            panel,
            registry,
            representation,
            landmark=landmark,
            bootstrap_replications=args.bootstrap_replications,
            seed=args.seed,
        )
        performance_parts.append(performance)
        prediction_parts.append(predictions)
    landmark_performance = pd.concat(performance_parts, ignore_index=True)
    all_predictions = pd.concat(prediction_parts, ignore_index=True)
    calibration = calibration_bins(all_predictions)

    print("[Q2] seven leave-module-out folds with fold-internal K selection", flush=True)
    (
        leave_module,
        state_alignment,
        leave_comparison,
        leave_starts,
        leave_warnings,
    ) = leave_module_validation(
        panel,
        registry,
        representation,
        seed=args.seed + 200_000,
        cap_per_stratum=args.leave_cap_per_stratum,
        k_values=args.leave_k,
        n_starts=args.leave_starts,
        n_iter=args.n_iter,
        tol=args.tol,
        bootstrap_replications=args.leave_bootstrap_replications,
    )
    warnings_out.extend(leave_warnings)
    comparison = pd.concat([comparison, leave_comparison], ignore_index=True)
    starts = pd.concat([starts, leave_starts], ignore_index=True)

    flow, bounds, post_activity = registration_flow_and_bounds(
        registry, panel, [14, 28, 56]
    )
    print("[Q2] sampling-fraction/frequency-weighted discrete-time termination model", flush=True)
    hazard_coefficients, hazard_metadata = discrete_time_termination_model(
        control_days_per_sequence=args.hazard_control_days
    )

    state_profiles = pd.DataFrame()
    if representation.hmm_model is not None:
        state_profiles = emission_profile_frame(representation.hmm_model, outer_bundle)
        state_profiles.insert(0, "partition", "untouched_2014J")
        state_profiles["state_semantics"] = "numbered platform-behavior state; no cognitive label"

    write_table("q2_sampling_audit.csv", sampling_audit, output_paths)
    write_table("q2_nested_model_selection.csv", comparison, output_paths)
    write_table("q2_hmm_random_starts.csv", starts, output_paths)
    write_table("q2_conditional_sequence_scores.csv", conditional_scores, output_paths)
    write_table("q2_landmark_performance.csv", landmark_performance, output_paths)
    write_table("q2_landmark_calibration.csv", calibration, output_paths)
    write_table("q2_leave_module_external_validation.csv", leave_module, output_paths)
    write_table("q2_leave_module_state_alignment.csv", state_alignment, output_paths)
    write_table("q2_state_emissions.csv", state_profiles, output_paths)
    write_table("q2_termination_flow.csv", flow, output_paths)
    write_table("q2_platform_active_population_bounds.csv", bounds, output_paths)
    write_table("q2_post_unregistration_activity.csv", post_activity, output_paths)
    write_table("q2_discrete_hazard_coefficients.csv", hazard_coefficients, output_paths)

    save_figures(comparison, calibration, leave_module, flow, output_paths)

    main_adverse = landmark_performance.loc[
        landmark_performance["validation_scope"].eq("untouched_2014J_landmark_28")
        & landmark_performance["target"].eq("adverse_final")
    ].copy()
    main_withdrawal = landmark_performance.loc[
        landmark_performance["validation_scope"].eq("untouched_2014J_landmark_28")
        & landmark_performance["target"].eq("withdrawal_next_28d_at_lm28")
    ].copy()
    development_student_ids = (
        set(partitions["inner_train"]["id_student"])
        | set(partitions["inner_validation"]["id_student"])
    )
    outer_sequence_ids = set(partitions["outer_test"]["sequence_id"])
    conditional_student_overlap = len(
        set(conditional_scores["id_student"]) & development_student_ids
    )
    conditional_non_outer_sequences = len(
        set(conditional_scores["sequence_id"]) - outer_sequence_ids
    )
    conditional_summary = {
        "n_complete_followup_sequences": int(len(conditional_scores)),
        "n_complete_followup_students": int(conditional_scores["id_student"].nunique()),
        "eligible_outer_sequences_before_complete_window_filter": int(
            len(partitions["outer_test"])
        ),
        "development_student_overlap": int(conditional_student_overlap),
        "non_outer_sequence_count": int(conditional_non_outer_sequences),
        "outer_sequence_membership_rule": (
            "subset of the student-disjoint untouched 2014J outer_test only"
        ),
        "mc_loglik_per_future_day": (
            float(conditional_scores["mc_conditional_loglik"].sum() / (len(conditional_scores) * MAIN_HORIZON))
            if len(conditional_scores)
            else None
        ),
        "hmm_loglik_per_future_day": (
            float(conditional_scores["hmm_conditional_loglik"].sum() / (len(conditional_scores) * MAIN_HORIZON))
            if len(conditional_scores) and conditional_scores["hmm_conditional_loglik"].notna().all()
            else None
        ),
        "censoring_note": (
            "Observation likelihood uses registrations with a complete future window; the separate termination model retains informative dropout."
        ),
    }

    expected_counts = {
        "official_registrations": 32593,
        "pre_course_unregistrations": 2678,
        "during_course_unregistrations": 7393,
        "post_unregistration_activity_registrations": 1279,
    }
    observed_counts = {
        "official_registrations": int(len(registry)),
        "pre_course_unregistrations": int(
            registry["unregistration_timing"].eq("pre_course").sum()
        ),
        "during_course_unregistrations": int(
            registry["unregistration_timing"].eq("during_course").sum()
        ),
        "post_unregistration_activity_registrations": int(
            registry["has_post_unregistration_vle_activity"].sum()
        ),
    }
    withdrawal_rows = landmark_performance.loc[
        landmark_performance["target"].astype(str).str.startswith("withdrawal_next_")
    ].copy()
    adverse_final_rows = landmark_performance.loc[
        landmark_performance["target"].eq("adverse_final")
    ].copy()
    expected_landmarks = {14, 28, 56}
    expected_withdrawal_targets = {
        "withdrawal_next_28d_at_lm14",
        "withdrawal_next_28d_at_lm28",
        "withdrawal_next_28d_at_lm56",
    }
    audit_checks = {
        "official_registration_count_preserved": observed_counts["official_registrations"] == expected_counts["official_registrations"],
        "pre_course_zero_length_count_preserved": observed_counts["pre_course_unregistrations"] == expected_counts["pre_course_unregistrations"],
        "during_course_unregistration_count_preserved": observed_counts["during_course_unregistrations"] == expected_counts["during_course_unregistrations"],
        "post_unregistration_activity_count_preserved": observed_counts["post_unregistration_activity_registrations"] == expected_counts["post_unregistration_activity_registrations"],
        "main_student_overlap_inner_train_validation_zero": partition_metadata["student_overlap_inner_train_validation"] == 0,
        "main_student_overlap_development_outer_zero": partition_metadata["student_overlap_development_outer"] == 0,
        "outer_test_is_2014J_only": set(partitions["outer_test"]["code_presentation"]) == {"2014J"},
        "sampling_declares_no_outcome_columns": bool((~sampling_audit["outcome_columns_used"]).all()),
        "all_landmark_probabilities_finite": bool(
            np.isfinite(all_predictions["predicted_probability"]).all()
            and all_predictions["predicted_probability"].between(0, 1).all()
        ),
        "leave_module_student_overlap_zero": bool(
            leave_module["student_overlap_train_test"].eq(0).all()
        ),
        "conditional_forecast_uses_complete_future_window": bool(
            conditional_scores["future_days"].eq(MAIN_HORIZON).all()
        ),
        "conditional_sequences_are_outer_subset": bool(
            conditional_non_outer_sequences == 0
        ),
        "conditional_students_disjoint_from_development": bool(
            conditional_student_overlap == 0
        ),
        "withdrawal_targets_cover_landmarks_14_28_56": bool(
            set(withdrawal_rows["landmark_days"].astype(int)) == expected_landmarks
            and set(withdrawal_rows["target"]) == expected_withdrawal_targets
        ),
        "withdrawal_forecast_horizon_always_28_days": bool(
            not withdrawal_rows.empty
            and withdrawal_rows["forecast_horizon_days"].eq(MAIN_HORIZON).all()
        ),
        "withdrawal_time_scopes_are_exact_28_day_windows": bool(
            not withdrawal_rows.empty
            and (
                withdrawal_rows["outcome_time_scope"]
                == withdrawal_rows["landmark_days"].astype(int).map(
                    lambda value: f"[{value},{value + MAIN_HORIZON}) course days"
                )
            ).all()
        ),
        "adverse_final_covers_landmarks_14_28_56": bool(
            not adverse_final_rows.empty
            and set(adverse_final_rows["landmark_days"].astype(int)) == expected_landmarks
        ),
        "adverse_final_forecast_horizon_is_empty": bool(
            not adverse_final_rows.empty
            and adverse_final_rows["forecast_horizon_days"].isna().all()
        ),
        "adverse_final_scope_is_course_final": bool(
            not adverse_final_rows.empty
            and adverse_final_rows["outcome_time_scope"].eq("course_final_outcome").all()
        ),
        "states_remain_numbered": bool(
            state_profiles.empty or state_profiles["state"].str.match(r"State \d+").all()
        ),
    }
    audit_warnings = [
        "HMM estimation uses an outcome-blind capacity sample; outcome validation uses the full eligible cohort.",
        "Leave-module folds use a reduced K grid; they are external-validity stress tests, not final state-count discovery.",
        "2014J simultaneously indexes a later presentation and calendar period, so presentation and year transport are not separately identified.",
        "Platform-Inactive cannot distinguish off-platform learning from disengagement.",
        "All outcome and termination relations are associational, not causal strategy effects.",
    ]
    if args.main_starts < 10:
        audit_warnings.insert(
            1,
            "Main HMM uses fewer than 10 random starts per K; convergence tables must accompany claims.",
        )
    audit_status = "PASS_WITH_WARNINGS" if all(audit_checks.values()) else "FAIL"
    audit_path = RESULTS_DIR / "q2_independent_audit.json"
    write_json(
        audit_path,
        {
            "schema_version": SCHEMA_VERSION,
            "generated_by": str(SCRIPT_PATH.relative_to(PROJECT_ROOT)).replace("\\", "/"),
            "generated_at": utc_now(),
            "question_id": "q2",
            "status": audit_status,
            "checks": audit_checks,
            "hard_audit_details": {
                "conditional_development_student_overlap": int(conditional_student_overlap),
                "conditional_non_outer_sequence_count": int(conditional_non_outer_sequences),
                "withdrawal_landmarks": sorted(
                    withdrawal_rows["landmark_days"].astype(int).unique().tolist()
                ),
                "withdrawal_forecast_horizons": sorted(
                    withdrawal_rows["forecast_horizon_days"].dropna().astype(int).unique().tolist()
                ),
                "adverse_final_forecast_horizon_nonempty_count": int(
                    adverse_final_rows["forecast_horizon_days"].notna().sum()
                ),
            },
            "expected_counts": expected_counts,
            "observed_counts": observed_counts,
            "warnings": audit_warnings + warnings_out,
        },
    )
    output_paths.append(audit_path)

    summary_path = RESULTS_DIR / "q2_run_summary.json"
    write_json(
        summary_path,
        {
            "schema_version": SCHEMA_VERSION,
            "generated_by": str(SCRIPT_PATH.relative_to(PROJECT_ROOT)).replace("\\", "/"),
            "generated_at": utc_now(),
            "question_id": "q2",
            "status": "computed_external_replication" if audit_status != "FAIL" else "audit_failed",
            "dataset": "OULAD",
            "analysis_unit": "student-module-presentation-course-day",
            "partition": partition_metadata,
            "capacity_design": {
                "main_cap_per_module_presentation_partition": args.main_cap_per_stratum,
                "main_k_grid": args.main_k,
                "main_random_starts": args.main_starts,
                "leave_module_cap_per_stratum": args.leave_cap_per_stratum,
                "leave_module_k_grid": args.leave_k,
                "leave_module_random_starts": args.leave_starts,
                "selection_uses_future_outcomes": False,
            },
            "observation_encoder": representation.encoder.metadata(),
            "selected_sequence_representation": {
                "kind": representation.chosen_kind,
                "inner_selected_k": representation.selected_k,
                "inner_validation_hmm_minus_mc_loglik_per_day": representation.inner_delta_vs_mc,
                "minimum_substantive_margin": MC_IMPROVEMENT_MARGIN,
                "selection_status": representation.selection_status,
                "refit_status": representation.refit_status,
                "outer_2014J_history_mc_loglik_per_day": outer_mc_ll,
                "outer_2014J_history_hmm_loglik_per_day": outer_hmm_ll,
                "state_naming": "State 1..K only; emissions are platform-behavior evidence",
            },
            "conditional_future_prediction": conditional_summary,
            "main_adverse_outcome_performance": main_adverse.to_dict(orient="records"),
            "main_withdrawal_horizon_performance": main_withdrawal.to_dict(orient="records"),
            "termination_process": hazard_metadata,
            "official_termination_counts": observed_counts,
            "official_contract_assertion": official_contract_counts,
            "leave_module_folds": int(leave_module["held_out_module"].nunique()),
            "audit_status": audit_status,
            "claim_boundary": [
                "Hidden states are numbered platform-behavior states, not observed cognition.",
                "Platform-Inactive is not equivalent to no learning.",
                "Administrative termination is modeled separately and never encoded as a final latent state.",
                "Outcome associations and simulated scenarios do not identify causal strategy effects.",
                "Generalization is limited to the observed OULAD modules/presentations and the stated encoding.",
            ],
            "warnings": audit_warnings + warnings_out,
        },
    )
    output_paths.append(summary_path)

    md_path = RESULTS_DIR / "Q2_OULAD复现与外部效度说明.md"
    observed_main = main_adverse.loc[main_adverse["representation"].eq("Observable-history")].iloc[0]
    hmm_main = main_adverse.loc[
        main_adverse["representation"].eq("Observable+numbered-HMM-history")
    ]
    hmm_text = (
        f"编号 HMM 增量表征在外测中的 AUC 为 {hmm_main.iloc[0]['auc']:.3f}、Brier 为 {hmm_main.iloc[0]['brier']:.3f}。"
        if not hmm_main.empty
        else "HMM 未通过严格选择/重拟合门，因此没有把潜在状态表征带入结局模型。"
    )
    hazard_focus = hazard_coefficients.loc[
        hazard_coefficients["term"].isin(
            ["inactive_prior_per10pp", "assessment_prior_per10pp", "communication_prior_per10pp"]
        )
    ]
    hazard_lines = "\n".join(
        f"- `{row.term}`：OR={row.odds_ratio:.3f}，学生聚类稳健95% CI [{row.or_ci_low:.3f}, {row.or_ci_high:.3f}]。"
        for row in hazard_focus.itertuples(index=False)
    )
    md_text = f"""# Q2：OULAD 复现与外部效度结果

## 结论先行

本分析采用严格的三层时间切分：2013B/2013J 用于内层训练，2014B 仅用于模型选择，2014J 在选定模型并重新拟合前保持未触碰；三部分学生标识无重叠。HMM 的状态始终写作 State 1--K，未命名为“专注”或“深度思考”。内层模型选择允许可观测 MC 获胜，最终表征为 **{representation.chosen_kind}**，内层候选 K 为 **{representation.selected_k}**，固定 K 重拟合状态为 `{representation.refit_status}`。

## 固定 landmark 外测

以开课后前28日为历史、随后28日为预测窗。2014J 未触碰外测中，可观测历史表征对官方不利结局（Fail/Withdrawn）的 AUC 为 {observed_main['auc']:.3f}（学生聚类bootstrap 95% CI {observed_main['auc_ci_low']:.3f}--{observed_main['auc_ci_high']:.3f}），Brier 为 {observed_main['brier']:.3f}。{hmm_text}

14日、28日与56日 landmark 的行政注销目标均采用各自 landmark 后固定28日的区间；`adverse_final` 则始终是整门课程结束时的 Fail/Withdrawn 终局，不把它误写成28日结局。三个 landmark 使用同一个已锁定的观测/HMM测量模型，只重新估计下游结局关联，避免用敏感性结果反向挑选状态数。完整结果见 `q2_landmark_performance.csv`，其中 `outcome_time_scope` 明确区分两类时间口径，校准截距、斜率、ECE 与分箱数据均已保留。

## 信息性终止

全部32,593条官方注册均进入终止流程审计：2,678条课前注销没有合法第0日，因此只进入总体界限而不制造状态；7,393条课中注销作为行政终止事件；1,279条注册存在注销后VLE活动，这些点击仅进入窗外审计，不回填主序列。离散时间风险模型使用此前7日的严格滞后观测，非事件人日采用确定性近似等距抽样，并按各注册非事件人日总数与入样数之比赋予频率权重；标准误按学生聚类：

{hazard_lines}

这些比值是条件关联，不是策略干预的因果效应。

## 跨课程外部效度

AAA--GGG七个模块逐一留出。每一折均在其余模块内部重新划分学生互斥的训练/验证集、重新估计观测阈值并选择 K；被留出模块及其中学生不进入拟合。折间若 K 不同，只报告发射分布的Hungarian/JS匹配，不强行声称状态一一等价。AUC、Brier、校准与学生聚类区间见 `q2_leave_module_external_validation.csv`。

## 证据边界

- Platform-Inactive 只表示平台没有日志，不能区分平台外学习与偏离学习。
- 结局标签不参与HMM训练、抽样或状态命名，避免用结局循环定义“有利状态”。
- HMM采用结局盲容量样本估计；全部合格学生用于2014J和留模块外测。
- 2014J同时代表较晚开课批次与日历时期，批次迁移和年份迁移不能完全分离。
- 本节支持预测/描述性外部效度，不支持因果策略优化结论。

独立自审计状态：**{audit_status}**。
"""
    md_path.write_text(md_text, encoding="utf-8")
    output_paths.append(md_path)

    finished_at = utc_now()
    manifest_path = RESULTS_DIR / "q2_execution_manifest.json"
    write_json(
        manifest_path,
        {
            "schema_version": SCHEMA_VERSION,
            "generated_by": str(SCRIPT_PATH.relative_to(PROJECT_ROOT)).replace("\\", "/"),
            "question_id": "q2",
            "status": audit_status,
            "started_at": started_at,
            "finished_at": finished_at,
            "elapsed_seconds": time.perf_counter() - started_clock,
            "run_command": [sys.executable, str(SCRIPT_PATH), *sys.argv[1:]],
            "working_directory": str(Path.cwd()),
            "arguments": vars(args),
            "python": {
                "implementation": platform.python_implementation(),
                "version": platform.python_version(),
                "platform": platform.platform(),
            },
            "source_code": artifact_record(SCRIPT_PATH),
            "dependencies": [artifact_record(SCRIPT_PATH.with_name("sequence_models.py"))],
            "inputs": [
                artifact_record(PANEL_PATH),
                artifact_record(SENSITIVITY_PATH),
                artifact_record(POST_ACTIVITY_PATH),
                artifact_record(QUALITY_PATH),
            ],
            "outputs": [artifact_record(path) for path in sorted(output_paths)],
            "random_seed": args.seed,
            "rerun_note": "Any source, input, or argument change requires a complete rerun.",
        },
    )
    print(
        f"[Q2 {audit_status}] selected={representation.chosen_kind}, K={representation.selected_k}, elapsed={time.perf_counter() - started_clock:.1f}s",
        flush=True,
    )
    print(f"summary: {summary_path.relative_to(PROJECT_ROOT)}", flush=True)
    return 0 if audit_status != "FAIL" else 2


if __name__ == "__main__":
    raise SystemExit(main())
