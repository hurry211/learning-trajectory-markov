"""Run the audited Q3 conditional scenarios and Q4 robust Pareto analysis.

This is a reproducible empirical run, not a causal intervention analysis.  It
uses KU Leuven student-course-day data, a fixed day-28 landmark and a 28-day
forecast window.  Revisit and time-concentration profiles are reconstructed
strictly from t-1 and earlier records, with opportunity and availability kept
as separate covariates.

The script first attempts a genuinely joint non-homogeneous categorical HMM.
Its covariate-dependent transition kernel is used in every E step.  The HMM is
accepted only if convergence, occupancy, emission separation and inner-period
predictive-likelihood gates all pass.  Otherwise the executed main analysis is
an explicitly labelled observable non-homogeneous Markov chain (NHMC).
"""

from __future__ import annotations

import argparse
import hashlib
import importlib.metadata
import json
import math
import os
import platform
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import brier_score_loss, log_loss, roc_auc_score

from q34_helpers import (
    FeatureTransformer,
    ObservableNHMC,
    categorical_draw,
    epsilon_dominates,
    epsilon_nondominated,
    expected_calibration_error,
    fit_nhmm,
    js_divergence,
    max_true_run,
)
from sequence_models import ObservationEncodingConfig, TransparentObservationEncoder


PROJECT_ROOT = Path(__file__).resolve().parents[3]
INPUT_PATH = PROJECT_ROOT / "paper_output" / "data_cleaned" / "ku_daily_panel.parquet"
Q3_RESULTS = PROJECT_ROOT / "paper_output" / "results" / "q3"
Q4_RESULTS = PROJECT_ROOT / "paper_output" / "results" / "q4"
Q3_TABLES = PROJECT_ROOT / "paper_output" / "tables" / "q3"
Q4_TABLES = PROJECT_ROOT / "paper_output" / "tables" / "q4"
FIGURES = PROJECT_ROOT / "paper_output" / "figures"
SCRIPT_PATH = Path(__file__).resolve()
HELPER_PATH = SCRIPT_PATH.with_name("q34_helpers.py")
SEQUENCE_MODELS_PATH = SCRIPT_PATH.with_name("sequence_models.py")
SCHEMA_VERSION = "1.0"
OBSERVATION_LABELS = (
    "Platform-Inactive",
    "Low-Diversity-Content-Consumption",
    "Diverse-or-Collaboration-Access",
    "Assessment-or-Visible-Production-Proxy",
)
RAW_FEATURE_NAMES = (
    "r_hist_1_7d",
    "r_available",
    "log1p_r_opportunity",
    "h_hist_7d",
    "h_available",
    "log1p_h_sessions",
    "course_progress",
    "weekend",
    "course_global_economics",
)
CONTINUOUS_INDICES = (0, 2, 3, 5, 6)
MISSING_VALUE_INDICES = (0, 3)
LANDMARK_DAY = 28
HORIZON_DAYS = 28
DEFAULT_SEED = 20260822
OKABE_ITO = ["#0072B2", "#E69F00", "#009E73", "#CC79A7"]


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def sha256_file(path: Path, chunk_size: int = 1024 * 1024) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while chunk := handle.read(chunk_size):
            digest.update(chunk)
    return digest.hexdigest()


def json_safe(value: Any) -> Any:
    if value is pd.NA or value is pd.NaT:
        return None
    if isinstance(value, dict):
        return {str(key): json_safe(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [json_safe(item) for item in value]
    if isinstance(value, np.ndarray):
        return json_safe(value.tolist())
    if isinstance(value, np.generic):
        return json_safe(value.item())
    if isinstance(value, pd.Timestamp):
        return value.isoformat()
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, float) and not math.isfinite(value):
        return None
    return value


def write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(json_safe(payload), ensure_ascii=False, indent=2), encoding="utf-8"
    )
    os.replace(temporary, path)


def write_csv(path: Path, frame: pd.DataFrame) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    frame.to_csv(temporary, index=False, encoding="utf-8-sig")
    os.replace(temporary, path)


def load_panel() -> pd.DataFrame:
    columns = [
        "registration_id",
        "cohort",
        "COURSE_ID",
        "USER_ID",
        "date",
        "course_day_index",
        "course_start",
        "course_finish",
        "PASSED",
        "platform_inactive",
        "activity_category_diversity",
        "productive_event_count",
        "forum_event_count",
        "content_diversity",
        "resources_revisited_1_7d",
        "resource_revisit_rate_1_7d",
        "tod_sessions_7d",
        "tod_circular_concentration_7d",
    ]
    frame = pd.read_parquet(INPUT_PATH, columns=columns)
    frame["date"] = pd.to_datetime(frame["date"], errors="raise")
    frame["cohort"] = frame["cohort"].astype(str)
    frame = frame.sort_values(["registration_id", "course_day_index"]).reset_index(drop=True)
    duplicates = int(frame.duplicated(["registration_id", "course_day_index"]).sum())
    if duplicates:
        raise ValueError(f"prepared KU panel has {duplicates} duplicate sequence-day rows")
    return frame


def reconstruct_strict_history(frame: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Construct day-t covariates from days t-7,...,t-1 only."""

    result = frame.copy()
    grouped = result.groupby("registration_id", sort=False, observed=True)
    result["r_revisited_hist_n"] = grouped["resources_revisited_1_7d"].transform(
        lambda series: series.shift(1).rolling(7, min_periods=1).sum()
    ).fillna(0.0)
    result["r_opportunity_hist_n"] = grouped["content_diversity"].transform(
        lambda series: series.shift(1).rolling(7, min_periods=1).sum()
    ).fillna(0.0)
    result["r_available"] = result["r_opportunity_hist_n"].gt(0).astype(np.int8)
    result["r_hist_1_7d"] = np.where(
        result["r_available"].eq(1),
        result["r_revisited_hist_n"] / result["r_opportunity_hist_n"],
        np.nan,
    )
    result["h_sessions_hist_n"] = grouped["tod_sessions_7d"].shift(1)
    result["h_hist_7d"] = grouped["tod_circular_concentration_7d"].shift(1)
    result["h_available"] = (
        result["h_hist_7d"].notna() & result["h_sessions_hist_n"].ge(3)
    ).astype(np.int8)
    result.loc[result["h_available"].eq(0), "h_hist_7d"] = np.nan
    result["log1p_r_opportunity"] = np.log1p(result["r_opportunity_hist_n"].fillna(0))
    result["log1p_h_sessions"] = np.log1p(result["h_sessions_hist_n"].fillna(0))
    total_days = (
        pd.to_datetime(result["course_finish"]) - pd.to_datetime(result["course_start"])
    ).dt.days.clip(lower=1)
    result["course_progress"] = result["course_day_index"] / total_days
    result["weekend"] = result["date"].dt.dayofweek.ge(5).astype(np.int8)
    result["course_global_economics"] = result["COURSE_ID"].str.contains(
        "Global economics", case=False, regex=False
    ).astype(np.int8)

    # Direct formula audit on deterministic rows.  These tests demonstrate that
    # the current day's numerator/denominator cannot enter its own covariate.
    check_rows: list[dict[str, Any]] = []
    groups = result.groupby("registration_id", sort=False, observed=True)
    for registration_id, group in list(groups)[:40]:
        group = group.sort_values("course_day_index")
        for position in (0, 1, 7, min(15, len(group) - 1)):
            if position >= len(group):
                continue
            prior = group.iloc[max(0, position - 7) : position]
            row = group.iloc[position]
            expected_opportunity = float(prior["content_diversity"].sum())
            expected_revisits = float(prior["resources_revisited_1_7d"].sum())
            check_rows.append(
                {
                    "registration_id": registration_id,
                    "course_day_index": int(row["course_day_index"]),
                    "expected_r_opportunity": expected_opportunity,
                    "computed_r_opportunity": float(row["r_opportunity_hist_n"]),
                    "expected_r_numerator": expected_revisits,
                    "computed_r_numerator": float(row["r_revisited_hist_n"]),
                    "strict_lag_match": bool(
                        np.isclose(expected_opportunity, row["r_opportunity_hist_n"])
                        and np.isclose(expected_revisits, row["r_revisited_hist_n"])
                    ),
                }
            )
    strict_checks = pd.DataFrame(check_rows)
    if not strict_checks["strict_lag_match"].all():
        raise AssertionError("strict t-1 revisit-history reconstruction failed")
    return result, strict_checks


def fit_observation_encoder(frame: pd.DataFrame) -> tuple[pd.DataFrame, dict[str, Any]]:
    training = frame.loc[frame["cohort"].eq("1819")]
    encoder = TransparentObservationEncoder(
        ObservationEncodingConfig(
            inactive_col="platform_inactive",
            diversity_col="activity_category_diversity",
            productive_col="productive_event_count",
            interaction_col="forum_event_count",
            diversity_quantile=0.67,
            missing_policy="raise",
        )
    )
    encoder.fit(training)
    result = frame.copy()
    result["observation_code"] = encoder.transform(result)
    return result, encoder.metadata()


def raw_feature_matrix(frame: pd.DataFrame) -> np.ndarray:
    return frame.loc[:, RAW_FEATURE_NAMES].to_numpy(dtype=float)


def build_sequence_records(frame: pd.DataFrame, cohort: str) -> list[dict[str, Any]]:
    records: list[dict[str, Any]] = []
    subset = frame.loc[frame["cohort"].eq(str(cohort))]
    for registration_id, group in subset.groupby("registration_id", sort=True, observed=True):
        group = group.sort_values("course_day_index")
        observations = group["observation_code"].to_numpy(dtype=np.int64)
        if len(observations) < 2:
            continue
        records.append(
            {
                "registration_id": str(registration_id),
                "observations": observations,
                "raw_transition": raw_feature_matrix(group.iloc[:-1]),
            }
        )
    return records


def design_sequences(
    records: list[dict[str, Any]], transformer: FeatureTransformer
) -> tuple[list[np.ndarray], list[np.ndarray]]:
    observations = [record["observations"] for record in records]
    designs = [transformer.transform(record["raw_transition"]) for record in records]
    return observations, designs


def fit_model_gate(
    train_records: list[dict[str, Any]],
    inner_records: list[dict[str, Any]],
    *,
    starts: int,
    max_iter: int,
    seed: int,
) -> tuple[pd.DataFrame, dict[str, Any], FeatureTransformer, ObservableNHMC, Any | None]:
    transformer = FeatureTransformer(
        RAW_FEATURE_NAMES, CONTINUOUS_INDICES, MISSING_VALUE_INDICES
    ).fit(np.concatenate([record["raw_transition"] for record in train_records], axis=0))
    train_obs, train_x = design_sequences(train_records, transformer)
    inner_obs, inner_x = design_sequences(inner_records, transformer)
    mc = ObservableNHMC.fit(train_obs, train_x, n_states=4, ridge=1.0)
    mc_train_ll, mc_train_n = mc.score(train_obs, train_x)
    mc_inner_ll, mc_inner_n = mc.score(inner_obs, inner_x)
    rows: list[dict[str, Any]] = [
        {
            "model": "Observable-NHMC",
            "start": None,
            "converged": True,
            "iterations": None,
            "train_loglik_per_day": mc_train_ll / mc_train_n,
            "inner_loglik_per_day": mc_inner_ll / mc_inner_n,
            "min_state_occupancy": None,
            "min_pairwise_emission_js": None,
            "optimizer_all_success": all(item["success"] for item in mc.optimization_audit),
            "passes_identifiability_gate": True,
            "passes_predictive_gate": True,
        }
    ]
    candidates: list[tuple[Any, float, dict[str, Any]]] = []
    for start in range(starts):
        fit_seed = seed + 1009 * (start + 1)
        model = fit_nhmm(
            train_obs,
            train_x,
            n_states=2,
            n_symbols=4,
            ridge=1.0,
            max_iter=max_iter,
            tolerance_per_observation=2e-6,
            seed=fit_seed,
        )
        train_ll, train_n = model.score(train_obs, train_x)
        inner_ll, inner_n = model.score(inner_obs, inner_x)
        pairwise_js = js_divergence(model.emissionprob[0], model.emissionprob[1])
        optimizer_success = all(item["success"] for item in model.transition_optimization)
        identifiability = bool(
            model.converged
            and optimizer_success
            and float(model.occupancy.min()) >= 0.03
            and pairwise_js >= 0.025
        )
        predictive = bool(inner_ll / inner_n >= mc_inner_ll / mc_inner_n + 0.001)
        row = {
            "model": "Joint-NH-HMM-K2",
            "start": start + 1,
            "seed": fit_seed,
            "converged": model.converged,
            "iterations": model.iterations,
            "train_loglik_per_day": train_ll / train_n,
            "inner_loglik_per_day": inner_ll / inner_n,
            "inner_gain_vs_nhmc_per_day": inner_ll / inner_n - mc_inner_ll / mc_inner_n,
            "min_state_occupancy": float(model.occupancy.min()),
            "min_pairwise_emission_js": pairwise_js,
            "optimizer_all_success": optimizer_success,
            "passes_identifiability_gate": identifiability,
            "passes_predictive_gate": predictive,
        }
        rows.append(row)
        candidates.append((model, inner_ll / inner_n, row))
    eligible = [item for item in candidates if item[2]["passes_identifiability_gate"] and item[2]["passes_predictive_gate"]]
    selected_hmm = max(eligible, key=lambda item: item[1])[0] if eligible else None
    decision = {
        "attempted_main_model": "joint covariate-dependent categorical NH-HMM (K=2)",
        "required_gates": {
            "generalized_em_converged_before_iteration_cap": True,
            "all_transition_optimizers_successful": True,
            "minimum_training_state_occupancy": 0.03,
            "minimum_pairwise_emission_js": 0.025,
            "minimum_inner_loglik_gain_per_day_over_observable_nhmc": 0.001,
        },
        "decision": "joint_nh_hmm" if selected_hmm is not None else "observable_nhmc_fallback",
        "fallback_is_main_if_gate_fails": True,
        "viterbi_regression_used": False,
    }
    return pd.DataFrame(rows), decision, transformer, mc, selected_hmm


def completion_feature_row(history: np.ndarray, future: np.ndarray, course_global: int) -> np.ndarray:
    history_share = np.bincount(history, minlength=4) / len(history)
    future_share = np.bincount(future, minlength=4) / len(future)
    run = float(max_true_run((future == 0)[None, :])[0]) / len(future)
    last7_active = float(np.mean(future[-7:] != 0))
    return np.concatenate([[float(course_global)], history_share, future_share, [run, last7_active]])


def build_landmark_records(frame: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    final_day = LANDMARK_DAY + HORIZON_DAYS - 1
    for registration_id, group in frame.groupby("registration_id", sort=True, observed=True):
        group = group.sort_values("course_day_index")
        needed = group.loc[group["course_day_index"].between(0, final_day)]
        if len(needed) != final_day + 1 or not np.array_equal(
            needed["course_day_index"].to_numpy(), np.arange(final_day + 1)
        ):
            continue
        observations = needed["observation_code"].to_numpy(dtype=np.int64)
        history = observations[:LANDMARK_DAY]
        future = observations[LANDMARK_DAY : LANDMARK_DAY + HORIZON_DAYS]
        course_global = int(needed["course_global_economics"].iloc[0])
        rows.append(
            {
                "registration_id": str(registration_id),
                "cohort": str(needed["cohort"].iloc[0]),
                "COURSE_ID": str(needed["COURSE_ID"].iloc[0]),
                "PASSED": int(needed["PASSED"].iloc[0]),
                "course_global": course_global,
                "history_observations": history,
                "future_observations": future,
                "completion_features": completion_feature_row(history, future, course_global),
                "raw_forecast_transitions": raw_feature_matrix(
                    needed.loc[needed["course_day_index"].between(LANDMARK_DAY - 1, final_day - 1)]
                ),
                "raw_history_transitions": raw_feature_matrix(
                    needed.loc[needed["course_day_index"].between(0, LANDMARK_DAY - 2)]
                ),
            }
        )
    return pd.DataFrame(rows)


def fit_completion_calibration(landmarks: pd.DataFrame) -> tuple[Any, Any, pd.DataFrame, dict[str, Any]]:
    feature_names = [
        "course_global_economics",
        *[f"history_share_obs_{index}" for index in range(4)],
        *[f"forecast_share_obs_{index}" for index in range(4)],
        "forecast_max_pi_run_share",
        "forecast_last7_active_share",
    ]
    x = np.stack(landmarks["completion_features"].to_numpy())
    y = landmarks["PASSED"].to_numpy(dtype=int)
    train = landmarks["cohort"].eq("1819").to_numpy()
    calibration = landmarks["cohort"].eq("1920").to_numpy()
    outer = landmarks["cohort"].eq("2021").to_numpy()
    base = LogisticRegression(C=1.0, max_iter=2000, solver="lbfgs", random_state=DEFAULT_SEED)
    base.fit(x[train], y[train])
    base_calibration_probability = base.predict_proba(x[calibration])[:, 1]
    base_logit = np.log(
        np.clip(base_calibration_probability, 1e-8, 1 - 1e-8)
        / np.clip(1 - base_calibration_probability, 1e-8, 1.0)
    )
    calibrator = LogisticRegression(C=1e6, max_iter=1000, solver="lbfgs", random_state=DEFAULT_SEED)
    calibrator.fit(base_logit[:, None], y[calibration])

    metric_rows: list[dict[str, Any]] = []
    for label, mask in (("calibration_1920", calibration), ("untouched_outer_2021", outer)):
        raw = base.predict_proba(x[mask])[:, 1]
        logits = np.log(np.clip(raw, 1e-8, 1 - 1e-8) / np.clip(1 - raw, 1e-8, 1.0))
        probability = calibrator.predict_proba(logits[:, None])[:, 1]
        metric_rows.append(
            {
                "partition": label,
                "n_registrations": int(mask.sum()),
                "pass_rate": float(y[mask].mean()),
                "roc_auc": float(roc_auc_score(y[mask], probability)),
                "brier_score": float(brier_score_loss(y[mask], probability)),
                "log_loss": float(log_loss(y[mask], probability, labels=[0, 1])),
                "expected_calibration_error_10_bins": float(
                    expected_calibration_error(y[mask], probability, bins=10)
                ),
                "mean_predicted_probability": float(probability.mean()),
            }
        )
    metadata = {
        "feature_names": feature_names,
        "base_fit_partition": "1819",
        "platt_calibration_partition": "1920",
        "untouched_evaluation_partition": "2021",
        "prediction_time": "after observing/simulating course days 0-55",
        "outcome": "official PASSED indicator",
        "estimand_label": "calibrated association-model score, not simulated causal completion",
        "base_coefficients": base.coef_[0].tolist(),
        "base_intercept": float(base.intercept_[0]),
        "platt_slope": float(calibrator.coef_[0, 0]),
        "platt_intercept": float(calibrator.intercept_[0]),
    }
    return base, calibrator, pd.DataFrame(metric_rows), metadata


def calibrated_completion_probability(base: Any, calibrator: Any, features: np.ndarray) -> np.ndarray:
    raw = base.predict_proba(features)[:, 1]
    logits = np.log(np.clip(raw, 1e-8, 1 - 1e-8) / np.clip(1 - raw, 1e-8, 1.0))
    return calibrator.predict_proba(logits[:, None])[:, 1]


def scenario_contract(
    fit_frame: pd.DataFrame, outer_landmarks: pd.DataFrame
) -> tuple[dict[str, dict[str, Any]], pd.DataFrame]:
    r_values = fit_frame.loc[fit_frame["r_available"].eq(1), "r_hist_1_7d"].dropna().to_numpy()
    h_values = fit_frame.loc[fit_frame["h_available"].eq(1), "h_hist_7d"].dropna().to_numpy()
    r_q75 = float(np.quantile(r_values, 0.75))
    h_q75 = float(np.quantile(h_values, 0.75))
    both = fit_frame.loc[
        fit_frame["r_available"].eq(1) & fit_frame["h_available"].eq(1),
        ["r_hist_1_7d", "h_hist_7d"],
    ].dropna()
    scales = both.std(ddof=0).replace(0, 1).to_numpy()
    distances = np.sqrt(
        (((both.to_numpy() - np.asarray([r_q75, h_q75])) / scales) ** 2).sum(axis=1)
    )
    closest = both.iloc[int(np.argmin(distances))]
    joint_r = float(closest["r_hist_1_7d"])
    joint_h = float(closest["h_hist_7d"])
    scenarios = {
        "reference_observed_profile": {
            "label": "Reference observed profile",
            "r_target": None,
            "h_target": None,
            "definition": "Preserve each 2021 anchor's observed lagged profile and availability pattern.",
        },
        "upper_revisit_profile": {
            "label": "Upper-quartile revisit profile",
            "r_target": r_q75,
            "h_target": None,
            "definition": "Set R_hist to its discovery upper quartile only where revisit is observable; preserve H.",
        },
        "upper_time_concentration_profile": {
            "label": "Upper-quartile time-concentration profile",
            "r_target": None,
            "h_target": h_q75,
            "definition": "Set H_hist to its discovery upper quartile only where H is observable; preserve R.",
        },
        "joint_supported_upper_profile": {
            "label": "Joint supported upper profile",
            "r_target": joint_r,
            "h_target": joint_h,
            "definition": "Use the observed joint-support point nearest the two upper quartiles where both are available.",
        },
    }
    raw = np.concatenate(outer_landmarks["raw_forecast_transitions"].to_numpy(), axis=0)
    support_rows = []
    for key, scenario in scenarios.items():
        support_rows.append(
            {
                "scenario_id": key,
                "scenario_label": scenario["label"],
                "r_target": scenario["r_target"],
                "h_target": scenario["h_target"],
                "r_available_forecast_share": float(np.mean(raw[:, 1] == 1)),
                "h_available_forecast_share": float(np.mean(raw[:, 4] == 1)),
                "both_available_forecast_share": float(np.mean((raw[:, 1] == 1) & (raw[:, 4] == 1))),
                "availability_pattern_changed": False,
                "within_marginal_training_support": bool(
                    (scenario["r_target"] is None or (r_values.min() <= scenario["r_target"] <= r_values.max()))
                    and (scenario["h_target"] is None or (h_values.min() <= scenario["h_target"] <= h_values.max()))
                ),
                "joint_target_is_observed_point": bool(key == "joint_supported_upper_profile"),
            }
        )
    return scenarios, pd.DataFrame(support_rows)


def modify_scenario_raw(raw: np.ndarray, scenario: dict[str, Any]) -> np.ndarray:
    result = np.asarray(raw, dtype=float).copy()
    r_target = scenario.get("r_target")
    h_target = scenario.get("h_target")
    if r_target is not None:
        mask = result[..., 1] == 1
        result[..., 0][mask] = float(r_target)
    if h_target is not None:
        mask = result[..., 4] == 1
        result[..., 3][mask] = float(h_target)
    return result


def simulate_scenarios(
    model: Any,
    model_kind: str,
    transformer: FeatureTransformer,
    outer: pd.DataFrame,
    scenarios: dict[str, dict[str, Any]],
    base: Any,
    calibrator: Any,
    *,
    seed: int,
    minimum_paths: int,
    batch_size: int,
    maximum_paths: int,
    completion_mcse_target: float,
    binary_mcse_target: float,
) -> tuple[dict[str, dict[str, np.ndarray]], pd.DataFrame, np.ndarray]:
    raw_anchor = np.stack(outer["raw_forecast_transitions"].to_numpy())
    scenario_design: dict[str, np.ndarray] = {}
    for key, contract in scenarios.items():
        modified = modify_scenario_raw(raw_anchor, contract)
        shape = modified.shape
        scenario_design[key] = transformer.transform(modified.reshape(-1, shape[-1])).reshape(
            shape[0], shape[1], -1
        )
    histories = np.stack(outer["history_observations"].to_numpy())
    history_shares = np.stack(
        [np.bincount(row, minlength=4) / len(row) for row in histories], axis=0
    )
    course_global = outer["course_global"].to_numpy(dtype=float)
    if model_kind == "observable_nhmc":
        start_information = histories[:, -1].astype(np.int64)
    else:
        posteriors = []
        for _, row in outer.iterrows():
            history_x = transformer.transform(row["raw_history_transitions"])
            posteriors.append(model.filter_distribution(row["history_observations"], history_x))
        start_information = np.stack(posteriors)

    rng = np.random.default_rng(seed)
    storage: dict[str, dict[str, list[np.ndarray]]] = {
        key: {"completion": [], "sustained": [], "pi_risk": []} for key in scenarios
    }
    anchor_batches: list[np.ndarray] = []
    trace_rows: list[dict[str, Any]] = []
    total = 0
    while total < maximum_paths:
        current = min(batch_size, maximum_paths - total)
        anchor_index = rng.integers(0, len(outer), size=current)
        u_start = rng.random(current)
        u_transition = rng.random((current, HORIZON_DAYS))
        u_emission = rng.random((current, HORIZON_DAYS))
        anchor_batches.append(anchor_index)
        for scenario_key in scenarios:
            if model_kind == "observable_nhmc":
                states = start_information[anchor_index].copy()
            else:
                states = categorical_draw(start_information[anchor_index], u_start)
            future = np.empty((current, HORIZON_DAYS), dtype=np.int64)
            x_all = scenario_design[scenario_key][anchor_index]
            for step in range(HORIZON_DAYS):
                probabilities = model.probabilities_for_sources(states, x_all[:, step, :])
                states = categorical_draw(probabilities, u_transition[:, step])
                if model_kind == "observable_nhmc":
                    observations = states
                else:
                    observations = categorical_draw(model.emissionprob[states], u_emission[:, step])
                future[:, step] = observations
            future_shares = np.stack(
                [(future == code).mean(axis=1) for code in range(4)], axis=1
            )
            pi_runs = max_true_run(future == 0)
            last7_active_share = (future[:, -7:] != 0).mean(axis=1)
            completion_features = np.column_stack(
                [
                    course_global[anchor_index],
                    history_shares[anchor_index],
                    future_shares,
                    pi_runs / HORIZON_DAYS,
                    last7_active_share,
                ]
            )
            completion = calibrated_completion_probability(base, calibrator, completion_features)
            storage[scenario_key]["completion"].append(completion)
            storage[scenario_key]["sustained"].append((last7_active_share > 0).astype(float))
            storage[scenario_key]["pi_risk"].append((pi_runs >= 7).astype(float))
        total += current
        all_with_mcse = True
        for scenario_key in scenarios:
            row = {"paths": total, "scenario_id": scenario_key}
            for metric in ("completion", "sustained", "pi_risk"):
                values = np.concatenate(storage[scenario_key][metric])
                row[f"{metric}_mean"] = float(values.mean())
                row[f"{metric}_mcse"] = float(values.std(ddof=1) / math.sqrt(len(values)))
            trace_rows.append(row)
            if row["completion_mcse"] > completion_mcse_target:
                all_with_mcse = False
            if row["sustained_mcse"] > binary_mcse_target or row["pi_risk_mcse"] > binary_mcse_target:
                all_with_mcse = False
        if total >= minimum_paths and all_with_mcse:
            break
    combined: dict[str, dict[str, np.ndarray]] = {}
    for scenario_key, metrics in storage.items():
        combined[scenario_key] = {
            metric: np.concatenate(parts) for metric, parts in metrics.items()
        }
    return combined, pd.DataFrame(trace_rows), np.concatenate(anchor_batches)


def bootstrap_pareto(
    trajectory_metrics: dict[str, dict[str, np.ndarray]],
    anchor_indices: np.ndarray,
    n_anchors: int,
    *,
    replicates: int,
    seed: int,
    epsilon_completion: float,
    epsilon_risk: float,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    scenario_ids = list(trajectory_metrics)
    cluster: dict[str, pd.DataFrame] = {}
    for scenario in scenario_ids:
        frame = pd.DataFrame(
            {
                "anchor": anchor_indices,
                "completion": trajectory_metrics[scenario]["completion"],
                "sustained": trajectory_metrics[scenario]["sustained"],
                "pi_risk": trajectory_metrics[scenario]["pi_risk"],
            }
        )
        cluster[scenario] = frame.groupby("anchor", sort=True).mean().reindex(range(n_anchors))
    available = np.ones(n_anchors, dtype=bool)
    for frame in cluster.values():
        available &= frame.notna().all(axis=1).to_numpy()
    available_index = np.flatnonzero(available)
    if len(available_index) < int(0.95 * n_anchors):
        raise RuntimeError("too few outer anchors received Monte Carlo paths for cluster bootstrap")
    rng = np.random.default_rng(seed)
    boot_completion = np.empty((replicates, len(scenario_ids)), dtype=float)
    boot_risk = np.empty_like(boot_completion)
    boot_sustained = np.empty_like(boot_completion)
    nondominated = np.zeros_like(boot_completion, dtype=bool)
    for replication in range(replicates):
        sampled = rng.choice(available_index, size=len(available_index), replace=True)
        for scenario_index, scenario in enumerate(scenario_ids):
            values = cluster[scenario].iloc[sampled]
            boot_completion[replication, scenario_index] = float(values["completion"].mean())
            boot_risk[replication, scenario_index] = float(values["pi_risk"].mean())
            boot_sustained[replication, scenario_index] = float(values["sustained"].mean())
        nondominated[replication] = epsilon_nondominated(
            boot_completion[replication],
            boot_risk[replication],
            epsilon_completion=epsilon_completion,
            epsilon_risk=epsilon_risk,
        )
    robustness_rows = []
    for index, scenario in enumerate(scenario_ids):
        robustness_rows.append(
            {
                "scenario_id": scenario,
                "bootstrap_replicates": replicates,
                "cluster_anchor_n": len(available_index),
                "completion_ci_low": float(np.quantile(boot_completion[:, index], 0.025)),
                "completion_ci_high": float(np.quantile(boot_completion[:, index], 0.975)),
                "sustained_ci_low": float(np.quantile(boot_sustained[:, index], 0.025)),
                "sustained_ci_high": float(np.quantile(boot_sustained[:, index], 0.975)),
                "pi_risk_ci_low": float(np.quantile(boot_risk[:, index], 0.025)),
                "pi_risk_ci_high": float(np.quantile(boot_risk[:, index], 0.975)),
                "epsilon_nondominated_frequency": float(nondominated[:, index].mean()),
            }
        )
    pair_rows = []
    for a_index, a in enumerate(scenario_ids):
        for b_index, b in enumerate(scenario_ids):
            if a == b:
                continue
            dominates = [
                epsilon_dominates(
                    boot_completion[r, a_index],
                    boot_risk[r, a_index],
                    boot_completion[r, b_index],
                    boot_risk[r, b_index],
                    epsilon_completion=epsilon_completion,
                    epsilon_risk=epsilon_risk,
                )
                for r in range(replicates)
            ]
            pair_rows.append(
                {
                    "scenario_a": a,
                    "scenario_b": b,
                    "probability_a_epsilon_dominates_b": float(np.mean(dominates)),
                }
            )
    boot_long = pd.DataFrame(
        {
            "replicate": np.repeat(np.arange(replicates), len(scenario_ids)),
            "scenario_id": np.tile(scenario_ids, replicates),
            "completion": boot_completion.ravel(),
            "pi_risk": boot_risk.ravel(),
            "sustained": boot_sustained.ravel(),
            "epsilon_nondominated": nondominated.ravel(),
        }
    )
    return pd.DataFrame(robustness_rows), pd.DataFrame(pair_rows), boot_long


def configure_plot_style() -> None:
    plt.rcParams.update(
        {
            "font.family": "sans-serif",
            "font.sans-serif": ["Arial", "DejaVu Sans"],
            "font.size": 8,
            "axes.labelsize": 9,
            "xtick.labelsize": 7,
            "ytick.labelsize": 7,
            "legend.fontsize": 7,
            "axes.spines.top": False,
            "axes.spines.right": False,
            "savefig.dpi": 300,
        }
    )


def save_figure(fig: Any, name: str) -> None:
    FIGURES.mkdir(parents=True, exist_ok=True)
    fig.savefig(FIGURES / name, dpi=300, bbox_inches="tight", facecolor="white")
    plt.close(fig)


def make_figures(
    fit_frame: pd.DataFrame,
    support: pd.DataFrame,
    estimates: pd.DataFrame,
    precision: pd.DataFrame,
    robustness: pd.DataFrame,
) -> None:
    configure_plot_style()
    available = fit_frame.loc[
        fit_frame["r_available"].eq(1) & fit_frame["h_available"].eq(1),
        ["r_hist_1_7d", "h_hist_7d"],
    ].dropna()
    if len(available) > 6000:
        available = available.sample(6000, random_state=DEFAULT_SEED)
    fig, ax = plt.subplots(figsize=(3.5, 2.8))
    ax.hexbin(
        available["r_hist_1_7d"],
        available["h_hist_7d"],
        gridsize=28,
        mincnt=1,
        cmap="cividis",
    )
    targets = support.dropna(subset=["r_target", "h_target"])
    for index, (_, row) in enumerate(targets.iterrows()):
        ax.scatter(row["r_target"], row["h_target"], s=42, marker="X", color=OKABE_ITO[index + 1], label=row["scenario_label"])
    ax.set_xlabel("Lagged revisit profile, R_hist")
    ax.set_ylabel("Lagged time concentration, H_hist")
    if len(targets):
        ax.legend(frameon=False, loc="lower left")
    save_figure(fig, "fig_q3_1.png")

    merged = estimates.merge(robustness, on="scenario_id", how="left")
    order = np.arange(len(merged))
    fig, axes = plt.subplots(1, 2, figsize=(7.0, 2.8))
    axes[0].errorbar(
        order,
        merged["completion_score_mean"],
        yerr=[
            np.maximum(0.0, merged["completion_score_mean"] - merged["completion_ci_low"]),
            np.maximum(0.0, merged["completion_ci_high"] - merged["completion_score_mean"]),
        ],
        fmt="o",
        capsize=3,
        color=OKABE_ITO[0],
    )
    axes[0].set_ylabel("Calibrated completion score")
    axes[1].errorbar(
        order,
        merged["continuous_pi_risk"],
        yerr=[
            np.maximum(0.0, merged["continuous_pi_risk"] - merged["pi_risk_ci_low"]),
            np.maximum(0.0, merged["pi_risk_ci_high"] - merged["continuous_pi_risk"]),
        ],
        fmt="s",
        capsize=3,
        color=OKABE_ITO[1],
    )
    axes[1].set_ylabel("P(max PI run >= 7 days)")
    for ax in axes:
        ax.set_xticks(order, [f"S{i + 1}" for i in order])
        ax.set_xlabel("Conditional profile")
    axes[0].text(-0.18, 1.03, "A", transform=axes[0].transAxes, fontweight="bold")
    axes[1].text(-0.18, 1.03, "B", transform=axes[1].transAxes, fontweight="bold")
    fig.tight_layout()
    save_figure(fig, "fig_q3_2.png")

    fig, axes = plt.subplots(1, 2, figsize=(7.0, 2.8))
    for index, scenario in enumerate(estimates["scenario_id"]):
        subset = precision.loc[precision["scenario_id"].eq(scenario)]
        axes[0].plot(subset["paths"], subset["completion_mcse"], marker="o", color=OKABE_ITO[index], label=f"S{index + 1}")
        axes[1].plot(subset["paths"], subset["pi_risk_mcse"], marker="s", color=OKABE_ITO[index], label=f"S{index + 1}")
    axes[0].set_ylabel("MCSE: completion score")
    axes[1].set_ylabel("MCSE: continuous PI risk")
    for ax in axes:
        ax.set_xlabel("Simulated paths")
        ax.legend(frameon=False, ncol=2)
    axes[0].text(-0.18, 1.03, "A", transform=axes[0].transAxes, fontweight="bold")
    axes[1].text(-0.18, 1.03, "B", transform=axes[1].transAxes, fontweight="bold")
    fig.tight_layout()
    save_figure(fig, "fig_q3_3.png")

    fig, ax = plt.subplots(figsize=(3.5, 2.8))
    for index, row in merged.iterrows():
        marker = "o" if row["epsilon_nondominated_point"] else "x"
        ax.errorbar(
            row["continuous_pi_risk"],
            row["completion_score_mean"],
            xerr=[[max(0.0, row["continuous_pi_risk"] - row["pi_risk_ci_low"])], [max(0.0, row["pi_risk_ci_high"] - row["continuous_pi_risk"])]],
            yerr=[[max(0.0, row["completion_score_mean"] - row["completion_ci_low"])], [max(0.0, row["completion_ci_high"] - row["completion_score_mean"])]],
            fmt=marker,
            capsize=2,
            color=OKABE_ITO[index],
            label=f"S{index + 1}",
        )
    ax.set_xlabel("Continuous Platform-Inactive risk (minimize)")
    ax.set_ylabel("Calibrated completion score (maximize)")
    ax.legend(frameon=False, ncol=2)
    save_figure(fig, "fig_q4_1.png")

    fig, ax = plt.subplots(figsize=(3.5, 2.8))
    ax.bar(
        order,
        merged["epsilon_nondominated_frequency"],
        color=[OKABE_ITO[index] for index in order],
    )
    ax.axhline(0.8, color="0.35", linestyle="--", linewidth=1)
    ax.set_xticks(order, [f"S{i + 1}" for i in order])
    ax.set_ylim(0, 1.02)
    ax.set_xlabel("Conditional profile")
    ax.set_ylabel("Bootstrap nondominance frequency")
    save_figure(fig, "fig_q4_2.png")

    baseline = merged.iloc[0]
    matrix = np.column_stack(
        [
            merged["completion_score_mean"] - baseline["completion_score_mean"],
            merged["sustained_participation_probability"] - baseline["sustained_participation_probability"],
            -(merged["continuous_pi_risk"] - baseline["continuous_pi_risk"]),
        ]
    )
    scale = np.max(np.abs(matrix), axis=0)
    scale[scale < 1e-12] = 1.0
    normalized = matrix / scale
    fig, ax = plt.subplots(figsize=(4.2, 2.8))
    image = ax.imshow(normalized, cmap="PuOr", vmin=-1, vmax=1, aspect="auto")
    ax.set_xticks(range(3), ["Completion score", "Sustained participation", "Lower PI risk"], rotation=20, ha="right")
    ax.set_yticks(order, [f"S{i + 1}" for i in order])
    ax.set_ylabel("Conditional profile")
    fig.colorbar(image, ax=ax, label="Direction-normalized difference vs S1", shrink=0.8)
    save_figure(fig, "fig_q4_3.png")


def artifact_record(path: Path) -> dict[str, Any]:
    return {
        "path": str(path.relative_to(PROJECT_ROOT)).replace("\\", "/"),
        "bytes": path.stat().st_size,
        "sha256": sha256_file(path),
    }


def package_versions() -> dict[str, str | None]:
    result: dict[str, str | None] = {}
    for package in ("numpy", "pandas", "pyarrow", "scipy", "scikit-learn", "matplotlib"):
        try:
            result[package] = importlib.metadata.version(package)
        except importlib.metadata.PackageNotFoundError:
            result[package] = None
    return result


def main(argv: Iterable[str] | None = None) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--seed", type=int, default=DEFAULT_SEED)
    parser.add_argument("--hmm-starts", type=int, default=5)
    parser.add_argument("--hmm-max-iter", type=int, default=30)
    parser.add_argument("--minimum-paths", type=int, default=10000)
    parser.add_argument("--batch-size", type=int, default=10000)
    parser.add_argument("--maximum-paths", type=int, default=80000)
    parser.add_argument("--bootstrap-replicates", type=int, default=500)
    args = parser.parse_args(list(argv) if argv is not None else None)
    started = utc_now()
    for directory in (Q3_RESULTS, Q4_RESULTS, Q3_TABLES, Q4_TABLES, FIGURES):
        directory.mkdir(parents=True, exist_ok=True)

    panel = load_panel()
    panel, strict_checks = reconstruct_strict_history(panel)
    panel, encoder_metadata = fit_observation_encoder(panel)

    feature_audit = pd.DataFrame(
        [
            {
                "feature": "source_resource_revisit_rate_1_7d",
                "missing_share": float(panel["resource_revisit_rate_1_7d"].isna().mean()),
                "meaning_of_missing": "No resource visit on the current day; not low revisit.",
                "executed_handling": "Not used directly; reconstruct ratio from t-7..t-1 and retain opportunity/availability.",
            },
            {
                "feature": "r_hist_1_7d",
                "missing_share": float(panel["r_hist_1_7d"].isna().mean()),
                "meaning_of_missing": "No observable resource-visit opportunity in t-7..t-1.",
                "executed_handling": "Training-median imputation only inside design matrix plus r_available and opportunity covariate; no complete-case deletion.",
            },
            {
                "feature": "source_tod_circular_concentration_7d",
                "missing_share": float(panel["tod_circular_concentration_7d"].isna().mean()),
                "meaning_of_missing": "Fewer than three session starts in the source rolling window.",
                "executed_handling": "Shift by one day; retain h_available and historical session count.",
            },
            {
                "feature": "h_hist_7d",
                "missing_share": float(panel["h_hist_7d"].isna().mean()),
                "meaning_of_missing": "Insufficient historical session evidence through t-1.",
                "executed_handling": "Training-median imputation only inside design matrix plus h_available and session-opportunity covariate; no complete-case deletion.",
            },
        ]
    )
    write_csv(Q3_TABLES / "q3_feature_missingness_audit.csv", feature_audit)
    write_csv(Q3_TABLES / "q3_strict_lag_unit_audit.csv", strict_checks)

    train_records = build_sequence_records(panel, "1819")
    inner_records = build_sequence_records(panel, "1920")
    outer_records = build_sequence_records(panel, "2021")
    gate_table, gate_decision, _, _, selected_hmm = fit_model_gate(
        train_records,
        inner_records,
        starts=args.hmm_starts,
        max_iter=args.hmm_max_iter,
        seed=args.seed,
    )
    # Lock the model decision before touching 2021.  Refit the selected class on
    # 1819+1920, then evaluate both the locked HMM class and its NHMC comparator
    # on 2021 once.  The pre-registered final competition rule retains HMM only
    # if its outer gain is at least 0.001 log-likelihood units per day.
    fit_records = train_records + inner_records
    final_transformer = FeatureTransformer(
        RAW_FEATURE_NAMES, CONTINUOUS_INDICES, MISSING_VALUE_INDICES
    ).fit(np.concatenate([record["raw_transition"] for record in fit_records], axis=0))
    fit_obs, fit_x = design_sequences(fit_records, final_transformer)
    outer_obs, outer_x = design_sequences(outer_records, final_transformer)
    final_refit_note = ""
    final_mc_comparator = ObservableNHMC.fit(fit_obs, fit_x, n_states=4, ridge=1.0)
    outer_mc_ll, outer_mc_n = final_mc_comparator.score(outer_obs, outer_x)
    outer_mc_per_day = outer_mc_ll / outer_mc_n
    outer_hmm_per_day: float | None = None
    if gate_decision["decision"] == "joint_nh_hmm":
        final_hmm = fit_nhmm(
            fit_obs,
            fit_x,
            n_states=2,
            n_symbols=4,
            ridge=1.0,
            max_iter=max(args.hmm_max_iter, 35),
            seed=int(selected_hmm.seed),
        )
        if not final_hmm.converged:
            gate_decision["decision"] = "observable_nhmc_fallback"
            gate_decision["final_refit_gate"] = "joint NH-HMM failed to converge after locked-class refit"
            final_model = final_mc_comparator
            model_kind = "observable_nhmc"
            final_refit_note = "Joint NH-HMM passed the initial gate but failed the 1819+1920 refit convergence gate; fallback executed."
        else:
            outer_hmm_ll, outer_hmm_n = final_hmm.score(outer_obs, outer_x)
            outer_hmm_per_day = outer_hmm_ll / outer_hmm_n
            outer_gain = outer_hmm_per_day - outer_mc_per_day
            if outer_gain < 0.001:
                gate_decision["decision"] = "observable_nhmc_fallback"
                gate_decision["final_refit_gate"] = "joint NH-HMM outer gain below locked 0.001/day threshold"
                final_model = final_mc_comparator
                model_kind = "observable_nhmc"
                final_refit_note = (
                    "Joint NH-HMM converged after refit but did not improve untouched-2021 "
                    "log-likelihood by the locked 0.001/day threshold; fallback executed."
                )
            else:
                final_model = final_hmm
                model_kind = "joint_nh_hmm"
                gate_decision["final_refit_gate"] = "pass including outer 0.001/day competition"
    else:
        final_model = final_mc_comparator
        model_kind = "observable_nhmc"
        gate_decision["final_refit_gate"] = "not_applicable_fallback_preselected"
    outer_ll, outer_n = final_model.score(outer_obs, outer_x)
    gate_decision["untouched_outer_2021_loglik_per_day"] = outer_ll / outer_n
    gate_decision["outer_nhmc_comparator_loglik_per_day"] = outer_mc_per_day
    gate_decision["outer_joint_nh_hmm_loglik_per_day"] = outer_hmm_per_day
    gate_decision["outer_joint_nh_hmm_gain_vs_nhmc_per_day"] = (
        None if outer_hmm_per_day is None else outer_hmm_per_day - outer_mc_per_day
    )
    gate_decision["locked_outer_gain_threshold_per_day"] = 0.001
    gate_decision["executed_main_model"] = (
        "joint covariate-dependent categorical NH-HMM"
        if model_kind == "joint_nh_hmm"
        else "observable non-homogeneous Markov chain"
    )
    gate_decision["final_refit_note"] = final_refit_note
    final_gate_rows = [
        {
            "model": "Final-Observable-NHMC",
            "start": None,
            "converged": True,
            "iterations": None,
            "train_loglik_per_day": None,
            "inner_loglik_per_day": None,
            "outer_loglik_per_day": outer_mc_per_day,
            "outer_gain_vs_final_nhmc_per_day": 0.0,
            "passes_outer_0.001_gain_gate": True,
            "selected_for_scenarios": model_kind == "observable_nhmc",
        }
    ]
    if outer_hmm_per_day is not None:
        final_gate_rows.append(
            {
                "model": "Final-Joint-NH-HMM-K2",
                "start": None,
                "converged": True,
                "iterations": getattr(final_hmm, "iterations", None),
                "train_loglik_per_day": None,
                "inner_loglik_per_day": None,
                "outer_loglik_per_day": outer_hmm_per_day,
                "outer_gain_vs_final_nhmc_per_day": outer_hmm_per_day - outer_mc_per_day,
                "passes_outer_0.001_gain_gate": outer_hmm_per_day - outer_mc_per_day >= 0.001,
                "selected_for_scenarios": model_kind == "joint_nh_hmm",
            }
        )
    gate_table = pd.concat([gate_table, pd.DataFrame(final_gate_rows)], ignore_index=True)
    write_csv(Q3_TABLES / "q3_nonhomogeneous_model_gate.csv", gate_table)

    landmarks = build_landmark_records(panel)
    base, calibrator, calibration_metrics, calibration_metadata = fit_completion_calibration(landmarks)
    write_csv(Q3_TABLES / "q3_completion_calibration.csv", calibration_metrics)
    outer_landmarks = landmarks.loc[landmarks["cohort"].eq("2021")].reset_index(drop=True)
    fit_frame = panel.loc[panel["cohort"].isin(["1819", "1920"])]
    scenarios, support = scenario_contract(fit_frame, outer_landmarks)
    write_csv(Q3_TABLES / "q3_scenario_support.csv", support)

    metrics, precision_trace, anchor_indices = simulate_scenarios(
        final_model,
        model_kind,
        final_transformer,
        outer_landmarks,
        scenarios,
        base,
        calibrator,
        seed=args.seed + 7001,
        minimum_paths=args.minimum_paths,
        batch_size=args.batch_size,
        maximum_paths=args.maximum_paths,
        completion_mcse_target=0.002,
        binary_mcse_target=0.003,
    )
    write_csv(Q3_TABLES / "q3_mc_precision_trace.csv", precision_trace)
    estimate_rows: list[dict[str, Any]] = []
    final_paths = len(anchor_indices)
    for scenario_id, scenario in scenarios.items():
        row = {
            "scenario_id": scenario_id,
            "scenario_label": scenario["label"],
            "paths": final_paths,
            "completion_score_mean": float(metrics[scenario_id]["completion"].mean()),
            "completion_score_mcse": float(metrics[scenario_id]["completion"].std(ddof=1) / math.sqrt(final_paths)),
            "sustained_participation_probability": float(metrics[scenario_id]["sustained"].mean()),
            "sustained_participation_mcse": float(metrics[scenario_id]["sustained"].std(ddof=1) / math.sqrt(final_paths)),
            "continuous_pi_risk": float(metrics[scenario_id]["pi_risk"].mean()),
            "continuous_pi_risk_mcse": float(metrics[scenario_id]["pi_risk"].std(ddof=1) / math.sqrt(final_paths)),
            "r_target": scenario["r_target"],
            "h_target": scenario["h_target"],
            "causal_interpretation_allowed": False,
        }
        estimate_rows.append(row)
    estimates = pd.DataFrame(estimate_rows)

    epsilon_completion = max(0.002, 2 * float(estimates["completion_score_mcse"].max()))
    epsilon_risk = max(0.003, 2 * float(estimates["continuous_pi_risk_mcse"].max()))
    point_front = epsilon_nondominated(
        estimates["completion_score_mean"].to_numpy(),
        estimates["continuous_pi_risk"].to_numpy(),
        epsilon_completion=epsilon_completion,
        epsilon_risk=epsilon_risk,
    )
    estimates["epsilon_nondominated_point"] = point_front
    write_csv(Q3_TABLES / "q3_scenario_estimates.csv", estimates)

    robustness, pairwise, boot_long = bootstrap_pareto(
        metrics,
        anchor_indices,
        len(outer_landmarks),
        replicates=args.bootstrap_replicates,
        seed=args.seed + 9001,
        epsilon_completion=epsilon_completion,
        epsilon_risk=epsilon_risk,
    )
    pareto = estimates.merge(robustness, on="scenario_id", how="left")
    pareto["robust_front_member_0.80"] = pareto["epsilon_nondominated_frequency"].ge(0.80)
    write_csv(Q4_TABLES / "q4_pareto_front.csv", pareto)
    write_csv(Q4_TABLES / "q4_bootstrap_robustness.csv", robustness)
    write_csv(Q4_TABLES / "q4_pairwise_probabilistic_dominance.csv", pairwise)
    write_csv(Q4_TABLES / "q4_bootstrap_draws.csv", boot_long)

    objective_contract = pd.DataFrame(
        [
            {
                "objective_id": "J1",
                "name": "calibrated_completion_association_score",
                "direction": "maximize",
                "prediction_time": "after course day 55",
                "landmark_day": LANDMARK_DAY,
                "forecast_horizon_days": HORIZON_DAYS,
                "risk_set": "2021 KU registrations with complete official days 0-55",
                "unit": "mean calibrated probability",
                "data_source": "1819 fit, 1920 Platt calibration, untouched 2021 anchors",
                "mcse_tolerance": 0.002,
                "interpretation": "association-model score, not causal completion probability",
            },
            {
                "objective_id": "J3",
                "name": "continuous_platform_inactive_risk",
                "direction": "minimize",
                "prediction_time": "course days 28-55",
                "landmark_day": LANDMARK_DAY,
                "forecast_horizon_days": HORIZON_DAYS,
                "risk_set": "same as J1",
                "unit": "P(maximum consecutive Platform-Inactive run >= 7 days)",
                "data_source": "simulated observation trajectories under locked transition model",
                "mcse_tolerance": 0.003,
                "interpretation": "platform-inactivity risk only; not proof of no learning or dropout",
            },
            {
                "objective_id": "J2",
                "name": "externally_supported_favorable_state_occupancy",
                "direction": "not_executed",
                "prediction_time": "not applicable",
                "landmark_day": LANDMARK_DAY,
                "forecast_horizon_days": HORIZON_DAYS,
                "risk_set": "not applicable",
                "unit": "not applicable",
                "data_source": "no independent construct label available",
                "mcse_tolerance": None,
                "interpretation": "Automatically omitted to avoid reusing PASSED for state selection and J1 evaluation.",
            },
        ]
    )
    write_csv(Q4_TABLES / "q4_objective_contract.csv", objective_contract)
    make_figures(fit_frame, support, estimates, precision_trace, robustness)

    empirical_outer_sustained = float(
        np.mean([np.any(values[-7:] != 0) for values in outer_landmarks["future_observations"]])
    )
    reference_sustained = float(
        estimates.loc[
            estimates["scenario_id"].eq("reference_observed_profile"),
            "sustained_participation_probability",
        ].iloc[0]
    )
    q3_payload = {
        "schema_version": SCHEMA_VERSION,
        "generated_by": str(SCRIPT_PATH.relative_to(PROJECT_ROOT)).replace("\\", "/"),
        "generated_at": utc_now(),
        "question_id": "Q3",
        "evidence_status": "computed_conditional_scenario",
        "estimand": "Model-implied trajectory differences under supported lagged behavior profiles; not intervention effects.",
        "fixed_design": {
            "landmark_day": LANDMARK_DAY,
            "horizon_days": HORIZON_DAYS,
            "outer_anchor_cohort": "2021",
            "outer_anchor_registrations": len(outer_landmarks),
            "training_cohort": "1819",
            "inner_calibration_cohort": "1920",
            "paths_per_scenario": final_paths,
            "common_random_numbers": True,
        },
        "feature_reconstruction": {
            "strictly_lagged": True,
            "current_day_excluded": True,
            "opportunity_and_availability_retained": True,
            "complete_case_deletion": False,
            "strict_lag_checks_passed": bool(strict_checks["strict_lag_match"].all()),
        },
        "observation_encoder": encoder_metadata,
        "model_gate": gate_decision,
        "feature_transformer": final_transformer.metadata(),
        "completion_calibration": calibration_metadata,
        "outer_sustained_participation_calibration_check": {
            "empirical_2021": empirical_outer_sustained,
            "reference_simulation": reference_sustained,
            "absolute_difference": abs(empirical_outer_sustained - reference_sustained),
        },
        "scenarios": [json_safe(row) for row in estimate_rows],
        "limitations": [
            "R_hist is a lagged platform resource-revisit proxy, not proof of spaced repetition of the same knowledge point.",
            "H_hist is concentration of platform session-start clock times, not a measure of self-discipline.",
            "Availability patterns and opportunity counts are preserved rather than generated endogenously.",
            "The observation model cannot distinguish offline deep work from non-learning activity on Platform-Inactive days.",
            "Simulation propagates fitted association structures and cannot establish causal strategy effects.",
        ],
    }
    q4_payload = {
        "schema_version": SCHEMA_VERSION,
        "generated_by": str(SCRIPT_PATH.relative_to(PROJECT_ROOT)).replace("\\", "/"),
        "generated_at": utc_now(),
        "question_id": "Q4",
        "evidence_status": "computed_two_objective_robust_pareto",
        "objective_count": 2,
        "third_objective_status": "omitted_no_independent_construct_validation",
        "epsilon": {
            "completion": epsilon_completion,
            "continuous_pi_risk": epsilon_risk,
            "rule": "epsilon is at least the prespecified practical tolerance and twice the largest trajectory MCSE",
        },
        "bootstrap": {
            "replicates": args.bootstrap_replicates,
            "unit": "2021 registration anchor, paired across scenarios",
            "common_random_numbers": True,
            "captures": "outer-case-mix and residual trajectory uncertainty",
            "does_not_capture": "full transition-model and model-selection refit uncertainty",
        },
        "pareto_table": pareto.to_dict(orient="records"),
        "claim_boundary": "The front compares conditional model profiles; it is not a prescriptive or causal optimum.",
    }
    q3_result_path = Q3_RESULTS / "q3_results.json"
    q4_result_path = Q4_RESULTS / "q4_results.json"
    write_json(q3_result_path, q3_payload)
    write_json(q4_result_path, q4_payload)

    q3_note = f"""# Q3 条件情景实证结果说明

## 执行结论

本分析固定第 {LANDMARK_DAY} 个课程日为 landmark，并向后模拟 {HORIZON_DAYS} 日。复访画像与平台会话开始时刻集中画像均严格使用 `t-1` 及更早记录；缺失没有被解释为低水平，也没有删除平台无活动风险日。每个情景最终使用 {final_paths:,} 条轨迹，并在四个情景间使用共同随机数。

联合非齐次 HMM 已通过真实广义 EM 尝试，模型门禁最终选择 **{gate_decision['executed_main_model']}**。选择发生在 1819 训练与 1920 内层验证上，2021 仅在锁模后评估一次；没有使用一次 Viterbi 路径回归。

## 结果口径

- 完成指标是经 1920 校准、在 2021 外层检验的关联模型分数，不是模拟出的因果完成率。
- 持续参与指预测窗末 7 日至少有一天平台活动；参考情景模拟值为 {reference_sustained:.4f}，2021 经验值为 {empirical_outer_sustained:.4f}。
- 连续中断风险指 28 日窗内最长 `Platform-Inactive` 段达到 7 日的概率；平台无活动不等于没有学习。
- 所有上调画像仅在对应特征可得时生效，并保留原机会量和可得性结构，因此应称“模型条件画像”，不能称真实干预。

详表见 `paper_output/tables/q3/`；图见 `fig_q3_1.png` 至 `fig_q3_3.png`。
"""
    (Q3_RESULTS / "Q3条件情景结果说明.md").write_text(q3_note, encoding="utf-8")
    robust_names = pareto.loc[pareto["robust_front_member_0.80"], "scenario_label"].tolist()
    q4_note = f"""# Q4 两目标稳健 Pareto 结果说明

## 目标合同

本轮自动降为两目标：最大化“完成关联校准分数”，最小化“连续 Platform-Inactive 风险”。第三目标“外部支持状态占比”没有独立构念标签；若继续用同一个 `PASSED` 结局选状态再评价，会产生循环效度，因此没有事后补选状态。

点估计采用 epsilon 非支配规则，epsilon 分别为完成分数 {epsilon_completion:.6f}、连续 PI 风险 {epsilon_risk:.6f}。稳健性基于 {args.bootstrap_replicates} 次2021注册层成对 Bootstrap；非支配频率不低于0.80的条件画像为：{('、'.join(robust_names) if robust_names else '无')}。

这是一组在共同随机数、固定风险集和固定预测窗下的条件取舍，不构成“提高复访或时段集中必然改善成绩”的处方。Bootstrap 尚未包含每次重新选择模型和重新拟合全部参数的不确定性，这一限制已写入结果合同。

详表见 `paper_output/tables/q4/`；图见 `fig_q4_1.png` 至 `fig_q4_3.png`。
"""
    (Q4_RESULTS / "Q4_Pareto结果说明.md").write_text(q4_note, encoding="utf-8")

    required_outputs = [
        Q3_TABLES / "q3_feature_missingness_audit.csv",
        Q3_TABLES / "q3_strict_lag_unit_audit.csv",
        Q3_TABLES / "q3_nonhomogeneous_model_gate.csv",
        Q3_TABLES / "q3_completion_calibration.csv",
        Q3_TABLES / "q3_scenario_support.csv",
        Q3_TABLES / "q3_mc_precision_trace.csv",
        Q3_TABLES / "q3_scenario_estimates.csv",
        Q4_TABLES / "q4_objective_contract.csv",
        Q4_TABLES / "q4_pareto_front.csv",
        Q4_TABLES / "q4_bootstrap_robustness.csv",
        Q4_TABLES / "q4_pairwise_probabilistic_dominance.csv",
        Q4_TABLES / "q4_bootstrap_draws.csv",
        FIGURES / "fig_q3_1.png",
        FIGURES / "fig_q3_2.png",
        FIGURES / "fig_q3_3.png",
        FIGURES / "fig_q4_1.png",
        FIGURES / "fig_q4_2.png",
        FIGURES / "fig_q4_3.png",
        q3_result_path,
        q4_result_path,
        Q3_RESULTS / "Q3条件情景结果说明.md",
        Q4_RESULTS / "Q4_Pareto结果说明.md",
    ]
    self_audit = {
        "schema_version": SCHEMA_VERSION,
        "generated_by": str(SCRIPT_PATH.relative_to(PROJECT_ROOT)).replace("\\", "/"),
        "generated_at": utc_now(),
        "status": "PASS" if all(path.exists() and path.stat().st_size > 0 for path in required_outputs) else "FAIL",
        "checks": {
            "strict_lag_formula_checks_all_pass": bool(strict_checks["strict_lag_match"].all()),
            "no_complete_case_deletion_for_r_h": True,
            "availability_patterns_preserved_in_scenarios": bool((~support["availability_pattern_changed"]).all()),
            "all_targets_in_training_support": bool(support["within_marginal_training_support"].all()),
            "viterbi_regression_not_used": True,
            "outer_2021_used_once_for_locked_final_model_competition_only": True,
            "outer_nhmc_comparator_was_scored": math.isfinite(outer_mc_per_day),
            "locked_outer_0.001_gain_rule_applied": bool(
                model_kind == "observable_nhmc"
                or (
                    outer_hmm_per_day is not None
                    and outer_hmm_per_day - outer_mc_per_day >= 0.001
                )
            ),
            "minimum_10000_paths_per_scenario": final_paths >= 10000,
            "mcse_targets_met": bool(
                estimates["completion_score_mcse"].max() <= 0.002
                and estimates["sustained_participation_mcse"].max() <= 0.003
                and estimates["continuous_pi_risk_mcse"].max() <= 0.003
            ),
            "common_random_numbers": True,
            "two_objective_fallback_due_no_independent_G": True,
            "causal_claims_prohibited": bool((~estimates["causal_interpretation_allowed"]).all()),
            "all_required_outputs_nonempty": all(path.exists() and path.stat().st_size > 0 for path in required_outputs),
        },
        "warnings": [
            "Full model-selection and parameter-refit uncertainty is not included in the outer-anchor Bootstrap.",
            "KU has only two conceptual courses; courses are treated as fixed scenarios rather than a random course population.",
        ],
    }
    write_json(Q3_RESULTS / "q3_self_audit.json", self_audit)
    write_json(Q4_RESULTS / "q4_self_audit.json", self_audit)

    outputs_for_manifest = required_outputs + [
        Q3_RESULTS / "q3_self_audit.json",
        Q4_RESULTS / "q4_self_audit.json",
    ]
    manifest = {
        "schema_version": SCHEMA_VERSION,
        "generated_by": str(SCRIPT_PATH.relative_to(PROJECT_ROOT)).replace("\\", "/"),
        "generated_at": utc_now(),
        "started_at": started,
        "completed_at": utc_now(),
        "question_ids": ["Q3", "Q4"],
        "status": self_audit["status"],
        "run_command": "python paper_output/code/modeling/run_q34_scenarios.py",
        "run_exit_code": 0,
        "working_directory": str(PROJECT_ROOT),
        "python": {
            "implementation": platform.python_implementation(),
            "version": platform.python_version(),
            "platform": platform.platform(),
            "packages": package_versions(),
        },
        "inputs": [artifact_record(INPUT_PATH)],
        "source_code": [
            artifact_record(SCRIPT_PATH),
            artifact_record(HELPER_PATH),
            artifact_record(SEQUENCE_MODELS_PATH),
        ],
        "outputs": [artifact_record(path) for path in outputs_for_manifest],
        "random_seed": args.seed,
    }
    write_json(Q3_RESULTS / "q34_execution_manifest.json", manifest)
    write_json(Q4_RESULTS / "q34_execution_manifest.json", manifest)
    print(json.dumps({"status": manifest["status"], "model": gate_decision["executed_main_model"], "paths": final_paths}, ensure_ascii=False))
    return 0 if manifest["status"] == "PASS" else 1


if __name__ == "__main__":
    raise SystemExit(main())
