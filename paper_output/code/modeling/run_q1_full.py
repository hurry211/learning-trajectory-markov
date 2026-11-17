"""Publication-oriented Q1 evaluation on the KU Leuven daily panel.

This script implements a genuinely nested, outcome-blind comparison between an
observable first-order Markov chain (MC) and categorical hidden Markov models
(HMMs).  It deliberately keeps the 2021 cohort untouched until K has been
selected and the selected HMM has been refit on the two earlier cohorts.

Design contract
---------------
* 1819: HMM K-grid estimation; 1920: inner conditional validation.
* 1819+1920: final refit after K selection; 2021: untouched outer test.
* Every validation/test sequence is conditioned on a fixed 42-day prefix.  The
  suffix likelihood is ``log p(y_full) - log p(y_prefix)`` for the HMM and the
  exactly corresponding transition likelihood for the observable MC.
* K=2,...,6, at least ten random starts per K, strict convergence before the
  iteration cap, and a minimum posterior-occupancy screen.
* If a timing benchmark implies that the complete K-grid exceeds the declared
  wall-clock budget, an outcome-blind SHA-256 cap is applied within every 1819
  course stratum.  K selection may be capacity-adaptive, but the selected K is
  always refit on every 1819+1920 sequence and outer-tested on every 2021
  sequence.
* Hidden states remain State 1,...,State K.  Platform-Inactive never means
  "not learning".  The duration PPC is an HMM diagnostic, not an HSMM fit.

Only q1_full-specific result/table files and fig_q1_*.png are written.
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
import time
import traceback
import uuid
from concurrent.futures import ProcessPoolExecutor, as_completed
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable, Sequence

# Limit numerical-library thread fan-out before importing NumPy.  Random starts
# are parallelised explicitly and should not each create a BLAS thread pool.
os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from sequence_models import (
    ObservationEncodingConfig,
    SequenceBundle,
    TransparentObservationEncoder,
    _fit_categorical_hmm_once,
    canonicalize_hmm,
    categorical_hmm_parameter_count,
    dwell_time_posterior_predictive_check,
    fit_dirichlet_markov_chain,
    hsmm_status,
    hungarian_state_alignment,
    make_sequence_bundle,
)


PROJECT_ROOT = Path(__file__).resolve().parents[3]
SCRIPT_PATH = Path(__file__).resolve()
PANEL_PATH = PROJECT_ROOT / "paper_output" / "data_cleaned" / "ku_daily_panel.parquet"
PILOT_SUMMARY_PATH = PROJECT_ROOT / "paper_output" / "results" / "q1" / "q1_run_summary.json"
RESULTS_DIR = PROJECT_ROOT / "paper_output" / "results" / "q1_full"
TABLES_DIR = PROJECT_ROOT / "paper_output" / "tables" / "q1_full"
FIGURES_DIR = PROJECT_ROOT / "paper_output" / "figures"
PROGRESS_PATH = RESULTS_DIR / "q1_full_progress.jsonl"
CHECKPOINT_ROOT = PROJECT_ROOT / "paper_output" / "checkpoints" / "q1_full"
SCHEMA_VERSION = "1.0"
CHECKPOINT_SCHEMA_VERSION = "1.0"
DEFAULT_SEED = 20260822
OBSERVATION_COLUMNS = [
    "registration_id",
    "cohort",
    "COURSE_ID",
    "date",
    "course_day_index",
    "platform_inactive",
    "activity_category_diversity",
    "productive_event_count",
    "forum_event_count",
]


_ACTIVE_INVOCATION_ID: str | None = None
_ACTIVE_CHECKPOINT_CONTRACT_ID: str | None = None
_ACTIVE_PROGRESS_PATH = PROGRESS_PATH


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def sha256_file(path: Path, chunk_size: int = 1024 * 1024) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while chunk := handle.read(chunk_size):
            digest.update(chunk)
    return digest.hexdigest()


def artifact_record(path: Path) -> dict[str, Any]:
    return {
        "path": str(path.relative_to(PROJECT_ROOT)).replace("\\", "/"),
        "bytes": int(path.stat().st_size),
        "sha256": sha256_file(path),
    }


def portable_path(path: Path) -> str:
    try:
        return str(path.resolve().relative_to(PROJECT_ROOT.resolve())).replace("\\", "/")
    except ValueError:
        return str(path.resolve())


def canonical_json_bytes(value: Any) -> bytes:
    return json.dumps(
        json_safe(value),
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")


def sha256_payload(value: Any) -> str:
    return hashlib.sha256(canonical_json_bytes(value)).hexdigest()


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
        json.dumps(json_safe(payload), ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    os.replace(temporary, path)


def append_progress(stage: str, status: str, **details: Any) -> None:
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    row = {
        "timestamp": utc_now(),
        "invocation_id": _ACTIVE_INVOCATION_ID,
        "checkpoint_contract_id": _ACTIVE_CHECKPOINT_CONTRACT_ID,
        "stage": stage,
        "status": status,
        **details,
    }
    _ACTIVE_PROGRESS_PATH.parent.mkdir(parents=True, exist_ok=True)
    with _ACTIVE_PROGRESS_PATH.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(json_safe(row), ensure_ascii=False) + "\n")


def package_versions() -> dict[str, str | None]:
    versions: dict[str, str | None] = {}
    for package in (
        "numpy",
        "pandas",
        "pyarrow",
        "scipy",
        "scikit-learn",
        "hmmlearn",
        "matplotlib",
    ):
        try:
            versions[package] = importlib.metadata.version(package)
        except importlib.metadata.PackageNotFoundError:
            versions[package] = None
    return versions


def build_checkpoint_run_contract(args: argparse.Namespace) -> dict[str, Any]:
    """Build an immutable compatibility contract for safe checkpoint reuse.

    Generated timestamps and invocation identifiers are deliberately excluded:
    two invocations are compatible only when their parsed arguments, source,
    dependency, inputs, interpreter, platform, and package versions are exact.
    """

    input_paths = [PANEL_PATH]
    if PILOT_SUMMARY_PATH.exists():
        input_paths.append(PILOT_SUMMARY_PATH)
    payload = {
        "checkpoint_schema_version": CHECKPOINT_SCHEMA_VERSION,
        "parsed_arguments": vars(args),
        "random_seed": int(args.seed),
        "python": {
            "implementation": platform.python_implementation(),
            "version": platform.python_version(),
            "executable": str(Path(sys.executable).resolve()),
            "platform": platform.platform(),
        },
        "package_versions": package_versions(),
        "source_code": artifact_record(SCRIPT_PATH),
        "dependencies": [artifact_record(SCRIPT_PATH.with_name("sequence_models.py"))],
        "inputs": [artifact_record(path) for path in input_paths],
    }
    payload["contract_sha256"] = sha256_payload(payload)
    return payload


def sequence_bundle_fingerprint(bundle: SequenceBundle) -> str:
    """Hash exactly the encoded observations, boundaries, and symbol contract."""

    digest = hashlib.sha256()
    x = np.ascontiguousarray(bundle.X, dtype=np.int64)
    lengths = np.ascontiguousarray(bundle.lengths, dtype=np.int64)
    digest.update(x.tobytes(order="C"))
    digest.update(lengths.tobytes(order="C"))
    digest.update(canonical_json_bytes({
        "n_symbols": bundle.n_symbols,
        "observation_labels": list(bundle.observation_labels),
        "n_sequences": bundle.n_sequences,
        "n_observations": bundle.n_observations,
        "metadata": dict(bundle.metadata),
    }))
    return digest.hexdigest()


def stable_hash(value: str, seed: int) -> str:
    return hashlib.sha256(f"{seed}|{value}".encode("utf-8")).hexdigest()


def anonymised_sequence_id(value: Any) -> str:
    return hashlib.sha256(f"q1-full|{value}".encode("utf-8")).hexdigest()[:16]


def load_panel() -> pd.DataFrame:
    if not PANEL_PATH.exists():
        raise FileNotFoundError(PANEL_PATH)
    frame = pd.read_parquet(PANEL_PATH, columns=OBSERVATION_COLUMNS)
    frame["date"] = pd.to_datetime(frame["date"], errors="raise")
    frame["cohort"] = frame["cohort"].astype(str)
    if frame.duplicated(["registration_id", "date"]).any():
        raise ValueError("KU panel contains duplicate registration-day rows")
    if set(frame["cohort"].unique()) != {"1819", "1920", "2021"}:
        raise ValueError("expected exactly cohorts 1819, 1920, and 2021")
    if frame[OBSERVATION_COLUMNS].isna().any().loc[
        [
            "registration_id",
            "cohort",
            "COURSE_ID",
            "date",
            "course_day_index",
            "platform_inactive",
            "activity_category_diversity",
            "productive_event_count",
            "forum_event_count",
        ]
    ].any():
        raise ValueError("required Q1 fields contain missing values")
    return frame.sort_values(["registration_id", "date"], kind="mergesort").reset_index(drop=True)


def encoder_config() -> ObservationEncodingConfig:
    return ObservationEncodingConfig(
        inactive_col="platform_inactive",
        diversity_col="activity_category_diversity",
        productive_col="productive_event_count",
        interaction_col="forum_event_count",
        diversity_quantile=0.67,
        missing_policy="raise",
    )


def make_bundle(frame: pd.DataFrame, encoder: TransparentObservationEncoder, name: str) -> SequenceBundle:
    return make_sequence_bundle(
        frame,
        encoder,
        group_cols="registration_id",
        time_col="date",
        min_sequence_length=2,
        expected_step="1D",
        gap_policy="raise",
        duplicate_time_policy="raise",
        dataset_name=name,
    )


def deterministic_stratified_cap(
    frame: pd.DataFrame,
    *,
    cap_per_course: int | None,
    seed: int,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    groups = frame[["registration_id", "cohort", "COURSE_ID"]].drop_duplicates().copy()
    if groups["registration_id"].duplicated().any():
        raise ValueError("registration_id crosses cohort-course strata")
    groups["selection_hash"] = groups["registration_id"].astype(str).map(
        lambda value: stable_hash(value, seed)
    )
    groups = groups.sort_values(["cohort", "COURSE_ID", "selection_hash", "registration_id"])
    groups["rank"] = groups.groupby(["cohort", "COURSE_ID"], sort=True).cumcount() + 1
    if cap_per_course is None:
        groups["selected"] = True
        rule = "all 1819 registrations"
    else:
        groups["selected"] = groups["rank"].le(int(cap_per_course))
        rule = (
            "outcome-blind SHA-256(seed|registration_id) order within each 1819 course; "
            f"cap={cap_per_course}"
        )
    chosen = groups.loc[groups["selected"], "registration_id"]
    selected = frame.loc[frame["registration_id"].isin(chosen)].copy()
    audit = (
        groups.groupby(["cohort", "COURSE_ID"], sort=True)
        .agg(
            available_sequences=("registration_id", "size"),
            selected_sequences=("selected", "sum"),
        )
        .reset_index()
    )
    audit["sampling_rule"] = rule
    return selected, audit


def observation_distribution(bundle: SequenceBundle, partition: str) -> pd.DataFrame:
    codes = np.concatenate(bundle.observations)
    counts = np.bincount(codes, minlength=bundle.n_symbols)
    return pd.DataFrame(
        {
            "partition": partition,
            "observation_code": np.arange(bundle.n_symbols, dtype=int),
            "observation": bundle.observation_labels,
            "count": counts.astype(int),
            "share": counts / counts.sum(),
        }
    )


def bundle_summary(bundle: SequenceBundle) -> dict[str, Any]:
    lengths = np.asarray(bundle.lengths, dtype=float)
    return {
        "n_sequences": int(bundle.n_sequences),
        "n_observations": int(bundle.n_observations),
        "min_length": int(lengths.min()),
        "median_length": float(np.median(lengths)),
        "max_length": int(lengths.max()),
    }


def conditional_mc_sequence_scores(
    transmat: np.ndarray,
    bundle: SequenceBundle,
    landmark_days: int,
) -> pd.DataFrame:
    log_transition = np.log(np.asarray(transmat, dtype=float))
    rows: list[dict[str, Any]] = []
    for key, sequence in zip(bundle.group_keys, bundle.observations):
        if len(sequence) <= landmark_days:
            continue
        source = sequence[landmark_days - 1 : -1]
        target = sequence[landmark_days:]
        ll = float(log_transition[source, target].sum())
        rows.append(
            {
                "registration_id": key[0],
                "conditional_loglik": ll,
                "scored_observations": int(len(target)),
            }
        )
    if not rows:
        raise ValueError("no sequence extends beyond the fixed landmark")
    return pd.DataFrame(rows)


def conditional_hmm_sequence_scores(
    model: Any,
    bundle: SequenceBundle,
    landmark_days: int,
) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for key, sequence in zip(bundle.group_keys, bundle.observations):
        if len(sequence) <= landmark_days:
            continue
        full_x = sequence.reshape(-1, 1)
        prefix_x = sequence[:landmark_days].reshape(-1, 1)
        conditional_ll = float(model.score(full_x) - model.score(prefix_x))
        rows.append(
            {
                "registration_id": key[0],
                "conditional_loglik": conditional_ll,
                "scored_observations": int(len(sequence) - landmark_days),
            }
        )
    if not rows:
        raise ValueError("no sequence extends beyond the fixed landmark")
    return pd.DataFrame(rows)


def score_summary(scores: pd.DataFrame) -> dict[str, float | int]:
    total_ll = float(scores["conditional_loglik"].sum())
    n_obs = int(scores["scored_observations"].sum())
    per_obs = total_ll / n_obs
    return {
        "total_loglik": total_ll,
        "n_sequences": int(len(scores)),
        "n_scored_observations": n_obs,
        "loglik_per_observation": per_obs,
        "negative_log_loss_per_observation": -per_obs,
        "perplexity": float(math.exp(-per_obs)),
    }


def posterior_occupancy(model: Any, bundle: SequenceBundle, start_position: int = 0) -> np.ndarray:
    posterior = model.predict_proba(bundle.X, lengths=bundle.lengths)
    selected: list[np.ndarray] = []
    cursor = 0
    for length in bundle.lengths:
        block = posterior[cursor : cursor + length]
        cursor += length
        if length > start_position:
            selected.append(block[start_position:])
    if not selected:
        raise ValueError("no posterior rows after requested start position")
    values = np.vstack(selected)
    return values.mean(axis=0)


_WORKER_BUNDLE: SequenceBundle | None = None

FitTask = tuple[int, int, int, int, float, float]


def _initialise_worker(bundle: SequenceBundle) -> None:
    global _WORKER_BUNDLE
    _WORKER_BUNDLE = bundle


def _fit_worker(task: FitTask) -> dict[str, Any]:
    if _WORKER_BUNDLE is None:
        raise RuntimeError("worker bundle not initialised")
    k, start_index, seed, n_iter, tol, prior_strength = task
    started = time.perf_counter()
    try:
        model, warning_messages = _fit_categorical_hmm_once(
            _WORKER_BUNDLE,
            n_states=k,
            seed=seed,
            n_iter=n_iter,
            tol=tol,
            prior_strength=prior_strength,
        )
        train_ll = float(model.score(_WORKER_BUNDLE.X, lengths=_WORKER_BUNDLE.lengths))
        iterations = int(model.monitor_.iter)
        history = list(model.monitor_.history)
        final_delta = (
            float(history[-1] - history[-2]) if len(history) >= 2 else float("nan")
        )
        return {
            "status": "fit",
            "k": int(k),
            "start_index": int(start_index),
            "seed": int(seed),
            "model": model,
            "iterations": iterations,
            "hmmlearn_converged": bool(model.monitor_.converged),
            "hit_iteration_cap": bool(iterations >= n_iter),
            "converged_before_cap": bool(model.monitor_.converged and iterations < n_iter),
            "train_loglik": train_ll,
            "train_loglik_per_observation": train_ll / _WORKER_BUNDLE.n_observations,
            "final_loglik_delta": final_delta,
            "warnings": " | ".join(warning_messages),
            "error": "",
            "elapsed_seconds": time.perf_counter() - started,
        }
    except Exception as exc:
        return {
            "status": "failed",
            "k": int(k),
            "start_index": int(start_index),
            "seed": int(seed),
            "model": None,
            "iterations": 0,
            "hmmlearn_converged": False,
            "hit_iteration_cap": False,
            "converged_before_cap": False,
            "train_loglik": float("nan"),
            "train_loglik_per_observation": float("nan"),
            "final_loglik_delta": float("nan"),
            "warnings": "",
            "error": f"{type(exc).__name__}: {exc}",
            "traceback": traceback.format_exc(limit=5),
            "elapsed_seconds": time.perf_counter() - started,
        }


def fit_task_payload(task: FitTask) -> dict[str, Any]:
    k, start_index, seed, n_iter, tol, prior_strength = task
    return {
        "k": int(k),
        "start_index": int(start_index),
        "seed": int(seed),
        "n_iter": int(n_iter),
        "tol": float(tol),
        "prior_strength": float(prior_strength),
    }


def checkpoint_stage_contract(
    bundle: SequenceBundle,
    *,
    run_contract: dict[str, Any],
    tasks: Sequence[FitTask],
    k_values: Sequence[int],
    n_starts: int,
    min_occupancy: float,
    workers: int,
    stage_name: str,
) -> dict[str, Any]:
    payload = {
        "checkpoint_schema_version": CHECKPOINT_SCHEMA_VERSION,
        "run_contract_sha256": run_contract["contract_sha256"],
        "stage_name": stage_name,
        "bundle": {
            "fingerprint_sha256": sequence_bundle_fingerprint(bundle),
            "n_sequences": bundle.n_sequences,
            "n_observations": bundle.n_observations,
            "n_symbols": bundle.n_symbols,
            "observation_labels": list(bundle.observation_labels),
        },
        "k_values": [int(value) for value in k_values],
        "n_starts": int(n_starts),
        "minimum_posterior_occupancy": float(min_occupancy),
        # Workers are part of the exact invocation contract even though each
        # start has its own deterministic seed and is mathematically independent.
        "workers": int(workers),
        "task_schedule": [fit_task_payload(task) for task in tasks],
    }
    payload["stage_contract_sha256"] = sha256_payload(payload)
    return payload


def checkpoint_task_stem(task: FitTask) -> str:
    task_data = fit_task_payload(task)
    return (
        f"k{task_data['k']:02d}_start{task_data['start_index']:03d}_"
        f"seed{task_data['seed']:010d}"
    )


def checkpoint_paths(
    checkpoint_root: Path,
    run_contract: dict[str, Any],
    stage_contract: dict[str, Any],
    task: FitTask,
) -> tuple[Path, Path]:
    stage_dir = (
        checkpoint_root
        / str(run_contract["contract_sha256"])
        / str(stage_contract["stage_name"])
    )
    stem = checkpoint_task_stem(task)
    return stage_dir / f"{stem}.json", stage_dir / f"{stem}.npz"


def _checkpoint_record_payload(record: dict[str, Any]) -> dict[str, Any]:
    return {
        key: value
        for key, value in record.items()
        if key not in {"model", "checkpoint_reused", "checkpoint_load_seconds"}
    }


def save_fit_checkpoint(
    checkpoint_root: Path,
    run_contract: dict[str, Any],
    stage_contract: dict[str, Any],
    task: FitTask,
    record: dict[str, Any],
) -> list[Path]:
    """Atomically persist one completed fit without using pickle."""

    metadata_path, arrays_path = checkpoint_paths(
        checkpoint_root, run_contract, stage_contract, task
    )
    metadata_path.parent.mkdir(parents=True, exist_ok=True)
    record_payload = _checkpoint_record_payload(record)
    body: dict[str, Any] = {
        "checkpoint_schema_version": CHECKPOINT_SCHEMA_VERSION,
        "run_contract_sha256": run_contract["contract_sha256"],
        "stage_contract_sha256": stage_contract["stage_contract_sha256"],
        "stage_name": stage_contract["stage_name"],
        "task": fit_task_payload(task),
        "record": record_payload,
        "model_storage": None,
    }
    written = [metadata_path]
    model = record.get("model")
    if record.get("status") == "fit" and model is not None:
        temporary_arrays = arrays_path.with_suffix(arrays_path.suffix + ".tmp")
        with temporary_arrays.open("wb") as handle:
            np.savez_compressed(
                handle,
                startprob=np.asarray(model.startprob_, dtype=np.float64),
                transmat=np.asarray(model.transmat_, dtype=np.float64),
                emissionprob=np.asarray(model.emissionprob_, dtype=np.float64),
            )
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary_arrays, arrays_path)
        body["model_storage"] = {
            "format": "numpy_npz_no_pickle",
            "filename": arrays_path.name,
            "bytes": int(arrays_path.stat().st_size),
            "sha256": sha256_file(arrays_path),
        }
        written.append(arrays_path)
    # Failed fits deliberately have no reusable parameter archive.  Their JSON
    # remains an audit trace, but the loader below always schedules them again.
    body["payload_sha256"] = sha256_payload(body)
    write_json(metadata_path, body)
    return written


def _rehydrate_categorical_hmm(
    task: FitTask,
    n_symbols: int,
    startprob: np.ndarray,
    transmat: np.ndarray,
    emissionprob: np.ndarray,
) -> Any:
    from hmmlearn.hmm import CategoricalHMM

    task_data = fit_task_payload(task)
    prior = 1.0 + float(task_data["prior_strength"])
    model = CategoricalHMM(
        n_components=int(task_data["k"]),
        n_features=int(n_symbols),
        startprob_prior=prior,
        transmat_prior=prior,
        emissionprob_prior=prior,
        algorithm="viterbi",
        random_state=int(task_data["seed"]),
        n_iter=int(task_data["n_iter"]),
        tol=float(task_data["tol"]),
        verbose=False,
        params="ste",
        init_params="ste",
        implementation="log",
    )
    model.startprob_ = np.asarray(startprob, dtype=np.float64).copy()
    model.transmat_ = np.asarray(transmat, dtype=np.float64).copy()
    model.emissionprob_ = np.asarray(emissionprob, dtype=np.float64).copy()
    return model


def load_fit_checkpoint(
    checkpoint_root: Path,
    run_contract: dict[str, Any],
    stage_contract: dict[str, Any],
    task: FitTask,
    bundle: SequenceBundle,
) -> tuple[dict[str, Any] | None, str, list[Path]]:
    """Return a verified successful fit or a reason requiring recomputation."""

    started = time.perf_counter()
    metadata_path, expected_arrays_path = checkpoint_paths(
        checkpoint_root, run_contract, stage_contract, task
    )
    if not metadata_path.exists():
        return None, "missing", []
    try:
        metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
        recorded_digest = metadata.pop("payload_sha256")
        if recorded_digest != sha256_payload(metadata):
            raise ValueError("metadata payload SHA-256 mismatch")
        if metadata.get("checkpoint_schema_version") != CHECKPOINT_SCHEMA_VERSION:
            raise ValueError("checkpoint schema version mismatch")
        if metadata.get("run_contract_sha256") != run_contract["contract_sha256"]:
            raise ValueError("run contract mismatch")
        if metadata.get("stage_contract_sha256") != stage_contract["stage_contract_sha256"]:
            raise ValueError("stage contract mismatch")
        if metadata.get("stage_name") != stage_contract["stage_name"]:
            raise ValueError("stage name mismatch")
        if metadata.get("task") != fit_task_payload(task):
            raise ValueError("fit task or seed mismatch")
        record_payload = metadata.get("record")
        if not isinstance(record_payload, dict):
            raise ValueError("checkpoint record is missing")
        if record_payload.get("status") != "fit":
            return None, "recorded_failed_fit_not_reusable", [metadata_path]
        for key, expected in (
            ("k", fit_task_payload(task)["k"]),
            ("start_index", fit_task_payload(task)["start_index"]),
            ("seed", fit_task_payload(task)["seed"]),
        ):
            if record_payload.get(key) != expected:
                raise ValueError(f"record {key} mismatch")
        storage = metadata.get("model_storage")
        if not isinstance(storage, dict) or storage.get("format") != "numpy_npz_no_pickle":
            raise ValueError("safe NPZ model storage is missing")
        if storage.get("filename") != expected_arrays_path.name:
            raise ValueError("model archive filename mismatch")
        if not expected_arrays_path.exists():
            raise ValueError("model archive is missing")
        if int(storage.get("bytes", -1)) != int(expected_arrays_path.stat().st_size):
            raise ValueError("model archive byte size mismatch")
        if storage.get("sha256") != sha256_file(expected_arrays_path):
            raise ValueError("model archive SHA-256 mismatch")
        with np.load(expected_arrays_path, allow_pickle=False) as archive:
            if set(archive.files) != {"startprob", "transmat", "emissionprob"}:
                raise ValueError("model archive keys are invalid")
            startprob = np.asarray(archive["startprob"], dtype=np.float64)
            transmat = np.asarray(archive["transmat"], dtype=np.float64)
            emissionprob = np.asarray(archive["emissionprob"], dtype=np.float64)
        k = int(fit_task_payload(task)["k"])
        if startprob.shape != (k,):
            raise ValueError("start probability shape mismatch")
        if transmat.shape != (k, k):
            raise ValueError("transition matrix shape mismatch")
        if emissionprob.shape != (k, bundle.n_symbols):
            raise ValueError("emission matrix shape mismatch")
        for name, array in (
            ("startprob", startprob),
            ("transmat", transmat),
            ("emissionprob", emissionprob),
        ):
            if not np.isfinite(array).all() or np.any(array < 0.0):
                raise ValueError(f"{name} contains invalid probabilities")
        if not np.isclose(startprob.sum(), 1.0, rtol=1e-10, atol=1e-10):
            raise ValueError("start probabilities do not sum to one")
        if not np.allclose(transmat.sum(axis=1), 1.0, rtol=1e-10, atol=1e-10):
            raise ValueError("transition probabilities do not sum to one")
        if not np.allclose(emissionprob.sum(axis=1), 1.0, rtol=1e-10, atol=1e-10):
            raise ValueError("emission probabilities do not sum to one")
        model = _rehydrate_categorical_hmm(
            task, bundle.n_symbols, startprob, transmat, emissionprob
        )
        verified_loglik = float(model.score(bundle.X, lengths=bundle.lengths))
        recorded_loglik = float(record_payload["train_loglik"])
        if not np.isclose(
            verified_loglik,
            recorded_loglik,
            rtol=1e-12,
            atol=1e-7,
        ):
            raise ValueError(
                "recomputed training log likelihood does not match checkpoint"
            )
        record = dict(record_payload)
        record["train_loglik"] = verified_loglik
        record["train_loglik_per_observation"] = (
            verified_loglik / bundle.n_observations
        )
        record["model"] = model
        record["checkpoint_reused"] = True
        record["checkpoint_load_seconds"] = time.perf_counter() - started
        return record, "verified_success_reused", [metadata_path, expected_arrays_path]
    except Exception as exc:
        return (
            None,
            f"rejected_{type(exc).__name__}: {exc}",
            [path for path in (metadata_path, expected_arrays_path) if path.exists()],
        )


@dataclass
class FitStageResult:
    records: list[dict[str, Any]]
    selected_by_k: dict[int, dict[str, Any]]
    starts_table: pd.DataFrame
    summary_table: pd.DataFrame
    checkpoint_audit: dict[str, Any]
    checkpoint_input_paths: list[Path]


def state_alignment_distances(reference: Any, candidate: Any) -> tuple[float, float, float]:
    alignment = hungarian_state_alignment(reference.emissionprob_, candidate.emissionprob_)
    ordered = alignment.sort_values("reference_state_index")
    order = ordered["candidate_state_index"].to_numpy(dtype=int)
    candidate_transition = candidate.transmat_[np.ix_(order, order)]
    candidate_emission = candidate.emissionprob_[order]
    transition_rmse = float(np.sqrt(np.mean((reference.transmat_ - candidate_transition) ** 2)))
    emission_rmse = float(np.sqrt(np.mean((reference.emissionprob_ - candidate_emission) ** 2)))
    return float(ordered["js_divergence"].mean()), emission_rmse, transition_rmse


def fit_multistart_grid(
    bundle: SequenceBundle,
    *,
    k_values: Sequence[int],
    n_starts: int,
    seed: int,
    n_iter: int,
    tol: float,
    prior_strength: float,
    min_occupancy: float,
    workers: int,
    stage_name: str,
    checkpoint_run_contract: dict[str, Any],
    checkpoint_root: Path = CHECKPOINT_ROOT,
    test_interrupt_after_new_fits: int | None = None,
) -> FitStageResult:
    seed_rng = np.random.default_rng(seed)
    tasks: list[FitTask] = []
    for k in k_values:
        for start_index in range(n_starts):
            start_seed = int(seed_rng.integers(0, np.iinfo(np.int32).max))
            tasks.append(
                (
                    int(k),
                    int(start_index),
                    start_seed,
                    int(n_iter),
                    float(tol),
                    float(prior_strength),
                )
            )

    stage_contract = checkpoint_stage_contract(
        bundle,
        run_contract=checkpoint_run_contract,
        tasks=tasks,
        k_values=k_values,
        n_starts=n_starts,
        min_occupancy=min_occupancy,
        workers=workers,
        stage_name=stage_name,
    )

    append_progress(
        stage_name,
        "started",
        n_sequences=bundle.n_sequences,
        n_observations=bundle.n_observations,
        k_values=list(k_values),
        n_starts=n_starts,
        workers=workers,
        n_iter=n_iter,
        tol=tol,
        stage_contract_sha256=stage_contract["stage_contract_sha256"],
    )
    records: list[dict[str, Any]] = []
    pending_tasks: list[FitTask] = []
    checkpoint_input_paths: list[Path] = []
    checkpoint_files_written: list[Path] = []
    rejected_details: list[dict[str, Any]] = []
    missing_checkpoints = 0
    rejected_checkpoints = 0
    reused_successful_fits = 0
    newly_computed_fits = 0
    newly_computed_successful_fits = 0
    newly_computed_failed_fits = 0

    for task in tasks:
        record, reuse_status, inspected_paths = load_fit_checkpoint(
            checkpoint_root,
            checkpoint_run_contract,
            stage_contract,
            task,
            bundle,
        )
        if record is not None:
            records.append(record)
            checkpoint_input_paths.extend(inspected_paths)
            reused_successful_fits += 1
            append_progress(
                stage_name,
                "checkpoint_reused",
                completed=len(records),
                total=len(tasks),
                k=record["k"],
                start_index=record["start_index"],
                seed=record["seed"],
                checkpoint_load_seconds=record["checkpoint_load_seconds"],
            )
        else:
            pending_tasks.append(task)
            if reuse_status == "missing":
                missing_checkpoints += 1
            else:
                rejected_checkpoints += 1
                rejected_details.append(
                    {
                        "task": fit_task_payload(task),
                        "reason": reuse_status,
                        "inspected_paths": [portable_path(path) for path in inspected_paths],
                    }
                )
                append_progress(
                    stage_name,
                    "checkpoint_rejected",
                    task=fit_task_payload(task),
                    reason=reuse_status,
                )

    append_progress(
        stage_name,
        "checkpoint_scan_completed",
        requested_fits=len(tasks),
        reused_successful_fits=reused_successful_fits,
        pending_fits=len(pending_tasks),
        missing_checkpoints=missing_checkpoints,
        rejected_checkpoints=rejected_checkpoints,
    )

    def accept_new_record(record: dict[str, Any], task: FitTask) -> None:
        nonlocal newly_computed_fits
        nonlocal newly_computed_successful_fits
        nonlocal newly_computed_failed_fits
        record["checkpoint_reused"] = False
        record["checkpoint_load_seconds"] = 0.0
        written_paths = save_fit_checkpoint(
            checkpoint_root,
            checkpoint_run_contract,
            stage_contract,
            task,
            record,
        )
        checkpoint_files_written.extend(written_paths)
        records.append(record)
        newly_computed_fits += 1
        if record["status"] == "fit":
            newly_computed_successful_fits += 1
        else:
            newly_computed_failed_fits += 1
        append_progress(
            stage_name,
            "fit_completed",
            completed=len(records),
            total=len(tasks),
            k=record["k"],
            start_index=record["start_index"],
            seed=record["seed"],
            checkpoint_reused=False,
            checkpoint_metadata=portable_path(written_paths[0]),
            converged_before_cap=record["converged_before_cap"],
            iterations=record["iterations"],
            elapsed_seconds=record["elapsed_seconds"],
        )
        if (
            test_interrupt_after_new_fits is not None
            and newly_computed_fits >= test_interrupt_after_new_fits
        ):
            raise RuntimeError(
                "intentional checkpoint interruption requested by test hook"
            )

    if workers <= 1:
        _initialise_worker(bundle)
        for task in pending_tasks:
            record = _fit_worker(task)
            accept_new_record(record, task)
    elif pending_tasks:
        with ProcessPoolExecutor(
            max_workers=workers,
            initializer=_initialise_worker,
            initargs=(bundle,),
        ) as executor:
            futures = {
                executor.submit(_fit_worker, task): task for task in pending_tasks
            }
            for future in as_completed(futures):
                task = futures[future]
                record = future.result()
                accept_new_record(record, task)

    if len(records) != len(tasks):
        raise RuntimeError(
            f"fit stage {stage_name} completed {len(records)} of {len(tasks)} tasks"
        )
    # Multiprocessing completion order and the mix of reused/new records must
    # never influence tie handling or any downstream table.
    records.sort(key=lambda item: (int(item["k"]), int(item["start_index"])))

    # Canonicalise and apply the occupancy screen without using validation data.
    for record in records:
        record["selected_within_k"] = False
        record["occupancy_screen_passed"] = False
        record["feasible"] = False
        record["min_posterior_state_occupancy"] = float("nan")
        record["mean_emission_js_to_selected"] = float("nan")
        record["emission_rmse_to_selected"] = float("nan")
        record["transition_rmse_to_selected"] = float("nan")
        if record["status"] != "fit" or record["model"] is None:
            continue
        model, _, _ = canonicalize_hmm(record["model"], bundle.observation_labels)
        record["model"] = model
        posterior = model.predict_proba(bundle.X, lengths=bundle.lengths)
        occupancy = posterior.mean(axis=0)
        minimum = float(occupancy.min())
        record["min_posterior_state_occupancy"] = minimum
        record["occupancy_screen_passed"] = bool(minimum >= min_occupancy)
        record["feasible"] = bool(
            record["converged_before_cap"] and record["occupancy_screen_passed"]
        )

    selected_by_k: dict[int, dict[str, Any]] = {}
    summary_rows: list[dict[str, Any]] = []
    for k in k_values:
        candidates = [record for record in records if record["k"] == int(k) and record["status"] == "fit"]
        feasible = [record for record in candidates if record["feasible"]]
        provisional = [record for record in candidates if record["occupancy_screen_passed"]]
        pool = feasible if feasible else provisional
        selected = (
            max(
                pool,
                key=lambda item: (
                    item["train_loglik"],
                    -int(item["start_index"]),
                ),
            )
            if pool
            else None
        )
        if selected is not None:
            selected["selected_within_k"] = True
            selected_by_k[int(k)] = selected
            for candidate in candidates:
                if candidate["model"] is not None:
                    js_mean, emission_rmse, transition_rmse = state_alignment_distances(
                        selected["model"], candidate["model"]
                    )
                    candidate["mean_emission_js_to_selected"] = js_mean
                    candidate["emission_rmse_to_selected"] = emission_rmse
                    candidate["transition_rmse_to_selected"] = transition_rmse
        strict_ll = np.asarray([item["train_loglik_per_observation"] for item in feasible], dtype=float)
        summary_rows.append(
            {
                "k": int(k),
                "requested_starts": int(n_starts),
                "successful_starts": int(len(candidates)),
                "strict_converged_starts": int(sum(item["converged_before_cap"] for item in candidates)),
                "occupancy_passed_starts": int(sum(item["occupancy_screen_passed"] for item in candidates)),
                "feasible_starts": int(len(feasible)),
                "selected_start_seed": selected["seed"] if selected else np.nan,
                "selected_start_strict": bool(selected and selected["feasible"]),
                "selected_train_loglik_per_observation": (
                    selected["train_loglik_per_observation"] if selected else np.nan
                ),
                "strict_start_loglik_sd": float(strict_ll.std(ddof=1)) if len(strict_ll) >= 2 else np.nan,
                "strict_start_loglik_range": float(strict_ll.max() - strict_ll.min()) if len(strict_ll) else np.nan,
            }
        )

    starts_rows: list[dict[str, Any]] = []
    for record in records:
        starts_rows.append({key: value for key, value in record.items() if key not in {"model", "traceback"}})
    starts_table = pd.DataFrame(starts_rows).sort_values(["k", "start_index"], ignore_index=True)
    summary_table = pd.DataFrame(summary_rows).sort_values("k", ignore_index=True)
    append_progress(
        stage_name,
        "completed",
        successful_fits=int(starts_table["status"].eq("fit").sum()),
        strict_converged_fits=int(starts_table["converged_before_cap"].sum()),
        reused_successful_fits=reused_successful_fits,
        newly_computed_fits=newly_computed_fits,
        rejected_checkpoints=rejected_checkpoints,
    )
    checkpoint_audit = {
        "checkpoint_schema_version": CHECKPOINT_SCHEMA_VERSION,
        "run_contract_sha256": checkpoint_run_contract["contract_sha256"],
        "stage_contract_sha256": stage_contract["stage_contract_sha256"],
        "stage_name": stage_name,
        "checkpoint_directory": portable_path(
            checkpoint_root / checkpoint_run_contract["contract_sha256"] / stage_name
        ),
        "requested_fits": len(tasks),
        "reused_successful_fits": reused_successful_fits,
        "newly_computed_fits": newly_computed_fits,
        "newly_computed_successful_fits": newly_computed_successful_fits,
        "newly_computed_failed_fits": newly_computed_failed_fits,
        "missing_checkpoints": missing_checkpoints,
        "rejected_checkpoints": rejected_checkpoints,
        "rejected_details": rejected_details,
        "checkpoint_files_written": sorted(
            {portable_path(path) for path in checkpoint_files_written}
        ),
        "reuse_rule": (
            "Only status=fit records with exact run/stage/task/seed contracts, "
            "valid JSON and NPZ hashes, valid probability arrays, and a matching "
            "recomputed training log likelihood are reusable. Failed fits are "
            "always recomputed."
        ),
    }
    return FitStageResult(
        records,
        selected_by_k,
        starts_table,
        summary_table,
        checkpoint_audit,
        sorted(set(checkpoint_input_paths), key=lambda path: str(path)),
    )


def long_matrix(matrix: np.ndarray, labels: Sequence[str], matrix_name: str) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for i, source in enumerate(labels):
        for j, target in enumerate(labels):
            rows.append(
                {
                    "matrix": matrix_name,
                    "from_index": i,
                    "from_label": source,
                    "to_index": j,
                    "to_label": target,
                    "probability": float(matrix[i, j]),
                }
            )
    return pd.DataFrame(rows)


def cluster_bootstrap_delta(
    merged: pd.DataFrame,
    *,
    replications: int,
    seed: int,
) -> dict[str, float | int]:
    delta = (
        merged["conditional_loglik_hmm"].to_numpy(dtype=float)
        - merged["conditional_loglik_mc"].to_numpy(dtype=float)
    )
    counts = merged["scored_observations"].to_numpy(dtype=float)
    point = float(delta.sum() / counts.sum())
    rng = np.random.default_rng(seed)
    n = len(merged)
    samples = np.empty(replications, dtype=float)
    for start in range(0, replications, 250):
        stop = min(start + 250, replications)
        indices = rng.integers(0, n, size=(stop - start, n))
        samples[start:stop] = delta[indices].sum(axis=1) / counts[indices].sum(axis=1)
    return {
        "replications": int(replications),
        "point_delta_loglik_per_observation": point,
        "ci_2.5pct": float(np.quantile(samples, 0.025)),
        "ci_97.5pct": float(np.quantile(samples, 0.975)),
        "bootstrap_probability_delta_positive": float(np.mean(samples > 0)),
    }


def save_model_comparison_figure(table: pd.DataFrame, selected_k: int | None, path: Path) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(10.2, 4.2), constrained_layout=True)
    inner = table.loc[table["evaluation_partition"].eq("1920-inner")].copy()
    labels = inner["model_label"].tolist()
    values = inner["conditional_loglik_per_observation"].to_numpy(dtype=float)
    colours = ["#5B6B7A" if label == "Observable MC" else "#2A9D8F" for label in labels]
    axes[0].bar(np.arange(len(labels)), values, color=colours)
    axes[0].set_xticks(np.arange(len(labels)), labels, rotation=35, ha="right")
    axes[0].set_ylabel("Conditional log-likelihood / observation")
    axes[0].set_title("Inner selection: 1920")
    axes[0].axhline(inner.loc[inner["model_label"].eq("Observable MC"), "conditional_loglik_per_observation"].iloc[0], color="#5B6B7A", linestyle="--", linewidth=1)
    if selected_k is not None:
        for index, label in enumerate(labels):
            if label == f"HMM K={selected_k}":
                axes[0].scatter(index, values[index], s=70, facecolors="none", edgecolors="#D1495B", linewidths=1.8, zorder=4)

    outer = table.loc[table["evaluation_partition"].eq("2021-outer")].copy()
    axes[1].bar(
        np.arange(len(outer)),
        outer["conditional_loglik_per_observation"],
        color=["#5B6B7A", "#E9C46A"][: len(outer)],
    )
    axes[1].set_xticks(np.arange(len(outer)), outer["model_label"], rotation=20, ha="right")
    axes[1].set_title("Untouched outer test: 2021")
    axes[1].set_ylabel("Conditional log-likelihood / observation")
    for axis in axes:
        axis.grid(axis="y", color="#D9D9D9", linewidth=0.6, alpha=0.8)
        axis.spines[["top", "right"]].set_visible(False)
    fig.suptitle("MC versus categorical HMM after a fixed 42-day landmark", fontsize=12)
    fig.savefig(path, dpi=320, bbox_inches="tight")
    plt.close(fig)


def save_state_figure(
    model: Any | None,
    observation_labels: Sequence[str],
    occupancy: pd.DataFrame,
    path: Path,
) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(11.2, 4.4), constrained_layout=True)
    if model is None:
        for axis in axes:
            axis.axis("off")
        axes[0].text(0.5, 0.5, "No HMM passed the strict gate", ha="center", va="center")
    else:
        image = axes[0].imshow(model.emissionprob_, vmin=0, vmax=1, cmap="YlGnBu", aspect="auto")
        axes[0].set_xticks(np.arange(len(observation_labels)), observation_labels, rotation=35, ha="right")
        states = [f"State {index + 1}" for index in range(model.n_components)]
        axes[0].set_yticks(np.arange(model.n_components), states)
        axes[0].set_title("Emission probabilities")
        for i in range(model.n_components):
            for j in range(len(observation_labels)):
                value = model.emissionprob_[i, j]
                axes[0].text(j, i, f"{value:.2f}", ha="center", va="center", color="white" if value > 0.55 else "black", fontsize=8)
        fig.colorbar(image, ax=axes[0], fraction=0.046, pad=0.04)

        partitions = occupancy["partition"].drop_duplicates().tolist()
        x = np.arange(model.n_components)
        width = 0.8 / max(len(partitions), 1)
        palette = ["#264653", "#2A9D8F", "#E9C46A"]
        for index, partition in enumerate(partitions):
            block = occupancy.loc[occupancy["partition"].eq(partition)].sort_values("state_index")
            axes[1].bar(x + (index - (len(partitions) - 1) / 2) * width, block["posterior_occupancy"], width=width, label=partition, color=palette[index % len(palette)])
        axes[1].set_xticks(x, states)
        axes[1].set_ylim(0, max(0.5, float(occupancy["posterior_occupancy"].max()) * 1.15))
        axes[1].set_ylabel("Smoothed posterior occupancy")
        axes[1].set_title("Numbered-state occupancy")
        axes[1].legend(frameon=False, fontsize=8)
        axes[1].grid(axis="y", color="#D9D9D9", linewidth=0.6)
        axes[1].spines[["top", "right"]].set_visible(False)
    fig.suptitle("State profiles are platform-evidence patterns, not cognitive labels", fontsize=11)
    fig.savefig(path, dpi=320, bbox_inches="tight")
    plt.close(fig)


def save_dwell_figure(dwell: pd.DataFrame, path: Path) -> None:
    fig, axis = plt.subplots(figsize=(7.4, 4.4), constrained_layout=True)
    if dwell.empty or "observed_uncensored_mean" not in dwell:
        axis.axis("off")
        axis.text(0.5, 0.5, "Duration PPC unavailable: no HMM passed the strict gate", ha="center", va="center")
    else:
        x = np.arange(len(dwell))
        observed = dwell["observed_uncensored_mean"].to_numpy(dtype=float)
        median = dwell["ppc_mean_median"].to_numpy(dtype=float)
        low = dwell["ppc_mean_2.5pct"].to_numpy(dtype=float)
        high = dwell["ppc_mean_97.5pct"].to_numpy(dtype=float)
        yerr = np.vstack([median - low, high - median])
        axis.errorbar(x, median, yerr=yerr, fmt="o", color="#2A9D8F", capsize=4, label="HMM PPC median and 95% interval")
        axis.scatter(x, observed, marker="D", color="#D1495B", label="Observed decoded spells", zorder=4)
        axis.set_xticks(x, dwell["state"])
        axis.set_ylabel("Uncensored spell mean (days)")
        axis.set_title("Geometric-duration posterior predictive diagnostic")
        axis.grid(axis="y", color="#D9D9D9", linewidth=0.6)
        axis.spines[["top", "right"]].set_visible(False)
        axis.legend(frameon=False, fontsize=8)
    fig.savefig(path, dpi=320, bbox_inches="tight")
    plt.close(fig)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--seed", type=int, default=DEFAULT_SEED)
    parser.add_argument("--landmark-days", type=int, default=42)
    parser.add_argument("--k-values", type=int, nargs="+", default=[2, 3, 4, 5, 6])
    parser.add_argument("--n-starts", type=int, default=10)
    parser.add_argument("--n-iter", type=int, default=300)
    parser.add_argument("--tol", type=float, default=1e-2)
    parser.add_argument("--hmm-prior-strength", type=float, default=0.1)
    parser.add_argument("--mc-alpha", type=float, default=0.5)
    parser.add_argument("--min-state-occupancy", type=float, default=0.01)
    parser.add_argument("--workers", type=int, default=2)
    parser.add_argument("--wall-clock-budget-minutes", type=float, default=45.0)
    parser.add_argument("--benchmark-seconds", type=float, default=68.3684427)
    parser.add_argument("--benchmark-iterations", type=int, default=100)
    parser.add_argument("--selection-cap-per-course", type=int, default=300)
    parser.add_argument("--dwell-ppc-replications", type=int, default=200)
    parser.add_argument("--dwell-sample-per-stratum", type=int, default=100)
    parser.add_argument("--bootstrap-replications", type=int, default=5000)
    parser.add_argument("--practical-delta-threshold", type=float, default=0.001)
    return parser


def main(argv: list[str] | None = None) -> int:
    global _ACTIVE_CHECKPOINT_CONTRACT_ID
    global _ACTIVE_INVOCATION_ID

    args = build_parser().parse_args(argv)
    args.k_values = sorted(set(int(value) for value in args.k_values))
    if args.k_values != [2, 3, 4, 5, 6]:
        raise ValueError("publication Q1 run requires exactly K=2,3,4,5,6")
    if args.n_starts < 10:
        raise ValueError("publication Q1 run requires at least 10 starts per K")
    if args.landmark_days < 2:
        raise ValueError("landmark must retain at least two conditioning days")
    if args.workers < 1:
        raise ValueError("workers must be positive")

    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    TABLES_DIR.mkdir(parents=True, exist_ok=True)
    FIGURES_DIR.mkdir(parents=True, exist_ok=True)
    CHECKPOINT_ROOT.mkdir(parents=True, exist_ok=True)
    checkpoint_run_contract = build_checkpoint_run_contract(args)
    _ACTIVE_INVOCATION_ID = uuid.uuid4().hex
    _ACTIVE_CHECKPOINT_CONTRACT_ID = checkpoint_run_contract["contract_sha256"]
    started_at = utc_now()
    start_clock = time.perf_counter()
    command = [sys.executable, str(SCRIPT_PATH), *sys.argv[1:]]
    # The progress ledger is append-only.  invocation_id and contract_id make
    # every resumed call distinguishable without destroying interrupted evidence.
    append_progress(
        "run",
        "started",
        command=command,
        checkpoint_resume_enabled=True,
        run_contract_sha256=checkpoint_run_contract["contract_sha256"],
    )

    frame = load_panel()
    pilot = json.loads(PILOT_SUMMARY_PATH.read_text(encoding="utf-8")) if PILOT_SUMMARY_PATH.exists() else {}
    cohort_counts = frame.groupby("cohort")["registration_id"].nunique().to_dict()
    if min(frame.groupby("registration_id").size()) <= args.landmark_days:
        raise ValueError("at least one sequence does not extend beyond the fixed landmark")
    append_progress("data", "validated", rows=len(frame), sequences=frame["registration_id"].nunique(), cohorts=cohort_counts)

    inner_train_full_frame = frame.loc[frame["cohort"].eq("1819")].copy()
    inner_validation_frame = frame.loc[frame["cohort"].eq("1920")].copy()
    development_frame = frame.loc[frame["cohort"].isin(["1819", "1920"])].copy()
    outer_frame = frame.loc[frame["cohort"].eq("2021")].copy()

    inner_encoder = TransparentObservationEncoder(encoder_config()).fit(inner_train_full_frame)
    inner_train_full_bundle = make_bundle(inner_train_full_frame, inner_encoder, "KU-1819-full")
    inner_validation_bundle = make_bundle(inner_validation_frame, inner_encoder, "KU-1920-inner-validation")

    # Capacity decision is determined only by an already measured, read-only
    # timing benchmark.  It never consults outcomes, validation likelihoods, or
    # the outer cohort.  K^2 scales the forward-backward work approximately.
    k5_equivalent = sum(k * k for k in args.k_values) / 25.0
    projected_grid_seconds = (
        args.benchmark_seconds
        * k5_equivalent
        * args.n_starts
        * (args.n_iter / args.benchmark_iterations)
        / args.workers
    )
    use_capacity_cap = projected_grid_seconds > args.wall_clock_budget_minutes * 60.0
    cap = args.selection_cap_per_course if use_capacity_cap else None
    inner_fit_frame, sampling_audit = deterministic_stratified_cap(
        inner_train_full_frame,
        cap_per_course=cap,
        seed=args.seed,
    )
    inner_fit_bundle = make_bundle(inner_fit_frame, inner_encoder, "KU-1819-K-grid-fit")
    capacity_payload = {
        "schema_version": SCHEMA_VERSION,
        "generated_by": str(SCRIPT_PATH.relative_to(PROJECT_ROOT)).replace("\\", "/"),
        "generated_at": utc_now(),
        "decision_uses_behavioral_outcomes": False,
        "benchmark": {
            "seconds": args.benchmark_seconds,
            "protocol": "KU 1819 full, K=5, one start, 100 iterations, read-only timing run",
            "iterations": args.benchmark_iterations,
        },
        "projection": {
            "k_squared_scaling_factor_relative_to_k5": k5_equivalent,
            "requested_starts_per_k": args.n_starts,
            "requested_iterations": args.n_iter,
            "workers": args.workers,
            "projected_grid_seconds": projected_grid_seconds,
            "wall_clock_budget_minutes": args.wall_clock_budget_minutes,
        },
        "decision": "capacity_adaptive_stratified_k_grid" if use_capacity_cap else "full_1819_k_grid",
        "cap_per_1819_course_stratum": cap,
        "grid_fit_sequences": inner_fit_bundle.n_sequences,
        "grid_fit_observations": inner_fit_bundle.n_observations,
        "full_1819_sequences": inner_train_full_bundle.n_sequences,
        "full_1819_observations": inner_train_full_bundle.n_observations,
        "guarantee": "selected K is refit on all 1819+1920 sequences; all 2021 sequences remain untouched for outer testing",
    }
    capacity_path = RESULTS_DIR / "q1_full_capacity_plan.json"
    write_json(capacity_path, capacity_payload)
    append_progress(
        "capacity",
        "decided",
        decision=capacity_payload["decision"],
        grid_sequences=inner_fit_bundle.n_sequences,
    )

    sampling_path = TABLES_DIR / "q1_full_sampling_audit.csv"
    sampling_audit.to_csv(sampling_path, index=False, encoding="utf-8-sig")

    # Inner baseline uses exactly the HMM K-grid estimation sample.  The final
    # outer baseline is refit below on the complete development sample.
    inner_mc = fit_dirichlet_markov_chain(inner_fit_bundle, alpha=args.mc_alpha)
    inner_mc_scores = conditional_mc_sequence_scores(inner_mc.transmat, inner_validation_bundle, args.landmark_days)
    inner_mc_summary = score_summary(inner_mc_scores)

    grid = fit_multistart_grid(
        inner_fit_bundle,
        k_values=args.k_values,
        n_starts=args.n_starts,
        seed=args.seed,
        n_iter=args.n_iter,
        tol=args.tol,
        prior_strength=args.hmm_prior_strength,
        min_occupancy=args.min_state_occupancy,
        workers=args.workers,
        stage_name="inner_k_grid",
        checkpoint_run_contract=checkpoint_run_contract,
    )

    comparison_rows: list[dict[str, Any]] = [
        {
            "evaluation_partition": "1920-inner",
            "model_label": "Observable MC",
            "model_family": "MC",
            "k": np.nan,
            "strictly_eligible": True,
            "selected_k": False,
            "conditional_total_loglik": inner_mc_summary["total_loglik"],
            "n_sequences": inner_mc_summary["n_sequences"],
            "n_scored_observations": inner_mc_summary["n_scored_observations"],
            "conditional_loglik_per_observation": inner_mc_summary["loglik_per_observation"],
            "negative_log_loss_per_observation": inner_mc_summary["negative_log_loss_per_observation"],
            "perplexity": inner_mc_summary["perplexity"],
        }
    ]
    for k in args.k_values:
        selected = grid.selected_by_k.get(k)
        if selected is None:
            conditional = {
                "total_loglik": np.nan,
                "n_sequences": inner_validation_bundle.n_sequences,
                "n_scored_observations": sum(max(0, length - args.landmark_days) for length in inner_validation_bundle.lengths),
                "loglik_per_observation": np.nan,
                "negative_log_loss_per_observation": np.nan,
                "perplexity": np.nan,
            }
        else:
            scores = conditional_hmm_sequence_scores(selected["model"], inner_validation_bundle, args.landmark_days)
            conditional = score_summary(scores)
        comparison_rows.append(
            {
                "evaluation_partition": "1920-inner",
                "model_label": f"HMM K={k}",
                "model_family": "Categorical HMM",
                "k": k,
                "strictly_eligible": bool(selected and selected["feasible"]),
                "selected_k": False,
                "conditional_total_loglik": conditional["total_loglik"],
                "n_sequences": conditional["n_sequences"],
                "n_scored_observations": conditional["n_scored_observations"],
                "conditional_loglik_per_observation": conditional["loglik_per_observation"],
                "negative_log_loss_per_observation": conditional["negative_log_loss_per_observation"],
                "perplexity": conditional["perplexity"],
            }
        )

    eligible_inner = [row for row in comparison_rows if row["model_family"] == "Categorical HMM" and row["strictly_eligible"] and np.isfinite(row["conditional_loglik_per_observation"])]
    if eligible_inner:
        best_inner = max(eligible_inner, key=lambda row: row["conditional_loglik_per_observation"])
        selected_k = int(best_inner["k"])
        best_inner["selected_k"] = True
        inner_selection_status = "selected_under_strict_gate"
    else:
        selected_k = None
        inner_selection_status = "no_hmm_passed_strict_gate"
    append_progress("inner_selection", "completed", selected_k=selected_k, status_detail=inner_selection_status)

    # Final preprocessing and MC are fit on every development sequence.  The
    # outer partition is deliberately not encoded or scored yet.
    final_encoder = TransparentObservationEncoder(encoder_config()).fit(development_frame)
    development_bundle = make_bundle(development_frame, final_encoder, "KU-1819-1920-final-development")
    final_mc = fit_dirichlet_markov_chain(development_bundle, alpha=args.mc_alpha)

    final_fit: FitStageResult | None = None
    final_hmm_record: dict[str, Any] | None = None
    outer_hmm_scores: pd.DataFrame | None = None
    outer_hmm_summary: dict[str, Any] | None = None
    if selected_k is not None:
        final_fit = fit_multistart_grid(
            development_bundle,
            k_values=[selected_k],
            n_starts=args.n_starts,
            seed=args.seed + 100_000,
            n_iter=args.n_iter,
            tol=args.tol,
            prior_strength=args.hmm_prior_strength,
            min_occupancy=args.min_state_occupancy,
            workers=args.workers,
            stage_name="final_full_development_refit",
            checkpoint_run_contract=checkpoint_run_contract,
        )
        candidate = final_fit.selected_by_k.get(selected_k)
        if candidate is not None and candidate["feasible"]:
            final_hmm_record = candidate
        else:
            append_progress("outer_test", "hmm_blocked", reason="final refit did not pass strict convergence and occupancy gates")

    # Only after final refitting is complete do we encode and score 2021.  Thus
    # the outer partition cannot affect preprocessing, K, start choice, or any
    # fitted probability.
    append_progress("outer_test", "started_after_final_refit", selected_k=selected_k)
    outer_bundle = make_bundle(outer_frame, final_encoder, "KU-2021-untouched-outer")
    outer_mc_scores = conditional_mc_sequence_scores(
        final_mc.transmat, outer_bundle, args.landmark_days
    )
    outer_mc_summary = score_summary(outer_mc_scores)
    if final_hmm_record is not None:
        outer_hmm_scores = conditional_hmm_sequence_scores(
            final_hmm_record["model"], outer_bundle, args.landmark_days
        )
        outer_hmm_summary = score_summary(outer_hmm_scores)

    comparison_rows.append(
        {
            "evaluation_partition": "2021-outer",
            "model_label": "Observable MC",
            "model_family": "MC",
            "k": np.nan,
            "strictly_eligible": True,
            "selected_k": False,
            "conditional_total_loglik": outer_mc_summary["total_loglik"],
            "n_sequences": outer_mc_summary["n_sequences"],
            "n_scored_observations": outer_mc_summary["n_scored_observations"],
            "conditional_loglik_per_observation": outer_mc_summary["loglik_per_observation"],
            "negative_log_loss_per_observation": outer_mc_summary["negative_log_loss_per_observation"],
            "perplexity": outer_mc_summary["perplexity"],
        }
    )
    if final_hmm_record is not None and outer_hmm_summary is not None:
        comparison_rows.append(
            {
                "evaluation_partition": "2021-outer",
                "model_label": f"HMM K={selected_k}",
                "model_family": "Categorical HMM",
                "k": selected_k,
                "strictly_eligible": True,
                "selected_k": True,
                "conditional_total_loglik": outer_hmm_summary["total_loglik"],
                "n_sequences": outer_hmm_summary["n_sequences"],
                "n_scored_observations": outer_hmm_summary["n_scored_observations"],
                "conditional_loglik_per_observation": outer_hmm_summary["loglik_per_observation"],
                "negative_log_loss_per_observation": outer_hmm_summary["negative_log_loss_per_observation"],
                "perplexity": outer_hmm_summary["perplexity"],
            }
        )

    comparison_table = pd.DataFrame(comparison_rows)
    comparison_path = TABLES_DIR / "q1_full_model_comparison.csv"
    comparison_table.to_csv(comparison_path, index=False, encoding="utf-8-sig")

    grid_summary = grid.summary_table.copy()
    grid_summary["selected_start_bic"] = np.nan
    for row_index, row in grid_summary.iterrows():
        if np.isfinite(row["selected_train_loglik_per_observation"]):
            train_ll = (
                float(row["selected_train_loglik_per_observation"])
                * inner_fit_bundle.n_observations
            )
            n_parameters = categorical_hmm_parameter_count(
                int(row["k"]), inner_fit_bundle.n_symbols
            )
            grid_summary.loc[row_index, "selected_start_bic"] = (
                -2.0 * train_ll
                + n_parameters * math.log(inner_fit_bundle.n_observations)
            )
    inner_lookup = comparison_table.loc[comparison_table["evaluation_partition"].eq("1920-inner") & comparison_table["model_family"].eq("Categorical HMM"), ["k", "conditional_loglik_per_observation", "selected_k"]]
    grid_summary = grid_summary.merge(inner_lookup, on="k", how="left")
    grid_summary_path = TABLES_DIR / "q1_full_k_selection.csv"
    grid_summary.to_csv(grid_summary_path, index=False, encoding="utf-8-sig")

    starts_frames = [grid.starts_table.assign(fit_stage="inner_k_grid")]
    if final_fit is not None:
        starts_frames.append(final_fit.starts_table.assign(fit_stage="final_full_development_refit"))
    starts_table = pd.concat(starts_frames, ignore_index=True, sort=False)
    starts_path = TABLES_DIR / "q1_full_hmm_random_starts.csv"
    starts_table.to_csv(starts_path, index=False, encoding="utf-8-sig")

    distributions = pd.concat(
        [
            observation_distribution(inner_fit_bundle, "1819-grid-fit"),
            observation_distribution(inner_validation_bundle, "1920-inner"),
            observation_distribution(development_bundle, "1819+1920-final-development"),
            observation_distribution(outer_bundle, "2021-outer"),
        ],
        ignore_index=True,
    )
    distribution_path = TABLES_DIR / "q1_full_observation_distribution.csv"
    distributions.to_csv(distribution_path, index=False, encoding="utf-8-sig")

    bootstrap_payload: dict[str, Any] = {
        "status": "not_computed_hmm_failed_gate",
        "practical_delta_threshold_nat_per_observation": args.practical_delta_threshold,
    }
    decision = "MC retained because no final HMM passed the strict gate"
    outer_scores_path = TABLES_DIR / "q1_full_outer_sequence_scores.csv"
    if outer_hmm_scores is not None:
        merged = outer_mc_scores.merge(
            outer_hmm_scores,
            on=["registration_id", "scored_observations"],
            suffixes=("_mc", "_hmm"),
            validate="one_to_one",
        )
        bootstrap_payload = {
            "status": "computed",
            **cluster_bootstrap_delta(
                merged,
                replications=args.bootstrap_replications,
                seed=args.seed + 200_000,
            ),
            "practical_delta_threshold_nat_per_observation": args.practical_delta_threshold,
        }
        delta = float(bootstrap_payload["point_delta_loglik_per_observation"])
        lower = float(bootstrap_payload["ci_2.5pct"])
        upper = float(bootstrap_payload["ci_97.5pct"])
        if delta >= args.practical_delta_threshold and lower > 0:
            decision = "HMM wins the preregistered outer predictive comparison"
        elif delta <= -args.practical_delta_threshold and upper < 0:
            decision = "Observable MC wins the preregistered outer predictive comparison"
        else:
            decision = "No practically and statistically decisive outer winner; retain MC for parsimony"
        id_to_course = outer_frame[["registration_id", "cohort", "COURSE_ID"]].drop_duplicates().set_index("registration_id")
        merged["cohort"] = merged["registration_id"].map(id_to_course["cohort"])
        merged["COURSE_ID"] = merged["registration_id"].map(id_to_course["COURSE_ID"])
        merged["sequence_hash"] = merged["registration_id"].map(anonymised_sequence_id)
        merged["delta_loglik_hmm_minus_mc"] = merged["conditional_loglik_hmm"] - merged["conditional_loglik_mc"]
        merged.drop(columns=["registration_id"]).to_csv(outer_scores_path, index=False, encoding="utf-8-sig")
    else:
        pd.DataFrame(
            columns=[
                "sequence_hash",
                "cohort",
                "COURSE_ID",
                "scored_observations",
                "conditional_loglik_mc",
                "conditional_loglik_hmm",
                "delta_loglik_hmm_minus_mc",
            ]
        ).to_csv(outer_scores_path, index=False, encoding="utf-8-sig")
    append_progress("outer_test", "completed", decision=decision, bootstrap=bootstrap_payload)

    state_model = final_hmm_record["model"] if final_hmm_record is not None else None
    state_labels = [f"State {index + 1}" for index in range(selected_k or 0)]
    if state_model is not None:
        emission_table = pd.DataFrame(state_model.emissionprob_, columns=outer_bundle.observation_labels)
        emission_table.insert(0, "state", state_labels)
        emission_table.insert(0, "state_index", np.arange(state_model.n_components, dtype=int))
        occupancy_rows: list[dict[str, Any]] = []
        for partition, bundle, start_position in (
            ("1819+1920-development", development_bundle, 0),
            ("2021-outer-full", outer_bundle, 0),
            ("2021-outer-post-landmark", outer_bundle, args.landmark_days),
        ):
            values = posterior_occupancy(state_model, bundle, start_position=start_position)
            for index, value in enumerate(values):
                occupancy_rows.append(
                    {
                        "partition": partition,
                        "state_index": index,
                        "state": f"State {index + 1}",
                        "posterior_occupancy": float(value),
                        "posterior_type": "retrospective smoothed; descriptive, not a forecast target",
                    }
                )
        occupancy_table = pd.DataFrame(occupancy_rows)
        transition_table = long_matrix(state_model.transmat_, state_labels, "selected_hmm_transition")
    else:
        emission_table = pd.DataFrame(columns=["state_index", "state", *outer_bundle.observation_labels])
        occupancy_table = pd.DataFrame(columns=["partition", "state_index", "state", "posterior_occupancy", "posterior_type"])
        transition_table = pd.DataFrame(columns=["matrix", "from_index", "from_label", "to_index", "to_label", "probability"])
    emission_path = TABLES_DIR / "q1_full_state_emissions.csv"
    occupancy_path = TABLES_DIR / "q1_full_state_occupancy.csv"
    transition_path = TABLES_DIR / "q1_full_state_transitions.csv"
    emission_table.to_csv(emission_path, index=False, encoding="utf-8-sig")
    occupancy_table.to_csv(occupancy_path, index=False, encoding="utf-8-sig")
    transition_table.to_csv(transition_path, index=False, encoding="utf-8-sig")

    dwell_payload: dict[str, Any] = {
        "status": "not_computed_hmm_failed_gate",
        "not_an_hsmm_fit": True,
    }
    dwell_table = pd.DataFrame()
    dwell_sampling_audit = pd.DataFrame()
    if state_model is not None:
        dwell_frame, dwell_sampling_audit = deterministic_stratified_cap(
            outer_frame,
            cap_per_course=args.dwell_sample_per_stratum,
            seed=args.seed + 300_000,
        )
        dwell_bundle = make_bundle(dwell_frame, final_encoder, "KU-2021-PPC-audit-sample")
        append_progress("duration_ppc", "started", sequences=dwell_bundle.n_sequences, replications=args.dwell_ppc_replications)
        diagnostic = dwell_time_posterior_predictive_check(
            state_model,
            dwell_bundle,
            n_replications=args.dwell_ppc_replications,
            random_state=args.seed + 400_000,
            minimum_uncensored_runs=20,
        )
        dwell_table = diagnostic.summary
        dwell_payload = {
            "status": "computed_hmm_diagnostic",
            "n_replications": diagnostic.n_replications,
            "sample_sequences": dwell_bundle.n_sequences,
            "sample_observations": dwell_bundle.n_observations,
            "diagnostic_counts": dwell_table["diagnostic"].value_counts().to_dict(),
            "caveats": list(diagnostic.caveats),
            "not_an_hsmm_fit": True,
        }
        append_progress("duration_ppc", "completed", diagnostics=dwell_payload["diagnostic_counts"])
    dwell_path = TABLES_DIR / "q1_full_dwell_ppc.csv"
    dwell_table.to_csv(dwell_path, index=False, encoding="utf-8-sig")
    dwell_sampling_path = TABLES_DIR / "q1_full_dwell_sampling_audit.csv"
    dwell_sampling_audit.to_csv(dwell_sampling_path, index=False, encoding="utf-8-sig")

    parameters_path = RESULTS_DIR / "q1_full_model_parameters.json"
    parameters_payload = {
        "schema_version": SCHEMA_VERSION,
        "generated_by": str(SCRIPT_PATH.relative_to(PROJECT_ROOT)).replace("\\", "/"),
        "generated_at": utc_now(),
        "observable_markov_chain_final": {
            "alpha": final_mc.alpha,
            "observation_labels": list(final_mc.observation_labels),
            "start_probability": final_mc.startprob,
            "transition_probability": final_mc.transmat,
        },
        "categorical_hmm_final": (
            {
                "status": "strictly_converged",
                "k": selected_k,
                "selected_start_seed": final_hmm_record["seed"],
                "iterations": final_hmm_record["iterations"],
                "observation_labels": list(outer_bundle.observation_labels),
                "state_labels": state_labels,
                "start_probability": state_model.startprob_,
                "transition_probability": state_model.transmat_,
                "emission_probability": state_model.emissionprob_,
                "state_semantics": "numbered platform-evidence states only",
            }
            if state_model is not None
            else {"status": "not_available_under_strict_gate", "k_selected_inner": selected_k}
        ),
        "hsmm": hsmm_status(),
    }
    write_json(parameters_path, parameters_payload)

    figure_comparison = FIGURES_DIR / "fig_q1_model_comparison.png"
    figure_states = FIGURES_DIR / "fig_q1_state_profiles.png"
    figure_dwell = FIGURES_DIR / "fig_q1_dwell_ppc.png"
    save_model_comparison_figure(comparison_table, selected_k, figure_comparison)
    save_state_figure(state_model, outer_bundle.observation_labels, occupancy_table, figure_states)
    save_dwell_figure(dwell_table, figure_dwell)

    outer_nll_reduction = None
    if outer_hmm_summary is not None:
        outer_nll_reduction = (
            outer_mc_summary["negative_log_loss_per_observation"]
            - outer_hmm_summary["negative_log_loss_per_observation"]
        ) / outer_mc_summary["negative_log_loss_per_observation"]

    fit_stage_results = [grid] + ([final_fit] if final_fit is not None else [])
    checkpoint_stage_audits = [
        stage.checkpoint_audit for stage in fit_stage_results
    ]
    checkpoint_totals = {
        key: int(sum(stage.checkpoint_audit[key] for stage in fit_stage_results))
        for key in (
            "requested_fits",
            "reused_successful_fits",
            "newly_computed_fits",
            "newly_computed_successful_fits",
            "newly_computed_failed_fits",
            "missing_checkpoints",
            "rejected_checkpoints",
        )
    }
    reused_checkpoint_paths = sorted(
        {
            path.resolve()
            for stage in fit_stage_results
            for path in stage.checkpoint_input_paths
        },
        key=lambda path: str(path),
    )

    summary_payload = {
        "schema_version": SCHEMA_VERSION,
        "generated_by": str(SCRIPT_PATH.relative_to(PROJECT_ROOT)).replace("\\", "/"),
        "generated_at": utc_now(),
        "question_id": "q1",
        "status": "computed_nested_outer_test" if state_model is not None else "computed_mc_hmm_blocked_by_gate",
        "source": {
            "dataset": "KU Leuven student-course-natural-day panel",
            "rows": int(len(frame)),
            "sequences": int(frame["registration_id"].nunique()),
            "cohort_sequence_counts": cohort_counts,
            "outcome_columns_loaded_or_used": [],
        },
        "design": {
            "inner_training_cohort": "1819",
            "inner_validation_cohort": "1920",
            "final_development_cohorts": ["1819", "1920"],
            "untouched_outer_cohort": "2021",
            "fixed_landmark_days": args.landmark_days,
            "forecast_target": "joint conditional likelihood of every post-landmark daily observation",
            "no_final_length_used_for_landmark": True,
            "sequence_boundary_rule": "no transition crosses a registration boundary",
            "capacity_decision": capacity_payload,
        },
        "encoder": {
            "inner_training_only": inner_encoder.metadata(),
            "final_development_only": final_encoder.metadata(),
        },
        "partitions": {
            "inner_grid_fit": bundle_summary(inner_fit_bundle),
            "inner_validation": bundle_summary(inner_validation_bundle),
            "final_development": bundle_summary(development_bundle),
            "untouched_outer": bundle_summary(outer_bundle),
        },
        "selection": {
            "k_values": args.k_values,
            "random_starts_per_k": args.n_starts,
            "maximum_iterations": args.n_iter,
            "absolute_loglik_tolerance": args.tol,
            "strict_convergence_rule": "hmmlearn_converged and iterations < maximum_iterations",
            "minimum_state_occupancy": args.min_state_occupancy,
            "within_k_start_rule": "highest training log likelihood among strictly converged occupancy-passing starts",
            "between_k_rule": "highest 1920 fixed-landmark conditional log likelihood among strictly eligible K",
            "status": inner_selection_status,
            "selected_k": selected_k,
            "pilot_candidate_including_provisional": pilot.get("model_selection", {}).get("best_metric_candidate_including_provisional_k"),
        },
        "outer_test": {
            "mc": outer_mc_summary,
            "hmm": outer_hmm_summary,
            "bootstrap_cluster_unit": "student-course registration",
            "bootstrap": bootstrap_payload,
            "relative_negative_log_loss_reduction_hmm_vs_mc": outer_nll_reduction,
            "decision": decision,
        },
        "checkpoint_resume": {
            "enabled": True,
            "invocation_id": _ACTIVE_INVOCATION_ID,
            "run_contract_sha256": checkpoint_run_contract["contract_sha256"],
            "stages": checkpoint_stage_audits,
            "totals": checkpoint_totals,
            "failed_fit_reuse_policy": "audit metadata retained; failed fits are always recomputed",
            "model_storage": "NumPy NPZ parameters with allow_pickle=False",
        },
        "duration_ppc": dwell_payload,
        "claim_boundary": [
            "States are numbered platform-evidence patterns, not validated cognitive states.",
            "Platform-Inactive does not imply no learning.",
            "The outer comparison is predictive/descriptive, not causal.",
            "Historical behavior profiles are not interventions in Q1.",
            "The HMM duration PPC is not an HSMM fit.",
        ],
        "hsmm": hsmm_status(),
    }
    summary_path = RESULTS_DIR / "q1_full_run_summary.json"
    write_json(summary_path, summary_payload)

    if use_capacity_cap:
        capacity_report_text = (
            f"全量 1819 数据上的只读基准计时为 K=5、单初值、100 次迭代 "
            f"{args.benchmark_seconds:.2f} 秒。按 K²、{args.n_starts} 个初值和 "
            f"{args.n_iter} 次迭代外推，在 {args.workers} 个进程下完整 K 网格约需 "
            f"{projected_grid_seconds/60:.1f} 分钟，超过预设 "
            f"{args.wall_clock_budget_minutes:.0f} 分钟预算。因此 K 网格按 SHA-256 "
            f"固定排序，在 1819 每个课程层抽取 {args.selection_cap_per_course} 条、共 "
            f"{inner_fit_bundle.n_sequences:,} 条序列；抽样不读取行为结果或结局。"
            f"状态数确定后，最终模型仍在全部开发序列上用 {args.n_starts} 个初值重估。"
        )
    else:
        capacity_report_text = (
            f"全量 1819 数据上的只读基准计时为 K=5、单初值、100 次迭代 "
            f"{args.benchmark_seconds:.2f} 秒。按 K²、{args.n_starts} 个初值和 "
            f"{args.n_iter} 次迭代外推，在 {args.workers} 个进程下完整 K 网格约需 "
            f"{projected_grid_seconds/60:.1f} 分钟，低于预设 "
            f"{args.wall_clock_budget_minutes:.0f} 分钟预算。因此 K 网格使用 1819 "
            f"全部 {inner_fit_bundle.n_sequences:,} 条序列；状态数确定后，最终模型在"
            f"全部开发序列上用 {args.n_starts} 个初值重估。"
        )

    # Independent-in-script contract checks.  Warnings keep scientific limits
    # visible but do not rewrite failed gates as success.
    checks: list[dict[str, Any]] = []

    def check(name: str, passed: bool, detail: str) -> None:
        checks.append({"check": name, "passed": bool(passed), "detail": detail})

    check("all_4280_sequences_accounted_for", development_bundle.n_sequences + outer_bundle.n_sequences == 4280, f"development={development_bundle.n_sequences}, outer={outer_bundle.n_sequences}")
    check("outer_year_not_in_development", set(development_frame["cohort"]) == {"1819", "1920"} and set(outer_frame["cohort"]) == {"2021"}, "disjoint cohort split")
    check("fixed_landmark_not_final_length_fraction", args.landmark_days == 42, f"landmark_days={args.landmark_days}")
    check("outcomes_not_loaded", not any(column in frame.columns for column in ["PASSED", "PASSED_FIRST_ATTEMPT", "SCORE_CATEGORY_FINAL"]), "model frame contains observation fields only")
    check("complete_k_grid", sorted(grid.summary_table["k"].astype(int).tolist()) == [2, 3, 4, 5, 6], "K=2..6")
    check("minimum_ten_starts_each_k", bool(grid.starts_table.groupby("k").size().ge(10).all()), str(grid.starts_table.groupby("k").size().to_dict()))
    check("strict_k_selection", selected_k is None or bool(grid.selected_by_k[selected_k]["feasible"]), f"selected_k={selected_k}")
    check("final_hmm_strict_or_blocked", state_model is None or bool(final_hmm_record and final_hmm_record["feasible"]), "provisional final fits are never reported as eligible")
    check("mc_always_outer_scored", outer_mc_summary["n_sequences"] == outer_bundle.n_sequences, f"outer MC sequences={outer_mc_summary['n_sequences']}")
    check("probability_constraints_mc", np.allclose(final_mc.startprob.sum(), 1) and np.allclose(final_mc.transmat.sum(axis=1), 1), "MC rows sum to one")
    check("probability_constraints_hmm", state_model is None or (np.allclose(state_model.startprob_.sum(), 1) and np.allclose(state_model.transmat_.sum(axis=1), 1) and np.allclose(state_model.emissionprob_.sum(axis=1), 1)), "HMM rows sum to one or HMM blocked")
    check("hsmm_not_implemented", not hsmm_status()["implemented"], "duration PPC is explicitly diagnostic only")
    check(
        "checkpoint_accounting_complete",
        checkpoint_totals["requested_fits"]
        == checkpoint_totals["reused_successful_fits"]
        + checkpoint_totals["newly_computed_fits"],
        str(checkpoint_totals),
    )
    check(
        "checkpoint_source_contract_unchanged",
        checkpoint_run_contract["source_code"] == artifact_record(SCRIPT_PATH)
        and checkpoint_run_contract["dependencies"]
        == [artifact_record(SCRIPT_PATH.with_name("sequence_models.py"))],
        f"contract={checkpoint_run_contract['contract_sha256']}",
    )
    check(
        "checkpoint_failed_fits_not_reused",
        all(
            not bool(row.checkpoint_reused)
            for row in starts_table.itertuples()
            if row.status != "fit"
        ),
        "failed checkpoints are audit-only and always recomputed",
    )
    check(
        "capacity_report_matches_decision",
        (
            ("每个课程层抽取" in capacity_report_text and "超过预设" in capacity_report_text)
            if use_capacity_cap
            else ("每个课程层抽取" not in capacity_report_text and "超过预设" not in capacity_report_text)
        ),
        f"decision={capacity_payload['decision']}",
    )
    warnings_list: list[str] = []
    if use_capacity_cap:
        warnings_list.append("K-grid used an outcome-blind capacity-adaptive sample; final selected-K refit and outer test used all eligible sequences.")
    if selected_k is None:
        warnings_list.append("No HMM K passed the strict inner convergence/occupancy gate; only MC is publication-eligible.")
    if selected_k is not None and state_model is None:
        warnings_list.append("An HMM K passed inner selection, but no final full-development start passed the strict gate.")
    if not dwell_table.empty and dwell_table["diagnostic"].ne("no_strong_misfit_signal").any():
        warnings_list.append("At least one numbered state has an HMM duration-PPC warning or insufficient uncensored spells.")
    audit_status = "PASS" if all(item["passed"] for item in checks) and not warnings_list else "PASS_WITH_WARNINGS" if all(item["passed"] for item in checks) else "FAIL"
    audit_path = RESULTS_DIR / "q1_full_self_audit.json"
    write_json(
        audit_path,
        {
            "schema_version": SCHEMA_VERSION,
            "generated_by": str(SCRIPT_PATH.relative_to(PROJECT_ROOT)).replace("\\", "/"),
            "generated_at": utc_now(),
            "status": audit_status,
            "checks": checks,
            "warnings": warnings_list,
        },
    )

    report_lines = [
        "# Q1 投稿级嵌套验证实证说明",
        "",
        "## 设计与数据使用",
        "",
        f"本次分析使用 KU Leuven 日级面板的全部 {len(frame):,} 个学生—课程日和 {frame['registration_id'].nunique():,} 条注册序列。1819 学年用于内层估计，1920 学年用于选择状态数，1819+1920 的全部 {development_bundle.n_sequences:,} 条序列随后用于最终重估；2021 学年的 {outer_bundle.n_sequences:,} 条序列在此之前保持未触碰，作为外层测试。",
        "",
        f"预测评价采用固定第 {args.landmark_days} 日地标：模型先读取每条新序列前 {args.landmark_days} 日，再计算余下日序列的条件联合对数似然。地标不依赖课程最终长度，不跨学生—课程边界。模型输入不含通过、成绩或其他结局字段。",
        "",
        "## 容量自适应与严格门禁",
        "",
        capacity_report_text,
        "",
        f"活动多样性阈值在内层训练集为 {inner_encoder.metadata()['diversity_threshold']:.0f}，最终重估时仅用 1819+1920 数据得到 {final_encoder.metadata()['diversity_threshold']:.0f}。K=2—6 均运行 {args.n_starts} 个随机初值；只有在 {args.n_iter} 次上限前收敛且所有状态后验占比均不低于 {args.min_state_occupancy:.1%} 的解，才可参与选模。",
        "",
        "## 内层选择与外层结果",
        "",
    ]
    if selected_k is None:
        report_lines.append("没有 HMM 状态数通过严格内层门禁。因此本轮不报告任何 HMM 外层优势，MC 是唯一可用于投稿主结果的 Q1 模型。")
    else:
        selected_inner_row = comparison_table.loc[comparison_table["evaluation_partition"].eq("1920-inner") & comparison_table["k"].eq(selected_k)].iloc[0]
        report_lines.append(
            f"严格内层选择得到 K={selected_k}：1920 后地标条件对数似然/观测为 {selected_inner_row['conditional_loglik_per_observation']:.6f}；同口径 MC 为 {inner_mc_summary['loglik_per_observation']:.6f}。K 的选择只使用 1920，没有查看 2021 外层结果。"
        )
        if state_model is None:
            report_lines.append("但 K 在全部 1819+1920 数据上的十初值重估未通过严格收敛/占比门禁，所以 HMM 外层指标被阻断，不能用 provisional 参数替代。")
        else:
            delta = float(bootstrap_payload["point_delta_loglik_per_observation"])
            report_lines.append(
                f"在未触碰的 2021 外层中，MC 条件对数似然/观测为 {outer_mc_summary['loglik_per_observation']:.6f}，HMM 为 {outer_hmm_summary['loglik_per_observation']:.6f}，HMM−MC 差值为 {delta:.6f} nat/观测；按学生—课程序列聚类的 {args.bootstrap_replications:,} 次 bootstrap 95% 区间为 [{bootstrap_payload['ci_2.5pct']:.6f}, {bootstrap_payload['ci_97.5pct']:.6f}]。HMM 相对负对数损失变化为 {outer_nll_reduction:.3%}。预注册判定：{decision}。"
            )
    report_lines.extend(
        [
            "",
            "## 状态稳定性与持续时间诊断",
            "",
        ]
    )
    if state_model is None:
        report_lines.append("由于最终 HMM 未通过门禁，不解释状态占比或持续时间。")
    else:
        final_start_block = starts_table.loc[starts_table["fit_stage"].eq("final_full_development_refit")]
        strict_count = int(final_start_block["feasible"].sum())
        median_js = float(final_start_block.loc[final_start_block["feasible"], "mean_emission_js_to_selected"].median()) if strict_count else float("nan")
        report_lines.append(
            f"最终 K={selected_k} 重估的 {args.n_starts} 个初值中，{strict_count} 个同时通过严格收敛和占比门禁；与入选解对齐后的发射分布平均 Jensen–Shannon 距离中位数为 {median_js:.6f}。状态仅写作 State 1—State {selected_k}，其含义限于平台行为证据结构。"
        )
        diagnostics = ", ".join(f"{row.state}: {row.diagnostic}" for row in dwell_table.itertuples())
        report_lines.append(
            f"持续时间 PPC 使用 2021 各课程层的固定哈希样本与 {args.dwell_ppc_replications} 次重复，诊断为：{diagnostics}。该检验只检查 HMM 的几何持续时间假设；它不是 HSMM，也没有把平台无活动解释为没有学习。"
        )
    report_lines.extend(
        [
            "",
            "## 可用于论文的结论边界",
            "",
            "本结果只支持比较平台行为序列的条件预测与识别编号状态。它不证明状态对应‘专注’或‘深度思考’，不把 Platform-Inactive 当成未学习，也不把预测关联解释为策略因果效应。HSMM 仍标记为 `needs_real_modeling`。",
            "",
            "## 主要证据文件",
            "",
            "- `q1_full_run_summary.json`：嵌套设计、选模规则、外层结果与结论边界。",
            "- `q1_full_execution_manifest.json`：代码、输入、表格、图形和结果的 SHA-256。",
            "- `../../tables/q1_full/q1_full_model_comparison.csv`：内外层 MC/HMM 比较。",
            "- `../../tables/q1_full/q1_full_hmm_random_starts.csv`：所有初值、收敛和稳定性记录。",
            "- `../../figures/fig_q1_model_comparison.png`、`fig_q1_state_profiles.png`、`fig_q1_dwell_ppc.png`：投稿图。",
            "",
        ]
    )
    report_path = RESULTS_DIR / "Q1投稿级实证说明.md"
    report_path.write_text("\n".join(report_lines), encoding="utf-8")

    append_progress(
        "run",
        "completed",
        audit_status=audit_status,
        decision=decision,
        checkpoint_totals=checkpoint_totals,
    )
    finished_at = utc_now()
    elapsed = time.perf_counter() - start_clock
    manifest_path = RESULTS_DIR / "q1_full_execution_manifest.json"
    output_candidates = sorted(
        [path for path in RESULTS_DIR.rglob("*") if path.is_file() and path != manifest_path]
        + [path for path in TABLES_DIR.rglob("*") if path.is_file()]
        + [figure_comparison, figure_states, figure_dwell],
        key=lambda path: str(path),
    )
    manifest_input_paths = [PANEL_PATH]
    if PILOT_SUMMARY_PATH.exists():
        manifest_input_paths.append(PILOT_SUMMARY_PATH)
    manifest_input_paths.extend(reused_checkpoint_paths)
    manifest_input_paths = sorted(
        {path.resolve() for path in manifest_input_paths},
        key=lambda path: str(path),
    )
    write_json(
        manifest_path,
        {
            "schema_version": SCHEMA_VERSION,
            "generated_by": str(SCRIPT_PATH.relative_to(PROJECT_ROOT)).replace("\\", "/"),
            "question_id": "q1",
            "status": "success" if audit_status != "FAIL" else "failed_audit",
            "started_at": started_at,
            "finished_at": finished_at,
            "elapsed_seconds": elapsed,
            "run_command": command,
            "working_directory": str(Path.cwd()),
            "random_seed": args.seed,
            "arguments": vars(args),
            "python": {
                "implementation": platform.python_implementation(),
                "version": platform.python_version(),
                "platform": platform.platform(),
                "cpu_count": os.cpu_count(),
            },
            "package_versions": package_versions(),
            "source_code": artifact_record(SCRIPT_PATH),
            "dependencies": [artifact_record(SCRIPT_PATH.with_name("sequence_models.py"))],
            "inputs": [artifact_record(path) for path in manifest_input_paths],
            "outputs": [artifact_record(path) for path in output_candidates],
            "checkpoint_resume": {
                "enabled": True,
                "invocation_id": _ACTIVE_INVOCATION_ID,
                "run_contract": checkpoint_run_contract,
                "stages": checkpoint_stage_audits,
                "totals": checkpoint_totals,
                "reused_checkpoint_inputs": [
                    artifact_record(path) for path in reused_checkpoint_paths
                ],
                "failed_fit_reuse_policy": "never reusable; recompute on every resumed invocation",
                "corrupt_or_mismatched_policy": "reject and recompute; never downgrade validation",
                "serialization_security": "fitted probabilities stored as NumPy NPZ and loaded with allow_pickle=False",
            },
            "rerun_rule": (
                "Exact-contract checkpoint reuse is permitted only when parsed "
                "arguments and seed, source/dependency/input SHA-256 values, "
                "runtime versions, stage bundle fingerprint, task schedule, "
                "checkpoint hashes, probability constraints, and recomputed "
                "training log likelihood all match. Any mismatch is rejected "
                "and that fit is recomputed."
            ),
        },
    )
    print(f"[Q1 FULL] selected_k={selected_k}; decision={decision}")
    print(f"audit={audit_status}; elapsed_seconds={elapsed:.3f}")
    print(f"summary={summary_path.relative_to(PROJECT_ROOT)}")
    return 0 if audit_status != "FAIL" else 3


if __name__ == "__main__":
    raise SystemExit(main())
