"""Leakage-resistant Markov-chain and categorical-HMM sequence engine.

This module is deliberately independent of the question-specific ``q1``--``q4``
scripts.  It implements the empirical core needed by the learning-state paper:

* a transparent, training-only observation encoder;
* a Dirichlet-smoothed Markov chain over observable behavior categories;
* multi-start categorical HMM selection for K=2,...,6;
* strict student-course sequence boundaries and explicit missing/gap policies;
* temporal and leave-course-out split helpers;
* stable emission-profile ordering and Hungarian state alignment; and
* a Viterbi-duration posterior-predictive diagnostic of the HMM geometric
  duration assumption.

The hidden states are always called ``State 1``, ..., ``State K`` here.  A state
must not be named "focused", "deep learning", or similar until external validity
evidence supports that interpretation.  Likewise, ``Platform-Inactive`` means
only that the platform recorded no event in the analysis window.

There is intentionally no fake HSMM implementation.  ``fit_hsmm`` raises an
explicit ``HSMMNeedsRealModelingError`` until a genuine explicit-duration model,
with likelihood and validation code, is supplied.
"""

from __future__ import annotations

import copy
import math
import warnings
from dataclasses import dataclass, field
from functools import cached_property
from typing import Any, Iterable, Iterator, Literal, Mapping, Sequence

import numpy as np
import pandas as pd
from scipy.optimize import linear_sum_assignment


OBSERVATION_LABELS: tuple[str, ...] = (
    "Platform-Inactive",
    "Low-Diversity-Content-Consumption",
    "Diverse-Interaction",
    "Assessment-or-Production",
)
UNKNOWN_OBSERVATION_LABEL = "Unknown-or-Missing"

MissingPolicy = Literal["raise", "explicit_unknown", "split"]
GapPolicy = Literal["raise", "split", "ignore"]
DuplicateTimePolicy = Literal["raise", "split"]


def _as_columns(columns: str | Sequence[str]) -> tuple[str, ...]:
    if isinstance(columns, str):
        return (columns,)
    values = tuple(columns)
    if not values:
        raise ValueError("at least one grouping column is required")
    return values


def _require_columns(frame: pd.DataFrame, columns: Iterable[str], context: str) -> None:
    missing = [column for column in columns if column not in frame.columns]
    if missing:
        raise KeyError(f"{context}: missing required columns {missing}")


def _coerce_boolean(series: pd.Series) -> pd.Series:
    """Coerce common boolean encodings while preserving invalid values as NA."""

    if pd.api.types.is_bool_dtype(series.dtype):
        return series.astype("boolean")
    numeric = pd.to_numeric(series, errors="coerce")
    result = pd.Series(pd.NA, index=series.index, dtype="boolean")
    result.loc[numeric == 0] = False
    result.loc[numeric == 1] = True
    text = series.astype("string").str.strip().str.lower()
    result.loc[text.isin(["false", "f", "no", "n"])] = False
    result.loc[text.isin(["true", "t", "yes", "y"])] = True
    return result


@dataclass(frozen=True)
class ObservationEncodingConfig:
    """Column contract for the transparent observation categories.

    ``diversity_quantile`` is learned from active, non-productive rows in the
    training partition only.  Presence thresholds such as ``productive > 0``
    are definitional, not estimated cut-points.  Click volume is intentionally
    absent: high click counts do not automatically imply high-quality study.
    """

    inactive_col: str = "platform_inactive"
    diversity_col: str = "activity_type_diversity"
    productive_col: str = "productive_event_count"
    interaction_col: str | None = None
    diversity_quantile: float = 0.67
    missing_policy: MissingPolicy = "raise"

    def __post_init__(self) -> None:
        if not 0.0 < self.diversity_quantile < 1.0:
            raise ValueError("diversity_quantile must be strictly between 0 and 1")
        if self.missing_policy not in {"raise", "explicit_unknown", "split"}:
            raise ValueError(f"unsupported missing_policy={self.missing_policy!r}")


class TransparentObservationEncoder:
    """Fit/transform encoder with no threshold leakage from validation data."""

    def __init__(self, config: ObservationEncodingConfig | None = None) -> None:
        self.config = config or ObservationEncodingConfig()
        self.diversity_threshold_: float | None = None
        self.fit_metadata_: dict[str, Any] | None = None

    @property
    def fitted(self) -> bool:
        return self.diversity_threshold_ is not None

    @property
    def labels(self) -> tuple[str, ...]:
        if self.config.missing_policy == "explicit_unknown":
            return OBSERVATION_LABELS + (UNKNOWN_OBSERVATION_LABEL,)
        return OBSERVATION_LABELS

    @property
    def n_symbols(self) -> int:
        return len(self.labels)

    def _required_columns(self) -> tuple[str, ...]:
        values = [
            self.config.inactive_col,
            self.config.diversity_col,
            self.config.productive_col,
        ]
        if self.config.interaction_col is not None:
            values.append(self.config.interaction_col)
        return tuple(values)

    def fit(self, training_frame: pd.DataFrame) -> "TransparentObservationEncoder":
        """Learn the relative-diversity threshold from training rows only."""

        _require_columns(training_frame, self._required_columns(), "encoder.fit")
        inactive = _coerce_boolean(training_frame[self.config.inactive_col])
        productive = pd.to_numeric(
            training_frame[self.config.productive_col], errors="coerce"
        )
        diversity = pd.to_numeric(
            training_frame[self.config.diversity_col], errors="coerce"
        )
        candidates = (
            inactive.eq(False)
            & productive.notna()
            & productive.le(0)
            & diversity.notna()
        )
        candidate_values = diversity.loc[candidates].to_numpy(dtype=float)
        if candidate_values.size:
            threshold = float(
                np.quantile(candidate_values, self.config.diversity_quantile)
            )
            reason = "training_quantile"
        else:
            # No active non-productive training row means that the diverse versus
            # low-diversity distinction is not estimable.  Infinity transparently
            # disables that branch instead of borrowing a threshold from validation.
            threshold = float("inf")
            reason = "not_estimable_no_active_nonproductive_training_rows"
        self.diversity_threshold_ = threshold
        self.fit_metadata_ = {
            "training_rows": int(len(training_frame)),
            "threshold_source_rows": int(candidate_values.size),
            "diversity_quantile": float(self.config.diversity_quantile),
            # JSON-safe: infinity is retained internally but reported as null.
            "diversity_threshold": threshold if np.isfinite(threshold) else None,
            "threshold_reason": reason,
            "missing_policy": self.config.missing_policy,
            "click_volume_used": False,
        }
        return self

    def transform(self, frame: pd.DataFrame) -> np.ndarray:
        """Encode rows; ``-1`` denotes a split point under ``missing_policy='split'``."""

        if not self.fitted:
            raise RuntimeError("encoder must be fit on a training partition first")
        _require_columns(frame, self._required_columns(), "encoder.transform")

        inactive = _coerce_boolean(frame[self.config.inactive_col])
        productive = pd.to_numeric(frame[self.config.productive_col], errors="coerce")
        diversity = pd.to_numeric(frame[self.config.diversity_col], errors="coerce")
        if self.config.interaction_col is None:
            interaction = pd.Series(0.0, index=frame.index)
        else:
            interaction = pd.to_numeric(
                frame[self.config.interaction_col], errors="coerce"
            )

        codes = np.full(len(frame), -1, dtype=np.int64)
        is_inactive = inactive.eq(True).fillna(False).to_numpy(dtype=bool)
        codes[is_inactive] = 0

        active = inactive.eq(False).fillna(False)
        active_known = (
            active
            & productive.notna()
            & (
                productive.gt(0)
                | (diversity.notna() & interaction.notna())
            )
        )
        production = active_known & productive.gt(0)
        diverse = (
            active_known
            & ~production
            & (
                diversity.gt(float(self.diversity_threshold_))
                | interaction.gt(0)
            )
        )
        low_diversity = active_known & ~production & ~diverse
        codes[low_diversity.to_numpy(dtype=bool)] = 1
        codes[diverse.to_numpy(dtype=bool)] = 2
        codes[production.to_numpy(dtype=bool)] = 3

        missing_mask = codes < 0
        if missing_mask.any():
            missing_rows = frame.index[missing_mask].tolist()
            if self.config.missing_policy == "raise":
                preview = missing_rows[:10]
                raise ValueError(
                    "observation fields are missing/invalid for rows "
                    f"{preview}; set missing_policy explicitly to 'split' or "
                    "'explicit_unknown' if that is scientifically justified"
                )
            if self.config.missing_policy == "explicit_unknown":
                codes[missing_mask] = len(OBSERVATION_LABELS)
            # Under 'split', -1 is preserved and handled by make_sequence_bundle.
        return codes

    def fit_transform(self, training_frame: pd.DataFrame) -> np.ndarray:
        return self.fit(training_frame).transform(training_frame)

    def metadata(self) -> dict[str, Any]:
        if self.fit_metadata_ is None:
            raise RuntimeError("encoder has not been fit")
        return copy.deepcopy(self.fit_metadata_)


@dataclass(frozen=True)
class SequenceBundle:
    """A collection of independent sequences with auditable row provenance."""

    observations: tuple[np.ndarray, ...]
    group_keys: tuple[tuple[Any, ...], ...]
    segment_ids: tuple[int, ...]
    row_ids: tuple[np.ndarray, ...]
    times: tuple[np.ndarray, ...]
    group_names: tuple[str, ...]
    n_symbols: int
    observation_labels: tuple[str, ...]
    metadata: Mapping[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        n = len(self.observations)
        parallel_lengths = [
            len(self.group_keys),
            len(self.segment_ids),
            len(self.row_ids),
            len(self.times),
        ]
        if any(value != n for value in parallel_lengths):
            raise ValueError("all SequenceBundle tuple fields must have equal length")
        if self.n_symbols < 1:
            raise ValueError("n_symbols must be positive")
        if len(self.observation_labels) != self.n_symbols:
            raise ValueError("observation_labels length must equal n_symbols")
        for index, sequence in enumerate(self.observations):
            array = np.asarray(sequence)
            if array.ndim != 1 or array.size == 0:
                raise ValueError(f"sequence {index} must be a non-empty 1D array")
            if np.any(array < 0) or np.any(array >= self.n_symbols):
                raise ValueError(f"sequence {index} contains an invalid observation code")
            if len(self.row_ids[index]) != len(array) or len(self.times[index]) != len(array):
                raise ValueError(f"sequence {index} provenance length mismatch")

    @property
    def n_sequences(self) -> int:
        return len(self.observations)

    @property
    def n_observations(self) -> int:
        return int(sum(len(sequence) for sequence in self.observations))

    @cached_property
    def lengths(self) -> list[int]:
        return [int(len(sequence)) for sequence in self.observations]

    @cached_property
    def X(self) -> np.ndarray:
        if not self.observations:
            return np.empty((0, 1), dtype=np.int64)
        return np.concatenate(self.observations).astype(np.int64, copy=False).reshape(-1, 1)


def sequence_bundle_from_arrays(
    sequences: Sequence[Sequence[int] | np.ndarray],
    n_symbols: int,
    *,
    observation_labels: Sequence[str] | None = None,
    group_name: str = "sequence_id",
    metadata: Mapping[str, Any] | None = None,
) -> SequenceBundle:
    """Create a bundle for simulation/tests while retaining independent boundaries."""

    arrays = tuple(np.asarray(sequence, dtype=np.int64) for sequence in sequences)
    labels = tuple(observation_labels or tuple(f"Observation {i}" for i in range(n_symbols)))
    keys = tuple((index,) for index in range(len(arrays)))
    rows = tuple(np.arange(len(array), dtype=np.int64) for array in arrays)
    times = tuple(np.arange(len(array), dtype=np.int64) for array in arrays)
    return SequenceBundle(
        observations=arrays,
        group_keys=keys,
        segment_ids=tuple(0 for _ in arrays),
        row_ids=rows,
        times=times,
        group_names=(group_name,),
        n_symbols=n_symbols,
        observation_labels=labels,
        metadata=dict(metadata or {}),
    )


def _normalise_group_key(key: Any) -> tuple[Any, ...]:
    return key if isinstance(key, tuple) else (key,)


def _time_values(series: pd.Series) -> tuple[np.ndarray, Literal["datetime", "numeric"]]:
    if pd.api.types.is_datetime64_any_dtype(series.dtype):
        values = pd.to_datetime(series, errors="coerce").to_numpy()
        return values, "datetime"
    numeric = pd.to_numeric(series, errors="coerce")
    if numeric.notna().all():
        return numeric.to_numpy(dtype=float), "numeric"
    datetimes = pd.to_datetime(series, errors="coerce")
    if datetimes.notna().all():
        return datetimes.to_numpy(), "datetime"
    bad = series.index[numeric.isna() & datetimes.isna()].tolist()[:10]
    raise ValueError(f"time column contains unparseable values at rows {bad}")


def _gap_breaks(
    times: np.ndarray,
    time_kind: Literal["datetime", "numeric"],
    expected_step: str | float | int | pd.Timedelta | None,
) -> np.ndarray:
    breaks = np.zeros(len(times), dtype=bool)
    if len(times):
        breaks[0] = True
    if expected_step is None or len(times) < 2:
        return breaks
    if time_kind == "datetime":
        step = pd.Timedelta(expected_step)
        deltas = pd.Series(times).diff().to_numpy()
        breaks[1:] = deltas[1:] > step.to_timedelta64()
    else:
        if isinstance(expected_step, (str, pd.Timedelta)):
            # KU uses dates while OULAD commonly uses integer days relative to
            # course start.  For numeric daily data, a timedelta is interpreted
            # in day units; callers using another numeric unit should pass a number.
            step = float(pd.Timedelta(expected_step) / pd.Timedelta(days=1))
        else:
            step = float(expected_step)
        breaks[1:] = np.diff(times.astype(float)) > step
    return breaks


def make_sequence_bundle(
    frame: pd.DataFrame,
    encoder: TransparentObservationEncoder,
    *,
    group_cols: str | Sequence[str],
    time_col: str,
    min_sequence_length: int = 2,
    expected_step: str | float | int | pd.Timedelta | None = "1D",
    gap_policy: GapPolicy = "split",
    duplicate_time_policy: DuplicateTimePolicy = "raise",
    dataset_name: str = "",
) -> SequenceBundle:
    """Encode a daily table without ever joining transitions across groups/gaps.

    Rows must already represent one analysis window per student-course.  If a
    calendar day is absent rather than explicitly encoded as Platform-Inactive,
    the default policy cuts the sequence at that gap.  It never inserts a hidden
    value or bridges the gap.
    """

    groups = _as_columns(group_cols)
    _require_columns(frame, groups + (time_col,), "make_sequence_bundle")
    if min_sequence_length < 1:
        raise ValueError("min_sequence_length must be at least 1")
    if gap_policy not in {"raise", "split", "ignore"}:
        raise ValueError(f"unsupported gap_policy={gap_policy!r}")
    if duplicate_time_policy not in {"raise", "split"}:
        raise ValueError(f"unsupported duplicate_time_policy={duplicate_time_policy!r}")
    if frame.empty:
        raise ValueError("cannot build sequences from an empty frame")
    if frame.loc[:, list(groups)].isna().any().any():
        raise ValueError("group identifiers cannot be missing")

    codes = encoder.transform(frame)
    work = frame.loc[:, list(groups) + [time_col]].copy()
    work["__sm_code"] = codes
    work["__sm_row_position"] = np.arange(len(frame), dtype=np.int64)
    work["__sm_original_index"] = frame.index.to_numpy(dtype=object)

    observations: list[np.ndarray] = []
    group_keys: list[tuple[Any, ...]] = []
    segment_ids: list[int] = []
    row_ids: list[np.ndarray] = []
    sequence_times: list[np.ndarray] = []
    dropped_short_rows = 0
    split_at_missing = 0
    split_at_gap = 0
    split_at_duplicate = 0

    grouper: str | list[str] = groups[0] if len(groups) == 1 else list(groups)
    for raw_key, group in work.groupby(grouper, sort=True, dropna=False):
        key = _normalise_group_key(raw_key)
        group = group.sort_values(time_col, kind="mergesort")
        times, time_kind = _time_values(group[time_col])
        duplicated = pd.Series(times).duplicated(keep=False).to_numpy(dtype=bool)
        if duplicated.any() and duplicate_time_policy == "raise":
            duplicate_preview = group.loc[duplicated, time_col].head(10).tolist()
            raise ValueError(
                f"duplicate analysis windows for group {key}: {duplicate_preview}; "
                "aggregate upstream or choose duplicate_time_policy='split'"
            )

        gap_starts = _gap_breaks(times, time_kind, expected_step)
        if gap_policy == "raise" and gap_starts[1:].any():
            locations = np.flatnonzero(gap_starts[1:])[:10] + 1
            raise ValueError(f"time gaps for group {key} before positions {locations.tolist()}")
        if gap_policy == "ignore":
            gap_starts[:] = False
            if len(gap_starts):
                gap_starts[0] = True

        group_codes = group["__sm_code"].to_numpy(dtype=np.int64)
        starts = gap_starts.copy()
        if duplicate_time_policy == "split" and len(times) > 1:
            duplicate_starts = np.zeros(len(times), dtype=bool)
            duplicate_starts[1:] = times[1:] == times[:-1]
            starts |= duplicate_starts
            split_at_duplicate += int(duplicate_starts.sum())
        if gap_policy == "split":
            split_at_gap += int(gap_starts[1:].sum())

        # -1 is a deliberate break under encoder missing_policy='split'.
        missing = group_codes < 0
        split_at_missing += int(missing.sum())
        segment_number = -1
        current_positions: list[int] = []

        def flush_segment() -> None:
            nonlocal segment_number, dropped_short_rows
            if not current_positions:
                return
            segment_number += 1
            pos = np.asarray(current_positions, dtype=np.int64)
            if len(pos) < min_sequence_length:
                dropped_short_rows += int(len(pos))
                current_positions.clear()
                return
            observations.append(group_codes[pos].copy())
            group_keys.append(key)
            segment_ids.append(segment_number)
            row_ids.append(group["__sm_original_index"].to_numpy(dtype=object)[pos].copy())
            sequence_times.append(np.asarray(times)[pos].copy())
            current_positions.clear()

        for position in range(len(group)):
            if missing[position]:
                flush_segment()
                continue
            if starts[position] and current_positions:
                flush_segment()
            current_positions.append(position)
        flush_segment()

    if not observations:
        raise ValueError("no sequences remain after missing/gap handling and length filtering")

    metadata = {
        "dataset_name": dataset_name,
        "source_rows": int(len(frame)),
        "retained_rows": int(sum(len(value) for value in observations)),
        "n_sequences": int(len(observations)),
        "n_original_groups": int(work.groupby(grouper, dropna=False).ngroups),
        "dropped_short_rows": int(dropped_short_rows),
        "split_at_missing_rows": int(split_at_missing),
        "split_at_time_gaps": int(split_at_gap),
        "split_at_duplicate_windows": int(split_at_duplicate),
        "expected_step": str(expected_step),
        "gap_policy": gap_policy,
        "duplicate_time_policy": duplicate_time_policy,
        "encoder": encoder.metadata(),
        "sequence_boundary_rule": "transitions are counted only within each retained segment",
    }
    return SequenceBundle(
        observations=tuple(observations),
        group_keys=tuple(group_keys),
        segment_ids=tuple(segment_ids),
        row_ids=tuple(row_ids),
        times=tuple(sequence_times),
        group_names=groups,
        n_symbols=encoder.n_symbols,
        observation_labels=encoder.labels,
        metadata=metadata,
    )


@dataclass(frozen=True)
class PreparedSplit:
    split_name: str
    encoder: TransparentObservationEncoder
    train: SequenceBundle
    validation: SequenceBundle
    metadata: Mapping[str, Any]


def temporal_train_validation_split(
    frame: pd.DataFrame,
    *,
    group_cols: str | Sequence[str],
    time_col: str,
    encoder_config: ObservationEncodingConfig | None = None,
    train_fraction: float = 0.75,
    min_train_rows: int = 5,
    min_validation_rows: int = 2,
    min_sequence_length: int = 2,
    expected_step: str | float | int | pd.Timedelta | None = "1D",
    gap_policy: GapPolicy = "split",
    duplicate_time_policy: DuplicateTimePolicy = "raise",
    dataset_name: str = "",
) -> PreparedSplit:
    """Split each student-course chronologically before fitting the encoder."""

    if not 0.0 < train_fraction < 1.0:
        raise ValueError("train_fraction must be between 0 and 1")
    if min_train_rows < min_sequence_length or min_validation_rows < min_sequence_length:
        raise ValueError("minimum split sizes must be >= min_sequence_length")
    groups = _as_columns(group_cols)
    _require_columns(frame, groups + (time_col,), "temporal_train_validation_split")
    work = frame.loc[:, list(groups) + [time_col]].copy()
    work["__sm_position"] = np.arange(len(frame), dtype=np.int64)

    train_positions: list[int] = []
    validation_positions: list[int] = []
    dropped_groups = 0
    cutoffs: list[dict[str, Any]] = []
    grouper: str | list[str] = groups[0] if len(groups) == 1 else list(groups)
    for raw_key, group in work.groupby(grouper, sort=True, dropna=False):
        group = group.sort_values(time_col, kind="mergesort")
        n_rows = len(group)
        if n_rows < min_train_rows + min_validation_rows:
            dropped_groups += 1
            continue
        cut = int(math.floor(n_rows * train_fraction))
        cut = max(min_train_rows, min(cut, n_rows - min_validation_rows))
        train_positions.extend(group["__sm_position"].iloc[:cut].tolist())
        validation_positions.extend(group["__sm_position"].iloc[cut:].tolist())
        cutoffs.append(
            {
                "group_key": list(_normalise_group_key(raw_key)),
                "train_rows": int(cut),
                "validation_rows": int(n_rows - cut),
                "last_train_time": str(group[time_col].iloc[cut - 1]),
                "first_validation_time": str(group[time_col].iloc[cut]),
            }
        )
    if not train_positions or not validation_positions:
        raise ValueError("no groups satisfy temporal split size requirements")

    train_frame = frame.iloc[np.asarray(train_positions, dtype=np.int64)].copy()
    validation_frame = frame.iloc[np.asarray(validation_positions, dtype=np.int64)].copy()
    encoder = TransparentObservationEncoder(encoder_config).fit(train_frame)
    train_bundle = make_sequence_bundle(
        train_frame,
        encoder,
        group_cols=groups,
        time_col=time_col,
        min_sequence_length=min_sequence_length,
        expected_step=expected_step,
        gap_policy=gap_policy,
        duplicate_time_policy=duplicate_time_policy,
        dataset_name=f"{dataset_name}:temporal_train",
    )
    validation_bundle = make_sequence_bundle(
        validation_frame,
        encoder,
        group_cols=groups,
        time_col=time_col,
        min_sequence_length=min_sequence_length,
        expected_step=expected_step,
        gap_policy=gap_policy,
        duplicate_time_policy=duplicate_time_policy,
        dataset_name=f"{dataset_name}:later_validation",
    )
    return PreparedSplit(
        split_name="within_group_later_period",
        encoder=encoder,
        train=train_bundle,
        validation=validation_bundle,
        metadata={
            "train_fraction": float(train_fraction),
            "dropped_short_groups": int(dropped_groups),
            "eligible_groups": int(len(cutoffs)),
            "encoder_fit_partition": "train_only",
            "cutoffs": cutoffs,
        },
    )


def iter_leave_course_out_splits(
    frame: pd.DataFrame,
    *,
    course_cols: str | Sequence[str],
    group_cols: str | Sequence[str],
    time_col: str,
    encoder_config: ObservationEncodingConfig | None = None,
    held_out_courses: Iterable[Any | tuple[Any, ...]] | None = None,
    min_sequence_length: int = 2,
    expected_step: str | float | int | pd.Timedelta | None = "1D",
    gap_policy: GapPolicy = "split",
    duplicate_time_policy: DuplicateTimePolicy = "raise",
    dataset_name: str = "",
) -> Iterator[PreparedSplit]:
    """Yield leakage-free leave-course-out folds.

    The observation threshold is refit for every fold using training courses
    only.  This function prepares folds; fitting and aligning fold-specific HMMs
    remains an explicit caller action.
    """

    courses = _as_columns(course_cols)
    groups = _as_columns(group_cols)
    _require_columns(frame, courses + groups + (time_col,), "iter_leave_course_out_splits")
    if held_out_courses is None:
        distinct = frame.loc[:, list(courses)].drop_duplicates()
        requested = [tuple(row) for row in distinct.itertuples(index=False, name=None)]
    else:
        requested = [_normalise_group_key(value) for value in held_out_courses]

    for held_out in requested:
        if len(held_out) != len(courses):
            raise ValueError(
                f"held-out key {held_out} has {len(held_out)} values, expected {len(courses)}"
            )
        mask = np.ones(len(frame), dtype=bool)
        for column, value in zip(courses, held_out):
            mask &= frame[column].eq(value).to_numpy(dtype=bool)
        if not mask.any():
            raise ValueError(f"held-out course {held_out} is absent")
        if mask.all():
            raise ValueError("leave-course-out requires at least one training course")
        train_frame = frame.loc[~mask].copy()
        holdout_frame = frame.loc[mask].copy()
        encoder = TransparentObservationEncoder(encoder_config).fit(train_frame)
        train_bundle = make_sequence_bundle(
            train_frame,
            encoder,
            group_cols=groups,
            time_col=time_col,
            min_sequence_length=min_sequence_length,
            expected_step=expected_step,
            gap_policy=gap_policy,
            duplicate_time_policy=duplicate_time_policy,
            dataset_name=f"{dataset_name}:leave_course_out_train",
        )
        holdout_bundle = make_sequence_bundle(
            holdout_frame,
            encoder,
            group_cols=groups,
            time_col=time_col,
            min_sequence_length=min_sequence_length,
            expected_step=expected_step,
            gap_policy=gap_policy,
            duplicate_time_policy=duplicate_time_policy,
            dataset_name=f"{dataset_name}:held_out_course={held_out}",
        )
        yield PreparedSplit(
            split_name="leave_course_out",
            encoder=encoder,
            train=train_bundle,
            validation=holdout_bundle,
            metadata={
                "course_columns": list(courses),
                "held_out_course": list(held_out),
                "encoder_fit_partition": "training_courses_only",
            },
        )


@dataclass(frozen=True)
class LogLikelihoodScore:
    total_log_likelihood: float
    n_observations: int
    n_transitions: int | None = None

    @property
    def per_observation(self) -> float:
        return self.total_log_likelihood / self.n_observations

    @property
    def per_transition(self) -> float | None:
        if not self.n_transitions:
            return None
        return self.total_log_likelihood / self.n_transitions


@dataclass(frozen=True)
class DirichletMarkovChain:
    start_counts: np.ndarray
    transition_counts: np.ndarray
    startprob: np.ndarray
    transmat: np.ndarray
    alpha: float
    start_alpha: float
    observation_labels: tuple[str, ...]

    @property
    def n_symbols(self) -> int:
        return int(len(self.startprob))

    @property
    def n_parameters(self) -> int:
        m = self.n_symbols
        return int((m - 1) + m * (m - 1))

    def score(self, bundle: SequenceBundle) -> LogLikelihoodScore:
        _check_bundle_symbols(bundle, self.n_symbols, self.observation_labels)
        log_start = np.log(self.startprob)
        log_transition = np.log(self.transmat)
        total = 0.0
        transitions = 0
        for sequence in bundle.observations:
            total += float(log_start[int(sequence[0])])
            if len(sequence) > 1:
                total += float(log_transition[sequence[:-1], sequence[1:]].sum())
                transitions += len(sequence) - 1
        return LogLikelihoodScore(
            total_log_likelihood=total,
            n_observations=bundle.n_observations,
            n_transitions=int(transitions),
        )


def fit_dirichlet_markov_chain(
    bundle: SequenceBundle,
    *,
    alpha: float = 0.5,
    start_alpha: float | None = None,
) -> DirichletMarkovChain:
    """Fit a first-order observable MC, never crossing a sequence boundary."""

    if alpha <= 0:
        raise ValueError("Dirichlet alpha must be positive")
    if start_alpha is None:
        start_alpha = alpha
    if start_alpha <= 0:
        raise ValueError("start_alpha must be positive")
    m = bundle.n_symbols
    start_counts = np.zeros(m, dtype=np.int64)
    transition_counts = np.zeros((m, m), dtype=np.int64)
    for sequence in bundle.observations:
        start_counts[int(sequence[0])] += 1
        if len(sequence) > 1:
            np.add.at(transition_counts, (sequence[:-1], sequence[1:]), 1)
    startprob = (start_counts + start_alpha) / (
        start_counts.sum() + start_alpha * m
    )
    transmat = transition_counts.astype(float) + alpha
    transmat /= transmat.sum(axis=1, keepdims=True)
    return DirichletMarkovChain(
        start_counts=start_counts,
        transition_counts=transition_counts,
        startprob=startprob,
        transmat=transmat,
        alpha=float(alpha),
        start_alpha=float(start_alpha),
        observation_labels=bundle.observation_labels,
    )


def _check_bundle_symbols(
    bundle: SequenceBundle,
    n_symbols: int,
    labels: Sequence[str] | None = None,
) -> None:
    if bundle.n_symbols != n_symbols:
        raise ValueError(
            f"bundle has {bundle.n_symbols} symbols but model expects {n_symbols}"
        )
    if labels is not None and tuple(labels) != bundle.observation_labels:
        raise ValueError("observation label order differs between model and bundle")


def categorical_hmm_parameter_count(n_states: int, n_symbols: int) -> int:
    """Free parameters: start + transition rows + categorical emission rows."""

    return int((n_states - 1) + n_states * (n_states - 1) + n_states * (n_symbols - 1))


def _observation_scores(labels: Sequence[str]) -> np.ndarray:
    default = {
        "Platform-Inactive": 0.0,
        "Low-Diversity-Content-Consumption": 1.0,
        "Diverse-Interaction": 2.0,
        "Assessment-or-Production": 3.0,
        "Unknown-or-Missing": 0.5,
    }
    return np.asarray([default.get(label, float(index)) for index, label in enumerate(labels)])


def canonical_state_order(
    emissionprob: np.ndarray,
    observation_labels: Sequence[str],
) -> np.ndarray:
    """Return a deterministic low-to-high platform-evidence state order.

    This ordering is descriptive, not a claim about cognitive depth.  Ties are
    broken by production probability, diverse-interaction probability, inverse
    inactivity, then the original numerical state.
    """

    emissions = np.asarray(emissionprob, dtype=float)
    if emissions.ndim != 2 or emissions.shape[1] != len(observation_labels):
        raise ValueError("emission matrix and observation labels are incompatible")
    scores = emissions @ _observation_scores(observation_labels)
    label_to_index = {label: index for index, label in enumerate(observation_labels)}
    productive = emissions[:, label_to_index.get("Assessment-or-Production", 0)]
    diverse = emissions[:, label_to_index.get("Diverse-Interaction", 0)]
    inactive = emissions[:, label_to_index.get("Platform-Inactive", 0)]
    original = np.arange(emissions.shape[0])
    return np.lexsort((original, -inactive, productive, diverse, scores)).astype(np.int64)


def _reorder_hmm(model: Any, order: Sequence[int]) -> Any:
    order_array = np.asarray(order, dtype=np.int64)
    k = int(model.n_components)
    if sorted(order_array.tolist()) != list(range(k)):
        raise ValueError("state order must be a permutation of 0,...,K-1")
    reordered = copy.deepcopy(model)
    reordered.startprob_ = np.asarray(model.startprob_)[order_array].copy()
    reordered.transmat_ = np.asarray(model.transmat_)[np.ix_(order_array, order_array)].copy()
    reordered.emissionprob_ = np.asarray(model.emissionprob_)[order_array].copy()
    return reordered


def canonicalize_hmm(
    model: Any,
    observation_labels: Sequence[str],
) -> tuple[Any, np.ndarray, dict[int, int]]:
    order = canonical_state_order(model.emissionprob_, observation_labels)
    old_to_new = {int(old): int(new) for new, old in enumerate(order)}
    return _reorder_hmm(model, order), order, old_to_new


def _jensen_shannon_divergence(left: np.ndarray, right: np.ndarray) -> float:
    epsilon = np.finfo(float).tiny
    p = np.clip(np.asarray(left, dtype=float), epsilon, None)
    q = np.clip(np.asarray(right, dtype=float), epsilon, None)
    p /= p.sum()
    q /= q.sum()
    midpoint = 0.5 * (p + q)
    return float(
        0.5 * np.sum(p * np.log(p / midpoint))
        + 0.5 * np.sum(q * np.log(q / midpoint))
    )


def hungarian_state_alignment(
    reference_emissions: np.ndarray,
    candidate_emissions: np.ndarray,
) -> pd.DataFrame:
    """Match candidate states to reference states by emission-profile JS divergence."""

    reference = np.asarray(reference_emissions, dtype=float)
    candidate = np.asarray(candidate_emissions, dtype=float)
    if reference.ndim != 2 or candidate.ndim != 2:
        raise ValueError("emission matrices must be two-dimensional")
    if reference.shape[1] != candidate.shape[1]:
        raise ValueError("emission matrices must have the same symbol columns")
    cost = np.empty((reference.shape[0], candidate.shape[0]), dtype=float)
    for i in range(reference.shape[0]):
        for j in range(candidate.shape[0]):
            cost[i, j] = _jensen_shannon_divergence(reference[i], candidate[j])
    reference_index, candidate_index = linear_sum_assignment(cost)
    result = pd.DataFrame(
        {
            "reference_state_index": reference_index.astype(int),
            "candidate_state_index": candidate_index.astype(int),
            "reference_state": [f"State {value + 1}" for value in reference_index],
            "candidate_state": [f"State {value + 1}" for value in candidate_index],
            "js_divergence": cost[reference_index, candidate_index],
        }
    )
    return result.sort_values("reference_state_index", ignore_index=True)


def align_hmm_to_reference(reference_model: Any, candidate_model: Any) -> tuple[Any, pd.DataFrame]:
    """Hungarian-align a same-K candidate model to a reference model."""

    if int(reference_model.n_components) != int(candidate_model.n_components):
        raise ValueError("full HMM reordering requires the same number of states")
    alignment = hungarian_state_alignment(
        reference_model.emissionprob_, candidate_model.emissionprob_
    )
    order = alignment["candidate_state_index"].to_numpy(dtype=np.int64)
    return _reorder_hmm(candidate_model, order), alignment


def _posterior_occupancy(model: Any, bundle: SequenceBundle) -> np.ndarray:
    posterior = model.predict_proba(bundle.X, lengths=bundle.lengths)
    return posterior.sum(axis=0) / posterior.shape[0]


@dataclass
class HMMSelectionResult:
    status: str
    best_k: int | None
    best_model: Any | None
    models_by_k: dict[int, Any]
    summary: pd.DataFrame
    starts: pd.DataFrame
    selection_metric: str
    observation_labels: tuple[str, ...]
    min_state_occupancy: float
    notes: tuple[str, ...]


def _fit_categorical_hmm_once(
    bundle: SequenceBundle,
    *,
    n_states: int,
    seed: int,
    n_iter: int,
    tol: float,
    prior_strength: float,
) -> tuple[Any, list[str]]:
    try:
        from hmmlearn.hmm import CategoricalHMM
    except ImportError as exc:  # pragma: no cover - environment-dependent path
        raise ImportError(
            "hmmlearn is required for HMM fitting; install hmmlearn>=0.3"
        ) from exc
    if prior_strength < 0:
        raise ValueError("prior_strength cannot be negative")
    # hmmlearn uses Dirichlet prior=1 as the ML/no-pseudocount boundary.
    prior = 1.0 + float(prior_strength)
    model = CategoricalHMM(
        n_components=n_states,
        n_features=bundle.n_symbols,
        startprob_prior=prior,
        transmat_prior=prior,
        emissionprob_prior=prior,
        algorithm="viterbi",
        random_state=int(seed),
        n_iter=int(n_iter),
        tol=float(tol),
        verbose=False,
        params="ste",
        init_params="ste",
        implementation="log",
    )
    with warnings.catch_warnings(record=True) as captured:
        warnings.simplefilter("always")
        model.fit(bundle.X, lengths=bundle.lengths)
    messages = [str(item.message) for item in captured]
    return model, messages


def fit_categorical_hmm_grid(
    train_bundle: SequenceBundle,
    validation_bundle: SequenceBundle | None = None,
    *,
    k_values: Iterable[int] = range(2, 7),
    n_starts: int = 10,
    random_state: int = 20260822,
    n_iter: int = 300,
    tol: float = 1e-4,
    prior_strength: float = 0.1,
    min_state_occupancy: float = 0.01,
    selection_metric: Literal["validation_loglik", "bic"] = "validation_loglik",
) -> HMMSelectionResult:
    """Fit K=2,...,6 categorical HMMs with training-only multi-start selection.

    For each K, the random start is selected by training log likelihood.  The
    later-period validation likelihood may then select K.  Therefore the supplied
    validation bundle is a model-selection set, not an untouched final test set.
    """

    if train_bundle.n_observations < 2:
        raise ValueError("HMM fitting requires at least two observations")
    if validation_bundle is not None:
        _check_bundle_symbols(
            validation_bundle,
            train_bundle.n_symbols,
            train_bundle.observation_labels,
        )
    values = sorted(set(int(value) for value in k_values))
    if not values or any(value < 2 or value > 6 for value in values):
        raise ValueError("k_values must be a non-empty subset of {2,3,4,5,6}")
    if n_starts < 1:
        raise ValueError("n_starts must be positive")
    if selection_metric == "validation_loglik" and validation_bundle is None:
        raise ValueError("validation_loglik selection requires a validation bundle")
    if not 0.0 <= min_state_occupancy < 1.0:
        raise ValueError("min_state_occupancy must be in [0,1)")

    seed_generator = np.random.default_rng(random_state)
    start_rows: list[dict[str, Any]] = []
    summary_rows: list[dict[str, Any]] = []
    models_by_k: dict[int, Any] = {}

    for k in values:
        # tuple fields: train_ll, hmmlearn_converged, hit_iteration_cap,
        # converged_before_cap, seed, model, captured_warnings
        fitted: list[tuple[float, bool, bool, bool, int, Any, list[str]]] = []
        for start_index in range(n_starts):
            seed = int(seed_generator.integers(0, np.iinfo(np.int32).max))
            try:
                model, warning_messages = _fit_categorical_hmm_once(
                    train_bundle,
                    n_states=k,
                    seed=seed,
                    n_iter=n_iter,
                    tol=tol,
                    prior_strength=prior_strength,
                )
                train_ll = float(model.score(train_bundle.X, lengths=train_bundle.lengths))
                hmmlearn_converged = bool(model.monitor_.converged)
                iterations = int(model.monitor_.iter)
                # hmmlearn defines ``converged`` as True when either tolerance is
                # met OR the configured iteration limit is reached.  A capped fit
                # is therefore retained as a provisional candidate but cannot pass
                # our strict convergence gate.
                hit_iteration_cap = bool(iterations >= int(n_iter))
                converged_before_cap = bool(
                    hmmlearn_converged and not hit_iteration_cap
                )
                fitted.append(
                    (
                        train_ll,
                        hmmlearn_converged,
                        hit_iteration_cap,
                        converged_before_cap,
                        seed,
                        model,
                        warning_messages,
                    )
                )
                start_rows.append(
                    {
                        "k": k,
                        "start_index": start_index,
                        "seed": seed,
                        "status": "fit",
                        "converged": converged_before_cap,
                        "hmmlearn_converged": hmmlearn_converged,
                        "hit_iteration_cap": hit_iteration_cap,
                        "converged_before_cap": converged_before_cap,
                        "iterations": iterations,
                        "train_loglik": train_ll,
                        "train_loglik_per_observation": train_ll / train_bundle.n_observations,
                        "warnings": " | ".join(warning_messages),
                        "error": "",
                    }
                )
            except Exception as exc:  # retain failures for an auditable starts table
                start_rows.append(
                    {
                        "k": k,
                        "start_index": start_index,
                        "seed": seed,
                        "status": "failed",
                        "converged": False,
                        "hmmlearn_converged": False,
                        "hit_iteration_cap": False,
                        "converged_before_cap": False,
                        "iterations": 0,
                        "train_loglik": np.nan,
                        "train_loglik_per_observation": np.nan,
                        "warnings": "",
                        "error": f"{type(exc).__name__}: {exc}",
                    }
                )
        if not fitted:
            summary_rows.append(
                {
                    "model": "CategoricalHMM",
                    "k": k,
                    "status": "all_starts_failed",
                    "train_n_sequences": train_bundle.n_sequences,
                    "train_n_observations": train_bundle.n_observations,
                    "validation_n_sequences": (
                        validation_bundle.n_sequences if validation_bundle is not None else 0
                    ),
                    "validation_n_observations": (
                        validation_bundle.n_observations if validation_bundle is not None else 0
                    ),
                    "converged": False,
                    "hmmlearn_converged": False,
                    "hit_iteration_cap": False,
                    "converged_before_cap": False,
                    "train_loglik": np.nan,
                    "train_loglik_per_observation": np.nan,
                    "validation_loglik": np.nan,
                    "validation_loglik_per_observation": np.nan,
                    "n_parameters": categorical_hmm_parameter_count(k, train_bundle.n_symbols),
                    "bic": np.nan,
                    "min_posterior_state_occupancy": np.nan,
                    "occupancy_screen_passed": False,
                    "feasible": False,
                    "selected_start_seed": np.nan,
                }
            )
            continue

        converged_fits = [value for value in fitted if value[3]]
        pool = converged_fits if converged_fits else fitted
        (
            train_ll,
            hmmlearn_converged,
            hit_iteration_cap,
            converged_before_cap,
            seed,
            raw_model,
            _,
        ) = max(pool, key=lambda value: value[0])
        model, _, _ = canonicalize_hmm(raw_model, train_bundle.observation_labels)
        models_by_k[k] = model
        occupancy = _posterior_occupancy(model, train_bundle)
        n_parameters = categorical_hmm_parameter_count(k, train_bundle.n_symbols)
        bic = -2.0 * train_ll + n_parameters * math.log(train_bundle.n_observations)
        if validation_bundle is None:
            validation_ll = np.nan
            validation_per_obs = np.nan
        else:
            validation_ll = float(
                model.score(validation_bundle.X, lengths=validation_bundle.lengths)
            )
            validation_per_obs = validation_ll / validation_bundle.n_observations
        occupancy_screen_passed = bool(occupancy.min() >= min_state_occupancy)
        feasible = bool(converged_before_cap and occupancy_screen_passed)
        summary_rows.append(
            {
                "model": "CategoricalHMM",
                "k": k,
                "status": (
                    "fit" if converged_before_cap else "provisional_iteration_cap"
                    if hit_iteration_cap
                    else "not_converged"
                ),
                "train_n_sequences": train_bundle.n_sequences,
                "train_n_observations": train_bundle.n_observations,
                "validation_n_sequences": (
                    validation_bundle.n_sequences if validation_bundle is not None else 0
                ),
                "validation_n_observations": (
                    validation_bundle.n_observations if validation_bundle is not None else 0
                ),
                "converged": bool(converged_before_cap),
                "hmmlearn_converged": bool(hmmlearn_converged),
                "hit_iteration_cap": bool(hit_iteration_cap),
                "converged_before_cap": bool(converged_before_cap),
                "train_loglik": train_ll,
                "train_loglik_per_observation": train_ll / train_bundle.n_observations,
                "validation_loglik": validation_ll,
                "validation_loglik_per_observation": validation_per_obs,
                "n_parameters": n_parameters,
                "bic": bic,
                "min_posterior_state_occupancy": float(occupancy.min()),
                "occupancy_screen_passed": occupancy_screen_passed,
                "feasible": feasible,
                "selected_start_seed": seed,
            }
        )

    summary = pd.DataFrame(summary_rows).sort_values("k", ignore_index=True)
    starts = pd.DataFrame(start_rows).sort_values(["k", "start_index"], ignore_index=True)
    starts["selected_within_k"] = False
    for row in summary.itertuples(index=False):
        seed = getattr(row, "selected_start_seed")
        if pd.notna(seed):
            mask = starts["k"].eq(int(row.k)) & starts["seed"].eq(int(seed))
            starts.loc[mask, "selected_within_k"] = True
    candidate_pool = summary.loc[
        summary["occupancy_screen_passed"].fillna(False)
        & summary["validation_loglik_per_observation"].notna()
    ].copy()
    summary["best_metric_candidate_including_provisional"] = False
    if not candidate_pool.empty:
        if selection_metric == "validation_loglik":
            candidate_row = candidate_pool.sort_values(
                ["validation_loglik_per_observation", "bic"],
                ascending=[False, True],
            ).iloc[0]
        else:
            candidate_row = candidate_pool.sort_values(
                ["bic", "train_loglik_per_observation"],
                ascending=[True, False],
            ).iloc[0]
        summary.loc[
            summary["k"].eq(int(candidate_row["k"])),
            "best_metric_candidate_including_provisional",
        ] = True
    feasible = summary.loc[summary["feasible"].fillna(False)].copy()
    if feasible.empty:
        best_k = None
        best_model = None
        status = "no_feasible_model"
    else:
        if selection_metric == "validation_loglik":
            best_row = feasible.sort_values(
                ["validation_loglik_per_observation", "bic"],
                ascending=[False, True],
            ).iloc[0]
        else:
            best_row = feasible.sort_values(
                ["bic", "train_loglik_per_observation"],
                ascending=[True, False],
            ).iloc[0]
        best_k = int(best_row["k"])
        best_model = models_by_k[best_k]
        status = "selected"
        summary["selected"] = summary["k"].eq(best_k)
    if "selected" not in summary.columns:
        summary["selected"] = False

    notes = (
        "Random starts are chosen within K using training likelihood only.",
        "hmmlearn reports converged=True at the iteration limit; hit_iteration_cap fits are provisional and infeasible for strict K selection.",
        "Later-period validation can select K; it is not an untouched final test.",
        "BIC uses the standard free-parameter count for a categorical HMM.",
        "Minimum posterior occupancy is a degeneracy screen, not proof of construct validity.",
        "State ordering reflects platform-observation emissions, not cognitive depth.",
    )
    return HMMSelectionResult(
        status=status,
        best_k=best_k,
        best_model=best_model,
        models_by_k=models_by_k,
        summary=summary,
        starts=starts,
        selection_metric=selection_metric,
        observation_labels=train_bundle.observation_labels,
        min_state_occupancy=float(min_state_occupancy),
        notes=notes,
    )


@dataclass
class MCHMMComparison:
    markov_model: DirichletMarkovChain
    hmm_selection: HMMSelectionResult
    table: pd.DataFrame


def fit_mc_hmm_comparison(
    train_bundle: SequenceBundle,
    validation_bundle: SequenceBundle,
    *,
    mc_alpha: float = 0.5,
    k_values: Iterable[int] = range(2, 7),
    n_starts: int = 10,
    random_state: int = 20260822,
    n_iter: int = 300,
    tol: float = 1e-4,
    hmm_prior_strength: float = 0.1,
    min_state_occupancy: float = 0.01,
    selection_metric: Literal["validation_loglik", "bic"] = "validation_loglik",
) -> MCHMMComparison:
    """Fit the observable MC baseline and the candidate categorical HMMs."""

    _check_bundle_symbols(
        validation_bundle, train_bundle.n_symbols, train_bundle.observation_labels
    )
    markov = fit_dirichlet_markov_chain(train_bundle, alpha=mc_alpha)
    mc_train = markov.score(train_bundle)
    mc_validation = markov.score(validation_bundle)
    mc_bic = (
        -2.0 * mc_train.total_log_likelihood
        + markov.n_parameters * math.log(train_bundle.n_observations)
    )
    hmm_result = fit_categorical_hmm_grid(
        train_bundle,
        validation_bundle,
        k_values=k_values,
        n_starts=n_starts,
        random_state=random_state,
        n_iter=n_iter,
        tol=tol,
        prior_strength=hmm_prior_strength,
        min_state_occupancy=min_state_occupancy,
        selection_metric=selection_metric,
    )
    mc_row = pd.DataFrame(
        [
            {
                "model": "Observable-MC",
                "k": np.nan,
                "status": "fit",
                "train_n_sequences": train_bundle.n_sequences,
                "train_n_observations": train_bundle.n_observations,
                "validation_n_sequences": validation_bundle.n_sequences,
                "validation_n_observations": validation_bundle.n_observations,
                "converged": True,
                "train_loglik": mc_train.total_log_likelihood,
                "train_loglik_per_observation": mc_train.per_observation,
                "validation_loglik": mc_validation.total_log_likelihood,
                "validation_loglik_per_observation": mc_validation.per_observation,
                "n_parameters": markov.n_parameters,
                "bic": mc_bic,
                "min_posterior_state_occupancy": np.nan,
                "feasible": True,
                "selected_start_seed": np.nan,
                "selected": False,
            }
        ]
    )
    table = pd.concat([mc_row, hmm_result.summary], ignore_index=True, sort=False)
    return MCHMMComparison(markov_model=markov, hmm_selection=hmm_result, table=table)


def emission_profile_frame(
    model: Any,
    bundle: SequenceBundle,
) -> pd.DataFrame:
    """Return numbered-state emission profiles and posterior occupancies."""

    if model.emissionprob_.shape[1] != bundle.n_symbols:
        raise ValueError("model emission symbols do not match bundle")
    occupancy = _posterior_occupancy(model, bundle)
    expected_score = model.emissionprob_ @ _observation_scores(bundle.observation_labels)
    records: list[dict[str, Any]] = []
    for state in range(int(model.n_components)):
        row: dict[str, Any] = {
            "state_index": state,
            "state": f"State {state + 1}",
            "posterior_occupancy": float(occupancy[state]),
            "platform_evidence_score": float(expected_score[state]),
        }
        for label, probability in zip(bundle.observation_labels, model.emissionprob_[state]):
            row[f"P({label})"] = float(probability)
        records.append(row)
    return pd.DataFrame(records)


def posterior_frame(model: Any, bundle: SequenceBundle) -> pd.DataFrame:
    """Map posterior probabilities and Viterbi states back to source rows."""

    posterior = model.predict_proba(bundle.X, lengths=bundle.lengths)
    decoded = model.predict(bundle.X, lengths=bundle.lengths)
    records: list[dict[str, Any]] = []
    offset = 0
    for sequence_index, length in enumerate(bundle.lengths):
        group_key = bundle.group_keys[sequence_index]
        for position in range(length):
            absolute = offset + position
            row: dict[str, Any] = {
                name: value for name, value in zip(bundle.group_names, group_key)
            }
            row.update(
                {
                    "segment_id": int(bundle.segment_ids[sequence_index]),
                    "position_in_segment": int(position),
                    "source_row_id": bundle.row_ids[sequence_index][position],
                    "time": bundle.times[sequence_index][position],
                    "observation_code": int(bundle.observations[sequence_index][position]),
                    "observation": bundle.observation_labels[
                        int(bundle.observations[sequence_index][position])
                    ],
                    "viterbi_state_index": int(decoded[absolute]),
                    "viterbi_state": f"State {int(decoded[absolute]) + 1}",
                }
            )
            for state in range(int(model.n_components)):
                row[f"P(State {state + 1})"] = float(posterior[absolute, state])
            records.append(row)
        offset += length
    return pd.DataFrame.from_records(records)


def simulate_categorical_hmm(
    startprob: Sequence[float],
    transmat: Sequence[Sequence[float]],
    emissionprob: Sequence[Sequence[float]],
    lengths: Sequence[int],
    *,
    random_state: int | np.random.Generator = 20260822,
) -> tuple[tuple[np.ndarray, ...], tuple[np.ndarray, ...]]:
    """Simulate independent observation/state sequences with exact boundaries."""

    start = np.asarray(startprob, dtype=float)
    transition = np.asarray(transmat, dtype=float)
    emission = np.asarray(emissionprob, dtype=float)
    k = len(start)
    if transition.shape != (k, k) or emission.ndim != 2 or emission.shape[0] != k:
        raise ValueError("start, transition, and emission dimensions are incompatible")
    if any(int(length) < 1 for length in lengths):
        raise ValueError("all simulated sequence lengths must be positive")
    for name, values in (
        ("startprob", start.reshape(1, -1)),
        ("transmat", transition),
        ("emissionprob", emission),
    ):
        if np.any(values < 0) or not np.allclose(values.sum(axis=1), 1.0):
            raise ValueError(f"{name} rows must be nonnegative and sum to one")
    rng = (
        random_state
        if isinstance(random_state, np.random.Generator)
        else np.random.default_rng(random_state)
    )
    observation_sequences: list[np.ndarray] = []
    state_sequences: list[np.ndarray] = []
    for raw_length in lengths:
        length = int(raw_length)
        states = np.empty(length, dtype=np.int64)
        observations = np.empty(length, dtype=np.int64)
        states[0] = rng.choice(k, p=start)
        observations[0] = rng.choice(emission.shape[1], p=emission[states[0]])
        for position in range(1, length):
            states[position] = rng.choice(k, p=transition[states[position - 1]])
            observations[position] = rng.choice(
                emission.shape[1], p=emission[states[position]]
            )
        observation_sequences.append(observations)
        state_sequences.append(states)
    return tuple(observation_sequences), tuple(state_sequences)


def _run_lengths_by_state(
    paths: Sequence[np.ndarray],
    n_states: int,
) -> tuple[list[list[int]], list[list[int]], np.ndarray]:
    all_runs = [[] for _ in range(n_states)]
    uncensored_runs = [[] for _ in range(n_states)]
    censored_counts = np.zeros(n_states, dtype=np.int64)
    for raw_path in paths:
        path = np.asarray(raw_path, dtype=np.int64)
        if not len(path):
            continue
        start = 0
        for stop in range(1, len(path) + 1):
            if stop == len(path) or path[stop] != path[start]:
                state = int(path[start])
                duration = int(stop - start)
                all_runs[state].append(duration)
                boundary_censored = start == 0 or stop == len(path)
                if boundary_censored:
                    censored_counts[state] += 1
                else:
                    uncensored_runs[state].append(duration)
                start = stop
    return all_runs, uncensored_runs, censored_counts


def _geometric_ks_distance(durations: Sequence[int], stay_probability: float) -> float:
    values = np.asarray(durations, dtype=np.int64)
    if values.size == 0:
        return float("nan")
    support = np.arange(1, int(values.max()) + 1)
    empirical = np.asarray([(values <= value).mean() for value in support])
    theoretical = 1.0 - np.power(stay_probability, support)
    return float(np.max(np.abs(empirical - theoretical)))


def _two_sided_ppc_p(simulated: np.ndarray, observed: float) -> float:
    finite = simulated[np.isfinite(simulated)]
    if not len(finite) or not np.isfinite(observed):
        return float("nan")
    return float(min(1.0, 2.0 * min(np.mean(finite >= observed), np.mean(finite <= observed))))


def _duration_diagnostic_label(
    n_uncensored_runs: int,
    minimum_uncensored_runs: int,
    calibration_p_values: Sequence[float],
) -> str:
    """Combine all reported PPC calibration checks into one conservative flag."""

    if n_uncensored_runs < minimum_uncensored_runs:
        return "insufficient_uncensored_runs"
    finite = [float(value) for value in calibration_p_values if np.isfinite(value)]
    if any(value < 0.05 for value in finite):
        return "geometric_duration_misfit_warning"
    return "no_strong_misfit_signal"


def _finite_quantile(values: np.ndarray, quantile: float) -> float:
    finite = np.asarray(values, dtype=float)
    finite = finite[np.isfinite(finite)]
    if not len(finite):
        return float("nan")
    return float(np.quantile(finite, quantile))


@dataclass
class DwellTimeDiagnostic:
    summary: pd.DataFrame
    n_replications: int
    caveats: tuple[str, ...]


def dwell_time_posterior_predictive_check(
    model: Any,
    bundle: SequenceBundle,
    *,
    n_replications: int = 200,
    random_state: int = 20260822,
    minimum_uncensored_runs: int = 10,
) -> DwellTimeDiagnostic:
    """Diagnose the HMM geometric-duration assumption using decoded spell lengths.

    Each replication simulates observations from the fitted HMM at the exact
    empirical sequence lengths and Viterbi-decodes those observations again.
    Runs touching a sequence boundary are reported as censored and excluded from
    the primary comparison.  A small PPC p-value is a warning, not an HSMM fit.
    """

    if n_replications < 1:
        raise ValueError("n_replications must be positive")
    if minimum_uncensored_runs < 1:
        raise ValueError("minimum_uncensored_runs must be positive")
    if model.emissionprob_.shape[1] != bundle.n_symbols:
        raise ValueError("model and bundle symbols differ")

    decoded_flat = model.predict(bundle.X, lengths=bundle.lengths)
    observed_paths: list[np.ndarray] = []
    offset = 0
    for length in bundle.lengths:
        observed_paths.append(decoded_flat[offset : offset + length].copy())
        offset += length
    all_runs, uncensored_runs, censored_counts = _run_lengths_by_state(
        observed_paths, int(model.n_components)
    )

    rng = np.random.default_rng(random_state)
    simulated_means = np.full((n_replications, model.n_components), np.nan)
    simulated_p90 = np.full((n_replications, model.n_components), np.nan)
    simulated_ks = np.full((n_replications, model.n_components), np.nan)
    for replication in range(n_replications):
        observations, _ = simulate_categorical_hmm(
            model.startprob_,
            model.transmat_,
            model.emissionprob_,
            bundle.lengths,
            random_state=rng,
        )
        simulated_bundle = sequence_bundle_from_arrays(
            observations,
            bundle.n_symbols,
            observation_labels=bundle.observation_labels,
        )
        decoded = model.predict(simulated_bundle.X, lengths=simulated_bundle.lengths)
        paths: list[np.ndarray] = []
        cursor = 0
        for length in simulated_bundle.lengths:
            paths.append(decoded[cursor : cursor + length].copy())
            cursor += length
        _, simulated_uncensored, _ = _run_lengths_by_state(paths, model.n_components)
        for state in range(model.n_components):
            values = np.asarray(simulated_uncensored[state], dtype=float)
            if len(values):
                simulated_means[replication, state] = float(values.mean())
                simulated_p90[replication, state] = float(np.quantile(values, 0.9))
                simulated_ks[replication, state] = _geometric_ks_distance(
                    values.astype(int), float(model.transmat_[state, state])
                )

    rows: list[dict[str, Any]] = []
    for state in range(model.n_components):
        all_values = np.asarray(all_runs[state], dtype=float)
        values = np.asarray(uncensored_runs[state], dtype=float)
        stay = float(model.transmat_[state, state])
        observed_mean = float(values.mean()) if len(values) else np.nan
        observed_p90 = float(np.quantile(values, 0.9)) if len(values) else np.nan
        observed_ks = _geometric_ks_distance(values.astype(int), stay)
        ppc_mean = _two_sided_ppc_p(simulated_means[:, state], observed_mean)
        ppc_p90 = _two_sided_ppc_p(simulated_p90[:, state], observed_p90)
        ks_tail = simulated_ks[:, state]
        finite_ks = ks_tail[np.isfinite(ks_tail)]
        ppc_ks = (
            float(np.mean(finite_ks >= observed_ks))
            if len(finite_ks) and np.isfinite(observed_ks)
            else np.nan
        )
        diagnostic = _duration_diagnostic_label(
            len(values),
            minimum_uncensored_runs,
            (ppc_mean, ppc_p90, ppc_ks),
        )
        rows.append(
            {
                "state_index": state,
                "state": f"State {state + 1}",
                "self_transition_probability": stay,
                "geometric_expected_duration": (
                    1.0 / (1.0 - stay) if stay < 1.0 else np.inf
                ),
                "all_decoded_runs": int(len(all_values)),
                "boundary_censored_runs": int(censored_counts[state]),
                "uncensored_runs": int(len(values)),
                "observed_uncensored_mean": observed_mean,
                "observed_uncensored_p90": observed_p90,
                "observed_ks_to_geometric": observed_ks,
                "ppc_mean_median": _finite_quantile(simulated_means[:, state], 0.5),
                "ppc_mean_2.5pct": _finite_quantile(simulated_means[:, state], 0.025),
                "ppc_mean_97.5pct": _finite_quantile(simulated_means[:, state], 0.975),
                "ppc_p_two_sided_mean": ppc_mean,
                "ppc_p_two_sided_p90": ppc_p90,
                "ppc_p_ks_upper_tail": ppc_ks,
                "diagnostic": diagnostic,
            }
        )
    caveats = (
        "Durations are spells in the Viterbi-decoded path, not directly observed cognitive states.",
        "Boundary spells are censored and excluded from the primary duration comparison.",
        "A non-small PPC p-value does not prove the geometric assumption.",
        "The diagnostic warns when any reported mean, p90, or KS calibration p-value is below 0.05.",
        "A misfit warning motivates a genuine explicit-duration HSMM; it is not itself an HSMM result.",
    )
    return DwellTimeDiagnostic(
        summary=pd.DataFrame(rows),
        n_replications=int(n_replications),
        caveats=caveats,
    )


class HSMMNeedsRealModelingError(NotImplementedError):
    """Raised when code attempts to report an HSMM that has not been implemented."""


def hsmm_status() -> dict[str, Any]:
    return {
        "status": "needs_real_modeling",
        "implemented": False,
        "required_components": [
            "explicit state-duration distributions",
            "a valid HSMM likelihood/inference implementation",
            "multi-start estimation and model selection",
            "time-out and leave-course-out validation",
            "duration posterior-predictive checks",
        ],
        "prohibited_shortcut": "Do not relabel an HMM dwell-time calculation as an HSMM.",
    }


def fit_hsmm(*args: Any, **kwargs: Any) -> Any:
    del args, kwargs
    raise HSMMNeedsRealModelingError(
        "needs_real_modeling: no explicit-duration HSMM is implemented. "
        "Use hsmm_status() for the required evidence contract."
    )


__all__ = [
    "OBSERVATION_LABELS",
    "UNKNOWN_OBSERVATION_LABEL",
    "ObservationEncodingConfig",
    "TransparentObservationEncoder",
    "SequenceBundle",
    "PreparedSplit",
    "DirichletMarkovChain",
    "LogLikelihoodScore",
    "HMMSelectionResult",
    "MCHMMComparison",
    "DwellTimeDiagnostic",
    "HSMMNeedsRealModelingError",
    "sequence_bundle_from_arrays",
    "make_sequence_bundle",
    "temporal_train_validation_split",
    "iter_leave_course_out_splits",
    "fit_dirichlet_markov_chain",
    "categorical_hmm_parameter_count",
    "canonical_state_order",
    "canonicalize_hmm",
    "hungarian_state_alignment",
    "align_hmm_to_reference",
    "fit_categorical_hmm_grid",
    "fit_mc_hmm_comparison",
    "emission_profile_frame",
    "posterior_frame",
    "simulate_categorical_hmm",
    "dwell_time_posterior_predictive_check",
    "hsmm_status",
    "fit_hsmm",
]
