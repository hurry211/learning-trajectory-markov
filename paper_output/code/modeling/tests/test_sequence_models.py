"""Small synthetic tests for sequence_models.py.

Run from the project root with:

    python -m unittest discover -s paper_output/code/modeling/tests -v
"""

from __future__ import annotations

import sys
import unittest
from pathlib import Path

import numpy as np
import pandas as pd


MODELING_DIR = Path(__file__).resolve().parents[1]
if str(MODELING_DIR) not in sys.path:
    sys.path.insert(0, str(MODELING_DIR))

from sequence_models import (  # noqa: E402
    HSMMNeedsRealModelingError,
    OBSERVATION_LABELS,
    ObservationEncodingConfig,
    TransparentObservationEncoder,
    dwell_time_posterior_predictive_check,
    fit_categorical_hmm_grid,
    fit_dirichlet_markov_chain,
    fit_hsmm,
    hungarian_state_alignment,
    iter_leave_course_out_splits,
    make_sequence_bundle,
    sequence_bundle_from_arrays,
    simulate_categorical_hmm,
    temporal_train_validation_split,
    _duration_diagnostic_label,
)


class ObservationAndBoundaryTests(unittest.TestCase):
    def test_markov_counts_never_cross_sequence_boundary(self) -> None:
        bundle = sequence_bundle_from_arrays(
            [[0, 1], [2, 3]],
            n_symbols=4,
            observation_labels=OBSERVATION_LABELS,
        )
        model = fit_dirichlet_markov_chain(bundle, alpha=0.5)

        self.assertEqual(int(model.transition_counts.sum()), 2)
        self.assertEqual(int(model.transition_counts[0, 1]), 1)
        self.assertEqual(int(model.transition_counts[2, 3]), 1)
        # This would be one if concatenated students were treated as one sequence.
        self.assertEqual(int(model.transition_counts[1, 2]), 0)
        np.testing.assert_allclose(model.transmat.sum(axis=1), 1.0)

    def test_training_only_quantile_and_temporal_split(self) -> None:
        rows: list[dict[str, object]] = []
        for student in ("s1", "s2"):
            for day in range(10):
                rows.append(
                    {
                        "course": "c1",
                        "student": student,
                        "day": pd.Timestamp("2025-01-01") + pd.Timedelta(days=day),
                        "platform_inactive": False,
                        # Validation values are extreme and must not change threshold.
                        "activity_type_diversity": float(day if day < 7 else 10_000),
                        "productive_event_count": 0.0,
                    }
                )
        frame = pd.DataFrame(rows)
        split = temporal_train_validation_split(
            frame,
            group_cols=["course", "student"],
            time_col="day",
            train_fraction=0.7,
            min_train_rows=5,
            min_validation_rows=2,
            encoder_config=ObservationEncodingConfig(diversity_quantile=0.5),
            dataset_name="synthetic",
        )

        # Training diversity for each student is 0,...,6, so the pooled median is 3.
        self.assertEqual(split.encoder.diversity_threshold_, 3.0)
        self.assertEqual(split.train.n_sequences, 2)
        self.assertEqual(split.validation.n_sequences, 2)
        self.assertEqual(split.train.n_observations, 14)
        self.assertEqual(split.validation.n_observations, 6)
        self.assertEqual(split.metadata["encoder_fit_partition"], "train_only")

    def test_missing_row_splits_instead_of_bridging(self) -> None:
        frame = pd.DataFrame(
            {
                "student": ["s1"] * 5,
                "day": pd.date_range("2025-01-01", periods=5, freq="D"),
                "platform_inactive": [True, False, False, False, False],
                "activity_type_diversity": [np.nan, 0.0, np.nan, 4.0, 0.0],
                "productive_event_count": [np.nan, 0.0, 0.0, 0.0, 1.0],
            }
        )
        encoder = TransparentObservationEncoder(
            ObservationEncodingConfig(missing_policy="split", diversity_quantile=0.5)
        ).fit(frame.iloc[[0, 1, 3, 4]])
        bundle = make_sequence_bundle(
            frame,
            encoder,
            group_cols="student",
            time_col="day",
            min_sequence_length=1,
        )

        self.assertEqual(bundle.lengths, [2, 2])
        self.assertEqual(bundle.metadata["split_at_missing_rows"], 1)
        model = fit_dirichlet_markov_chain(bundle, alpha=0.5)
        # Two sequences of length two produce exactly two within-segment transitions.
        self.assertEqual(int(model.transition_counts.sum()), 2)

    def test_leave_course_out_encoder_never_sees_held_out_threshold_values(self) -> None:
        rows: list[dict[str, object]] = []
        for course, base in (("train_course", 0.0), ("held_course", 10_000.0)):
            for day in range(4):
                rows.append(
                    {
                        "course": course,
                        "student": f"student_{course}",
                        "day": day,
                        "platform_inactive": False,
                        "activity_type_diversity": base + day,
                        "productive_event_count": 0.0,
                    }
                )
        frame = pd.DataFrame(rows)
        folds = list(
            iter_leave_course_out_splits(
                frame,
                course_cols="course",
                group_cols=["course", "student"],
                time_col="day",
                held_out_courses=["held_course"],
                encoder_config=ObservationEncodingConfig(diversity_quantile=0.5),
                expected_step=1,
            )
        )

        self.assertEqual(len(folds), 1)
        self.assertEqual(folds[0].encoder.diversity_threshold_, 1.5)
        self.assertEqual(folds[0].metadata["encoder_fit_partition"], "training_courses_only")


class HMMRecoveryTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.start = np.array([0.60, 0.40])
        cls.transition = np.array([[0.91, 0.09], [0.12, 0.88]])
        cls.emission = np.array([[0.88, 0.10, 0.02], [0.03, 0.12, 0.85]])
        observations, _ = simulate_categorical_hmm(
            cls.start,
            cls.transition,
            cls.emission,
            lengths=[70] * 60,
            random_state=314159,
        )
        cls.train_bundle = sequence_bundle_from_arrays(
            observations[:45],
            n_symbols=3,
            observation_labels=("low", "middle", "high"),
        )
        cls.validation_bundle = sequence_bundle_from_arrays(
            observations[45:],
            n_symbols=3,
            observation_labels=("low", "middle", "high"),
        )
        cls.selection = fit_categorical_hmm_grid(
            cls.train_bundle,
            cls.validation_bundle,
            k_values=[2],
            n_starts=5,
            random_state=271828,
            n_iter=150,
            tol=1e-5,
            min_state_occupancy=0.05,
        )

    def test_hmm_recovers_emission_and_transition_parameters(self) -> None:
        self.assertEqual(self.selection.status, "selected")
        self.assertEqual(self.selection.best_k, 2)
        estimated = self.selection.best_model
        alignment = hungarian_state_alignment(self.emission, estimated.emissionprob_)
        candidate_order = alignment["candidate_state_index"].to_numpy(dtype=int)
        aligned_emission = estimated.emissionprob_[candidate_order]
        aligned_transition = estimated.transmat_[np.ix_(candidate_order, candidate_order)]

        np.testing.assert_allclose(aligned_emission, self.emission, atol=0.10)
        np.testing.assert_allclose(aligned_transition, self.transition, atol=0.10)
        self.assertGreaterEqual(
            float(self.selection.summary.loc[0, "min_posterior_state_occupancy"]),
            0.05,
        )

    def test_dwell_time_ppc_runs_with_exact_sequence_lengths(self) -> None:
        diagnostic = dwell_time_posterior_predictive_check(
            self.selection.best_model,
            self.validation_bundle,
            n_replications=20,
            random_state=161803,
            minimum_uncensored_runs=3,
        )
        self.assertEqual(len(diagnostic.summary), 2)
        self.assertEqual(diagnostic.n_replications, 20)
        self.assertIn("boundary_censored_runs", diagnostic.summary.columns)
        self.assertIn("ppc_p_two_sided_mean", diagnostic.summary.columns)
        self.assertIn("ppc_p_two_sided_p90", diagnostic.summary.columns)
        self.assertIn("ppc_p_ks_upper_tail", diagnostic.summary.columns)
        for row in diagnostic.summary.itertuples(index=False):
            if row.uncensored_runs < 3:
                self.assertEqual(row.diagnostic, "insufficient_uncensored_runs")
                continue
            calibration = (
                row.ppc_p_two_sided_mean,
                row.ppc_p_two_sided_p90,
                row.ppc_p_ks_upper_tail,
            )
            if any(np.isfinite(value) and value < 0.05 for value in calibration):
                self.assertEqual(row.diagnostic, "geometric_duration_misfit_warning")

    def test_duration_diagnostic_uses_any_reported_calibration_statistic(self) -> None:
        self.assertEqual(
            _duration_diagnostic_label(20, 10, (0.0, 0.70, 0.80)),
            "geometric_duration_misfit_warning",
        )
        self.assertEqual(
            _duration_diagnostic_label(20, 10, (0.80, 0.01, 0.80)),
            "geometric_duration_misfit_warning",
        )
        self.assertEqual(
            _duration_diagnostic_label(20, 10, (0.80, 0.70, 0.01)),
            "geometric_duration_misfit_warning",
        )
        self.assertEqual(
            _duration_diagnostic_label(2, 10, (0.0, 0.0, 0.0)),
            "insufficient_uncensored_runs",
        )

    def test_iteration_cap_is_not_strict_convergence(self) -> None:
        capped = fit_categorical_hmm_grid(
            self.train_bundle,
            self.validation_bundle,
            k_values=[2],
            n_starts=1,
            random_state=123,
            n_iter=1,
            tol=1e-12,
            min_state_occupancy=0.0,
        )
        self.assertTrue(bool(capped.starts.loc[0, "hit_iteration_cap"]))
        self.assertFalse(bool(capped.starts.loc[0, "converged_before_cap"]))
        self.assertEqual(capped.summary.loc[0, "status"], "provisional_iteration_cap")
        self.assertFalse(bool(capped.summary.loc[0, "feasible"]))
        self.assertEqual(capped.status, "no_feasible_model")


class HSMMSafetyTest(unittest.TestCase):
    def test_hsmm_is_explicitly_not_implemented(self) -> None:
        with self.assertRaisesRegex(HSMMNeedsRealModelingError, "needs_real_modeling"):
            fit_hsmm(None)


if __name__ == "__main__":
    unittest.main(verbosity=2)
