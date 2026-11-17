"""Numerical helpers for the Q3--Q4 conditional-scenario analysis.

The module deliberately keeps the estimand descriptive.  Covariates used for
the transition from day t to t+1 must have been available before day t.  The
non-homogeneous HMM is estimated by a genuine generalized-EM loop in which the
time-varying transition kernel is used in every E step; a one-shot Viterbi
regression is never used as a substitute.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Sequence

import numpy as np
from scipy.optimize import minimize
from scipy.special import expit, logsumexp


TINY = np.finfo(float).tiny


def row_softmax_reference(eta: np.ndarray) -> np.ndarray:
    """Softmax probabilities with destination 0 as the reference category."""

    values = np.asarray(eta, dtype=float)
    zeros = np.zeros((*values.shape[:-1], 1), dtype=float)
    logits = np.concatenate([zeros, values], axis=-1)
    logits -= logits.max(axis=-1, keepdims=True)
    probabilities = np.exp(logits)
    probabilities /= probabilities.sum(axis=-1, keepdims=True)
    return probabilities


def categorical_draw(probabilities: np.ndarray, uniforms: np.ndarray) -> np.ndarray:
    """Vectorized inverse-CDF draw, one row of probabilities per draw."""

    probabilities = np.asarray(probabilities, dtype=float)
    uniforms = np.asarray(uniforms, dtype=float)
    cumulative = np.cumsum(probabilities, axis=1)
    return (uniforms[:, None] > cumulative).sum(axis=1).clip(0, probabilities.shape[1] - 1)


def js_divergence(p: np.ndarray, q: np.ndarray) -> float:
    p = np.asarray(p, dtype=float)
    q = np.asarray(q, dtype=float)
    p = p / p.sum()
    q = q / q.sum()
    m = 0.5 * (p + q)

    def kl(a: np.ndarray, b: np.ndarray) -> float:
        mask = a > 0
        return float(np.sum(a[mask] * np.log(a[mask] / b[mask])))

    return 0.5 * (kl(p, m) + kl(q, m))


def max_true_run(values: np.ndarray) -> np.ndarray:
    """Maximum consecutive True run for each row of a 2-D Boolean array."""

    array = np.asarray(values, dtype=bool)
    current = np.zeros(array.shape[0], dtype=np.int16)
    best = np.zeros(array.shape[0], dtype=np.int16)
    for column in range(array.shape[1]):
        current = np.where(array[:, column], current + 1, 0)
        best = np.maximum(best, current)
    return best


@dataclass
class FeatureTransformer:
    """Training-only imputation and scaling for transition covariates."""

    raw_names: tuple[str, ...]
    continuous_indices: tuple[int, ...]
    missing_value_indices: tuple[int, ...]
    medians: np.ndarray | None = None
    means: np.ndarray | None = None
    scales: np.ndarray | None = None

    def fit(self, raw: np.ndarray) -> "FeatureTransformer":
        values = np.asarray(raw, dtype=float)
        medians = np.zeros(values.shape[1], dtype=float)
        for index in range(values.shape[1]):
            finite = values[np.isfinite(values[:, index]), index]
            medians[index] = float(np.median(finite)) if finite.size else 0.0
        imputed = np.where(np.isfinite(values), values, medians)
        means = np.zeros(values.shape[1], dtype=float)
        scales = np.ones(values.shape[1], dtype=float)
        for index in self.continuous_indices:
            means[index] = float(imputed[:, index].mean())
            scale = float(imputed[:, index].std(ddof=0))
            scales[index] = scale if scale > 1e-8 else 1.0
        self.medians = medians
        self.means = means
        self.scales = scales
        return self

    def transform(self, raw: np.ndarray) -> np.ndarray:
        if self.medians is None or self.means is None or self.scales is None:
            raise RuntimeError("FeatureTransformer must be fitted first")
        values = np.asarray(raw, dtype=float)
        imputed = np.where(np.isfinite(values), values, self.medians)
        scaled = (imputed - self.means) / self.scales
        return np.column_stack([np.ones(len(values), dtype=float), scaled])

    @property
    def design_names(self) -> tuple[str, ...]:
        return ("intercept",) + self.raw_names

    def metadata(self) -> dict[str, Any]:
        if self.medians is None or self.means is None or self.scales is None:
            raise RuntimeError("FeatureTransformer must be fitted first")
        return {
            "raw_names": list(self.raw_names),
            "design_names": list(self.design_names),
            "continuous_indices": list(self.continuous_indices),
            "training_medians": self.medians.tolist(),
            "training_means": self.means.tolist(),
            "training_scales": self.scales.tolist(),
            "missing_value_indices": list(self.missing_value_indices),
        }


def _multinomial_objective(
    flat: np.ndarray,
    x: np.ndarray,
    expected_counts: np.ndarray,
    ridge: float,
) -> tuple[float, np.ndarray]:
    """Weighted multinomial cross-entropy and gradient."""

    n_destinations = expected_counts.shape[1]
    coefficients = flat.reshape(n_destinations - 1, x.shape[1])
    eta = x @ coefficients.T
    probabilities = row_softmax_reference(eta)
    weights = expected_counts.sum(axis=1)
    log_probabilities = np.log(np.clip(probabilities, TINY, 1.0))
    loss = -float(np.sum(expected_counts * log_probabilities))
    penalty = coefficients.copy()
    penalty[:, 0] = 0.0
    loss += 0.5 * ridge * float(np.sum(penalty * penalty))
    residual = probabilities[:, 1:] * weights[:, None] - expected_counts[:, 1:]
    gradient = residual.T @ x + ridge * penalty
    return loss, gradient.ravel()


def fit_transition_coefficients(
    x: np.ndarray,
    expected_xi: np.ndarray,
    initial: np.ndarray,
    *,
    ridge: float = 1.0,
    maxiter: int = 150,
) -> tuple[np.ndarray, list[dict[str, Any]]]:
    """Fit one multinomial transition regression for every source state."""

    n_states = expected_xi.shape[1]
    coefficients = np.asarray(initial, dtype=float).copy()
    audits: list[dict[str, Any]] = []
    for source in range(n_states):
        counts = expected_xi[:, source, :]
        effective = counts.sum(axis=1)
        keep = effective > 1e-12
        if int(keep.sum()) < max(20, x.shape[1] * 2):
            audits.append(
                {
                    "source_state": source,
                    "success": False,
                    "reason": "insufficient_effective_transitions",
                    "effective_rows": int(keep.sum()),
                    "effective_mass": float(effective.sum()),
                }
            )
            continue
        result = minimize(
            _multinomial_objective,
            coefficients[source].ravel(),
            args=(x[keep], counts[keep], ridge),
            method="L-BFGS-B",
            jac=True,
            options={"maxiter": int(maxiter), "ftol": 1e-11, "gtol": 1e-7},
        )
        coefficients[source] = result.x.reshape(n_states - 1, x.shape[1])
        audits.append(
            {
                "source_state": source,
                "success": bool(result.success),
                "reason": str(result.message),
                "iterations": int(result.nit),
                "objective": float(result.fun),
                "effective_rows": int(keep.sum()),
                "effective_mass": float(effective.sum()),
            }
        )
    return coefficients, audits


@dataclass
class ObservableNHMC:
    n_states: int
    coefficients: np.ndarray
    startprob: np.ndarray
    ridge: float
    optimization_audit: list[dict[str, Any]]

    @classmethod
    def fit(
        cls,
        observation_sequences: Sequence[np.ndarray],
        design_sequences: Sequence[np.ndarray],
        *,
        n_states: int,
        ridge: float = 1.0,
    ) -> "ObservableNHMC":
        x = np.concatenate(design_sequences, axis=0)
        expected = np.zeros((len(x), n_states, n_states), dtype=float)
        cursor = 0
        starts = np.full(n_states, 0.5, dtype=float)
        for observations, design in zip(observation_sequences, design_sequences):
            observations = np.asarray(observations, dtype=int)
            starts[observations[0]] += 1.0
            rows = np.arange(cursor, cursor + len(design))
            expected[rows, observations[:-1], observations[1:]] = 1.0
            cursor += len(design)
        initial = np.zeros((n_states, n_states - 1, x.shape[1]), dtype=float)
        coefficients, audit = fit_transition_coefficients(
            x, expected, initial, ridge=ridge
        )
        starts /= starts.sum()
        return cls(n_states, coefficients, starts, ridge, audit)

    def transition_matrices(self, x: np.ndarray) -> np.ndarray:
        matrices = np.empty((len(x), self.n_states, self.n_states), dtype=float)
        for source in range(self.n_states):
            eta = x @ self.coefficients[source].T
            matrices[:, source, :] = row_softmax_reference(eta)
        return matrices

    def score(
        self,
        observation_sequences: Sequence[np.ndarray],
        design_sequences: Sequence[np.ndarray],
    ) -> tuple[float, int]:
        total = 0.0
        observations_n = 0
        for observations, design in zip(observation_sequences, design_sequences):
            observations = np.asarray(observations, dtype=int)
            matrices = self.transition_matrices(design)
            total += float(np.log(np.clip(self.startprob[observations[0]], TINY, 1.0)))
            total += float(
                np.log(
                    np.clip(
                        matrices[np.arange(len(design)), observations[:-1], observations[1:]],
                        TINY,
                        1.0,
                    )
                ).sum()
            )
            observations_n += len(observations)
        return total, observations_n

    def probabilities_for_sources(self, sources: np.ndarray, x: np.ndarray) -> np.ndarray:
        matrices = self.transition_matrices(x)
        return matrices[np.arange(len(sources)), sources]


def _forward_backward(
    observations: np.ndarray,
    x: np.ndarray,
    startprob: np.ndarray,
    coefficients: np.ndarray,
    emissionprob: np.ndarray,
) -> tuple[float, np.ndarray, np.ndarray, np.ndarray]:
    n = len(observations)
    k = len(startprob)
    transitions = np.empty((n - 1, k, k), dtype=float)
    for source in range(k):
        transitions[:, source, :] = row_softmax_reference(x @ coefficients[source].T)
    alpha = np.empty((n, k), dtype=float)
    scales = np.empty(n, dtype=float)
    alpha[0] = startprob * emissionprob[:, observations[0]]
    scales[0] = max(float(alpha[0].sum()), TINY)
    alpha[0] /= scales[0]
    for t in range(n - 1):
        alpha[t + 1] = (alpha[t] @ transitions[t]) * emissionprob[:, observations[t + 1]]
        scales[t + 1] = max(float(alpha[t + 1].sum()), TINY)
        alpha[t + 1] /= scales[t + 1]
    beta = np.ones((n, k), dtype=float)
    for t in range(n - 2, -1, -1):
        beta[t] = transitions[t] @ (emissionprob[:, observations[t + 1]] * beta[t + 1])
        beta[t] /= scales[t + 1]
    gamma = alpha * beta
    gamma /= gamma.sum(axis=1, keepdims=True)
    xi = np.empty((n - 1, k, k), dtype=float)
    for t in range(n - 1):
        xi[t] = (
            alpha[t, :, None]
            * transitions[t]
            * (emissionprob[:, observations[t + 1]] * beta[t + 1])[None, :]
        )
        denominator = max(float(xi[t].sum()), TINY)
        xi[t] /= denominator
    return float(np.log(scales).sum()), gamma, xi, alpha


@dataclass
class NHMMFit:
    n_states: int
    n_symbols: int
    coefficients: np.ndarray
    startprob: np.ndarray
    emissionprob: np.ndarray
    ridge: float
    converged: bool
    iterations: int
    loglik_trace: list[float]
    occupancy: np.ndarray
    transition_optimization: list[dict[str, Any]]
    seed: int

    def transition_matrices(self, x: np.ndarray) -> np.ndarray:
        matrices = np.empty((len(x), self.n_states, self.n_states), dtype=float)
        for source in range(self.n_states):
            matrices[:, source, :] = row_softmax_reference(x @ self.coefficients[source].T)
        return matrices

    def score(
        self,
        observation_sequences: Sequence[np.ndarray],
        design_sequences: Sequence[np.ndarray],
    ) -> tuple[float, int]:
        total = 0.0
        count = 0
        for observations, design in zip(observation_sequences, design_sequences):
            ll, _, _, _ = _forward_backward(
                np.asarray(observations, dtype=int),
                design,
                self.startprob,
                self.coefficients,
                self.emissionprob,
            )
            total += ll
            count += len(observations)
        return total, count

    def filter_distribution(self, observations: np.ndarray, design: np.ndarray) -> np.ndarray:
        _, _, _, alpha = _forward_backward(
            np.asarray(observations, dtype=int),
            design,
            self.startprob,
            self.coefficients,
            self.emissionprob,
        )
        return alpha[-1]

    def probabilities_for_sources(self, sources: np.ndarray, x: np.ndarray) -> np.ndarray:
        matrices = self.transition_matrices(x)
        return matrices[np.arange(len(sources)), sources]


def fit_nhmm(
    observation_sequences: Sequence[np.ndarray],
    design_sequences: Sequence[np.ndarray],
    *,
    n_states: int = 2,
    n_symbols: int = 4,
    ridge: float = 1.0,
    max_iter: int = 35,
    tolerance_per_observation: float = 2e-6,
    seed: int = 20260822,
) -> NHMMFit:
    """Generalized-EM fit of a categorical non-homogeneous HMM."""

    if n_states != 2:
        raise ValueError("This audited implementation currently supports K=2 only")
    rng = np.random.default_rng(seed)
    p = design_sequences[0].shape[1]
    coefficients = np.zeros((n_states, n_states - 1, p), dtype=float)
    coefficients[0, 0, 0] = -2.4 + rng.normal(0, 0.15)
    coefficients[1, 0, 0] = 2.4 + rng.normal(0, 0.15)
    coefficients[:, :, 1:] = rng.normal(0, 0.03, size=(n_states, n_states - 1, p - 1))
    emissionprob = np.asarray(
        [[0.78, 0.17, 0.04, 0.01], [0.30, 0.35, 0.29, 0.06]], dtype=float
    )
    emissionprob += rng.uniform(0.0, 0.025, size=emissionprob.shape)
    emissionprob /= emissionprob.sum(axis=1, keepdims=True)
    startprob = np.asarray([0.70, 0.30], dtype=float)
    n_observations = int(sum(len(sequence) for sequence in observation_sequences))
    loglik_trace: list[float] = []
    converged = False
    optimizer_audit: list[dict[str, Any]] = []
    occupancy = np.full(n_states, np.nan)
    for iteration in range(1, max_iter + 1):
        start_counts = np.full(n_states, 0.5, dtype=float)
        emission_counts = np.full((n_states, n_symbols), 0.25, dtype=float)
        xi_parts: list[np.ndarray] = []
        x_parts: list[np.ndarray] = []
        gamma_mass = np.zeros(n_states, dtype=float)
        total_ll = 0.0
        for observations, design in zip(observation_sequences, design_sequences):
            observations = np.asarray(observations, dtype=int)
            ll, gamma, xi, _ = _forward_backward(
                observations, design, startprob, coefficients, emissionprob
            )
            total_ll += ll
            start_counts += gamma[0]
            gamma_mass += gamma.sum(axis=0)
            for symbol in range(n_symbols):
                emission_counts[:, symbol] += gamma[observations == symbol].sum(axis=0)
            xi_parts.append(xi)
            x_parts.append(design)
        loglik_trace.append(float(total_ll))
        occupancy = gamma_mass / gamma_mass.sum()
        startprob = start_counts / start_counts.sum()
        emissionprob = emission_counts / emission_counts.sum(axis=1, keepdims=True)
        coefficients, optimizer_audit = fit_transition_coefficients(
            np.concatenate(x_parts, axis=0),
            np.concatenate(xi_parts, axis=0),
            coefficients,
            ridge=ridge,
            maxiter=120,
        )
        if len(loglik_trace) >= 2:
            gain_per_observation = (loglik_trace[-1] - loglik_trace[-2]) / n_observations
            if gain_per_observation < -1e-5:
                break
            if 0.0 <= gain_per_observation < tolerance_per_observation:
                converged = True
                break
    return NHMMFit(
        n_states=n_states,
        n_symbols=n_symbols,
        coefficients=coefficients,
        startprob=startprob,
        emissionprob=emissionprob,
        ridge=ridge,
        converged=converged,
        iterations=len(loglik_trace),
        loglik_trace=loglik_trace,
        occupancy=occupancy,
        transition_optimization=optimizer_audit,
        seed=seed,
    )


def expected_calibration_error(y: np.ndarray, p: np.ndarray, bins: int = 10) -> float:
    y = np.asarray(y, dtype=float)
    p = np.asarray(p, dtype=float)
    edges = np.linspace(0.0, 1.0, bins + 1)
    result = 0.0
    for index in range(bins):
        if index == bins - 1:
            mask = (p >= edges[index]) & (p <= edges[index + 1])
        else:
            mask = (p >= edges[index]) & (p < edges[index + 1])
        if mask.any():
            result += float(mask.mean() * abs(y[mask].mean() - p[mask].mean()))
    return result


def epsilon_nondominated(
    completion: np.ndarray,
    pi_risk: np.ndarray,
    *,
    epsilon_completion: float,
    epsilon_risk: float,
) -> np.ndarray:
    """Return the epsilon-nondominated mask (completion max, PI risk min)."""

    completion = np.asarray(completion, dtype=float)
    pi_risk = np.asarray(pi_risk, dtype=float)
    keep = np.ones(len(completion), dtype=bool)
    for candidate in range(len(completion)):
        for challenger in range(len(completion)):
            if candidate == challenger:
                continue
            nonworse = (
                completion[challenger] >= completion[candidate] - epsilon_completion
                and pi_risk[challenger] <= pi_risk[candidate] + epsilon_risk
            )
            meaningfully_better = (
                completion[challenger] > completion[candidate] + epsilon_completion
                or pi_risk[challenger] < pi_risk[candidate] - epsilon_risk
            )
            if nonworse and meaningfully_better:
                keep[candidate] = False
                break
    return keep


def epsilon_dominates(
    completion_a: float,
    risk_a: float,
    completion_b: float,
    risk_b: float,
    *,
    epsilon_completion: float,
    epsilon_risk: float,
) -> bool:
    nonworse = (
        completion_a >= completion_b - epsilon_completion
        and risk_a <= risk_b + epsilon_risk
    )
    better = (
        completion_a > completion_b + epsilon_completion
        or risk_a < risk_b - epsilon_risk
    )
    return bool(nonworse and better)


__all__ = [
    "FeatureTransformer",
    "ObservableNHMC",
    "NHMMFit",
    "fit_nhmm",
    "categorical_draw",
    "max_true_run",
    "js_divergence",
    "expected_calibration_error",
    "epsilon_nondominated",
    "epsilon_dominates",
]
