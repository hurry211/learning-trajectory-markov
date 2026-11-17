"""Future-window run-event probabilities; no empirical data needed."""
from __future__ import annotations
import numpy as np


def run_risk(kernels: np.ndarray, starts: np.ndarray, run_length: int) -> np.ndarray:
    """P(at least m successive future state-0 observations).

    kernels: [anchor, future_step, source, destination]. Starts are source-state
    probabilities at the landmark. The run counter starts at ZERO: historical
    zero observations do not contribute to the future-only event.
    """
    p = np.asarray(kernels, dtype=float)
    s = np.asarray(starts, dtype=float)
    if p.ndim != 4 or p.shape[-1] != p.shape[-2] or s.shape != (p.shape[0], p.shape[-1]):
        raise ValueError("invalid kernel/start shapes")
    if not isinstance(run_length, (int, np.integer)) or run_length < 1:
        raise ValueError("run_length must be a positive integer")
    if not np.all(np.isfinite(p)) or np.any(p < 0) or not np.allclose(p.sum(-1), 1):
        raise ValueError("kernels must be stochastic")
    if not np.all(np.isfinite(s)) or np.any(s < 0) or not np.allclose(s.sum(-1), 1):
        raise ValueError("starts must be distributions")
    n, horizon, states, _ = p.shape
    if run_length > horizon:
        return np.zeros(n)
    mass = np.zeros((n, states, run_length))
    mass[:, :, 0] = s
    for t in range(horizon):
        new = np.zeros_like(mass)
        new[:, 1:, 0] = np.einsum("as,asd->ad", mass.sum(2), p[:, t, :, 1:])
        if run_length > 1:
            new[:, 0, 1:] = np.einsum("asr,as->ar", mass[:, :, :-1], p[:, t, :, 0])
        mass = new
    return np.clip(1 - mass.sum((1, 2)), 0, 1)


def row_tv_bound(p: np.ndarray, q: np.ndarray) -> np.ndarray:
    """Coupling bound for any path event, for identical starting distributions."""
    delta = (0.5 * np.abs(p - q).sum(-1)).max(-1)
    return 1 - np.prod(1 - delta, axis=-1)


def paired_summary(a: np.ndarray, reference: np.ndarray) -> dict:
    d = np.asarray(a, dtype=float) - np.asarray(reference, dtype=float)
    return {"difference": float(d.mean()), "paired_mcse": float(d.std(ddof=1) / np.sqrt(d.size)),
            "paths": int(d.size)}
