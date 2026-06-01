"""Per-API tolerances and comparison metrics for the rocQuantum oracle harness.

This module is intentionally library-agnostic: nothing in here imports
``cuquantum`` or any rocQuantum package. It works on plain NumPy arrays so it
can run on any node, including a CI runner without GPUs.

All metric functions return a ``MetricResult`` dataclass with a boolean
``passed`` flag and a human-readable ``detail`` string. Replay scripts
aggregate these into a final pass/fail tally.
"""
from __future__ import annotations

import dataclasses
from typing import Any

import numpy as np

# Default tolerances per dtype. Override per call from the CLI.
_DEFAULT_TOL = {
    "complex64":  {"atol": 1e-5,  "rtol": 1e-4,  "overlap": 1.0 - 1e-5},
    "complex128": {"atol": 1e-10, "rtol": 1e-9,  "overlap": 1.0 - 1e-9},
    "float32":    {"atol": 1e-5,  "rtol": 1e-4},
    "float64":    {"atol": 1e-10, "rtol": 1e-9},
}


@dataclasses.dataclass
class MetricResult:
    passed: bool
    detail: str
    score: float = float("nan")  # numeric value of the underlying metric

    def __bool__(self) -> bool:  # convenient `if result: ...`
        return self.passed


def _tol(dtype: Any, **overrides: float) -> dict[str, float]:
    name = np.dtype(dtype).name
    base = dict(_DEFAULT_TOL.get(name, {"atol": 1e-6, "rtol": 1e-5}))
    base.update({k: v for k, v in overrides.items() if v is not None})
    return base


# ---------------------------------------------------------------------------
# Pure-state (state-vector) comparisons
# ---------------------------------------------------------------------------

def state_vector_overlap(a: np.ndarray, b: np.ndarray, *, atol: float | None = None,
                         rtol: float | None = None, overlap: float | None = None) -> MetricResult:
    """Compare two state vectors up to a global phase.

    Returns ``passed`` iff |<a|b>| >= overlap_threshold *and* both states have
    unit norm to within ``atol``.
    """
    tol = _tol(a.dtype, atol=atol, rtol=rtol, overlap=overlap)
    a = np.asarray(a).reshape(-1)
    b = np.asarray(b).reshape(-1)
    if a.shape != b.shape:
        return MetricResult(False, f"shape mismatch {a.shape} vs {b.shape}")
    norm_a = float(np.linalg.norm(a))
    norm_b = float(np.linalg.norm(b))
    if abs(norm_a - 1.0) > tol["atol"] or abs(norm_b - 1.0) > tol["atol"]:
        return MetricResult(
            False,
            f"non-unit norms |a|={norm_a:.3e} |b|={norm_b:.3e} (atol={tol['atol']:.1e})",
        )
    inner = abs(complex(np.vdot(a, b)))
    threshold = tol["overlap"]
    return MetricResult(
        inner >= threshold,
        f"|<a|b>|={inner:.12f} threshold={threshold:.12f}",
        score=inner,
    )


def amplitudes_close(a: np.ndarray, b: np.ndarray, *, atol: float | None = None,
                     rtol: float | None = None) -> MetricResult:
    """Element-wise absolute + relative comparison."""
    tol = _tol(a.dtype, atol=atol, rtol=rtol)
    a = np.asarray(a)
    b = np.asarray(b)
    if a.shape != b.shape:
        return MetricResult(False, f"shape mismatch {a.shape} vs {b.shape}")
    diff = np.abs(a - b)
    max_abs = float(diff.max(initial=0.0))
    denom = np.maximum(np.abs(a), np.abs(b))
    with np.errstate(invalid="ignore", divide="ignore"):
        rel = np.where(denom > 0, diff / denom, 0.0)
    max_rel = float(rel.max(initial=0.0))
    ok = np.allclose(a, b, atol=tol["atol"], rtol=tol["rtol"])
    return MetricResult(
        bool(ok),
        f"max_abs={max_abs:.3e} max_rel={max_rel:.3e} atol={tol['atol']:.1e} rtol={tol['rtol']:.1e}",
        score=max_abs,
    )


# ---------------------------------------------------------------------------
# Expectation values and gradients
# ---------------------------------------------------------------------------

def expectation_close(a: np.ndarray | complex | float, b: np.ndarray | complex | float,
                      *, atol: float | None = None, rtol: float | None = None) -> MetricResult:
    """Compare expectation values; relative tolerance is relaxed near zero."""
    a = np.asarray(a)
    b = np.asarray(b)
    tol = _tol(a.dtype, atol=atol, rtol=rtol)
    diff = np.abs(a - b)
    scale = np.maximum(np.abs(a), np.abs(b))
    # Use absolute when |E| is tiny, relative when it is not.
    threshold = np.maximum(tol["atol"], tol["rtol"] * scale)
    ok = bool(np.all(diff <= threshold))
    max_abs = float(diff.max(initial=0.0))
    return MetricResult(ok, f"max|dE|={max_abs:.3e} (mixed atol/rtol)", score=max_abs)


def gradient_close(a: np.ndarray, b: np.ndarray, *, atol: float | None = None,
                   rtol: float | None = None) -> MetricResult:
    return amplitudes_close(a, b, atol=atol, rtol=rtol)


# ---------------------------------------------------------------------------
# Density-matrix and channel comparisons
# ---------------------------------------------------------------------------

def trace_distance(rho_a: np.ndarray, rho_b: np.ndarray, *, atol: float | None = None) -> MetricResult:
    """Trace distance D(rho_a, rho_b) = 0.5 * ||rho_a - rho_b||_1.

    Computes the nuclear norm via SVD (rho_a - rho_b is Hermitian if both inputs
    are valid density matrices, but we don't assume it).
    """
    tol = _tol(rho_a.dtype, atol=atol)
    diff = rho_a - rho_b
    s = np.linalg.svd(diff, compute_uv=False)
    d = 0.5 * float(s.sum())
    ok = d <= tol["atol"] * max(1.0, np.linalg.norm(rho_a))
    return MetricResult(ok, f"trace_distance={d:.3e} atol={tol['atol']:.1e}", score=d)


# ---------------------------------------------------------------------------
# Statistical / sampling comparisons
# ---------------------------------------------------------------------------

def chi_square_distribution(samples_a: np.ndarray, samples_b: np.ndarray,
                            *, alpha: float = 0.001) -> MetricResult:
    """Chi-square test on two empirical bit-string distributions.

    ``samples_a`` and ``samples_b`` are 1D integer arrays of equal-length-encoded
    bit strings (or any small categorical). We pool the bins and assert the
    null (same distribution) is not rejected at significance ``alpha``.
    """
    try:
        from scipy.stats import chisquare
    except ImportError:
        return MetricResult(False, "scipy.stats unavailable; install scipy")

    bins = np.union1d(np.unique(samples_a), np.unique(samples_b))
    obs = np.array([np.sum(samples_a == v) for v in bins], dtype=float)
    exp = np.array([np.sum(samples_b == v) for v in bins], dtype=float)
    # Re-normalise expected to the observed total for chisquare's contract.
    if exp.sum() == 0:
        return MetricResult(False, "empty reference distribution")
    exp = exp * (obs.sum() / exp.sum())
    # Drop bins with expected count < 5 to keep chi-square valid.
    keep = exp >= 5
    if keep.sum() < 2:
        return MetricResult(False, f"too few bins with expected>=5 ({int(keep.sum())})")
    stat, p = chisquare(obs[keep], exp[keep])
    return MetricResult(
        bool(p > alpha),
        f"chi2={stat:.3f} p={p:.4f} alpha={alpha} bins={int(keep.sum())}",
        score=p,
    )


def detector_marginal_close(probs_a: np.ndarray, probs_b: np.ndarray, n_shots: int,
                            *, n_sigma: float = 3.0) -> MetricResult:
    """Compare per-detector marginal probabilities within `n_sigma` of the
    binomial standard error 1/sqrt(N). Used by DEM-sampler verification.
    """
    diff = np.abs(probs_a - probs_b)
    se = 1.0 / np.sqrt(max(n_shots, 1))
    threshold = n_sigma * se
    max_diff = float(diff.max(initial=0.0))
    ok = bool(np.all(diff <= threshold))
    return MetricResult(
        ok,
        f"max|dp|={max_diff:.4f} threshold={threshold:.4f} (n={n_shots}, {n_sigma} sigma)",
        score=max_diff,
    )


# ---------------------------------------------------------------------------
# Pauli expansion (cuPauliProp)
# ---------------------------------------------------------------------------

def pauli_expansion_close(xz_a: np.ndarray, coefs_a: np.ndarray,
                          xz_b: np.ndarray, coefs_b: np.ndarray,
                          *, atol: float | None = None) -> MetricResult:
    """Compare two Pauli expansions on their common support.

    xz_<x> are uint64 packed (X|Z) representations of the Pauli strings;
    coefs_<x> are the matching coefficients.
    """
    tol = _tol(coefs_a.dtype, atol=atol)
    # Build dicts keyed by the xz tuple.
    def _to_dict(xz, c):
        return {tuple(map(int, xz[i])): complex(c[i]) for i in range(len(c))}

    da = _to_dict(xz_a, coefs_a)
    db = _to_dict(xz_b, coefs_b)
    keys = set(da) | set(db)
    max_abs = 0.0
    missing = 0
    for k in keys:
        ca = da.get(k, 0.0 + 0.0j)
        cb = db.get(k, 0.0 + 0.0j)
        if k not in da or k not in db:
            if max(abs(ca), abs(cb)) > tol["atol"]:
                missing += 1
        max_abs = max(max_abs, abs(ca - cb))
    ok = max_abs <= tol["atol"] and missing == 0
    return MetricResult(
        ok,
        f"max|dc|={max_abs:.3e} atol={tol['atol']:.1e} missing_above_tol={missing}",
        score=max_abs,
    )


# ---------------------------------------------------------------------------
# Dispatch table: API name (matching oracle metadata['api']) -> metric fn
# ---------------------------------------------------------------------------

METRICS = {
    # cuStateVec
    "custatevec.apply_matrix":          state_vector_overlap,
    "custatevec.apply_pauli_rotation":  state_vector_overlap,
    "custatevec.compute_expect_pauli":  expectation_close,
    "custatevec.compute_expect":        expectation_close,
    "custatevec.sampler":               chi_square_distribution,
    "custatevec.measure":               state_vector_overlap,
    # cuTensorNet
    "cutensornet.contract":             amplitudes_close,
    "cutensornet.contract_gradient":    gradient_close,
    "cutensornet.tensor_svd":           amplitudes_close,
    "cutensornet.network_state.expect": expectation_close,
    "cutensornet.network_state.sample": chi_square_distribution,
    # cuDensityMat
    "cudensitymat.compute_action":      trace_distance,
    "cudensitymat.eigenspectrum":       amplitudes_close,
    # cuStabilizer
    "custabilizer.dem_sampling":        detector_marginal_close,
    # cuPauliProp
    "cupauliprop.expectation":          expectation_close,
    "cupauliprop.expansion":            pauli_expansion_close,
}


def metric_for(api_name: str):
    if api_name not in METRICS:
        raise KeyError(
            f"no metric registered for API {api_name!r}; "
            f"add one to verification/metrics.py::METRICS"
        )
    return METRICS[api_name]
