"""Common helpers for replay/*.py scripts.

Replay scripts walk the ``oracle/`` directory and run rocQuantum on each
captured input, then diff against the captured output via
``verification.metrics``.

The rocQuantum import surface is intentionally centralised in
``load_rocquantum``: change that one function when your package layout
changes. We deliberately avoid importing rocQuantum at module load so this
file works on a CI runner without ROCm available.
"""
from __future__ import annotations

import argparse
import importlib
import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable, Iterator

# Bring in the shared (capture-side) loader. The capture/ package only
# depends on numpy so it is safe to import here.
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from capture._common import load_case  # noqa: E402

# verification/ is on sys.path so we can import metrics directly.
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from metrics import MetricResult, metric_for  # noqa: E402


@dataclass
class ReplayResult:
    api: str
    param_id: str
    passed: bool
    detail: str
    score: float = float("nan")


@dataclass
class ReplayReport:
    results: list[ReplayResult] = field(default_factory=list)

    def add(self, r: ReplayResult) -> None:
        self.results.append(r)

    @property
    def n_passed(self) -> int:
        return sum(1 for r in self.results if r.passed)

    @property
    def n_failed(self) -> int:
        return sum(1 for r in self.results if not r.passed)

    def print_summary(self) -> None:
        for r in self.results:
            tag = "PASS" if r.passed else "FAIL"
            print(f"  {tag} {r.api} {r.param_id}: {r.detail}")
        print(f"summary: {self.n_passed} passed, {self.n_failed} failed,"
              f" {len(self.results)} total")

    def exit_code(self) -> int:
        return 0 if self.n_failed == 0 else 1


import os


class _FallbackSentinel:
    """Marker returned by ``load_rocquantum()`` in fallback mode.

    Runners that receive it should rely on their NumPy reference path rather
    than dispatching into rocQuantum. Accessing any attribute raises a clear
    error so accidental real-binding calls fail loudly instead of silently.
    """

    def __bool__(self) -> bool:  # truthy so `if rq:` works
        return True

    def __repr__(self) -> str:
        return "<rocquantum-fallback>"

    def __getattr__(self, name: str):
        raise AttributeError(
            f"rocquantum is in fallback mode; attribute {name!r} is not "
            f"available. Either install rocquantum or rewrite the runner to "
            f"use its NumPy reference path."
        )


_FALLBACK = _FallbackSentinel()


def load_rocquantum() -> Any:
    """Import rocQuantum.

    Returns a sentinel ``_FALLBACK`` object when the environment variable
    ``ROCQUANTUM_SKIP_IMPORT=1`` is set, which lets the harness exercise the
    runners' NumPy reference paths on hardware that does not have rocQuantum
    installed (e.g. the H100 development node).

    Raises ImportError on failure otherwise, with a helpful message.
    """
    if os.environ.get("ROCQUANTUM_SKIP_IMPORT") == "1":
        return _FALLBACK
    candidates = ["rocquantum", "rocquantum_python", "rocq"]
    last_err: Exception | None = None
    for name in candidates:
        try:
            return importlib.import_module(name)
        except Exception as e:  # ImportError, RuntimeError, etc.
            last_err = e
    raise ImportError(
        "Could not import any of "
        + ", ".join(repr(n) for n in candidates)
        + f". Last error: {last_err!r}. "
        + "Set ROCQUANTUM_SKIP_IMPORT=1 (or pass --use-fallback) to exercise "
        + "the NumPy reference paths only, or edit "
        + "verification/replay/_common.py::load_rocquantum to match your "
        + "rocQuantum package layout."
    )


def iter_oracle(oracle_root: Path, library: str) -> Iterator[Path]:
    """Yield every .npz under <oracle_root>/<library>/."""
    base = Path(oracle_root) / library
    if not base.exists():
        return
    yield from sorted(base.rglob("*.npz"))


def common_argparser(library: str) -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description=f"Replay rocQuantum {library} against oracle")
    p.add_argument("--oracle", type=Path, required=True,
                   help="oracle root, e.g. verification/oracle/v1.0.0")
    p.add_argument("--filter", default=None, help="substring filter on param_id")
    p.add_argument("--rtol", type=float, default=None)
    p.add_argument("--atol", type=float, default=None)
    p.add_argument("--alpha", type=float, default=None,
                   help="significance level for distribution metrics")
    p.add_argument("--list", action="store_true", help="list cases, do not run")
    p.add_argument("--strict", action="store_true",
                   help="exit non-zero if no cases were found.")
    p.add_argument("--use-fallback", action="store_true",
                   help="skip importing rocquantum; runners must use NumPy "
                        "reference paths. Useful for harness self-test.")
    return p


def replay_loop(library: str, runners: dict[str, Callable[[dict], dict]],
                argv: list[str] | None = None) -> int:
    """Generic replay loop. Each runner takes the captured inputs dict and
    must return a dict shaped like the captured outputs."""
    p = common_argparser(library)
    args = p.parse_args(argv)
    if args.use_fallback:
        os.environ["ROCQUANTUM_SKIP_IMPORT"] = "1"
    report = ReplayReport()

    cases = list(iter_oracle(args.oracle, library))
    if not cases:
        print(f"[replay] no oracle cases found under {args.oracle}/{library}")
        return 1 if args.strict else 0

    # Lazy import of rocQuantum: only when we actually need to run.
    rq = None

    if args.list:
        n_listed = 0
        for npz in cases:
            case = load_case(npz)
            if args.filter and args.filter not in case["param_id"]:
                continue
            print(f"  {case['api']} {case['param_id']}")
            n_listed += 1
        print(f"listed {n_listed} cases under {args.oracle}/{library}")
        return 0

    for npz in cases:
        case = load_case(npz)
        api = case["api"]
        pid = case["param_id"]
        if args.filter and args.filter not in pid:
            continue

        if api not in runners:
            report.add(ReplayResult(api, pid, False, f"no runner registered for {api}"))
            continue

        if rq is None:
            rq = load_rocquantum()

        try:
            outputs = runners[api](case["inputs"])
        except Exception as e:
            report.add(ReplayResult(api, pid, False, f"runner raised: {e!r}"))
            continue

        try:
            result = _compare(api, case["outputs"], outputs,
                              rtol=args.rtol, atol=args.atol, alpha=args.alpha,
                              metadata=case.get("metadata", {}))
        except Exception as e:
            report.add(ReplayResult(api, pid, False,
                                    f"metric raised: {type(e).__name__}: {e}"))
            continue
        report.add(ReplayResult(api, pid, result.passed, result.detail, result.score))

    report.print_summary()
    return report.exit_code()


def _compare(api: str, expected: dict, actual: dict, *,
             rtol: float | None, atol: float | None, alpha: float | None,
             metadata: dict) -> MetricResult:
    fn = metric_for(api)
    # Each metric expects a particular signature; we route on api here. Keep
    # this small and explicit rather than getting clever.
    if api in ("custatevec.apply_matrix", "custatevec.measure"):
        return fn(expected["psi_out"], actual["psi_out"], atol=atol, rtol=rtol)
    if api == "custatevec.compute_expect_pauli":
        return fn(expected["expectation"], actual["expectation"], atol=atol, rtol=rtol)
    if api == "custatevec.compute_expect":
        return fn(expected["expectation"], actual["expectation"], atol=atol, rtol=rtol)
    if api == "custatevec.sampler":
        return fn(expected["samples"], actual["samples"], alpha=alpha or 0.001)
    if api == "cutensornet.contract":
        return fn(expected["result"], actual["result"], atol=atol, rtol=rtol)
    if api == "cutensornet.contract_gradient":
        return fn(expected["grad"], actual["grad"], atol=atol, rtol=rtol)
    if api == "cutensornet.tensor_svd":
        # Compare singular values (the unitary factors are gauge-dependent).
        return fn(expected["s"], actual["s"], atol=atol, rtol=rtol)
    if api == "cutensornet.network_state.expect":
        return fn(expected["expectation"], actual["expectation"], atol=atol, rtol=rtol)
    if api == "cutensornet.network_state.sample":
        return fn(expected["samples"], actual["samples"], alpha=alpha or 0.001)
    if api == "cudensitymat.compute_action":
        return fn(expected["rho_out"], actual["rho_out"], atol=atol)
    if api == "cudensitymat.eigenspectrum":
        return fn(expected["eigvals"], actual["eigvals"], atol=atol, rtol=rtol)
    if api == "custabilizer.dem_sampling":
        n_shots = int(expected.get("n_shots", actual.get("n_shots", 1)))
        return fn(expected["detector_marginals"], actual["detector_marginals"], n_shots)
    if api == "cupauliprop.expectation":
        return fn(expected["expectation"], actual["expectation"], atol=atol, rtol=rtol)
    if api == "cupauliprop.expansion":
        return fn(expected["xz"], expected["coefs"],
                  actual["xz"], actual["coefs"], atol=atol)
    return MetricResult(False, f"comparison logic missing for {api}")
