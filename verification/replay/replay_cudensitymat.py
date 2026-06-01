"""Replay cuDensityMat oracle data against rocQuantum. Scaffold only."""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from replay._common import load_rocquantum, replay_loop  # noqa: E402

LIBRARY = "cudensitymat"


def run_compute_action(inputs: dict) -> dict:
    rq = load_rocquantum()  # noqa: F841
    rho = np.asarray(inputs["rho"])
    n = int(inputs["n_qubits"])
    n_terms = int(inputs["n_terms"])
    coeffs = np.asarray(inputs["coeffs"])
    rho_out = np.zeros_like(rho)
    for i in range(n_terms):
        op = np.asarray(inputs[f"op_{i}"])
        eye_rest = np.eye(1 << (n - 1), dtype=op.dtype)
        full = np.kron(op, eye_rest)
        rho_out += coeffs[i] * (full @ rho - rho @ full)
    return {"rho_out": rho_out.astype("complex128")}


def run_eigenspectrum(inputs: dict) -> dict:
    rq = load_rocquantum()  # noqa: F841
    h = np.asarray(inputs["hamiltonian"])
    k = int(inputs["k"])
    w, _ = np.linalg.eigh(h)
    return {"eigvals": w[:k].astype("float64")}


RUNNERS = {
    "cudensitymat.compute_action": run_compute_action,
    "cudensitymat.eigenspectrum":  run_eigenspectrum,
}


def main(argv=None) -> int:
    return replay_loop(LIBRARY, RUNNERS, argv)


if __name__ == "__main__":
    raise SystemExit(main())
