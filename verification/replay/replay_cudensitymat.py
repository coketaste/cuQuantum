"""Replay cuDensityMat oracle data against rocQuantum. Scaffold only."""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from replay._common import load_rocquantum, replay_loop  # noqa: E402

LIBRARY = "cudensitymat"


def _build_full_hamiltonian(dims: tuple[int, ...],
                            h_per_mode: list[np.ndarray]) -> np.ndarray:
    """H_full = sum_i I ⊗ ... ⊗ H_i ⊗ ... ⊗ I."""
    d_total = int(np.prod(dims))
    H_full = np.zeros((d_total, d_total), dtype="complex128")
    for i, h_i in enumerate(h_per_mode):
        op = np.array([[1.0]], dtype="complex128")
        for j, d in enumerate(dims):
            op = np.kron(op, h_i if j == i else np.eye(d, dtype="complex128"))
        H_full += op
    return H_full


def run_compute_action(inputs: dict) -> dict:
    """Replay: rho_out = H_full @ rho for H_full = sum_i H_i on mode i.

    Mirrors the capture's cuDensityMat ``Operator.compute_action`` semantics.
    """
    rq = load_rocquantum()  # noqa: F841
    dims = tuple(int(x) for x in np.asarray(inputs["hilbert_space_dims"]).tolist())
    rho = np.asarray(inputs["rho"], dtype="complex128")
    h_per_mode = [np.asarray(inputs[f"h_{i}"], dtype="complex128")
                  for i in range(len(dims))]
    H_full = _build_full_hamiltonian(dims, h_per_mode)
    rho_out = H_full @ rho
    return {"rho_out": rho_out.astype("complex128")}


def run_eigenspectrum(inputs: dict) -> dict:
    """Replay: k smallest eigenvalues of H_full via dense diagonalisation."""
    rq = load_rocquantum()  # noqa: F841
    dims = tuple(int(x) for x in np.asarray(inputs["hilbert_space_dims"]).tolist())
    k = int(np.asarray(inputs["k"]))
    h_per_mode = [np.asarray(inputs[f"h_{i}"], dtype="complex128")
                  for i in range(len(dims))]
    H_full = _build_full_hamiltonian(dims, h_per_mode)
    w = np.linalg.eigvalsh(H_full)
    return {"eigvals": np.sort(w)[:k].astype("float64")}


RUNNERS = {
    "cudensitymat.compute_action": run_compute_action,
    "cudensitymat.eigenspectrum":  run_eigenspectrum,
}


def main(argv=None) -> int:
    return replay_loop(LIBRARY, RUNNERS, argv)


if __name__ == "__main__":
    raise SystemExit(main())
