"""Replay cuStateVec oracle data against rocQuantum.

Run on an MI3xx node:
    python verification/replay/replay_custatevec.py --oracle verification/oracle/v1.0.0

The runners below are *placeholders*: they translate captured inputs into
rocQuantum API calls. You will need to adjust function names and module
paths to match your rocQuantum bindings. The hooks return a dict shaped
identically to the captured outputs so the dispatcher in
``replay/_common.py`` can compare them automatically.
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from replay._common import load_rocquantum, replay_loop  # noqa: E402

LIBRARY = "custatevec"


# ---------------------------------------------------------------------------
# Runners (rocQuantum-side)
# ---------------------------------------------------------------------------

def run_apply_matrix(inputs: dict) -> dict:
    """Apply ``gate`` to ``psi_in`` on rocQuantum and return ``psi_out``.

    Replace the body with your rocQuantum binding calls. The contract is:
    same inputs in -> same outputs out, with the same dict keys as the
    captured oracle.
    """
    rq = load_rocquantum()
    rocsv = getattr(rq, "rocstatevec", rq)  # adjust to your layout

    psi = np.asarray(inputs["psi_in"]).copy()
    gate = np.asarray(inputs["gate"])
    targets = np.asarray(inputs["targets"], dtype=np.int32)
    n_qubits = int(inputs["n_qubits"])
    adjoint = int(inputs["adjoint"])

    # ----- TODO: rocQuantum binding call ----------------------------------
    # Example pseudocode:
    #
    #   handle = rocsv.create()
    #   psi_d = hip.array(psi)
    #   gate_d = hip.array(gate)
    #   rocsv.apply_matrix(handle, psi_d, n_qubits, gate_d, targets, adjoint, ...)
    #   psi = psi_d.to_host()
    #   rocsv.destroy(handle)
    #
    # Until that exists, fall back to a NumPy reference so the harness is
    # exercised end-to-end. Remove this once the binding is wired up.
    psi = _numpy_apply_gate(psi, gate, targets.tolist(), n_qubits)
    return {"psi_out": psi}


def run_compute_expect_pauli(inputs: dict) -> dict:
    rq = load_rocquantum()  # noqa: F841 - unused until binding exists
    psi = np.asarray(inputs["psi"])
    n = int(inputs["n_qubits"])
    paulis = inputs["paulis"]
    basis = inputs["basis_qubits"]
    # TODO: rocsv.compute_expectations_on_pauli_basis(...)
    expect = np.array([_numpy_pauli_expectation(psi, paulis, basis, n)], dtype=np.float64)
    return {"expectation": expect}


def run_sampler(inputs: dict) -> dict:
    rq = load_rocquantum()  # noqa: F841
    psi = np.asarray(inputs["psi"])
    n = int(inputs["n_qubits"])
    n_shots = int(inputs["n_shots"])
    randnums = np.asarray(inputs["randnums"])
    bit_ordering = np.asarray(inputs["bit_ordering"])
    # TODO: rocsv.sampler_create / preprocess / sample
    samples = _numpy_sample(psi, n, n_shots, randnums, bit_ordering)
    return {"samples": samples}


# ---------------------------------------------------------------------------
# Reference helpers (used as placeholder until rocQuantum bindings are wired)
# ---------------------------------------------------------------------------

def _numpy_apply_gate(psi: np.ndarray, gate: np.ndarray,
                      targets: list[int], n_qubits: int) -> np.ndarray:
    """Apply gate on `targets` qubits of an n-qubit state via reshape+einsum.

    Used only as a placeholder reference. Replace with rocQuantum.
    """
    assert len(psi) == 1 << n_qubits
    k = len(targets)
    # cuStateVec convention: targets are little-endian qubit indices into the
    # linear amplitude array. Bit position 0 is the most rapidly varying.
    perm = list(range(n_qubits))
    # Move target axes to the front (reverse so axis-0 = targets[0])
    others = [q for q in range(n_qubits) if q not in targets]
    new_order = list(reversed(targets)) + list(reversed(others))
    psi_t = psi.reshape([2] * n_qubits).transpose(new_order)
    psi_t = psi_t.reshape((1 << k, 1 << (n_qubits - k)))
    psi_t = gate @ psi_t
    psi_t = psi_t.reshape([2] * n_qubits)
    inv_order = [new_order.index(i) for i in range(n_qubits)]
    psi_t = psi_t.transpose(inv_order)
    return psi_t.reshape(-1)


def _numpy_pauli_expectation(psi: np.ndarray, paulis: np.ndarray,
                             basis: np.ndarray, n_qubits: int) -> float:
    P = {0: np.eye(2, dtype="complex128"),
         1: np.array([[0, 1], [1, 0]], dtype="complex128"),
         2: np.array([[0, -1j], [1j, 0]], dtype="complex128"),
         3: np.array([[1, 0], [0, -1]], dtype="complex128")}
    op = np.eye(1, dtype="complex128")
    for q in range(n_qubits):
        if q in basis:
            op = np.kron(op, P[int(paulis[list(basis).index(q)])])
        else:
            op = np.kron(op, P[0])
    return float(np.real(np.vdot(psi, op @ psi)))


def _numpy_sample(psi: np.ndarray, n_qubits: int, n_shots: int,
                  randnums: np.ndarray, bit_ordering: np.ndarray) -> np.ndarray:
    p = np.abs(psi) ** 2
    cdf = np.cumsum(p)
    idx = np.searchsorted(cdf, randnums)
    return idx.astype(np.int64)


# ---------------------------------------------------------------------------
# Driver
# ---------------------------------------------------------------------------

RUNNERS = {
    "custatevec.apply_matrix":         run_apply_matrix,
    "custatevec.compute_expect_pauli": run_compute_expect_pauli,
    "custatevec.sampler":              run_sampler,
}


def main(argv=None) -> int:
    return replay_loop(LIBRARY, RUNNERS, argv)


if __name__ == "__main__":
    raise SystemExit(main())
