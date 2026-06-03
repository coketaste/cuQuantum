"""Capture cuPauliProp oracle data on H100.

Back-propagates a single-site Z observable through a 1D nearest-neighbor
kicked-Ising circuit using cuPauliProp's pythonic API on the GPU. The system
size is kept small (n ≤ 6) so that an analytical state-vector reference is
feasible for cross-validation. The replay script can use the same exact
small-n reference to confirm rocQuantum's cuPauliProp port agrees with both
cuQuantum and the textbook expectation.
"""
from __future__ import annotations

import sys
from pathlib import Path
from typing import Sequence

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from capture._common import (  # noqa: E402
    OracleCase, common_argparser, get_environment_metadata, make_rng,
    write_case,
)

LIBRARY = "cupauliprop"


# ---------------------------------------------------------------------------
# Generators
# ---------------------------------------------------------------------------

def gen_kicked_ising_expect(_rng: np.random.Generator, *, small: bool):
    sizes = [4] if small else [4, 6]
    n_steps = 2 if small else 3
    angle = np.pi / 4
    for n in sizes:
        pid = f"n{n}_steps{n_steps}_obsZ_mid"
        yield pid, {
            "n_qubits": n,
            "n_trotter_steps": n_steps,
            "x_angle": angle,
            "zz_angle": -np.pi / 2,
            "obs_qubit": n // 2,
        }


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _packed_z_observable(n_qubits: int, obs_qubit: int) -> np.ndarray:
    """Encode Z_{obs_qubit} as a single Pauli string in the cuPauliProp
    bit-packed format. Layout matches the bindings sample."""
    from cuquantum.pauliprop.experimental import get_num_packed_integers
    num_packed = get_num_packed_integers(n_qubits)
    out = np.zeros(2 * num_packed, dtype=np.uint64)
    z_ptr = out[num_packed:]
    int_ind = obs_qubit // 64
    bit_ind = obs_qubit % 64
    z_ptr[int_ind] |= np.uint64(1) << np.uint64(bit_ind)
    return out


def _build_kicked_ising_circuit(n_qubits: int, n_steps: int,
                                x_angle: float, zz_angle: float):
    """1D nearest-neighbor kicked Ising: each Trotter step is an X-rotation
    layer on every qubit followed by a ZZ-rotation layer on every
    (i, i+1) pair."""
    from cuquantum.pauliprop.experimental import PauliRotationGate
    gates = []
    for _ in range(n_steps):
        for q in range(n_qubits):
            gates.append(PauliRotationGate(x_angle, ["X"], [q]))
        for q in range(n_qubits - 1):
            gates.append(PauliRotationGate(zz_angle, ["Z", "Z"], [q, q + 1]))
    return gates


def _statevector_reference(n_qubits: int, n_steps: int,
                           x_angle: float, zz_angle: float,
                           obs_qubit: int) -> complex:
    """Exact NumPy state-vector reference for the same circuit, used for
    capture-side sanity checking and the replay NumPy fallback."""
    def pauli_mat(s: str) -> np.ndarray:
        P = {
            "I": np.eye(2, dtype="complex128"),
            "X": np.array([[0, 1], [1, 0]], dtype="complex128"),
            "Y": np.array([[0, -1j], [1j, 0]], dtype="complex128"),
            "Z": np.array([[1, 0], [0, -1]], dtype="complex128"),
        }
        out = P[s[0]]
        for c in s[1:]:
            out = np.kron(out, P[c])
        return out

    def local_rot(q: int, p: str, angle: float) -> np.ndarray:
        P = pauli_mat(p)
        u = np.cos(angle / 2) * np.eye(2, dtype="complex128") - 1j * np.sin(angle / 2) * P
        op = np.eye(1, dtype="complex128")
        for i in range(n_qubits):
            op = np.kron(op, u if i == q else np.eye(2, dtype="complex128"))
        return op

    def pair_rot(q1: int, q2: int, p1: str, p2: str, angle: float) -> np.ndarray:
        base = list("I" * n_qubits)
        base[q1] = p1
        base[q2] = p2
        P = pauli_mat("".join(base))
        return np.cos(angle / 2) * np.eye(1 << n_qubits, dtype="complex128") - 1j * np.sin(angle / 2) * P

    psi = np.zeros(1 << n_qubits, dtype="complex128")
    psi[0] = 1.0
    for _ in range(n_steps):
        for q in range(n_qubits):
            psi = local_rot(q, "X", x_angle) @ psi
        for q in range(n_qubits - 1):
            psi = pair_rot(q, q + 1, "Z", "Z", zz_angle) @ psi

    z_string = ("I" * obs_qubit) + "Z" + ("I" * (n_qubits - obs_qubit - 1))
    Z = pauli_mat(z_string)
    return complex(np.vdot(psi, Z @ psi))


# ---------------------------------------------------------------------------
# Runner
# ---------------------------------------------------------------------------

def run_kicked_ising_expect(inputs: dict) -> dict:
    """Back-propagate Z_{obs_qubit} through the kicked-Ising circuit using
    cuPauliProp on the GPU. For small systems the expansion stays exact when
    the truncation thresholds are zero, so the result is comparable to the
    state-vector reference up to floating-point precision."""
    import cupy as cp
    from cuquantum.pauliprop.experimental import (
        LibraryHandle, PauliExpansion, PauliExpansionOptions, Truncation,
    )

    n = int(inputs["n_qubits"])
    steps = int(inputs["n_trotter_steps"])
    x_angle = float(inputs["x_angle"])
    zz_angle = float(inputs["zz_angle"])
    obs_qubit = int(inputs["obs_qubit"])

    handle = LibraryHandle()

    xz_h = _packed_z_observable(n, obs_qubit)
    xz = cp.asarray(xz_h.reshape(1, -1))
    coefs = cp.asarray(np.array([1.0], dtype=np.float64))

    options = PauliExpansionOptions(memory_limit="50%", blocking=True)
    expansion = PauliExpansion(handle, n, 1, xz, coefs, options=options)

    # No truncation: keep all terms so the result is exact for small n.
    truncation = Truncation(pauli_coeff_cutoff=0.0, pauli_weight_cutoff=n)

    circuit = _build_kicked_ising_circuit(n, steps, x_angle, zz_angle)
    for gate_index in range(len(circuit) - 1, -1, -1):
        expansion = expansion.apply_gate(
            circuit[gate_index],
            truncation=truncation,
            adjoint=True,
            sort_order=None,
            keep_duplicates=False,
        )

    trace_sig, trace_exp = expansion.trace_with_zero_state()
    expec = complex(trace_sig * np.exp2(trace_exp))
    cp.cuda.Stream.null.synchronize()
    return {"expectation": np.asarray(expec, dtype="complex128")}


# ---------------------------------------------------------------------------
# Driver
# ---------------------------------------------------------------------------

API_TABLE = [
    ("cupauliprop.expectation", gen_kicked_ising_expect, run_kicked_ising_expect),
]


def main(argv=None) -> int:
    p = common_argparser(LIBRARY)
    args = p.parse_args(argv)
    rng = make_rng(args.seed)
    env = get_environment_metadata()
    n = 0
    for api, gen, runner in API_TABLE:
        for pid, inputs in gen(rng, small=args.small):
            if args.filter and args.filter not in pid:
                continue
            print(f"[capture] {api} {pid}")
            if args.dry_run:
                continue
            outputs = runner(inputs)
            case = OracleCase(
                api=api, param_id=pid, inputs=inputs, outputs=outputs,
                params={"seed": args.seed},
                metadata={
                    "library": LIBRARY,
                    "kind": "deterministic",
                    "backend": "cuquantum-gpu",
                },
            )
            write_case(args.out, LIBRARY, case, env)
            n += 1
    print(f"[capture] wrote {n} cases")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
