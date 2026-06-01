"""Capture cuPauliProp oracle data on H100.

Scaffold: a small 1D kicked-Ising-like circuit and a single-Z observable
back-propagated through it. The reference is computed via NumPy on a small
state vector (n=6) so we can verify rocQuantum against an exact ground truth.
For larger sizes this oracle no longer applies and you must rely on
self-consistency between the two implementations.
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from capture._common import (  # noqa: E402
    OracleCase, common_argparser, get_environment_metadata, make_rng,
    write_case,
)

LIBRARY = "cupauliprop"


def _pauli_matrix(s: str) -> np.ndarray:
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


def gen_kicked_ising_expect(_rng: np.random.Generator, *, small: bool):
    sizes = [4] if small else [4, 6]
    n_steps = 2 if small else 4
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


def run_kicked_ising_expect(inputs: dict) -> dict:
    """Reference: build the full unitary on n_qubits, evolve |0...0>, take <Z_k>.

    Only valid for small n. It exists to give the rocQuantum cuPauliProp
    implementation a deterministic correctness gate at small sizes.
    """
    n = int(inputs["n_qubits"])
    steps = int(inputs["n_trotter_steps"])
    x_angle = float(inputs["x_angle"])
    zz_angle = float(inputs["zz_angle"])
    obs_qubit = int(inputs["obs_qubit"])

    psi = np.zeros(1 << n, dtype="complex128")
    psi[0] = 1.0

    Rx = lambda q: _local_rotation(n, q, "X", x_angle)  # noqa: E731
    Rzz = lambda q1, q2: _local_pair_rotation(n, q1, q2, "Z", "Z", zz_angle)  # noqa: E731

    for _ in range(steps):
        # X layer
        for q in range(n):
            psi = Rx(q) @ psi
        # ZZ chain (1D nearest-neighbour)
        for q in range(n - 1):
            psi = Rzz(q, q + 1) @ psi

    # <Z_obs_qubit>
    z_string = ("I" * obs_qubit) + "Z" + ("I" * (n - obs_qubit - 1))
    Z = _pauli_matrix(z_string)
    ev = complex(np.vdot(psi, Z @ psi))
    return {"expectation": np.asarray(ev)}


def _local_rotation(n: int, q: int, pauli: str, angle: float) -> np.ndarray:
    P = _pauli_matrix(pauli)
    U = np.cos(angle / 2) * np.eye(2, dtype="complex128") - 1j * np.sin(angle / 2) * P
    op = np.eye(1, dtype="complex128")
    for i in range(n):
        op = np.kron(op, U if i == q else np.eye(2, dtype="complex128"))
    return op


def _local_pair_rotation(n: int, q1: int, q2: int, p1: str, p2: str, angle: float) -> np.ndarray:
    base = "I" * n
    base = list(base)
    base[q1] = p1
    base[q2] = p2
    P = _pauli_matrix("".join(base))
    return np.cos(angle / 2) * np.eye(1 << n, dtype="complex128") - 1j * np.sin(angle / 2) * P


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
                metadata={"library": LIBRARY, "kind": "deterministic"},
            )
            write_case(args.out, LIBRARY, case, env)
            n += 1
    print(f"[capture] wrote {n} cases")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
