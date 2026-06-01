"""Capture cuDensityMat oracle data on H100.

Scaffold only. Two APIs are wired up:
- compute_action: Liouvillian action on a small density matrix.
- eigenspectrum: Krylov spectrum of an operator.

Both use small system sizes so the resulting ``.npz`` files stay small. The
runners are deliberately sketchy; flesh them out as you build the rocQuantum
counterparts and learn which corner cases matter to your codebase.
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from capture._common import (  # noqa: E402
    OracleCase, common_argparser, get_environment_metadata, make_rng,
    stable_hash, write_case,
)

LIBRARY = "cudensitymat"


def gen_compute_action(rng: np.random.Generator, *, small: bool):
    sizes = [(2, 4)] if small else [(2, 4), (3, 5)]
    for n_qubits, n_terms in sizes:
        d = 1 << n_qubits
        rho = rng.standard_normal((d, d)) + 1j * rng.standard_normal((d, d))
        rho = rho @ rho.conj().T
        rho /= np.trace(rho).real
        coeffs = rng.standard_normal(n_terms) + 1j * rng.standard_normal(n_terms)
        # Per-term local 2x2 Hermitian operators (Pauli-ish).
        ops = []
        for _ in range(n_terms):
            h = rng.standard_normal((2, 2)) + 1j * rng.standard_normal((2, 2))
            h = h + h.conj().T
            ops.append(h.astype("complex128"))
        pid = f"n{n_qubits}_t{n_terms}_{stable_hash(rho, coeffs, *ops)}"
        yield pid, {
            "rho": rho.astype("complex128"),
            "n_qubits": n_qubits,
            "n_terms": n_terms,
            "coeffs": coeffs.astype("complex128"),
            **{f"op_{i}": op for i, op in enumerate(ops)},
        }


def gen_eigenspectrum(rng: np.random.Generator, *, small: bool):
    sizes = [4] if small else [4, 8]
    for d in sizes:
        h = rng.standard_normal((d, d)) + 1j * rng.standard_normal((d, d))
        h = (h + h.conj().T) / 2
        pid = f"d{d}_{stable_hash(h)}"
        yield pid, {"hamiltonian": h.astype("complex128"), "k": min(3, d - 1)}


def run_compute_action(inputs: dict) -> dict:
    """CPU oracle: build full L explicitly, apply once. Replace with
    cuquantum.densitymat.Operator.compute_action when you want to specifically
    validate the cuDensityMat code path."""
    rho = np.asarray(inputs["rho"])
    n_qubits = int(inputs["n_qubits"])
    n_terms = int(inputs["n_terms"])
    coeffs = np.asarray(inputs["coeffs"])
    rho_out = np.zeros_like(rho)
    for i in range(n_terms):
        op = np.asarray(inputs[f"op_{i}"])
        # Embed op on qubit 0 of an n_qubit system.
        eye_rest = np.eye(1 << (n_qubits - 1), dtype=op.dtype)
        full = np.kron(op, eye_rest)
        rho_out += coeffs[i] * (full @ rho - rho @ full)
    return {"rho_out": rho_out.astype("complex128")}


def run_eigenspectrum(inputs: dict) -> dict:
    h = np.asarray(inputs["hamiltonian"])
    k = int(inputs["k"])
    w, _ = np.linalg.eigh(h)
    return {"eigvals": w[:k].astype("float64")}


API_TABLE = [
    ("cudensitymat.compute_action", gen_compute_action, run_compute_action),
    ("cudensitymat.eigenspectrum",  gen_eigenspectrum,  run_eigenspectrum),
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
