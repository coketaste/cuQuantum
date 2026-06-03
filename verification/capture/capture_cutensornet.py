"""Capture cuTensorNet oracle data on H100.

Currently scaffolds three APIs: contract (forward), tensor_svd, and the
high-level NetworkState expectation. Adding the gradient pathway is a one-line
extension once the forward case is solid (use cuquantum.tensornet.Network and
contract_path / autotune / contract / gradient on the same operands).
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

LIBRARY = "cutensornet"


# ---------------------------------------------------------------------------
# Generators
# ---------------------------------------------------------------------------

def gen_contract(rng: np.random.Generator, *, small: bool):
    cases = [
        ("ij,jk->ik", [(3, 2), (2, 3)]),
        ("abc,cde->abde", [(2, 3, 4), (4, 5, 6)]),
    ]
    if not small:
        cases.append(
            ("ab,bc,cd,de->ae",
             [(4, 5), (5, 6), (6, 7), (7, 8)]))
    for einsum, shapes in cases:
        operands = [
            (rng.standard_normal(s) + 1j * rng.standard_normal(s)).astype("complex128")
            for s in shapes
        ]
        pid = f"{einsum.replace(',', '_').replace('->', '__')}_{stable_hash(*operands)}"
        yield pid, {
            "einsum": einsum,
            "n_operands": len(operands),
            **{f"operand_{i}": op for i, op in enumerate(operands)},
        }


def gen_tensor_svd(rng: np.random.Generator, *, small: bool):
    shapes = [(8, 8)] if small else [(8, 8), (16, 32), (4, 4, 4, 4)]
    for shape in shapes:
        a = (rng.standard_normal(shape) + 1j * rng.standard_normal(shape)).astype("complex128")
        pid = f"shape{'x'.join(map(str, shape))}_{stable_hash(a)}"
        yield pid, {"tensor": a, "shape": np.asarray(shape, dtype=np.int64)}


def gen_network_state_expect(rng: np.random.Generator, *, small: bool):
    # Build a small QFT-like circuit and compute one Pauli expectation.
    qubits = [3] if small else [3, 5]
    for n in qubits:
        pid = f"qft_n{n}"
        yield pid, {"n_qubits": n, "pauli": "Z" + "I" * (n - 1)}


# ---------------------------------------------------------------------------
# Runners
# ---------------------------------------------------------------------------

def run_contract(inputs: dict) -> dict:
    """cuTensorNet contract on GPU via cupy operands."""
    import cupy as cp
    from cuquantum.tensornet import contract
    operands_d = [cp.asarray(inputs[f"operand_{i}"])
                  for i in range(int(inputs["n_operands"]))]
    out_d = contract(str(inputs["einsum"]), *operands_d)
    cp.cuda.Stream.null.synchronize()
    return {"result": cp.asnumpy(out_d)}


def run_tensor_svd(inputs: dict) -> dict:
    """cuTensorNet tensor.decompose with the GESVD method on a GPU operand.

    The input tensor is reshaped to a matrix (the standard MPS-style
    bipartition) so the result is directly comparable to ``np.linalg.svd``.
    """
    import cupy as cp
    from cuquantum.tensornet.tensor import decompose, SVDMethod

    a = np.asarray(inputs["tensor"])
    shape = tuple(map(int, inputs["shape"]))
    if len(shape) > 2:
        a2 = a.reshape(int(np.prod(shape[: len(shape) // 2])),
                       int(np.prod(shape[len(shape) // 2:])))
    else:
        a2 = a
    a_d = cp.asarray(a2)
    u_d, s_d, vh_d = decompose(
        "ij->ik,kj", a_d,
        method=SVDMethod(partition=None),
    )
    cp.cuda.Stream.null.synchronize()
    return {
        "u": cp.asnumpy(u_d),
        "s": cp.asnumpy(s_d).astype("float64"),
        "vh": cp.asnumpy(vh_d),
    }


def run_network_state_expect(inputs: dict) -> dict:
    """cuTensorNet NetworkState expectation, default (cuTensorNet) backend."""
    import cupy as cp
    import qiskit
    from cuquantum.tensornet.experimental import NetworkState, TNConfig
    n = int(inputs["n_qubits"])
    circuit = qiskit.circuit.library.QFTGate(n).definition
    state = NetworkState.from_circuit(
        circuit, dtype="complex128", config=TNConfig(num_hyper_samples=4),
        backend="cupy",
    )
    try:
        ev = state.compute_expectation({str(inputs["pauli"]): 1.0})
    finally:
        state.free()
    cp.cuda.Stream.null.synchronize()
    return {"expectation": np.asarray(complex(ev))}


# ---------------------------------------------------------------------------
# Driver
# ---------------------------------------------------------------------------

API_TABLE = [
    ("cutensornet.contract",             gen_contract,              run_contract),
    ("cutensornet.tensor_svd",           gen_tensor_svd,            run_tensor_svd),
    ("cutensornet.network_state.expect", gen_network_state_expect,  run_network_state_expect),
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
                metadata={"library": LIBRARY, "kind": "deterministic",
                          "backend": "cuquantum-gpu"},
            )
            write_case(args.out, LIBRARY, case, env)
            n += 1
    print(f"[capture] wrote {n} cases")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
