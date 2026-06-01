"""Capture cuStateVec oracle data on H100.

Run:
    python verification/capture/capture_custatevec.py --out verification/oracle/v1.0.0

This is the *worked example* the rest of the harness is modeled on. Three
APIs are exercised: apply_matrix, compute_expectations_on_pauli_basis, sampler.
Each generator yields parametrised cases; each runner calls the cuQuantum
binding and returns the outputs as plain NumPy arrays.

Adding a new API: implement gen_<name>() and run_<name>() and append the pair
to API_TABLE at the bottom.
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from capture._common import (  # noqa: E402
    OracleCase, common_argparser, get_environment_metadata, make_rng,
    random_statevector, random_unitary, stable_hash, write_case,
)

LIBRARY = "custatevec"


# ---------------------------------------------------------------------------
# Test-case generators
# ---------------------------------------------------------------------------

def gen_apply_matrix(rng: np.random.Generator, *, small: bool):
    qubits = [2, 4] if small else [2, 4, 6, 8]
    target_arities = [1, 2]
    dtypes = ["complex128"] if small else ["complex64", "complex128"]
    for n_qubits in qubits:
        for k in target_arities:
            if k > n_qubits:
                continue
            for dtype in dtypes:
                psi = random_statevector(rng, n_qubits, dtype)
                gate = random_unitary(rng, 1 << k, dtype)
                targets = list(range(k))  # least-significant qubits
                pid = f"n{n_qubits}_k{k}_{dtype}_{stable_hash(psi, gate)}"
                yield pid, {
                    "psi_in": psi,
                    "gate": gate,
                    "targets": np.asarray(targets, dtype=np.int32),
                    "n_qubits": n_qubits,
                    "adjoint": 0,
                    "dtype": dtype,
                }


def gen_compute_expect_pauli(rng: np.random.Generator, *, small: bool):
    qubits = [3, 5] if small else [3, 5, 7]
    pauli_strings = [["I"], ["X", "Y"], ["Z", "Z", "X"]]
    for n_qubits in qubits:
        for paulis in pauli_strings:
            if len(paulis) > n_qubits:
                continue
            psi = random_statevector(rng, n_qubits, "complex128")
            basis_qubits = list(range(len(paulis)))
            pid = f"n{n_qubits}_{''.join(paulis)}_{stable_hash(psi)}"
            yield pid, {
                "psi": psi,
                "n_qubits": n_qubits,
                "paulis": np.asarray([_pauli_to_int(p) for p in paulis], dtype=np.int32),
                "basis_qubits": np.asarray(basis_qubits, dtype=np.int32),
                "dtype": "complex128",
            }


def gen_sampler(rng: np.random.Generator, *, small: bool):
    qubits = [4] if small else [4, 6, 8]
    n_shots = 5000 if small else 50000
    for n_qubits in qubits:
        psi = random_statevector(rng, n_qubits, "complex128")
        randnums = rng.uniform(0.0, 1.0, n_shots).astype(np.float64)
        bit_ordering = list(range(n_qubits))
        pid = f"n{n_qubits}_shots{n_shots}_{stable_hash(psi, randnums)}"
        yield pid, {
            "psi": psi,
            "n_qubits": n_qubits,
            "n_shots": n_shots,
            "randnums": randnums,
            "bit_ordering": np.asarray(bit_ordering, dtype=np.int32),
        }


# ---------------------------------------------------------------------------
# Runners (call the cuQuantum binding once per case and return outputs)
# ---------------------------------------------------------------------------

# Pauli enum is 0=I, 1=X, 2=Y, 3=Z, matching cusv.Pauli. Stored as int32 in
# the oracle so downstream consumers don't need to import cuStateVec to read.
_PAULI_INT = {"I": 0, "X": 1, "Y": 2, "Z": 3}


def _pauli_to_int(p: str) -> int:
    return _PAULI_INT[p]


def _data_and_compute_type(dtype_name: str):
    """Return (cudaDataType, ComputeType) for a state-vector dtype name."""
    from cuquantum.bindings.custatevec import ComputeType
    from cuda.bindings.runtime import cudaDataType
    table = {
        "complex64":  (cudaDataType.CUDA_C_32F, ComputeType.COMPUTE_32F),
        "complex128": (cudaDataType.CUDA_C_64F, ComputeType.COMPUTE_64F),
    }
    return table[dtype_name]


def run_apply_matrix(inputs: dict) -> dict:
    import cupy as cp
    from cuquantum.bindings import custatevec as cusv

    dtype = str(inputs["dtype"])
    data_type, compute_type = _data_and_compute_type(dtype)
    sv_d = cp.asarray(inputs["psi_in"]).copy()
    gate_d = cp.asarray(inputs["gate"])
    targets = list(map(int, np.asarray(inputs["targets"]).tolist()))
    n_qubits = int(inputs["n_qubits"])
    adjoint = int(inputs["adjoint"])

    handle = cusv.create()
    try:
        ws_size = cusv.apply_matrix_get_workspace_size(
            handle, data_type, n_qubits,
            gate_d.data.ptr, data_type, cusv.MatrixLayout.ROW,
            adjoint, len(targets), 0, compute_type,
        )
        ws = cp.cuda.alloc(ws_size) if ws_size > 0 else None
        ws_ptr = ws.ptr if ws is not None else 0
        cusv.apply_matrix(
            handle,
            sv_d.data.ptr, data_type, n_qubits,
            gate_d.data.ptr, data_type, cusv.MatrixLayout.ROW, adjoint,
            targets, len(targets),
            [], 0, 0,                       # no controls
            compute_type, ws_ptr, ws_size,
        )
    finally:
        cusv.destroy(handle)
    cp.cuda.Stream.null.synchronize()
    return {"psi_out": cp.asnumpy(sv_d)}


def run_compute_expect_pauli(inputs: dict) -> dict:
    import cupy as cp
    from cuquantum.bindings import custatevec as cusv

    data_type, _ = _data_and_compute_type("complex128")
    psi = cp.asarray(inputs["psi"])
    n_qubits = int(inputs["n_qubits"])
    pauli_ints = list(map(int, np.asarray(inputs["paulis"]).tolist()))
    basis_qubits = list(map(int, np.asarray(inputs["basis_qubits"]).tolist()))

    paulis = [[cusv.Pauli(p) for p in pauli_ints]]
    basis_bits = [basis_qubits]
    n_basis_bits = [len(basis_qubits)]
    expect = np.empty((1,), dtype=np.float64)

    handle = cusv.create()
    try:
        cusv.compute_expectations_on_pauli_basis(
            handle, psi.data.ptr, data_type, n_qubits,
            expect.ctypes.data, paulis, len(paulis),
            basis_bits, n_basis_bits,
        )
    finally:
        cusv.destroy(handle)
    cp.cuda.Stream.null.synchronize()
    return {"expectation": expect}


def run_sampler(inputs: dict) -> dict:
    import cupy as cp
    from cuquantum.bindings import custatevec as cusv

    data_type, _ = _data_and_compute_type("complex128")
    psi = cp.asarray(inputs["psi"])
    n_qubits = int(inputs["n_qubits"])
    n_shots = int(inputs["n_shots"])
    randnums = np.ascontiguousarray(np.asarray(inputs["randnums"], dtype=np.float64))
    bit_ordering = list(map(int, np.asarray(inputs["bit_ordering"]).tolist()))

    bit_strings = np.zeros(n_shots, dtype=np.int64)
    handle = cusv.create()
    try:
        sampler, ws_size = cusv.sampler_create(
            handle, psi.data.ptr, data_type, n_qubits, n_shots,
        )
        ws = cp.cuda.alloc(ws_size) if ws_size > 0 else None
        ws_ptr = ws.ptr if ws is not None else 0
        try:
            cusv.sampler_preprocess(handle, sampler, ws_ptr, ws_size)
            cusv.sampler_sample(
                handle, sampler, bit_strings.ctypes.data,
                bit_ordering, len(bit_ordering),
                randnums.ctypes.data, n_shots,
                cusv.SamplerOutput.ASCENDING_ORDER,
            )
        finally:
            cusv.sampler_destroy(sampler)
    finally:
        cusv.destroy(handle)
    cp.cuda.Stream.null.synchronize()
    return {"samples": bit_strings}


# ---------------------------------------------------------------------------
# Driver
# ---------------------------------------------------------------------------

API_TABLE = [
    ("custatevec.apply_matrix",         gen_apply_matrix,         run_apply_matrix),
    ("custatevec.compute_expect_pauli", gen_compute_expect_pauli, run_compute_expect_pauli),
    ("custatevec.sampler",              gen_sampler,              run_sampler),
]


def main(argv=None) -> int:
    p = common_argparser(LIBRARY)
    args = p.parse_args(argv)
    rng = make_rng(args.seed)
    env = get_environment_metadata()
    n_total = 0
    n_skipped = 0
    for api, gen, runner in API_TABLE:
        for pid, inputs in gen(rng, small=args.small):
            if args.filter and args.filter not in pid:
                n_skipped += 1
                continue
            print(f"[capture] {api} {pid}")
            if args.dry_run:
                continue
            outputs = runner(inputs)
            case = OracleCase(
                api=api, param_id=pid, inputs=inputs, outputs=outputs,
                params={"seed": args.seed},
                metadata={"library": LIBRARY, "kind": _kind_for(api)},
            )
            write_case(args.out, LIBRARY, case, env)
            n_total += 1
    print(f"[capture] wrote {n_total} cases, skipped {n_skipped}")
    return 0


def _kind_for(api: str) -> str:
    if api.endswith("sampler"):
        return "distribution"
    return "deterministic"


if __name__ == "__main__":
    raise SystemExit(main())
