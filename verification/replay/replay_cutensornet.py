"""Replay cuTensorNet oracle data against rocQuantum. Scaffold only -
adapt the runner bodies to your rocQuantum tensor-network bindings.
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from replay._common import load_rocquantum, replay_loop  # noqa: E402

LIBRARY = "cutensornet"


def run_contract(inputs: dict) -> dict:
    rq = load_rocquantum()  # noqa: F841
    einsum = str(inputs["einsum"])
    n = int(inputs["n_operands"])
    operands = [np.asarray(inputs[f"operand_{i}"]) for i in range(n)]
    # TODO: rq.tensornet.contract(einsum, *operands)
    out = np.einsum(einsum, *operands)
    return {"result": np.asarray(out)}


def run_tensor_svd(inputs: dict) -> dict:
    rq = load_rocquantum()  # noqa: F841
    a = np.asarray(inputs["tensor"])
    shape = tuple(map(int, np.asarray(inputs["shape"])))
    if len(shape) > 2:
        a = a.reshape(np.prod(shape[: len(shape) // 2]),
                      np.prod(shape[len(shape) // 2:]))
    u, s, vh = np.linalg.svd(a, full_matrices=False)
    return {"u": u, "s": s.astype("float64"), "vh": vh}


def run_network_state_expect(inputs: dict) -> dict:
    rq = load_rocquantum()  # noqa: F841
    # TODO: rq.tensornet.NetworkState.from_circuit(...).compute_expectation(pauli)
    # Placeholder: the captured value is the truth, so an identity passthrough
    # keeps the harness running while the rocQuantum binding is being built.
    raise NotImplementedError(
        "Wire rocQuantum NetworkState here; remove this raise to enable.")


RUNNERS = {
    "cutensornet.contract":             run_contract,
    "cutensornet.tensor_svd":           run_tensor_svd,
    "cutensornet.network_state.expect": run_network_state_expect,
}


def main(argv=None) -> int:
    return replay_loop(LIBRARY, RUNNERS, argv)


if __name__ == "__main__":
    raise SystemExit(main())
