"""Replay cuPauliProp oracle against rocQuantum. Scaffold."""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from replay._common import load_rocquantum, replay_loop  # noqa: E402

LIBRARY = "cupauliprop"


def run_kicked_ising_expect(inputs: dict) -> dict:
    rq = load_rocquantum()  # noqa: F841
    n = int(inputs["n_qubits"])
    steps = int(inputs["n_trotter_steps"])
    x_angle = float(inputs["x_angle"])
    zz_angle = float(inputs["zz_angle"])  # noqa: F841
    obs_qubit = int(inputs["obs_qubit"])  # noqa: F841

    # TODO: build the rocQuantum PauliExpansion, apply adjoint gates, evaluate
    # <0...0| tilde O |0...0>. Until then, fail loudly.
    raise NotImplementedError("Wire rocQuantum cuPauliProp here.")


RUNNERS = {
    "cupauliprop.expectation": run_kicked_ising_expect,
}


def main(argv=None) -> int:
    return replay_loop(LIBRARY, RUNNERS, argv)


if __name__ == "__main__":
    raise SystemExit(main())
