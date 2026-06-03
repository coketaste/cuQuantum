"""Replay cuPauliProp oracle against rocQuantum. Scaffold."""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from replay._common import load_rocquantum, replay_loop  # noqa: E402

LIBRARY = "cupauliprop"


def run_kicked_ising_expect(inputs: dict) -> dict:
    """Replay kicked-Ising expectation on rocQuantum.

    In fallback mode we recompute the reference via the same exact small-n
    state-vector evolution used in the capture script. Replace with the
    rocQuantum cuPauliProp call once it exists.
    """
    rq = load_rocquantum()
    n = int(inputs["n_qubits"])
    steps = int(inputs["n_trotter_steps"])
    x_angle = float(inputs["x_angle"])
    zz_angle = float(inputs["zz_angle"])
    obs_qubit = int(inputs["obs_qubit"])

    # TODO: switch to rq.pauliprop.* once the rocQuantum binding lands.
    # Reuse the capture-side helpers via direct import for the harness self-test.
    import sys
    from pathlib import Path
    sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "capture"))
    from capture_cupauliprop import run_kicked_ising_expect as _ref  # type: ignore
    return _ref({
        "n_qubits": n, "n_trotter_steps": steps,
        "x_angle": x_angle, "zz_angle": zz_angle, "obs_qubit": obs_qubit,
    })


RUNNERS = {
    "cupauliprop.expectation": run_kicked_ising_expect,
}


def main(argv=None) -> int:
    return replay_loop(LIBRARY, RUNNERS, argv)


if __name__ == "__main__":
    raise SystemExit(main())
