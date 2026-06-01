"""Replay cuStabilizer oracle (DEM sampler) against rocQuantum. Scaffold."""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from replay._common import load_rocquantum, replay_loop  # noqa: E402

LIBRARY = "custabilizer"


def run_dem_sampling(inputs: dict) -> dict:
    rq = load_rocquantum()  # noqa: F841
    distance = int(inputs["distance"])    # noqa: F841
    rounds = int(inputs["rounds"])        # noqa: F841
    prob = float(inputs["prob"])          # noqa: F841
    n_shots = int(inputs["n_shots"])
    seed = int(inputs["seed"])            # noqa: F841

    # TODO: build the same surface-code DEM and sample on rocQuantum.
    #
    # Placeholder: return zeros so the diff fails loudly; replace with the
    # rocQuantum DEM sampler call.
    n_detectors = (distance ** 2) * rounds  # rough upper bound; refine
    marginals = np.zeros(n_detectors, dtype="float64")
    return {
        "detector_marginals": marginals,
        "n_shots": np.asarray(n_shots, dtype=np.int64),
    }


RUNNERS = {
    "custabilizer.dem_sampling": run_dem_sampling,
}


def main(argv=None) -> int:
    return replay_loop(LIBRARY, RUNNERS, argv)


if __name__ == "__main__":
    raise SystemExit(main())
