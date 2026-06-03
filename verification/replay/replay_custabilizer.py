"""Replay cuStabilizer oracle (DEM sampler) against rocQuantum. Scaffold."""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from replay._common import load_rocquantum, replay_loop  # noqa: E402

LIBRARY = "custabilizer"


def run_dem_sampling(inputs: dict) -> dict:
    """Replay DEM sampling on rocQuantum.

    In fallback mode (no rocQuantum) we use ``stim`` itself with a different
    seed; the harness then validates that two independent ``n_shots`` runs of
    the same DEM agree on per-detector marginals within the chi-square
    threshold (detector_marginal_close in metrics.py).
    """
    rq = load_rocquantum()
    distance = int(inputs["distance"])
    rounds = int(inputs["rounds"])
    prob = float(inputs["prob"])
    n_shots = int(inputs["n_shots"])
    seed = int(inputs["seed"]) + 1   # different seed for an independent run

    # TODO: when rocQuantum is available, replace the stim path with the
    # rocQuantum DEM sampler call. Until then we exercise the harness against
    # a known-good independent sample to demonstrate wiring correctness.
    import stim
    circuit = stim.Circuit.generated(
        "surface_code:rotated_memory_z",
        distance=distance, rounds=rounds,
        after_clifford_depolarization=prob,
        before_round_data_depolarization=prob,
        before_measure_flip_probability=prob,
    )
    dem = circuit.detector_error_model(
        decompose_errors=True, approximate_disjoint_errors=True,
    ).flattened()
    sampler = dem.compile_sampler(seed=seed)
    detectors, _, _ = sampler.sample(shots=n_shots, return_errors=False)
    detectors = np.asarray(detectors, dtype=np.uint8)
    marginals = detectors.mean(axis=0).astype("float64")
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
