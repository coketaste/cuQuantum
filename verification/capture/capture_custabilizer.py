"""Capture cuStabilizer oracle data on H100.

Currently exercises DEM sampling against a stim-generated surface-code
detector error model. cuStabilizer cross-validates against stim in its own
test suite (test_correctness_wrt_stim_dem_surface_code), so we use stim
itself as the deterministic oracle and capture per-detector marginal
probabilities.
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from capture._common import (  # noqa: E402
    OracleCase, common_argparser, get_environment_metadata, write_case,
)

LIBRARY = "custabilizer"


def gen_dem_sampling(_rng, *, small: bool):
    cases = [(3, 0.001), (5, 0.005)] if small else [(3, 0.001), (5, 0.005), (7, 0.005)]
    n_shots = 50_000 if small else 200_000
    for distance, prob in cases:
        pid = f"surface_d{distance}_p{prob:.3g}_shots{n_shots}"
        yield pid, {
            "distance": distance,
            "rounds": distance,
            "prob": prob,
            "n_shots": n_shots,
            "seed": 0,
        }


def run_dem_sampling(inputs: dict) -> dict:
    """cuStabilizer GPU DEM sampler.

    Builds the surface-code DEM with stim (this is parameter prep, not the
    sampling itself), then samples on the H100 via ``cuquantum.stabilizer
    .DEMSampler`` and returns per-detector marginals. cuStabilizer's own
    test suite cross-validates this against stim's sampler.
    """
    import cupy as cp
    import stim
    from cuquantum.stabilizer import DEMSampler, Options

    distance = int(inputs["distance"])
    rounds = int(inputs["rounds"])
    prob = float(inputs["prob"])
    n_shots = int(inputs["n_shots"])
    seed = int(inputs["seed"])

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

    sampler = DEMSampler(dem, n_shots, options=Options(device_id=0), seed=seed)
    sampler.sample(n_shots, seed=seed)
    detectors_d = sampler.get_outcomes(bit_packed=False)
    cp.cuda.Stream.null.synchronize()
    marginals = (cp.sum(detectors_d, axis=0).get() / float(n_shots)).astype("float64")
    return {
        "detector_marginals": marginals,
        "n_shots": np.asarray(n_shots, dtype=np.int64),
    }


API_TABLE = [
    ("custabilizer.dem_sampling", gen_dem_sampling, run_dem_sampling),
]


def main(argv=None) -> int:
    p = common_argparser(LIBRARY)
    args = p.parse_args(argv)
    env = get_environment_metadata()
    n = 0
    for api, gen, runner in API_TABLE:
        for pid, inputs in gen(None, small=args.small):
            if args.filter and args.filter not in pid:
                continue
            print(f"[capture] {api} {pid}")
            if args.dry_run:
                continue
            outputs = runner(inputs)
            case = OracleCase(
                api=api, param_id=pid, inputs=inputs, outputs=outputs,
                params={"seed": args.seed},
                metadata={"library": LIBRARY, "kind": "distribution",
                          "backend": "cuquantum-gpu"},
            )
            write_case(args.out, LIBRARY, case, env)
            n += 1
    print(f"[capture] wrote {n} cases")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
