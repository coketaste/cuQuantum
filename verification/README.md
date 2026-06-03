# rocQuantum vs cuQuantum verification harness

This directory implements the verification methodology described at the end of
[../paper/](../paper/): use cuQuantum on NVIDIA hardware as a *correctness oracle*
for rocQuantum on AMD hardware, then compare performance separately.

The workflow is **capture once on H100, replay forever on MI3xx**. We do not
require both libraries to be importable in the same Python process - that is
fragile across CUDA / ROCm and would force us to give up on either CuPy or
HIP-Python.

## Layout

```
verification/
  README.md            (this file)
  metrics.py           per-API tolerance and metric definitions
  compare_oracles.py   diff two oracle directories case-by-case
  capture/             scripts that run cuQuantum on H100 and dump oracle data
    _common.py
    capture_custatevec.py
    capture_cutensornet.py
    capture_cudensitymat.py
    capture_custabilizer.py
    capture_cupauliprop.py
  replay/              scripts that load oracle data and run rocQuantum
    _common.py
    replay_custatevec.py
    replay_cutensornet.py
    replay_cudensitymat.py
    replay_custabilizer.py
    replay_cupauliprop.py
  oracle/              captured (input, output) artefacts (use Git LFS)
    .gitkeep
  benchmarks/          performance harness wrappers
    schema.json
    h100_runner.sh
    mi300_runner.sh
    compare.py
  ci/                  example CI configurations
    correctness.yml
    performance.yml
```

## Phases

### Phase 0 - environment health

Sanity-check the cuQuantum install on H100 before anything else.

```bash
source /home/ysha/cuQuantum/.venv/bin/activate
cd /home/ysha/cuQuantum/python/tests
python -m pytest cuquantum_tests/bindings/test_custatevec.py::TestApply -v -k "not cffi" --maxfail=3
```

If this passes, your oracle environment is ready.

### Phase 1 - oracle capture (run on H100)

All five capture scripts call the real cuQuantum GPU APIs on the H100; every
captured case is tagged ``metadata.backend = "cuquantum-gpu"`` so it's
machine-verifiable that the oracle is a true GPU reference, not a CPU
stand-in. Specifically:

| Library | API | GPU code path |
|---|---|---|
| cuStateVec | apply_matrix, compute_expect_pauli, sampler | `cuquantum.bindings.custatevec` |
| cuTensorNet | contract, tensor_svd, network_state.expect | `cuquantum.tensornet.{contract, tensor.decompose, experimental.NetworkState}` |
| cuDensityMat | compute_action, eigenspectrum | `Operator.compute_action`, `OperatorSpectrumSolver` |
| cuStabilizer | dem_sampling | `cuquantum.stabilizer.DEMSampler` |
| cuPauliProp | expectation | `cuquantum.pauliprop.experimental.PauliExpansion` back-prop |


Each `capture/capture_<lib>.py` is a small program that:

1. Sets a deterministic RNG seed.
2. Iterates over a parametrised set of (problem-size, dtype, gate, ...) tuples.
3. Calls the cuQuantum API.
4. Writes a self-contained `.npz` file to `oracle/<sdk-version>/<lib>/<test-name>/<param-id>.npz`
   containing inputs, outputs, the API parameters, and a metadata dict
   (cuQuantum version, CUDA version, GPU SKU, driver, host hostname).

Run all capture scripts from the repo root after activating the cuQuantum venv:

```bash
python verification/capture/capture_custatevec.py --out verification/oracle/v1.0.0
python verification/capture/capture_cutensornet.py --out verification/oracle/v1.0.0
python verification/capture/capture_cudensitymat.py --out verification/oracle/v1.0.0
python verification/capture/capture_custabilizer.py --out verification/oracle/v1.0.0
python verification/capture/capture_cupauliprop.py --out verification/oracle/v1.0.0
```

The captures are small (<50 MB total for the default seeds and sizes); they go
into Git LFS so other developers can verify rocQuantum without owning H100.

### Phase 2 - rocQuantum replay (run on MI3xx)

Each `replay/replay_<lib>.py` script:

1. Walks `oracle/<sdk-version>/<lib>/` and loads each `.npz`.
2. Calls the rocQuantum equivalent with the same inputs.
3. Computes the per-API metric defined in `metrics.py`.
4. Reports pass/fail per case and a summary at the end; exits non-zero on any
   tolerance violation.

```bash
python verification/replay/replay_custatevec.py --oracle verification/oracle/v1.0.0
```

The rocQuantum import is centralised in `replay/_common.py::load_rocquantum()`;
adjust that one function if your package layout changes.

#### Phase 2b - symmetric oracle capture (alternative)

When the H100 and AMD nodes can't share a Python process (the common case),
use each replay script's `--capture-to <dir>` flag to *persist* the rocQuantum
outputs in the same `.npz + .json` oracle format instead of comparing in
process:

```bash
# on the AMD node, after pulling the H100 oracle
for L in custatevec cutensornet cudensitymat custabilizer cupauliprop; do
    python verification/replay/replay_$L.py \
        --oracle    verification/oracle/v1.0.0-cuquantum-cu13 \
        --capture-to verification/oracle/rocquantum-mi300
done
```

Then on either node compare the two trees offline (no GPU library required):

```bash
python verification/compare_oracles.py \
    --left  verification/oracle/v1.0.0-cuquantum-cu13 \
    --right verification/oracle/rocquantum-mi300
```

`compare_oracles.py` matches cases by `<library>/<api>/<param_id>`, runs the
same per-API metric from `metrics.py`, and reports keys present on only one
side. Pass `--strict` to fail on missing keys.

### Phase 3 - distribution sanity (statistical APIs)

For samplers, trajectory unravellers, DEM samplers we capture seeds and
user-provided uniforms, but the *output distribution* is what we compare.
`metrics.py::sample_distribution_test` runs a chi-square (and KS for ordered
data) at a configurable significance level and returns a pass/fail. The replay
script automatically dispatches to it for any oracle file whose
`metadata['kind'] == 'distribution'`.

### Phase 4 - performance characterisation

We run `nv_quantum_benchmarks` on H100 and the rocQuantum equivalent on AMD
with **identical parameters**, then compare the resulting JSON.

```bash
verification/benchmarks/h100_runner.sh   verification/benchmarks/results/h100/
verification/benchmarks/mi300_runner.sh  verification/benchmarks/results/mi300/
python verification/benchmarks/compare.py \
    --left verification/benchmarks/results/h100 \
    --right verification/benchmarks/results/mi300
```

`compare.py` reports absolute time on each platform, *effective bandwidth
fraction* (`measured / peak_HBM_bandwidth`), and speedup curves.

### Phase 5 - distributed (later)

Multi-node MPI / NCCL / RCCL tests follow the same capture-and-replay model
but with one oracle file per (rank, test) pair.

## Tolerances and metrics

Defined in [metrics.py](metrics.py):

| Object | Metric | Default tol |
|---|---|---|
| State vector | overlap modulo global phase | `\|<a\|b>\| > 1 - 1e-9` (c128), `1 - 1e-5` (c64) |
| Amplitudes | element-wise abs+rel | `atol=1e-10, rtol=1e-9` (c128) |
| Expectation value | absolute, with relaxation near 0 | `atol=1e-9` (c128) |
| Density matrix | trace distance | `< 1e-9` (c128) |
| Bit-string samples | chi-square | `p > 0.001` |
| DEM samples | per-detector marginals | `\|p_a - p_b\| < 3 sigma` |
| Pauli expansion | coefficient diff on common support | `atol=1e-9` |
| Gradient | per-parameter rel | `rtol=1e-7` (c128) |

Override via CLI: `--rtol`, `--atol`, `--alpha` on every capture/replay script.

## What this is NOT

- It is not a port of cuQuantum's pytest suite to ROCm.
- It is not a substitute for rocQuantum having its own internal unit tests.
- It is not a performance gate during correctness debugging - keep those
  pipelines separate (see paper chapter 60).

## Adding new tests

1. Add a parametrised generator function `gen_<api>()` to the matching
   `capture_<lib>.py`. Yield `(param_id_str, inputs_dict)` tuples.
2. Implement the cuQuantum call in `capture_<lib>.py::run_<api>(inputs)`.
3. Implement the rocQuantum call in `replay_<lib>.py::run_<api>(inputs)`.
4. Pick or define the right metric in `metrics.py::METRICS[<api>]`.

That's it - the file naming and oracle directory layout do the rest.

## See also

- [paper/](../paper/) - in-depth chapters on each cuQuantum library.
- [TESTS.md](../TESTS.md) - the upstream test catalogue this harness draws from.
- [paper/60-benchmarks.md](../paper/60-benchmarks.md) - the
  `nv_quantum_benchmarks` harness wrapped by `benchmarks/`.
