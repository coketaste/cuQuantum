# H100 reference benchmark baseline

Recorded with `bash verification/benchmarks/h100_runner.sh` against
`nv-quantum-benchmarks 0.6.1` and `cuquantum 26.3.2` on an NVIDIA H100
PCIe (driver 580.95.05, CUDA 13.0). Raw nested JSON dumps live under
`verification/benchmarks/results/h100/data/` and are gitignored; this file
captures the headline GPU timings so the AMD side has a stable target.

- Host: `rocm-framework-h100-pcie`, AMD EPYC 9534 (256 logical cores)
- GPU 0: NVIDIA H100 PCIe, SM count 114, base clock 1755 MHz
- Settings: `NWARMUPS=2`, `NREPEATS=10` (sampler: 6), `CUDA_VISIBLE_DEVICES=0`
- Date: 2026-06-03

## custatevec.apply_matrix (--ntargets 1, layout=row, location=device)

| n_qubits | precision | CPU time (s) | GPU time (s) |
|---------:|:----------|-------------:|-------------:|
|       20 | single    |    3.822e-06 |    1.209e-05 |
|       20 | double    |    3.855e-06 |    1.383e-05 |
|       24 | single    |    4.122e-06 |    1.547e-04 |
|       24 | double    |    4.312e-06 |    3.015e-04 |
|       28 | single    |    6.203e-06 |    2.352e-03 |
|       28 | double    |    8.613e-06 |    4.705e-03 |

## custatevec.apply_matrix with controls (targets=0,1; controls=2,3; single)

| n_qubits | CPU time (s) | GPU time (s) |
|---------:|-------------:|-------------:|
|       22 |    3.872e-06 |    1.438e-05 |
|       26 |    4.323e-06 |    2.981e-04 |

## custatevec.cusv_sampler (output=ascending, single)

| n_qubits | n_shots | CPU time (s) | GPU time (s) |
|---------:|--------:|-------------:|-------------:|
|       20 |   1 000 |    6.108e-05 |    6.974e-05 |
|       20 | 100 000 |    4.086e-04 |    4.180e-04 |
|       24 |   1 000 |    6.802e-05 |    1.498e-04 |
|       24 | 100 000 |    5.879e-04 |    6.708e-04 |

## cutensornet.tensor_decompose (gesvd, single, expr="ab->ax,xb")

| shape       | CPU time (s) | GPU time (s) |
|:------------|-------------:|-------------:|
| 256x256     |    1.587e-02 |    1.587e-02 |
| 512x512     |    3.978e-02 |    3.979e-02 |
| 1024x1024   |    1.083e-01 |    1.083e-01 |

## circuit qft + qiskit/cutn (compute-mode=amplitude, single)

| n_qubits | CPU time (s) | GPU time (s) |
|---------:|-------------:|-------------:|
|       12 |    3.246e-04 |    4.003e-04 |
|       16 |    5.072e-04 |    6.967e-04 |
|       20 |    7.752e-04 |    1.689e-03 |

## Reproducing

```bash
source .venv/bin/activate
bash verification/benchmarks/h100_runner.sh
# results -> verification/benchmarks/results/h100/{sweep.log, data/*.json}
```

To regenerate this table from a fresh run, look in
`verification/benchmarks/results/h100/sweep.log`.

## Cross-platform comparison

`compare.py` and `verification/benchmarks/schema.json` currently describe a
flat per-record schema. The actual upstream `nv-quantum-benchmarks 0.6.1`
JSON is nested as `{nqubits: {sim_config_hash: record}}`, so a small
normalization step will be needed before `compare.py` can pair MI3xx runs
against this table. That normalizer is a TODO for the AMD side; the
configurations in `h100_runner.sh` and `mi300_runner.sh` are intentionally
mirrored so the matching pairs are obvious.
