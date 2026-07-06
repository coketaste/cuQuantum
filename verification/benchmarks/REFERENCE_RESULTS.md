# H100 reference benchmark baseline

Recorded with `bash verification/benchmarks/h100_runner.sh` against
`nv-quantum-benchmarks 0.6.1` and `cuquantum 26.3.2` on an NVIDIA H100
PCIe (driver 580.95.05, CUDA 13.0).

There are **three** artifacts in play; only the last two are committed:

| Artifact | Location | Tracked? | Purpose |
|---|---|---|---|
| Raw nested JSON dumps | `verification/benchmarks/results/h100/data/*.json` | **no** (gitignored) | Faithful upstream output, regenerable via `h100_runner.sh`. Large, noisy (per-record CPU memory, sim_config hashes), unstable byte-for-byte across runs. |
| Normalized reference (machine-readable) | `verification/benchmarks/reference/h100.json` + `.csv` | **yes** | Small (~13 KB), deterministic, sorted by `record_key`, diffable in git. The lib-by-lib source-of-truth for rocQuantum verification. |
| Headline table (human-readable) | this file | **yes** | At-a-glance summary for review. |

This means a `git pull` on the rocQuantum side is enough to get the
H100 baseline — you don't need to ship raw JSON around.

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

## circuit qft + qiskit/cutn (compute-mode=statevector, single)

`h100_runner.sh` also sweeps `--compute-mode statevector` (full evolved
state materialized, not just one amplitude) over n=8..20 — this is the
side of the cross-platform QFT comparison that lines up against
rocQuantum's `bench_tensornet_qft` sweep (see "Cross-platform rocTensorNet
vs cuTensorNet comparison" below). Not yet captured on this H100 baseline
run; re-run `h100_runner.sh` to populate it.

## Reproducing on H100

```bash
source .venv/bin/activate
bash verification/benchmarks/h100_runner.sh                  # raw JSON sweep
python verification/benchmarks/normalize_results.py \
    --in  verification/benchmarks/results/h100/data \
    --out verification/benchmarks/reference/h100.json \
    --csv verification/benchmarks/reference/h100.csv         # refresh reference
```

The normalizer is deterministic (values are rounded to 6 sig figs and
records are sorted by `record_key`), so re-running it against an
unchanged sweep yields a byte-identical file — any change you see in
`git diff` is a real platform-level perf change worth reviewing.

## Lib-by-lib verification on AMD (rocQuantum side)

```bash
# 1. on the AMD host, after running the mirrored MI3xx sweep
python verification/benchmarks/normalize_results.py \
    --in  verification/benchmarks/results/mi300/data \
    --out verification/benchmarks/reference/mi300.json

# 2. compare per library — start with custatevec, then cutensornet, ...
python verification/benchmarks/compare_perf.py \
    --reference verification/benchmarks/reference/h100.json \
    --comparand verification/benchmarks/reference/mi300.json \
    --library   custatevec
```

`compare_perf.py` joins the two sides on `record_key` (a stable identifier
derived from `library.api.n_qubits.precision.<sorted config kv>`), prints
a per-library table of matched pairs with speedup ratios, and flags
records that only exist on one side (`L---` reference-only, `---R`
comparand-only). The AMD-side `mi300.json` is **not** committed here —
that lives in the rocQuantum repo — but the comparator works the same
way regardless of which file lives where.

## Cross-platform rocTensorNet vs cuTensorNet comparison

Two workloads have a directly comparable counterpart on the other
platform: `tensor_decompose` (SVD) and `qft`/`qft_mps` (QFT, approximate —
whole-network contraction on NVIDIA vs MPS+SVD-truncation on AMD; see the
caveat in rocQuantum's `benchmarks/bench_tensornet_qft.cpp` header). Each
side runs independently, on its own hardware, with no cross-machine
access required:

**NVIDIA operator** (this repo, on the H100/GPU node):

```bash
bash verification/benchmarks/h100_runner.sh
python verification/benchmarks/normalize_results.py \
    --in  verification/benchmarks/results/h100/data \
    --out verification/benchmarks/reference/h100.json \
    --csv verification/benchmarks/reference/h100.csv
```

**AMD operator** (rocQuantum checkout, on the MI300X/GPU node) — build the
tensornet benchmarks first (see rocQuantum's `CLAUDE.md`), then run the
self-contained driver, which builds, runs, and normalizes in one step
(it shells out to `normalize_rocquantum.py` itself — no separate
normalize command needed):

```bash
cmake -S . -B build -DROCQ_BUILD_TESTS=ON -DROCQ_ENABLE_COMPAT=OFF \
      -DROCQ_BUILD_BENCHMARKS=ON
cmake --build build -j --target bench_tensornet_decompose bench_tensornet_qft

ROCQ_BUILD_DIR=build \
  /path/to/cuQuantum/verification/benchmarks/mi300_runner.sh /tmp/mi300_results
```

This writes `/tmp/mi300_results/reference/mi300.json` — copy it (or the
whole `reference/` dir) to wherever `compare_perf.py` will run, e.g. next
to `reference/h100.json` in this repo:

```bash
python verification/benchmarks/compare_perf.py \
    --reference verification/benchmarks/reference/h100.json \
    --comparand /tmp/mi300_results/reference/mi300.json
```

`tensor_decompose` records won't match across platforms unless precision
is aligned (rocTensorNet runs complex128 by default; the H100 baseline
above uses single/real fp32) — that shows as reference-only /
comparand-only rows, not a bug. `qft_mps` (AMD) and `qft` (NVIDIA,
statevector mode) sit in separate library groups (`cutensornet_mps_approx`
vs `cutensornet`) by design and are meant to be read side-by-side, not
joined as a literal speedup ratio.
