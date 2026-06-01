#!/usr/bin/env bash
# Run the cuQuantum benchmark harness on H100 and dump JSON for compare.py.
#
# Usage:  verification/benchmarks/h100_runner.sh <out_dir>
#
# Re-runs are idempotent: each invocation appends to the same JSON file.
set -euo pipefail

OUT_DIR="${1:-verification/benchmarks/results/h100}"
mkdir -p "$OUT_DIR"
JSON="$OUT_DIR/h100.json"

NREPEAT="${NREPEAT:-50}"
NWARMUP="${NWARMUP:-3}"

echo "[h100] writing $JSON"

# 1) apply_matrix sweep
for N in 22 24 26 28 30; do
  for DTYPE in complex64 complex128; do
    python -m nv_quantum_benchmarks api apply_matrix \
      --nqubits "$N" --ntargets 1 \
      --precision-sv "$DTYPE" --precision-mat "$DTYPE" \
      --layout row --location device \
      --nrepeat "$NREPEAT" --nwarmup "$NWARMUP" --flush-l2 \
      --benchmark-data "$JSON"
  done
done

# 2) cuStateVec sampler
for N in 20 24; do
  for SHOTS in 1000 100000 1000000; do
    python -m nv_quantum_benchmarks api cusv_sampler \
      --nqubits "$N" --nshots "$SHOTS" \
      --nrepeat "$NREPEAT" --nwarmup "$NWARMUP" \
      --benchmark-data "$JSON"
  done
done

# 3) Tensor decompose
for SHAPE in 16,256 64,512 32,32,32,32; do
  python -m nv_quantum_benchmarks api tensor_decompose \
    --algorithm svd --shape "$SHAPE" \
    --nrepeat "$NREPEAT" --nwarmup "$NWARMUP" \
    --benchmark-data "$JSON"
done

# 4) End-to-end circuits
for NQ in 22 26 30; do
  for BACKEND in aer-cusv cutn; do
    python -m nv_quantum_benchmarks circuit qft \
      --frontend qiskit --backend "$BACKEND" --nqubits "$NQ" \
      --benchmark-data "$JSON" || true
  done
done

echo "[h100] done -> $JSON"
