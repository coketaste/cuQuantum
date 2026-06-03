#!/usr/bin/env bash
# Run the rocQuantum benchmark harness on MI3xx and write all artefacts under
# <out_dir>/, matching the layout of h100_runner.sh so the two trees can be
# diffed.
#
# Usage:
#   verification/benchmarks/mi300_runner.sh [out_dir]
#
# Output layout (out_dir defaults to verification/benchmarks/results/mi300):
#   out_dir/
#     sweep.log                  human-readable timing summary (tee'd live)
#     data/<benchmark>.json      raw benchmark output, ideally in the same
#                                {nqubits: {sim_config_hash: record}} nesting
#                                as nv-quantum-benchmarks 0.6.1 so the same
#                                downstream tooling works
#
# This is a PLACEHOLDER. Replace the rocquantum invocations below with the
# actual rocQuantum benchmark front-end once it is available. The flag set
# below mirrors what h100_runner.sh passes to nv-quantum-benchmarks 0.6.1
# so the matching configurations are obvious.

set -euo pipefail

OUT_DIR="${1:-verification/benchmarks/results/mi300}"
mkdir -p "$OUT_DIR"
LOG="$OUT_DIR/sweep.log"

NREPEATS="${NREPEATS:-10}"
NWARMUPS="${NWARMUPS:-2}"
export ROCR_VISIBLE_DEVICES="${ROCR_VISIBLE_DEVICES:-0}"

CACHE_DIR="$(realpath "$OUT_DIR")"

cat <<EOF >&2
[mi300] runner is a placeholder. Replace the rocquantum_benchmarks calls
below with the real rocQuantum CLI. Configurations were chosen to mirror
verification/benchmarks/h100_runner.sh so per-benchmark pairs line up.
EOF

# Sketch — uncomment and adapt once a rocquantum benchmark CLI exists.
#
# {
#   echo "===== rocstatevec.apply_matrix ====="
#   for N in 20 24 28; do
#     for PREC in single double; do
#       python -m rocquantum_benchmarks api --benchmark apply_matrix \
#           --precision "$PREC" --nqubits "$N" --ntargets 1 \
#           --layout row --location device \
#           --nwarmups "$NWARMUPS" --nrepeats "$NREPEATS" \
#           --cachedir "$CACHE_DIR"
#     done
#   done
#
#   # ... mirror the rest of h100_runner.sh's blocks here ...
# } | tee "$LOG"

echo "[mi300] placeholder exited cleanly (no rocquantum CLI invoked)" >&2
