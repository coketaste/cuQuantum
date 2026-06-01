#!/usr/bin/env bash
# Run the rocQuantum benchmark harness on MI3xx and dump JSON in the schema
# defined by verification/benchmarks/schema.json so compare.py can pair runs.
#
# Usage:  verification/benchmarks/mi300_runner.sh <out_dir>
#
# This is a placeholder: replace the rocquantum CLI invocation below with
# whatever rocQuantum's equivalent looks like, ensuring that ``run_key``
# values match the H100 side exactly.
set -euo pipefail

OUT_DIR="${1:-verification/benchmarks/results/mi300}"
mkdir -p "$OUT_DIR"
JSON="$OUT_DIR/mi300.json"

NREPEAT="${NREPEAT:-50}"
NWARMUP="${NWARMUP:-3}"

cat <<EOF >&2
[mi300] this runner is a placeholder.
Replace the rocquantum CLI calls below with your own benchmark front-end.
The only invariant compare.py requires is the schema in
verification/benchmarks/schema.json - in particular, "run_key" values must
be byte-equal to the H100 side for matched comparisons.
EOF

# Example skeleton; uncomment once you have rocquantum's CLI:
# for N in 22 24 26 28 30; do
#   for DTYPE in complex64 complex128; do
#     python -m rocquantum_benchmarks api apply_matrix \
#       --nqubits "$N" --ntargets 1 \
#       --precision-sv "$DTYPE" --precision-mat "$DTYPE" \
#       --layout row --location device \
#       --nrepeat "$NREPEAT" --nwarmup "$NWARMUP" --flush-l2 \
#       --benchmark-data "$JSON"
#   done
# done

echo "[mi300] done -> $JSON"
