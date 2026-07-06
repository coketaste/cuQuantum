#!/usr/bin/env bash
# Run the rocQuantum tensornet benchmark suite on MI3xx and write all
# artefacts under <out_dir>/, matching the layout of h100_runner.sh so the
# two trees can be diffed via compare_perf.py.
#
# Usage:
#   verification/benchmarks/mi300_runner.sh [out_dir]
#
# Output layout (out_dir defaults to verification/benchmarks/results/mi300):
#   out_dir/
#     sweep.log                  human-readable timing summary (tee'd live)
#     data/*.jsonl                raw rocQuantum benchmark output (one JSON
#                                 object per line, already in the normalized
#                                 verification/benchmarks/schema.json shape —
#                                 see bench_tensornet_decompose.cpp and
#                                 bench_tensornet_qft.cpp in the rocQuantum
#                                 checkout)
#     reference/mi300.json       normalized reference (mirrors
#                                 reference/h100.json), written by
#                                 normalize_rocquantum.py
#     reference/mi300.csv        CSV mirror of the same
#
# The two workloads run here are the ones that have a directly comparable
# NVIDIA-side counterpart in h100_runner.sh:
#   - tensor_decompose (SVD)  : bench_tensornet_decompose, same matrix shapes
#                               (256/512/1024) as h100_runner.sh's
#                               cutensornet.tensor_decompose sweep. See that
#                               binary's header comment for the precision
#                               caveat (complex128 here vs real fp32 on H100).
#   - qft_mps (QFT, MPS path) : bench_tensornet_qft, an approximate analogue
#                               of h100_runner.sh's `circuit --benchmark qft
#                               --backend cutn` sweep — same logical circuit,
#                               different contraction strategy (MPS+SVD
#                               truncation here vs whole-network contraction
#                               there). Recorded under the distinct
#                               "cutensornet_mps_approx"/"qft_mps" library/api
#                               keys so compare_perf.py never joins it with
#                               cuTensorNet's own "cutensornet"/"qft" records
#                               as if they were the same measurement.
#
# rocTensorNet's other benchmarks (mps_random, dmrg, tdvp, net_chain, gemm,
# permuted, batched — see rocQuantum/benchmarks/METHODOLOGY.md) have no
# cuTensorNet upstream counterpart and are intentionally out of scope here;
# they stay in rocQuantum's own CSV-based cross-engine benchmark suite.
#
# Environment overrides:
#   ROCQ_BUILD_DIR     (default ../../../rocQuantum/build relative to this
#                       script) — path to a rocQuantum build directory
#                       containing benchmarks/bench_tensornet_decompose and
#                       benchmarks/bench_tensornet_qft. Build it with:
#                         cmake -S . -B build -DROCQ_BUILD_BENCHMARKS=ON \
#                               -DROCQ_ENABLE_TENSORNET=ON
#                         cmake --build build -j --target \
#                               bench_tensornet_decompose bench_tensornet_qft
#   NREPEATS           (default 10)   passed to bench_tensornet_decompose
#   NWARMUPS           (default 2)    passed to bench_tensornet_decompose
#   QFT_QMIN/QFT_QMAX  (default 8/20) qubit sweep range for bench_tensornet_qft
#   QFT_MAXBOND        (default 64)   MPS max bond dim for bench_tensornet_qft
#   ROCR_VISIBLE_DEVICES             pinned to "0" if unset, to keep numbers
#                                    stable across runs

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
OUT_DIR="${1:-$SCRIPT_DIR/results/mi300}"
mkdir -p "$OUT_DIR/data"
LOG="$OUT_DIR/sweep.log"

NREPEATS="${NREPEATS:-10}"
NWARMUPS="${NWARMUPS:-2}"
QFT_QMIN="${QFT_QMIN:-8}"
QFT_QMAX="${QFT_QMAX:-20}"
QFT_MAXBOND="${QFT_MAXBOND:-64}"
export ROCR_VISIBLE_DEVICES="${ROCR_VISIBLE_DEVICES:-0}"

ROCQ_BUILD_DIR="${ROCQ_BUILD_DIR:-$SCRIPT_DIR/../../../rocQuantum/build}"
BENCH_DECOMPOSE="$ROCQ_BUILD_DIR/benchmarks/bench_tensornet_decompose"
BENCH_QFT="$ROCQ_BUILD_DIR/benchmarks/bench_tensornet_qft"

for bin in "$BENCH_DECOMPOSE" "$BENCH_QFT"; do
  if [[ ! -x "$bin" ]]; then
    cat >&2 <<EOF
[mi300] missing benchmark binary: $bin

Build rocQuantum's tensornet benchmarks first:
  cmake -S . -B build -DROCQ_BUILD_TESTS=ON -DROCQ_ENABLE_COMPAT=OFF \\
        -DROCQ_BUILD_BENCHMARKS=ON
  cmake --build build -j --target bench_tensornet_decompose bench_tensornet_qft

Then either set ROCQ_BUILD_DIR to that build directory, or run this script
from a location where ../../../rocQuantum/build resolves to it.
EOF
    exit 1
  fi
done

{
  echo "================================================================"
  echo " rocQuantum tensornet benchmark sweep on $(hostname) — $(date)"
  echo " ROCR_VISIBLE_DEVICES=$ROCR_VISIBLE_DEVICES NREPEATS=$NREPEATS NWARMUPS=$NWARMUPS"
  echo " ROCQ_BUILD_DIR=$ROCQ_BUILD_DIR"
  echo " out=$OUT_DIR"
  echo "================================================================"

  echo
  echo "===== roctensornet.tensor_decompose (gesvd, shapes 256/512/1024) ====="
  "$BENCH_DECOMPOSE" "$NWARMUPS" "$NREPEATS" | tee "$OUT_DIR/data/tensor_decompose.jsonl" \
    | python3 -c '
import json, sys
for line in sys.stdin:
    r = json.loads(line)
    shape = r["config"]["shape"]
    gpu_time_s = r["gpu_time_s"]
    print("  shape={} gpu_time_s={:.6g}".format(shape, gpu_time_s))
'

  echo
  echo "===== roctensornet.qft_mps (n_qubits=$QFT_QMIN..$QFT_QMAX, max_bond=$QFT_MAXBOND) ====="
  "$BENCH_QFT" "$QFT_QMIN" "$QFT_QMAX" "$QFT_MAXBOND" "$NWARMUPS" "$NREPEATS" \
    | tee "$OUT_DIR/data/qft_mps.jsonl" \
    | python3 -c '
import json, sys
for line in sys.stdin:
    r = json.loads(line)
    print("  n_qubits={} gpu_time_s={:.6g}".format(r["n_qubits"], r["gpu_time_s"]))
'

  echo
  echo "===== normalizing to reference/mi300.json ====="
  python3 "$SCRIPT_DIR/normalize_rocquantum.py" \
    --in "$OUT_DIR/data" \
    --out "$OUT_DIR/reference/mi300.json" \
    --csv "$OUT_DIR/reference/mi300.csv"

  echo
  echo "================================================================"
  echo " sweep complete -> $LOG"
  echo " raw JSONL      -> $OUT_DIR/data/*.jsonl"
  echo " normalized     -> $OUT_DIR/reference/mi300.json"
  echo
  echo " Compare against an H100 reference with:"
  echo "   python3 $SCRIPT_DIR/compare_perf.py \\"
  echo "       --reference $SCRIPT_DIR/reference/h100.json \\"
  echo "       --comparand $OUT_DIR/reference/mi300.json"
  echo "================================================================"
} | tee "$LOG"
