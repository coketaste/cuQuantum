#!/usr/bin/env bash
# Run the nv-quantum-benchmarks suite on H100 and write all artefacts under
# <out_dir>/. Designed for nv-quantum-benchmarks 0.6.1; flag set verified
# against `nv-quantum-benchmarks {api,circuit} --help` on 2026-06-03.
#
# Usage:
#   verification/benchmarks/h100_runner.sh [out_dir]
#
# Output layout (out_dir defaults to verification/benchmarks/results/h100):
#   out_dir/
#     sweep.log                  human-readable timing summary (tee'd live)
#     data/<benchmark>.json      raw nv-quantum-benchmarks output (nested
#                                {nqubits: {sim_config_hash: record}})
#     circuits/                  pickled circuit fixtures cached by the suite
#
# Environment overrides:
#   NREPEATS   (default 10)    --nrepeats passed to every invocation
#   NWARMUPS   (default 2)     --nwarmups passed to every invocation
#   CUDA_VISIBLE_DEVICES       pinned to "0" if unset, to keep numbers stable

set -euo pipefail

OUT_DIR="${1:-verification/benchmarks/results/h100}"
mkdir -p "$OUT_DIR"
LOG="$OUT_DIR/sweep.log"

NREPEATS="${NREPEATS:-10}"
NWARMUPS="${NWARMUPS:-2}"
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}"

# Resolve --cachedir to an absolute path so nv-quantum-benchmarks writes
# data/*.json under $OUT_DIR regardless of the caller's CWD. (The suite
# default is `.`, which silently creates a root-level data/ directory.)
CACHE_DIR="$(realpath "$OUT_DIR")"

# Run one benchmark command, capturing the [CPU]/[GPU] averaged elapsed
# time lines and printing them with a one-line label. A non-zero exit from
# the underlying tool is reported but does not abort the whole sweep.
run() {
  local label="$1"; shift
  local raw status out
  set +e
  raw="$("$@" 2>&1)"
  status=$?
  set -e
  if [[ $status -ne 0 ]]; then
    local last_err
    last_err="$(echo "$raw" | grep -E 'Error|Assertion|Traceback' | tail -1)"
    echo "  $label : FAILED (exit=$status) ${last_err}"
    return 0
  fi
  out="$(echo "$raw" | grep -E '\[(CPU|GPU)\] Averaged' | sed 's/.*INFO\s*-\s*//' || true)"
  if [[ -z "$out" ]]; then
    echo "  $label : (no timing line emitted; check the suite invocation)"
  else
    echo "  $label :"
    echo "$out" | sed 's/^/    /'
  fi
}

{
  echo "================================================================"
  echo " nv-quantum-benchmarks sweep on $(hostname) — $(date)"
  echo " CUDA_VISIBLE_DEVICES=$CUDA_VISIBLE_DEVICES NREPEATS=$NREPEATS NWARMUPS=$NWARMUPS"
  echo " out=$CACHE_DIR"
  echo "================================================================"

  echo
  echo "===== custatevec.apply_matrix (--ntargets 1, layout=row, device) ====="
  for N in 20 24 28; do
    for PREC in single double; do
      run "nqubits=$N prec=$PREC" \
        nv-quantum-benchmarks api --benchmark apply_matrix \
            --precision "$PREC" --nqubits "$N" --ntargets 1 \
            --layout row --location device \
            --nwarmups "$NWARMUPS" --nrepeats "$NREPEATS" \
            --cachedir "$CACHE_DIR"
    done
  done

  echo
  echo "===== custatevec.apply_matrix with controls ====="
  # --ntargets/--ncontrols pick overlapping qubits by default, so we pass
  # explicit disjoint index lists.
  for N in 22 26; do
    run "nqubits=$N targets=0,1 controls=2,3" \
      nv-quantum-benchmarks api --benchmark apply_matrix \
          --precision single --nqubits "$N" \
          --targets 0,1 --controls 2,3 \
          --layout row --location device \
          --nwarmups "$NWARMUPS" --nrepeats "$NREPEATS" \
          --cachedir "$CACHE_DIR"
  done

  echo
  echo "===== custatevec.cusv_sampler (output=ascending) ====="
  for N in 20 24; do
    for SHOTS in 1000 100000; do
      run "nqubits=$N shots=$SHOTS" \
        nv-quantum-benchmarks api --benchmark cusv_sampler \
            --precision single --nqubits "$N" --nbit-ordering "$N" \
            --nshots "$SHOTS" --output-order ascending \
            --nwarmups "$NWARMUPS" --nrepeats $(( NREPEATS / 2 + 1 )) \
            --cachedir "$CACHE_DIR"
    done
  done

  echo
  echo "===== cutensornet.tensor_decompose (gesvd, single) ====="
  for SHAPE in 256,256 512,512 1024,1024; do
    run "shape=$SHAPE" \
      nv-quantum-benchmarks api --benchmark tensor_decompose \
          --precision single --algorithm gesvd \
          --expr "ab->ax,xb" --shape "$SHAPE" \
          --nwarmups "$NWARMUPS" --nrepeats "$NREPEATS" \
          --cachedir "$CACHE_DIR"
  done

  echo
  echo "===== circuit qft + qiskit/cutn (end-to-end cuTensorNet) ====="
  for NQ in 12 16 20; do
    run "nqubits=$NQ" \
      nv-quantum-benchmarks circuit --benchmark qft \
          --frontend qiskit --backend cutn --nqubits "$NQ" --ngpus 1 \
          --compute-mode amplitude \
          --nwarmups 1 --nrepeats 3 \
          --cachedir "$CACHE_DIR"
  done

  echo
  echo "================================================================"
  echo " sweep complete -> $LOG"
  echo " raw JSON       -> $CACHE_DIR/data/*.json"
  echo "================================================================"
} | tee "$LOG"
