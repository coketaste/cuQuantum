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
#   NREPEATS       (default 10)             --nrepeats passed to every invocation
#   NWARMUPS       (default 2)              --nwarmups passed to every invocation
#   CUDA_VISIBLE_DEVICES                    pinned to "0" if unset, to keep numbers stable
#   APPLY_MATRIX_NQUBITS (default "20 24 28 30")
#       qubit counts for the custatevec.apply_matrix sweep. At n<=28 the
#       state vector (<=4.3 GB double) is small enough that GPU kernel-launch/
#       sync overhead dominates over actual memory-bandwidth-bound work,
#       which is why GPU can look *slower* than the CPU reference path in
#       that regime -- this is expected, not a regression (see
#       METHODOLOGY.md-equivalent bandwidth discussion in rocQuantum). n=30
#       (8.6/17.2 GB single/double) is added by default so the sweep crosses
#       into the regime where GPU HBM bandwidth should start winning over
#       host DRAM bandwidth. Add "32" (34.4/68.7 GB) yourself if your host
#       RAM and the H100's HBM (80 GB on PCIe cards) both have headroom --
#       double precision at n=32 needs ~69 GB resident on each side
#       simultaneously (state + a working copy for CPU verification), which
#       exceeds many single-GPU/single-node configurations.

set -euo pipefail

OUT_DIR="${1:-verification/benchmarks/results/h100}"
mkdir -p "$OUT_DIR"
LOG="$OUT_DIR/sweep.log"

NREPEATS="${NREPEATS:-10}"
NWARMUPS="${NWARMUPS:-2}"
APPLY_MATRIX_NQUBITS="${APPLY_MATRIX_NQUBITS:-20 24 28 30}"
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
    last_err="$(echo "$raw" | grep -E 'Error|Assertion|Traceback' | tail -1 || true)"
    echo "  $label : FAILED (exit=$status) ${last_err}"
    echo "$raw" | tail -20 | sed 's/^/    | /'
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
  for N in $APPLY_MATRIX_NQUBITS; do
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
  echo "===== circuit qft + qiskit/cutn, statevector mode (for rocTensorNet MPS comparison) ====="
  # --compute-mode statevector (not amplitude, as in the block above) so the
  # full evolved state is materialized, matching what roctnMpsToStateVector
  # gives on the AMD side. Swept over the same qubit range as
  # bench_tensornet_qft's default sweep (rocQuantum/benchmarks) so
  # normalize_rocquantum.py's records line up n-for-n against this block once
  # both are fed to compare_perf.py. Recorded under cuTensorNet's normal
  # "cutensornet"/"qft" key — the AMD side intentionally uses the distinct
  # "cutensornet_mps_approx"/"qft_mps" key so compare_perf.py keeps the two
  # algorithms in separate table sections instead of joining them as if they
  # were the same measurement (see bench_tensornet_qft.cpp's header comment).
  for NQ in 8 10 12 14 16 18 20; do
    run "nqubits=$NQ" \
      nv-quantum-benchmarks circuit --benchmark qft \
          --frontend qiskit --backend cutn --nqubits "$NQ" --ngpus 1 \
          --compute-mode statevector \
          --nwarmups 1 --nrepeats 3 \
          --cachedir "$CACHE_DIR"
  done

  echo
  echo "================================================================"
  echo " sweep complete -> $LOG"
  echo " raw JSON       -> $CACHE_DIR/data/*.json"
  echo "================================================================"
} | tee "$LOG"
