"""Compare two benchmark JSON dumps produced under verification/benchmarks/schema.json.

Usage:
    python verification/benchmarks/compare.py \
        --left verification/benchmarks/results/h100 \
        --right verification/benchmarks/results/mi300 \
        --out verification/benchmarks/results/compare.csv

Records are matched by ``run_key``. Unmatched keys are reported but do not fail
the comparison (different platforms exercise slightly different sets of APIs
during early development).
"""
from __future__ import annotations

import argparse
import csv
import json
from collections import defaultdict
from pathlib import Path
from statistics import mean
from typing import Any


def load_records(root: Path) -> list[dict[str, Any]]:
    records: list[dict[str, Any]] = []
    for json_file in sorted(root.rglob("*.json")):
        try:
            data = json.loads(json_file.read_text())
        except Exception as e:
            print(f"[compare] could not parse {json_file}: {e!r}")
            continue
        if isinstance(data, list):
            records.extend(d for d in data if isinstance(d, dict))
        elif isinstance(data, dict):
            records.append(data)
    return records


def group_by_key(records: list[dict[str, Any]]) -> dict[str, list[dict[str, Any]]]:
    by_key: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for r in records:
        k = r.get("run_key")
        if not k:
            # Synthesize a run_key from the most distinctive fields.
            k = ".".join(str(r.get(f, "?")) for f in
                         ("library", "api_or_circuit", "dtype", "n_qubits",
                          "n_targets", "n_shots", "backend"))
        by_key[k].append(r)
    return by_key


def reduce_runs(runs: list[dict[str, Any]]) -> dict[str, Any]:
    """Aggregate multiple repeats into a single record."""
    out: dict[str, Any] = dict(runs[0])
    for f in ("cpu_time_s", "gpu_time_s", "effective_bandwidth_GBs"):
        vals = [r[f] for r in runs if f in r and r[f] is not None]
        if vals:
            out[f] = mean(vals)
    out["n_records"] = len(runs)
    return out


def main(argv=None) -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--left", type=Path, required=True, help="root for platform A (e.g. nvidia)")
    p.add_argument("--right", type=Path, required=True, help="root for platform B (e.g. amd)")
    p.add_argument("--out", type=Path, default=None, help="optional CSV output")
    args = p.parse_args(argv)

    left = group_by_key(load_records(args.left))
    right = group_by_key(load_records(args.right))

    keys = sorted(set(left) | set(right))
    rows: list[dict[str, Any]] = []
    for k in keys:
        lr = reduce_runs(left[k]) if k in left else None
        rr = reduce_runs(right[k]) if k in right else None
        row = {
            "run_key": k,
            "left_gpu_time_s": lr["gpu_time_s"] if lr else None,
            "right_gpu_time_s": rr["gpu_time_s"] if rr else None,
            "left_eff_bw_GBs": lr.get("effective_bandwidth_GBs") if lr else None,
            "right_eff_bw_GBs": rr.get("effective_bandwidth_GBs") if rr else None,
            "left_peak_bw_GBs": lr.get("peak_bandwidth_GBs") if lr else None,
            "right_peak_bw_GBs": rr.get("peak_bandwidth_GBs") if rr else None,
        }
        if lr and rr and lr["gpu_time_s"] and rr["gpu_time_s"]:
            row["speedup_right_over_left"] = lr["gpu_time_s"] / rr["gpu_time_s"]
        if lr and lr.get("effective_bandwidth_GBs") and lr.get("peak_bandwidth_GBs"):
            row["left_bw_efficiency"] = lr["effective_bandwidth_GBs"] / lr["peak_bandwidth_GBs"]
        if rr and rr.get("effective_bandwidth_GBs") and rr.get("peak_bandwidth_GBs"):
            row["right_bw_efficiency"] = rr["effective_bandwidth_GBs"] / rr["peak_bandwidth_GBs"]
        rows.append(row)

    # stdout summary
    print(f"{'run_key':<60} {'left(s)':>10} {'right(s)':>10} {'spdup':>8}"
          f" {'left_bw_eff':>11} {'right_bw_eff':>12}")
    for r in rows:
        print(
            f"{r['run_key']:<60}"
            f" {fmt(r.get('left_gpu_time_s')):>10}"
            f" {fmt(r.get('right_gpu_time_s')):>10}"
            f" {fmt(r.get('speedup_right_over_left')):>8}"
            f" {fmt(r.get('left_bw_efficiency')):>11}"
            f" {fmt(r.get('right_bw_efficiency')):>12}"
        )

    if args.out:
        args.out.parent.mkdir(parents=True, exist_ok=True)
        keys = sorted({k for r in rows for k in r.keys()})
        with args.out.open("w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=keys)
            w.writeheader()
            for r in rows:
                w.writerow(r)
        print(f"[compare] wrote {args.out}")
    return 0


def fmt(x):
    if x is None:
        return "-"
    if isinstance(x, float):
        return f"{x:.3g}"
    return str(x)


if __name__ == "__main__":
    raise SystemExit(main())
