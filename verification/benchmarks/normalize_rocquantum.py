"""Normalize rocQuantum benchmark JSONL output into the shared reference format.

Counterpart to ``normalize_results.py`` (which normalizes the NVIDIA-side
``nv-quantum-benchmarks`` nested JSON). rocQuantum's benchmarks
(``bench_tensornet_decompose``, ``bench_tensornet_qft``, see the sibling
rocQuantum checkout's ``benchmarks/``) already emit one flat JSON object per
line directly in this schema (see ``schema.json`` in this directory) — no
nested ``{nqubits: {hash: record}}`` unwrapping is needed, only:

  1. computing ``record_key`` with the exact same convention
     ``normalize_results.py``'s ``_record_key`` uses, so ``compare_perf.py``
     can join AMD and NVIDIA records byte-for-byte on the key, and
  2. wrapping the flat record list in the same top-level document shape
     (``platform``, ``host``, ``gpu``, ``captured_at``, ``records``) that
     ``reference/h100.json`` uses.

Usage::

    python verification/benchmarks/normalize_rocquantum.py \
        --in  verification/benchmarks/results/mi300/data \
        --out verification/benchmarks/reference/mi300.json \
        --csv verification/benchmarks/reference/mi300.csv

``--in`` is a directory of ``*.jsonl`` files (one JSON object per line, as
emitted by the rocQuantum benchmark binaries via shell redirection, e.g.
``bench_tensornet_decompose > data/tensor_decompose.jsonl``).
"""
from __future__ import annotations

import argparse
import csv
import datetime as dt
import json
import platform as _platform
import subprocess
from pathlib import Path
from typing import Any


def _round(x: float, sig: int = 6) -> float:
    """Round to ``sig`` significant figures — matches normalize_results.py's
    `_round` so re-runs of either side don't churn the committed JSON."""
    if x == 0 or x != x:  # 0 or NaN
        return x
    from math import floor, log10
    digits = sig - int(floor(log10(abs(x)))) - 1
    return round(x, digits)


def _record_key(rec: dict[str, Any]) -> str:
    """Identical convention to normalize_results.py's `_record_key`: stable,
    hash-free, byte-equal across platforms for the same logical config."""
    parts = [rec["library"], rec["api"]]
    if rec.get("n_qubits") is not None:
        parts.append(f"n={rec['n_qubits']}")
    if rec.get("precision"):
        parts.append(f"prec={rec['precision']}")
    cfg = rec.get("config", {})
    for k in sorted(cfg):
        v = cfg[k]
        if isinstance(v, list):
            v = ",".join(map(str, v))
        parts.append(f"{k}={v}")
    return ".".join(parts)


def _detect_gpu_name() -> str | None:
    """Best-effort AMD GPU name via rocminfo; returns None if unavailable
    (e.g. running this normalize step off-node from a saved data/ dir).
    rocminfo lists CPU agents before GPU agents in the same "Marketing Name:"
    format, so the agent's "Device Type:" line (which follows) must be
    checked too — otherwise this reports the host CPU model instead."""
    for exe in ("rocminfo", "/opt/rocm/bin/rocminfo"):
        try:
            out = subprocess.run([exe], capture_output=True, text=True, timeout=10).stdout
        except (FileNotFoundError, OSError, subprocess.TimeoutExpired):
            continue
        name = None
        for line in out.splitlines():
            line = line.strip()
            if line.startswith("Marketing Name:"):
                name = line.split(":", 1)[1].strip()
            elif line.startswith("Device Type:") and name is not None:
                if line.split(":", 1)[1].strip() == "GPU":
                    return name
                name = None
    return None


def _load_records(in_dir: Path) -> list[dict[str, Any]]:
    files = sorted(in_dir.glob("*.jsonl"))
    if not files:
        raise SystemExit(f"no *.jsonl found under {in_dir}")
    records: list[dict[str, Any]] = []
    for f in files:
        for lineno, line in enumerate(f.read_text().splitlines(), start=1):
            line = line.strip()
            if not line:
                continue
            try:
                rec = json.loads(line)
            except json.JSONDecodeError as e:
                raise SystemExit(f"{f}:{lineno}: invalid JSON: {e}") from e
            for req in ("library", "api", "n_qubits", "gpu_time_s"):
                if req not in rec:
                    raise SystemExit(f"{f}:{lineno}: record missing required field {req!r}")
            if "gpu_time_s" in rec:
                rec["gpu_time_s"] = _round(rec["gpu_time_s"])
            if "cpu_time_s" in rec:
                rec["cpu_time_s"] = _round(rec["cpu_time_s"])
            rec["record_key"] = _record_key(rec)
            records.append(rec)
    records.sort(key=lambda r: r["record_key"])
    return records


def normalize(in_dir: Path) -> dict[str, Any]:
    records = _load_records(in_dir)
    return {
        "platform": "amd",
        "host": _platform.node() or None,
        "gpu": _detect_gpu_name(),
        "captured_at": dt.datetime.now(dt.timezone.utc)
            .isoformat(timespec="seconds").replace("+00:00", "Z"),
        "n_records": len(records),
        "records": records,
    }


def write_csv(doc: dict[str, Any], csv_path: Path) -> None:
    fields = ["record_key", "library", "api", "kind", "n_qubits", "precision",
              "gpu_time_s", "cpu_time_s", "n_warmups", "n_repeats", "config"]
    with csv_path.open("w", newline="") as f:
        w = csv.writer(f)
        w.writerow(fields)
        for r in doc["records"]:
            row = [r.get(k) for k in fields[:-1]]
            row.append(json.dumps(r.get("config", {}), sort_keys=True))
            w.writerow(row)


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--in", dest="in_dir", required=True, type=Path,
                   help="directory of rocQuantum benchmark *.jsonl output.")
    p.add_argument("--out", required=True, type=Path,
                   help="path for the normalized JSON reference.")
    p.add_argument("--csv", type=Path, default=None,
                   help="optional CSV mirror for quick git diffs.")
    args = p.parse_args(argv)

    doc = normalize(args.in_dir)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(doc, indent=2, sort_keys=False) + "\n")
    n_decompose = sum(1 for r in doc["records"] if r["api"] == "tensor_decompose")
    n_qft = sum(1 for r in doc["records"] if r["api"] == "qft_mps")
    print(f"  wrote {args.out} ({args.out.stat().st_size} bytes, "
          f"{doc['n_records']} records, {n_decompose} tensor_decompose, "
          f"{n_qft} qft_mps)")
    if args.csv:
        args.csv.parent.mkdir(parents=True, exist_ok=True)
        write_csv(doc, args.csv)
        print(f"  wrote {args.csv} ({args.csv.stat().st_size} bytes)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
