"""Normalize nv-quantum-benchmarks raw output into a stable per-platform reference.

The upstream JSON format is nested as
``{nqubits: {sim_config_hash: record}}`` and bundles a lot of noise that
varies run-to-run (cpu_phy_mem, cpu_current_freq, sim_config_hash, full
run_env on every entry). For lib-by-lib cross-platform verification we
only need the *comparable* fields, in a deterministic, diffable layout.

Output (single JSON file, plus a CSV mirror):

    {
      "platform": "nvidia",
      "host": "rocm-framework-h100-pcie",
      "gpu": "NVIDIA H100 PCIe",
      "driver": "580.95.05",
      "cuquantum": "26.3.2",
      "library_versions": {"custatevec": 11301, ...},
      "captured_at": "2026-06-03T...",
      "records": [
        {
          "library": "custatevec",
          "api": "apply_matrix",
          "kind": "api",
          "n_qubits": 20,
          "precision": "single",
          "config": {"targets": [0], "controls": [], "layout": "row", ...},
          "gpu_time_s": 1.209e-05,
          "cpu_time_s": 3.82e-06,
          "n_warmups": 2,
          "n_repeats": 10
        },
        ...
      ]
    }

A ``record_key`` is derived deterministically as
``<library>.<api>.n=<n>.<sorted config kv>`` so the AMD-side
``compare_perf.py`` can join row-for-row.

Usage::

    python verification/benchmarks/normalize_results.py \
        --in  verification/benchmarks/results/h100/data \
        --out verification/benchmarks/reference/h100.json

The output JSON is small (~5-15 KB for the standard sweep) and
intentionally diffable in git so platform-level regressions are visible
in PR review.
"""
from __future__ import annotations

import argparse
import csv
import datetime as dt
import json
from pathlib import Path
from typing import Any, Iterable

# ---------------------------------------------------------------------------
# Mapping: which cuQuantum C library backs each benchmark / circuit backend
# ---------------------------------------------------------------------------
API_TO_LIBRARY: dict[str, str] = {
    "apply_matrix":                          "custatevec",
    "apply_generalized_permutation_matrix":  "custatevec",
    "cusv_sampler":                          "custatevec",
    "tensor_decompose":                      "cutensornet",
}

# Subset of ``circuit`` backends that actually exercise a cuQuantum library.
# Others (qsim*, cudaq*, qulacs*, pennylane-*, aer plain) hit different code
# paths and are recorded with library="external" so a rocQuantum side can
# decide whether to compare them.
BACKEND_TO_LIBRARY: dict[str, str] = {
    "aer-cusv":   "custatevec",
    "cusvaer":    "custatevec",
    "cudaq-cusv": "custatevec",
    "qsim-cusv":  "custatevec",
    "cutn":       "cutensornet",
}


# ---------------------------------------------------------------------------
# Config-key whitelist per benchmark (everything else is dropped)
# ---------------------------------------------------------------------------
CONFIG_KEYS_API: dict[str, tuple[str, ...]] = {
    "apply_matrix": ("targets", "controls", "layout", "location",
                     "adjoint", "flush_cache"),
    "apply_generalized_permutation_matrix": ("targets", "controls",
                                             "layout", "location"),
    "cusv_sampler": ("nshots", "bit_ordering", "output_order"),
    "tensor_decompose": ("shape", "expr", "is_complex", "method", "algorithm",
                         "check_reference"),
}

CONFIG_KEYS_CIRCUIT: tuple[str, ...] = (
    "frontend", "backend", "compute-mode", "nshots", "nfused", "precision",
    "ngpus", "ncputhreads", "nhypersamples",
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _sub(d: dict[str, Any], keys: Iterable[str]) -> dict[str, Any]:
    """Return ``{k: d[k]}`` for keys present in ``d``, in iteration order."""
    return {k: d[k] for k in keys if k in d}


def _round(x: float, sig: int = 6) -> float:
    """Round to ``sig`` significant figures so trailing FP noise doesn't
    churn the JSON byte-for-byte on every re-run."""
    if x == 0 or x != x:  # 0 or NaN
        return x
    from math import floor, log10
    digits = sig - int(floor(log10(abs(x)))) - 1
    return round(x, digits)


def _record_key(rec: dict[str, Any]) -> str:
    """Stable, hash-free identifier; should be byte-equal across runs and
    platforms for the same logical configuration."""
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


# ---------------------------------------------------------------------------
# Per-record extraction
# ---------------------------------------------------------------------------

def _api_record(n_qubits: int, raw: dict[str, Any]) -> dict[str, Any] | None:
    api_blob = raw.get("api") or {}
    name = api_blob.get("name")
    if name is None:
        return None
    library = API_TO_LIBRARY.get(name, "unknown")
    config_keys = CONFIG_KEYS_API.get(name, ())
    rec: dict[str, Any] = {
        "library": library,
        "api": name,
        "kind": "api",
        "n_qubits": n_qubits if n_qubits > 0 else None,
        "precision": api_blob.get("precision"),
        "config": _sub(api_blob, config_keys),
        "gpu_time_s": _round(raw["gpu_time"]),
        "cpu_time_s": _round(raw["cpu_time"]),
        "n_warmups": api_blob.get("nwarmups"),
        "n_repeats": api_blob.get("nrepeats"),
        "library_version": api_blob.get("lib_ver"),
    }
    rec["record_key"] = _record_key(rec)
    return rec


def _circuit_record(n_qubits: int, raw: dict[str, Any],
                    benchmark: str) -> dict[str, Any] | None:
    backend_blob = raw.get("backend") or {}
    frontend_blob = raw.get("frontend") or {}
    backend_name = backend_blob.get("name") if isinstance(backend_blob, dict) else backend_blob
    if backend_name is None:
        return None
    frontend_name = (frontend_blob.get("name")
                     if isinstance(frontend_blob, dict) else frontend_blob)
    library = BACKEND_TO_LIBRARY.get(backend_name, "external")
    config = {
        "frontend": frontend_name,
        "backend": backend_name,
        "compute_mode": backend_blob.get("compute_mode")
                        or str(raw.get("compute-mode", "")).rstrip("()"),
    }
    for k in ("nshots", "nfused", "ngpus", "ncputhreads", "nhypersamples"):
        if isinstance(backend_blob, dict) and backend_blob.get(k) is not None:
            config[k] = backend_blob[k]
    rec: dict[str, Any] = {
        "library": library,
        "api": benchmark,
        "kind": "circuit",
        "n_qubits": n_qubits,
        "precision": (backend_blob.get("precision")
                      if isinstance(backend_blob, dict) else None),
        "config": config,
        "gpu_time_s": _round(raw["gpu_time"]),
        "cpu_time_s": _round(raw["cpu_time"]),
        "library_version": (backend_blob.get("version")
                            if isinstance(backend_blob, dict) else None),
    }
    rec["record_key"] = _record_key(rec)
    return rec


# ---------------------------------------------------------------------------
# Driver
# ---------------------------------------------------------------------------

def normalize(in_dir: Path) -> dict[str, Any]:
    files = sorted(in_dir.glob("*.json"))
    if not files:
        raise SystemExit(f"no JSON found under {in_dir}")

    records: list[dict[str, Any]] = []
    env: dict[str, Any] = {}
    lib_versions: dict[str, int] = {}

    for f in files:
        benchmark = f.stem
        nested = json.loads(f.read_text())
        for nq_str, entries in nested.items():
            n_qubits = int(nq_str)
            for _hash, raw in entries.items():
                if "api" in raw:
                    rec = _api_record(n_qubits, raw)
                elif "backend" in raw:
                    rec = _circuit_record(n_qubits, raw, benchmark)
                else:
                    continue
                if rec is None:
                    continue
                records.append(rec)

                if not env:
                    re = raw.get("run_env", {})
                    env = {
                        "host": re.get("hostname"),
                        "gpu": re.get("gpu_name"),
                        "cpu": re.get("cpu_name"),
                        "driver": re.get("nvml_driver_ver"),
                        "cuda_runtime": re.get("gpu_runtime_ver"),
                    }
                if rec.get("library_version") and rec["library"] not in lib_versions:
                    lib_versions[rec["library"]] = rec["library_version"]

    records.sort(key=lambda r: r["record_key"])
    return {
        "platform": "nvidia",
        **env,
        "library_versions": lib_versions,
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
                   help="directory of raw nv-quantum-benchmarks JSON dumps "
                        "(typically <out>/data/).")
    p.add_argument("--out", required=True, type=Path,
                   help="path for the normalized JSON reference.")
    p.add_argument("--csv", type=Path, default=None,
                   help="optional CSV mirror for quick git diffs.")
    args = p.parse_args(argv)

    doc = normalize(args.in_dir)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(doc, indent=2, sort_keys=False) + "\n")
    print(f"  wrote {args.out} ({args.out.stat().st_size} bytes, "
          f"{doc['n_records']} records, "
          f"{sum(1 for r in doc['records'] if r['library']=='custatevec')} "
          "custatevec, "
          f"{sum(1 for r in doc['records'] if r['library']=='cutensornet')} "
          "cutensornet)")
    if args.csv:
        args.csv.parent.mkdir(parents=True, exist_ok=True)
        write_csv(doc, args.csv)
        print(f"  wrote {args.csv} ({args.csv.stat().st_size} bytes)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
