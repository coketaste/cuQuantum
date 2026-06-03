"""Compare two normalized benchmark references, library by library.

Both inputs must be JSON files produced by ``normalize_results.py`` (one
``records`` list, each with a ``record_key``). Typically the reference
side is the committed H100 baseline
(``verification/benchmarks/reference/h100.json``) and the comparand is a
fresh rocQuantum run on AMD, normalized into the same schema.

Output is a per-library table showing matched pairs and the speedup
ratio (left / right). Use ``--library custatevec`` to focus on one
library at a time; that's how rocQuantum lib-by-lib verification will
typically be driven.

Example::

    python verification/benchmarks/compare_perf.py \
        --reference verification/benchmarks/reference/h100.json \
        --comparand verification/benchmarks/reference/mi300.json \
        --library custatevec
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any


def _load(path: Path) -> tuple[dict[str, Any], dict[str, dict[str, Any]]]:
    doc = json.loads(path.read_text())
    by_key = {r["record_key"]: r for r in doc["records"]}
    return doc, by_key


def _fmt(x: float | None) -> str:
    if x is None:
        return "    --   "
    if x == 0 or abs(x) < 1e-9:
        return f"{x: .3e}"
    if abs(x) < 1e-3:
        return f"{x * 1e6:7.2f} µs"
    if abs(x) < 1:
        return f"{x * 1e3:7.2f} ms"
    return f"{x:7.3f} s "


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--reference", required=True, type=Path,
                   help="normalized JSON taken as the baseline (e.g. H100).")
    p.add_argument("--comparand", required=True, type=Path,
                   help="normalized JSON to diff against the baseline (e.g. MI3xx).")
    p.add_argument("--library", default=None,
                   help="restrict to one library (custatevec, cutensornet, ...).")
    p.add_argument("--api", default=None,
                   help="restrict to one api / circuit name.")
    args = p.parse_args(argv)

    ref_doc, ref = _load(args.reference)
    cmp_doc, cmp_ = _load(args.comparand)

    print(f"reference: {ref_doc.get('platform')!r:8} {ref_doc.get('gpu','?')}  "
          f"({len(ref)} records, captured {ref_doc.get('captured_at','?')})")
    print(f"comparand: {cmp_doc.get('platform')!r:8} {cmp_doc.get('gpu','?')}  "
          f"({len(cmp_)} records, captured {cmp_doc.get('captured_at','?')})")
    print()

    keys = sorted(set(ref) | set(cmp_))
    libs: dict[str, list[tuple[str, dict | None, dict | None]]] = {}
    for k in keys:
        r = ref.get(k)
        c = cmp_.get(k)
        lib = (r or c)["library"]
        if args.library and lib != args.library:
            continue
        if args.api and (r or c)["api"] != args.api:
            continue
        libs.setdefault(lib, []).append((k, r, c))

    if not libs:
        print("no matching records (after filters).")
        return 1

    n_pass = n_only_left = n_only_right = 0
    for lib in sorted(libs):
        rows = libs[lib]
        print(f"=== {lib} ({len(rows)} records) ===")
        print(f"  {'api':24} {'n_qubits':>8} {'precision':>9}  "
              f"{'ref gpu':>11}  {'cmp gpu':>11}  {'speedup':>8}  config")
        for k, r, c in rows:
            base = r or c
            api = base["api"]
            n = base.get("n_qubits")
            prec = base.get("precision") or ""
            ref_t = r["gpu_time_s"] if r else None
            cmp_t = c["gpu_time_s"] if c else None
            speedup = (ref_t / cmp_t) if (ref_t and cmp_t) else None
            cfg = json.dumps(base.get("config", {}), sort_keys=True)
            if len(cfg) > 60:
                cfg = cfg[:57] + "..."
            marker = "    "
            if r and c:
                n_pass += 1
            elif r and not c:
                marker = "L---"
                n_only_left += 1
            else:
                marker = "---R"
                n_only_right += 1
            speedup_s = f"{speedup:7.2f}x" if speedup else "       -"
            print(f"  {marker} {api:20} {str(n) if n else '-':>8} {prec:>9}  "
                  f"{_fmt(ref_t):>11}  {_fmt(cmp_t):>11}  {speedup_s:>8}  {cfg}")
        print()

    print(f"summary: {n_pass} matched, {n_only_left} only on reference, "
          f"{n_only_right} only on comparand")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
