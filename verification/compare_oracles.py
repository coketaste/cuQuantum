"""Compare two oracle directories case-by-case.

Use this when each side (cuQuantum on H100, rocQuantum on MI3xx) produces an
oracle independently and you want to diff them offline, without needing
either library to be importable at compare time.

Typical workflow:

    # On H100
    python3 verification/capture/capture_custatevec.py \
        --out verification/oracle/h100/v1.0.0 --small
    # On MI3xx
    python3 verification/replay/replay_custatevec.py \
        --oracle verification/oracle/h100/v1.0.0 \
        --capture-to verification/oracle/mi300/v1.0.0
    # Anywhere
    python3 verification/compare_oracles.py \
        --left  verification/oracle/h100/v1.0.0 \
        --right verification/oracle/mi300/v1.0.0

Matches are made on ``<library>/<api>/<param_id>``. Cases present on only
one side are reported but do not by themselves fail the run unless
``--strict`` is set. All comparisons use the per-API metric registered in
``verification/metrics.py``.
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

# Allow running this file directly (it lives one level under verification/).
ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))

from capture._common import load_case  # noqa: E402
from metrics import MetricResult       # noqa: E402
from replay._common import _compare    # reuse the dispatcher  # noqa: E402


def index_oracle(root: Path) -> dict[str, Path]:
    """Map relative case key (library/api/param_id) -> npz path."""
    out: dict[str, Path] = {}
    for npz in sorted(root.rglob("*.npz")):
        # Skip non-oracle npz (e.g. user scratch); a valid oracle case has a
        # sibling JSON sidecar.
        if not npz.with_suffix(".json").exists():
            continue
        rel = npz.relative_to(root).with_suffix("")
        out[str(rel)] = npz
    return out


def main(argv=None) -> int:
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--left", type=Path, required=True,
                   help="Reference oracle root (e.g. cuQuantum capture).")
    p.add_argument("--right", type=Path, required=True,
                   help="Comparand oracle root (e.g. rocQuantum capture).")
    p.add_argument("--filter", default=None,
                   help="Substring filter on the case key.")
    p.add_argument("--rtol", type=float, default=None)
    p.add_argument("--atol", type=float, default=None)
    p.add_argument("--alpha", type=float, default=None,
                   help="Significance for distribution metrics (default 0.001).")
    p.add_argument("--strict", action="store_true",
                   help="Treat keys missing on either side as failures.")
    p.add_argument("--quiet", action="store_true",
                   help="Only print the summary and any failures.")
    args = p.parse_args(argv)

    left = index_oracle(args.left)
    right = index_oracle(args.right)
    common = sorted(set(left) & set(right))
    only_left = sorted(set(left) - set(right))
    only_right = sorted(set(right) - set(left))

    if args.filter:
        common = [k for k in common if args.filter in k]
        only_left = [k for k in only_left if args.filter in k]
        only_right = [k for k in only_right if args.filter in k]

    n_pass = n_fail = 0
    for key in common:
        l = load_case(left[key])
        r = load_case(right[key])
        if l["api"] != r["api"]:
            print(f"  FAIL {key}: API mismatch {l['api']!r} vs {r['api']!r}")
            n_fail += 1
            continue
        try:
            result = _compare(l["api"], l["outputs"], r["outputs"],
                              rtol=args.rtol, atol=args.atol, alpha=args.alpha,
                              metadata=l.get("metadata", {}))
        except Exception as e:
            print(f"  FAIL {key}: metric raised {type(e).__name__}: {e}")
            n_fail += 1
            continue
        if result.passed:
            n_pass += 1
            if not args.quiet:
                print(f"  PASS {key}: {result.detail}")
        else:
            n_fail += 1
            print(f"  FAIL {key}: {result.detail}")

    if only_left:
        msg = "missing on right" if args.strict else "only on left (informational)"
        for k in only_left:
            print(f"  {('FAIL' if args.strict else 'INFO')} {k}: {msg}")
        if args.strict:
            n_fail += len(only_left)

    if only_right:
        msg = "missing on left" if args.strict else "only on right (informational)"
        for k in only_right:
            print(f"  {('FAIL' if args.strict else 'INFO')} {k}: {msg}")
        if args.strict:
            n_fail += len(only_right)

    n_total = n_pass + n_fail
    print(f"summary: {n_pass} passed, {n_fail} failed, {n_total} compared,"
          f" {len(only_left)} only-left, {len(only_right)} only-right")
    return 0 if n_fail == 0 else 1


if __name__ == "__main__":
    raise SystemExit(main())
