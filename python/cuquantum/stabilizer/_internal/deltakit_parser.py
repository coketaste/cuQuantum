# Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES
#
# SPDX-License-Identifier: BSD-3-Clause

"""Deltakit-stim text rewrite helpers for :mod:`cuquantum.stabilizer.circuit_converter`."""

from __future__ import annotations

import re
from decimal import Decimal, InvalidOperation


_LINE_RE = re.compile(
    r"^(\s*)"
    r"([A-Za-z_][A-Za-z0-9_]*)"
    r"(?:\[[^\]]*\])?"
    r"(?:\(([^)]*)\))?"
    r"(.*)$"
)

_UNSUPPORTED = frozenset({"SWAP", "MPP", "MPAD", "ISWAP", "CY", "ZCY"})


def _parse_args(raw: str | None) -> list[str]:
    if raw is None or not raw.strip():
        return []
    return [part.strip() for part in raw.split(",") if part.strip()]


def _emit(indent: str, name: str, args: list[str] | None, targets: list[str]) -> str:
    body = name if args is None else f"{name}({', '.join(args)})"
    if targets:
        return f"{indent}{body} {' '.join(targets)}"
    return f"{indent}{body}"


def _format_prob(value: Decimal) -> str:
    """Format a probability for Stim text (trim trailing zeros)."""
    text = format(value.normalize(), "f")
    if "." in text:
        text = text.rstrip("0").rstrip(".")
    return text or "0"


def _parse_probs(tokens: list[str]) -> list[Decimal]:
    return [Decimal(t) for t in tokens]


def _remap_propagation_args(
    args: list[str], *, line_no: int, gate: str
) -> list[str]:
    """Map deltakit CX/CZ propagation args to ``LEAKAGE_PROPAGATE`` rates."""
    if len(args) != 4:
        raise ValueError(
            f"line {line_no}: {gate} expected 0 or 4 args, got {len(args)}: {args!r}"
        )
    try:
        s01, s10, m01, m10 = _parse_probs(args)
    except InvalidOperation as exc:
        raise ValueError(
            f"line {line_no}: {gate} expected numeric propagation args, got {args!r}"
        ) from exc
    return [
        _format_prob(s01),
        _format_prob(s10),
        _format_prob((1 - s01) * m01),
        _format_prob((1 - s10) * m10),
    ]


def rewrite_deltakit_stim(
    circuit: str,
    *,
    reject_unsupported: bool,
) -> str:
    """Rewrite deltakit-stim text to cuStabilizer-compatible Stim text."""
    output: list[str] = []

    for line_no, line in enumerate(circuit.splitlines(), start=1):
        stripped = line.strip()
        if not stripped or stripped.startswith("#") or stripped == "}":
            output.append(line)
            continue

        match = _LINE_RE.match(line.rstrip("\n"))
        if match is None:
            output.append(line)
            continue

        indent, name, raw_args, rest = match.groups()
        name_upper = name.upper()
        args = _parse_args(raw_args)
        targets = rest.split()

        if name_upper in _UNSUPPORTED:
            message = (
                f"line {line_no}: unsupported deltakit/Stim gate {name!r} "
                "for cuStabilizer"
            )
            if reject_unsupported:
                raise ValueError(message)
            output.append(line)
            continue

        if name_upper == "LEAKAGE":
            output.append(_emit(indent, "LEAKAGE1", args, targets))
            continue

        if name_upper == "RELAX":
            output.append(_emit(indent, "LEAKAGE_RELAX", args, targets))
            continue

        if name_upper == "RL":
            if args:
                raise ValueError(
                    f"line {line_no}: RL expected 0 args, got {len(args)}: {args!r}"
                )
            output.append(_emit(indent, "LEAKAGE_RESET", None, targets))
            continue

        # CX/CZ: emit bare Clifford plus LEAKAGE_PROPAGATE (args remapped).
        if name_upper in {"CX", "CNOT", "ZCX", "CZ", "ZCZ"}:
            out_name = "CZ" if name_upper in {"CZ", "ZCZ"} else "CX"
            output.append(_emit(indent, out_name, None, targets))
            if len(args) == 4:
                prop_args = _remap_propagation_args(
                    args, line_no=line_no, gate=name
                )
            elif args:
                raise ValueError(
                    f"line {line_no}: {name} expected 0 or 4 args, got {len(args)}: "
                    f"{args!r}"
                )
            else:
                prop_args = ["0", "0", "0", "0"]
            output.append(_emit(indent, "LEAKAGE_PROPAGATE", prop_args, targets))
            continue

        output.append(line)

    return "\n".join(output) + ("\n" if circuit.endswith("\n") or circuit else "")
