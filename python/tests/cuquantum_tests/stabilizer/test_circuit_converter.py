# Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES
#
# SPDX-License-Identifier: BSD-3-Clause

"""Tests for cuStabilizer circuit converters (deltakit-stim dialect)."""

from __future__ import annotations

import re

import pytest

from cuquantum.stabilizer import Circuit, DeltakitParserOptions
from cuquantum.stabilizer.circuit_converter import DeltakitCircuitConverter

pytestmark = pytest.mark.custabilizer


def _convert(src: str, *, reject_unsupported: bool = True) -> str:
    return DeltakitCircuitConverter(
        src, options=DeltakitParserOptions(reject_unsupported=reject_unsupported)
    ).to_custabilizer_text()


def test_basic_gate_renames() -> None:
    src = """
H 0
LEAKAGE(0.01) 0
TICK
RELAX(0.02) 0 1
RL 0 1
"""
    out = _convert(src)
    assert "LEAKAGE1(0.01) 0" in out
    assert "LEAKAGE_RELAX(0.02) 0 1" in out
    assert "LEAKAGE_RESET 0 1" in out
    assert "LEAKAGE(0.01)" not in out
    assert not re.search(r"(?m)^RELAX\(", out)
    assert not re.search(r"(?m)^RL\b", out)


@pytest.mark.parametrize("gate", ["CX", "CZ"])
def test_bare_2q_emits_propagate(gate: str) -> None:
    out = _convert(f"{gate} 0 1\nLEAKAGE(0.1) 0 1\n")
    assert "LEAKAGE_PROPAGATE" in out


def test_rejects_cy_by_default() -> None:
    with pytest.raises(ValueError, match=r"line 1.*CY"):
        _convert("CY(0.1, 0.2, 0.3, 0.4) 0 1")


def test_reject_unsupported_false_passes_swap_through() -> None:
    out = _convert("SWAP 0 1\nLEAKAGE(0.1) 0\n", reject_unsupported=False)
    assert out == "SWAP 0 1\nLEAKAGE1(0.1) 0\n"


def test_rejects_unsupported_by_default() -> None:
    with pytest.raises(ValueError, match=r"line 1.*SWAP"):
        _convert("SWAP 0 1")


def test_converter_accepts_options_dict_and_caches() -> None:
    converter = DeltakitCircuitConverter(
        "SWAP 0 1\nLEAKAGE(0.1) 0\n", options={"reject_unsupported": False}
    )
    assert converter.options == DeltakitParserOptions(reject_unsupported=False)
    assert converter.to_custabilizer_text() is converter.to_custabilizer_text()


def test_converter_rejects_bad_options_type() -> None:
    with pytest.raises(TypeError, match="DeltakitParserOptions"):
        DeltakitCircuitConverter("H 0\n", options=object())


def test_converted_circuit_parses() -> None:
    src = """
R 0 1
CX(0.001, 0.001, 0.001, 0.001) 0 1
LEAKAGE(0.001) 0 1
CZ(0.001, 0.0, 0.0, 0.001) 0 1
LEAKAGE(0.001) 0 1
M 0 1
"""
    circuit = Circuit.from_deltakit_stim(src)
    assert circuit.num_qubits >= 2
    assert circuit.num_measurements >= 2
