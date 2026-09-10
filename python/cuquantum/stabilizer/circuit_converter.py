# Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES
#
# SPDX-License-Identifier: BSD-3-Clause

"""Deltakit-stim conversion.

Users drive conversion via :meth:`~cuquantum.stabilizer.Circuit.from_deltakit_stim`
and :meth:`~cuquantum.stabilizer.Circuit.from_deltakit_stim_file`. The
:class:`DeltakitCircuitConverter` class backs those entry points and is
available for pipelines that need the intermediate cuStabilizer-compatible
text.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

from ._internal import deltakit_parser


__all__ = ["DeltakitParserOptions"]


@dataclass
class DeltakitParserOptions:
    """Options controlling deltakit-stim conversion.

    Args:
        reject_unsupported: Raise :class:`ValueError` when an unsupported
            deltakit/Stim gate is encountered. If ``False``, leave that line
            unchanged in the output text.

    .. note::

        Deltakit-stim ``CX(s01, s10, m01, m10)`` applies spread then move per
        direction, with move only if the partner is still unleaked after
        spread. The effective rates are ``P(spread)=s`` and
        ``P(move)=(1-s)*m``. cuStabilizer ``LEAKAGE_PROPAGATE`` uses mutually
        exclusive rates, so each direction is rewritten as
        ``(s, (1-s)*m)``. For example, ``CX(0.03, 0, 0.05, 0)`` becomes
        ``LEAKAGE_PROPAGATE(0.03, 0, 0.0485, 0)``.
    """

    reject_unsupported: bool = True


class DeltakitCircuitConverter:
    """Convert deltakit-stim text into cuStabilizer-compatible Stim text.

    Gate renames include ``LEAKAGE`` → ``LEAKAGE1``, ``RELAX`` →
    ``LEAKAGE_RELAX``, and ``RL`` → ``LEAKAGE_RESET``. Two-qubit Clifford
    gates with leakage-propagation arguments are split into a bare ``CX``/``CZ``
    followed by ``LEAKAGE_PROPAGATE``.

    Not part of the documented public API — the intended entry points are
    :meth:`~cuquantum.stabilizer.Circuit.from_deltakit_stim` and
    :meth:`~cuquantum.stabilizer.Circuit.from_deltakit_stim_file`.

    Args:
        circuit: Deltakit-stim circuit text.
        options: A :class:`DeltakitParserOptions` instance, a compatible
            dictionary, or ``None`` for defaults.
    """

    def __init__(
        self,
        circuit: str,
        *,
        options: DeltakitParserOptions | dict | None = None,
    ):
        if not isinstance(circuit, str):
            raise TypeError(f"circuit must be str, got {type(circuit).__name__}")
        if options is None:
            options = DeltakitParserOptions()
        elif isinstance(options, dict):
            options = DeltakitParserOptions(**options)
        elif not isinstance(options, DeltakitParserOptions):
            raise TypeError(
                "options must be DeltakitParserOptions, dict, or None; "
                f"got {type(options).__name__}"
            )
        self.circuit = circuit
        self.options = options
        self._custabilizer_text: str | None = None

    @classmethod
    def from_file(
        cls,
        path: Path | str,
        *,
        options: DeltakitParserOptions | dict | None = None,
    ) -> "DeltakitCircuitConverter":
        """Create a converter from a deltakit-stim circuit file."""
        return cls(Path(path).read_text(), options=options)

    def to_custabilizer_text(self) -> str:
        """Return cuStabilizer-compatible Stim text."""
        if self._custabilizer_text is None:
            self._custabilizer_text = deltakit_parser.rewrite_deltakit_stim(
                self.circuit,
                reject_unsupported=self.options.reject_unsupported,
            )
        return self._custabilizer_text
