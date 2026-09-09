# Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES
#
# SPDX-License-Identifier: BSD-3-Clause

"""
Configuration and default conditions for the operator-product simplification passes.
"""

from collections.abc import Callable
from dataclasses import dataclass, field


@dataclass
class SimplifierConfig:
    """
    Configuration for the operator-product simplification passes run at compile time.

    Each field holds one condition callable per simplification layer. An empty tuple
    disables the pass; ``None`` at position ``i`` skips the pass for layer ``i``. The
    passes are applied interleaved by layer, i.e. ``kron_cond[0]``, ``sum_cond[0]``,
    ``kron_cond[1]``, ``sum_cond[1]``, ... The shorter tuple is padded with ``None``.

    Attributes:
        kron_cond: Conditions for the intra-operator product (kron) pass, with contract
            ``kron_cond(modes_duals) -> bool``.
        sum_cond: Conditions for the inter-operator product (sum) pass, with contract
            ``sum_cond(modes_duals_src, modes_duals_dst) -> bool``. A condition must never
            return ``True`` for a ``modes_duals_dst`` spanning more than one base operator
            (``len(modes_duals_dst) > 1``): the merge folds the source only into
            ``modes_duals_dst``'s first base operator, so approving a multi-base-op
            destination would discard the rest of it. ``default_sum_cond`` enforces this by
            construction; a custom condition must too, or ``_sum_simplify`` raises
            ``ValueError``.

    Examples:
        Enable both passes with their defaults::

            Operator(dims, simplify=True)

        Disable the sum pass but keep the default kron pass::

            Operator(dims, simplify=SimplifierConfig(sum_cond=()))

        Loosen the kron pass to merge base operators spanning up to 3 modes instead of
        the default cap of 2 (``default_kron_cond``'s cap is not itself parametrized, so
        widening it means writing the condition, mirroring ``default_kron_cond`` but with
        the cap raised)::

            def kron_cond_up_to_3_modes(modes_duals):
                combined = set(modes_duals[0]) | set(modes_duals[-1])
                if len(combined) > 3:
                    return False
                return not any(set(md) & combined for md in modes_duals[1:-1])

            Operator(dims, simplify=SimplifierConfig(kron_cond=(kron_cond_up_to_3_modes,)))

        A custom sum_cond stricter than the default: only absorb a source product whose
        combined mode-dual footprint exactly matches the destination's, rather than merely
        being contained in it (still respects the single-base-op-destination invariant
        above)::

            def sum_cond_exact_match(modes_duals_src, modes_duals_dst):
                if len(modes_duals_dst) > 1:  # required: see the invariant above
                    return False
                combined_src = set().union(*modes_duals_src)
                return combined_src == set(modes_duals_dst[0])

            Operator(dims, simplify=SimplifierConfig(sum_cond=(sum_cond_exact_match,)))
    """

    kron_cond: tuple[Callable | None, ...] = field(
        default_factory=lambda: (SimplifierConfig.default_kron_cond,))
    sum_cond: tuple[Callable | None, ...] = field(
        default_factory=lambda: (SimplifierConfig.default_sum_cond,))

    @staticmethod
    def default_kron_cond(modes_duals: tuple[tuple[tuple[int, bool], ...], ...]) -> bool:
        """
        Check if the base operators at the start and the end can be merged.
        """
        modes_duals_src = set(modes_duals[0])
        modes_duals_dst = set(modes_duals[-1])
        modes_duals_combined = modes_duals_src | modes_duals_dst

        if len(modes_duals_combined) > 2:  # the combined base operator can't act on more than 2 modes
            return False

        for modes_duals_ in modes_duals[1:-1]:
            if set(modes_duals_) & modes_duals_combined:  # overlapping mode-dual pairs in between
                return False

        return True

    @staticmethod
    def default_sum_cond(modes_duals_src: tuple[tuple[tuple[int, bool], ...], ...],
                                      modes_duals_dst: tuple[tuple[tuple[int, bool], ...], ...]
                                      ) -> bool:
        """
        Check if the source operator product can be summed into the destination operator product.
        """
        # The destination operator product must contain a single base operator.
        if len(modes_duals_dst) > 1:
            return False

        for modes_duals_src_ in modes_duals_src:
            # Mode-dual pairs of each base operator in the source operator product must be
            # a subset of the mode-dual pairs of the single base operator in the
            # destination operator product.
            if not (set(modes_duals_src_) <= set(modes_duals_dst[0])):
                return False

        return True
