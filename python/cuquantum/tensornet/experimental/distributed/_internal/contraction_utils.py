# Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES
#
# SPDX-License-Identifier: BSD-3-Clause

"""Expression parsing, validation, and collective-agreement helpers for
distributed binary contraction."""

from __future__ import annotations


from ...._internal import decomposition_utils, einsum_parser


def parse_binary_expr(expr):
    """Parse a binary contraction expression, returning (input_modes, output_modes)
    as normalized integer mode labels (matching the descriptor creation contract).
    """
    input_modes, output_modes = einsum_parser.parse_einsum_str(expr)
    if output_modes is None:
        raise ValueError(
            "DistributedBinaryContraction requires an explicit output "
            "(expression must contain '->')"
        )
    if len(input_modes) != 2:
        raise ValueError(
            "DistributedBinaryContraction supports exactly two inputs"
        )
    if any(Ellipsis in term for term in input_modes) or Ellipsis in output_modes:
        raise NotImplementedError(
            "Ellipsis is not supported in DistributedBinaryContraction"
        )

    # Normalize user mode labels (which may be arbitrary hashables, e.g.
    # single characters) to ordinal int labels, the same way
    # distributed_decompose does: fold inputs and output into one list so
    # they all get mapped through a single ordinal assignment.
    morpher = einsum_parser.select_morpher(False)
    all_modes, _, _, _, _ = einsum_parser.map_modes(
        input_modes + [output_modes], None, 0, morpher
    )
    input_modes, output_modes = all_modes[:2], all_modes[2]
    return input_modes, output_modes


def output_global_shape(input_modes, output_modes, a_shape, b_shape) -> tuple[int, ...]:
    """Derive the binary-contraction output global shape from an einsum expr."""
    size_dict = {}
    for modes, shape in zip(input_modes, (a_shape, b_shape)):
        if len(modes) != len(shape):
            raise ValueError(
                f"operand rank {len(shape)} does not match modes {''.join(map(str, modes))}"
            )
        for mode, extent in zip(modes, shape):
            previous = size_dict.get(mode)
            if previous is None:
                size_dict[mode] = extent
            elif previous != extent:
                raise ValueError(
                    f"inconsistent extent for mode {mode}: {previous} vs {extent}"
                )

    return tuple(size_dict[mode] for mode in output_modes)


def scalar_dtype_for(dtype_name: str) -> str:
    """C/D scalar dtype for alpha/beta, following cuTENSOR scalar rules.

    Restricted to the dtypes distributed_decompose/contraction support
    (float32, float64, complex64, complex128), for which the scalar type
    always equals the data type (unlike float16/bfloat16, whose scalar type
    is float32 -- not reachable here).
    """
    if dtype_name not in decomposition_utils.DECOMPOSITION_DTYPE_NAMES:
        raise TypeError(
            f"DistributedBinaryContraction does not support dtype {dtype_name!r}; "
            f"expected one of {decomposition_utils.DECOMPOSITION_DTYPE_NAMES}"
        )
    return dtype_name
