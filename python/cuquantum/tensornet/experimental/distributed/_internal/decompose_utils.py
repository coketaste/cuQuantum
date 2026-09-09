# Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES
#
# SPDX-License-Identifier: BSD-3-Clause

"""Expression parsing and result-shaping helpers for distributed_decompose."""

from __future__ import annotations

from types import SimpleNamespace

import numpy

from cuquantum.bindings import cutensornet as cutn

from ...._internal import decomposition_utils, einsum_parser
from . import tensor_utils


def parse_distributed_decompose(subscripts: str, global_shape: tuple[int, ...]):
    """Parse a decomposition expression against global extents."""
    inputs_raw, outputs_raw = decomposition_utils.parse_decomposition_subscripts(
        subscripts
    )
    if len(inputs_raw) != 1:
        raise ValueError(
            "distributed_decompose currently supports a single input operand"
        )

    morpher = einsum_parser.select_morpher(False)
    inputs = [einsum_parser.parse_single(term) for term in inputs_raw]
    outputs = [einsum_parser.parse_single(term) for term in outputs_raw]

    if any(Ellipsis in term for term in inputs + outputs):
        raise NotImplementedError(
            "Ellipsis is not supported in distributed_decompose"
        )

    fake = SimpleNamespace(shape=global_shape)
    einsum_parser.check_einsum_with_operands(inputs, [fake], morpher)

    all_modes, _, _, _, _ = einsum_parser.map_modes(
        inputs + outputs, None, 0, morpher
    )
    inputs = all_modes[:1]
    outputs = all_modes[1:]

    contracted = set(einsum_parser.infer_output_mode_labels(outputs))
    if contracted != set(inputs[0]):
        raise ValueError(
            "The contracted outcome from the right hand side of the expression "
            "does not match the input"
        )

    size_dict = {mode: extent for mode, extent in zip(inputs[0], global_shape)}
    mid_extent = decomposition_utils.compute_mid_extent(size_dict, inputs, outputs)
    return inputs, outputs, size_dict, mid_extent


def output_global_shapes(outputs, size_dict, mid_extent):
    shared = list(set(outputs[0]) & set(outputs[1]))[0]
    return [
        tuple(size_dict[m] if m != shared else mid_extent for m in modes)
        for modes in outputs
    ]


# Singular values carry the real type of the operand's precision; the library
# writes that type into the s buffer, so a wider buffer would reinterpret raw
# bits rather than widen values.
SVD_S_DTYPE = {
    "float32": "float32",
    "float64": "float64",
    "complex64": "float32",
    "complex128": "float64",
}


def s_dtype_for(dtype_name):
    try:
        return SVD_S_DTYPE[dtype_name]
    except KeyError:
        raise ValueError(f"dtype {dtype_name} not supported") from None


def query_realized_layout(handle, desc, num_modes):
    """Read the realized global extents, rank-local extents, and local
    element strides that the library published into an output descriptor
    during execution."""
    _, _, extents, _ = cutn.get_tensor_details(handle, desc)
    buf = numpy.empty(num_modes, dtype=numpy.int64)
    cutn.tensor_descriptor_get_attribute(
        handle,
        desc,
        cutn.TensorDescriptorAttribute.LOCAL_EXTENTS,
        buf.ctypes.data,
        buf.nbytes,
    )
    local_extents = tuple(int(extent) for extent in buf)
    cutn.tensor_descriptor_get_attribute(
        handle,
        desc,
        cutn.TensorDescriptorAttribute.ELEMENT_STRIDES,
        buf.ctypes.data,
        buf.nbytes,
    )
    element_strides = [int(stride) for stride in buf]
    return tuple(int(extent) for extent in extents), local_extents, element_strides


def rebind_realized(handle, desc, tensor):
    """Rebind an output whose bond was truncated below the planned capacity.

    The library compacts the kept data inside the capacity buffer and
    publishes the realized extents and local strides in the output
    descriptor; the rebound tensor is a zero-copy view of that buffer with
    the realized shape and strides (the view keeps the capacity allocation
    alive). The distribution itself is unchanged by truncation — slab modes
    stay slab and an explicit block-cyclic bond keeps its block size under
    the owner-stable compaction — so the original block sizes are rebound to
    the realized global shape."""
    from ..base import BlockCyclic
    from ..tensor import DistributedTensor

    global_shape, local_extents, element_strides = query_realized_layout(
        handle, desc, len(tensor.global_shape)
    )
    if global_shape == tensor.global_shape:
        return tensor
    distribution = BlockCyclic(
        tensor.distribution._process_grid,
        tensor.distribution.block_sizes,
        first_process=tensor.distribution._first_process,
    )
    # The view shape comes from the descriptor's realized LOCAL_EXTENTS;
    # construction re-derives the shard shape from the distribution's
    # ownership math, so a disagreement between the two raises instead of
    # misplacing data.
    local_view = tensor_utils.as_strided_view(
        tensor.local, local_extents, element_strides
    )
    return DistributedTensor(local_view, distribution, global_shape)
