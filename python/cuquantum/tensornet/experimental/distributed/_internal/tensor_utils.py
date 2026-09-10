# Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES
#
# SPDX-License-Identifier: BSD-3-Clause

"""Shard layout, allocation, and descriptor-argument helpers."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from typing import Any

import numpy
from nvmath.internal import typemaps
from nvmath.internal import utils as nvmath_utils
from nvmath.internal.tensor_wrapper import wrap_operand

from cuquantum.bindings import cutensornet as cutn

from . import runtime




def same_distribution(d1, d2) -> bool:
    """True if two BlockCyclic distributions describe identical layouts."""
    return (
        d1._process_grid == d2._process_grid
        and tuple(d1.block_sizes) == tuple(d2.block_sizes)
        and d1._first_process == d2._first_process
    )


def as_shape(shape: Sequence[int], name: str = "shape") -> tuple[int, ...]:
    try:
        shape = tuple(shape)
    except TypeError as exc:
        raise TypeError(f"{name} must be a sequence of integers") from exc
    if any(type(extent) is not int for extent in shape):
        raise TypeError(f"{name} must be a sequence of integers, got {shape}")
    if any(extent < 0 for extent in shape):
        raise ValueError(f"{name} dimensions must be nonnegative, got {shape}")
    return shape


def dtype_name(dtype: Any) -> str:
    """Normalize a dtype spec to the name string nvmath's allocators take.

    Tensor call sites pass the wrapper's already-normalized ``.dtype``
    string; this fallback exists for BARE dtype objects (empty_local's
    public ``dtype`` parameter). numpy.dtype() cannot interpret e.g.
    torch.float64, so torch dtypes take the same normalization nvmath's
    own torch holder uses (tensor_ifc_torch: str(dtype).split(".")[-1])."""
    if isinstance(dtype, str):
        return dtype
    if type(dtype).__module__.split(".", 1)[0] == "torch":
        return str(dtype).split(".")[-1]
    return numpy.dtype(dtype).name


def local_shape(local: Any) -> tuple[int, ...]:
    try:
        return tuple(int(extent) for extent in local.shape)
    except Exception as exc:
        raise TypeError(
            f"local must be an array-like object with a .shape attribute, got {type(local)}"
        ) from exc


def as_strided_view(local: Any, shape: Sequence[int], element_strides: Sequence[int]):
    """Zero-copy view of ``local``'s allocation with the given shape and
    element strides.

    The view holds a reference to the backing allocation (``.base`` for CuPy,
    shared storage for torch), so it stays valid after the capacity-shaped
    array object goes out of scope.
    """
    shape = tuple(int(extent) for extent in shape)
    module_name = type(local).__module__.split(".", 1)[0]
    if module_name == "cupy":
        import cupy

        strides_bytes = tuple(
            int(stride) * local.itemsize for stride in element_strides
        )
        return cupy.lib.stride_tricks.as_strided(
            local, shape=shape, strides=strides_bytes
        )
    if module_name == "torch":
        return local.as_strided(
            shape, tuple(int(stride) for stride in element_strides)
        )
    raise TypeError(
        f"Unsupported array package for strided views: "
        f"{type(local).__module__}.{type(local).__name__}"
    )


def canonical_element_strides(shape: Sequence[int]) -> list[int]:
    """Canonical packed (column-major) element strides for a local shape —
    the layout that null element strides select in the native descriptor."""
    strides = []
    acc = 1
    for extent in shape:
        strides.append(acc)
        acc *= extent
    return strides


def is_canonical_local(shape: Sequence[int], element_strides: Sequence[int]) -> bool:
    """True if a local shard is in the canonical packed layout (or empty —
    a zero-size shard has no meaningful strides).

    The stride of an extent-one mode never contributes to an address (its
    only index is 0), so it is excluded from the comparison; a strict
    comparison would misclassify e.g. a (1, n) C-order shard, whose layout
    is physically identical to the canonical one.
    """
    if 0 in shape:
        return True
    canonical = canonical_element_strides(shape)
    return all(
        extent <= 1 or int(stride) == expected
        for extent, stride, expected in zip(shape, element_strides, canonical)
    )


def as_canonical_local(local: Any, *, stream_holder=None):
    """Return ``local`` in the canonical packed (column-major) layout,
    copying only when it is not already canonical.

    The copy is device work: for device arrays it is performed under the
    array's device and ``stream_holder``'s stream, ordering it against the
    native call that consumes the result."""
    shape = local_shape(local)
    wrapped = wrap_operand(local)
    if is_canonical_local(shape, wrapped.strides):
        return local
    module_name = type(local).__module__.split(".", 1)[0]
    if module_name == "numpy":
        return numpy.asfortranarray(local)
    if module_name == "cupy":
        import cupy

        with nvmath_utils.device_ctx(int(wrapped.device_id)), stream_holder.ctx:
            return cupy.asfortranarray(local)
    if module_name == "torch":
        holder = nvmath_utils.create_empty_tensor(
            type(wrapped),
            shape,
            wrapped.dtype,
            wrapped.device_id,
            stream_holder,
            verify_strides=False,
            strides=canonical_element_strides(shape),
        )
        with stream_holder.ctx:
            holder.tensor.copy_(local)
        return holder.tensor
    raise TypeError(
        f"Unsupported array package for canonical staging: "
        f"{type(local).__module__}.{type(local).__name__}"
    )


def allocate_local(
    distribution,
    global_shape: Sequence[int],
    *,
    dtype,
    like,
    rank: int | None = None,
    stream_holder=None,
):
    """Allocate an empty canonical local shard for ``distribution`` and
    ``global_shape``, stream-ordered on ``stream_holder``."""
    from ..base import BlockCyclic

    if not isinstance(distribution, BlockCyclic):
        raise TypeError(
            "distribution must be an experimental distributed.BlockCyclic"
        )
    global_shape = as_shape(global_shape, "global_shape")
    if rank is None:
        rank = runtime.get_process_group().rank
    shape = distribution.shape(rank, global_shape)
    # nvmath's TensorHolder.empty allocates under the target device and the
    # execution stream; canonical strides give the packed column-major layout.
    wrapped_like = wrap_operand(like)
    holder = nvmath_utils.create_empty_tensor(
        type(wrapped_like),
        shape,
        dtype_name(dtype),
        wrapped_like.device_id,
        stream_holder,
        verify_strides=False,
        strides=canonical_element_strides(shape),
    )
    return holder.tensor


def allocate_distributed(
    distribution,
    global_shape: Sequence[int],
    *,
    dtype,
    like,
    stream_holder=None,
):
    """Allocate a fresh ``DistributedTensor`` for ``distribution``: a
    canonical packed local shard in ``like``'s array package (whose memory
    pool / current allocator serves the allocation), stream-ordered on
    ``stream_holder``."""
    from ..tensor import DistributedTensor

    global_shape = as_shape(global_shape, "global_shape")
    local = allocate_local(
        distribution, global_shape, dtype=dtype, like=like, stream_holder=stream_holder
    )
    return DistributedTensor(local, distribution, global_shape)


def infer_global_shape(
    distribution,
    local_shapes: Mapping[int, Sequence[int]],
) -> tuple[int, ...]:
    """Reconstruct a global shape from per-rank local shapes (host-side).

    Requires at least one local shape for every active-grid rank
    ``0 .. process_grid.size - 1``. Replica ranks, if provided, must match
    their active-grid owners. Every provided shape is re-validated against
    the inferred global shape.
    """
    from ..base import BlockCyclic

    if not isinstance(distribution, BlockCyclic):
        raise TypeError(
            "distribution must be an experimental distributed.BlockCyclic"
        )
    if not local_shapes:
        raise ValueError("local_shapes must contain at least one rank entry")

    grid = distribution._process_grid
    ndim = distribution.ndim
    active_size = grid.size
    required = set(range(active_size))
    provided_active = {int(rank) % active_size for rank in local_shapes}
    if provided_active != required:
        missing = sorted(required - provided_active)
        raise ValueError(
            f"local_shapes must cover every active-grid rank; missing {missing}"
        )

    extents_by_mode: list[dict[int, int]] = [dict() for _ in range(ndim)]
    for rank, shape in local_shapes.items():
        if type(rank) is not int:
            raise TypeError(f"local_shapes keys must be integers, got {rank}")
        shape = as_shape(shape, f"local_shapes[{rank}]")
        if len(shape) != ndim:
            raise ValueError(
                f"local_shapes[{rank}] dimensionality ({len(shape)}) does not "
                f"match distribution dimensionality ({ndim})"
            )
        coords = grid.rank_to_coords(rank)
        for mode, (extent, process_count, coord, first) in enumerate(
            zip(
                shape,
                grid.shape,
                coords,
                distribution._first_process,
            )
        ):
            relative = (coord - first) % process_count
            previous = extents_by_mode[mode].get(relative)
            if previous is None:
                extents_by_mode[mode][relative] = extent
            elif previous != extent:
                raise ValueError(
                    f"inconsistent local extent for mode {mode}, relative "
                    f"coordinate {relative}: {previous} vs {extent}"
                )

    global_shape_list = []
    for mode, process_count in enumerate(grid.shape):
        extents = extents_by_mode[mode]
        missing_rel = sorted(set(range(process_count)) - set(extents))
        if missing_rel:
            raise ValueError(
                f"missing process coordinates for mode {mode}: {missing_rel}"
            )
        global_shape_list.append(sum(extents[rel] for rel in range(process_count)))
    global_shape = tuple(global_shape_list)

    for rank, shape in local_shapes.items():
        expected = distribution.shape(rank, global_shape)
        if as_shape(shape, f"local_shapes[{rank}]") != expected:
            raise ValueError(
                f"local_shapes[{rank}]={tuple(shape)} is inconsistent with "
                f"inferred global_shape {global_shape} (expected {expected})"
            )
    return global_shape


def pack_distributed_descriptor_spec_args(
    distribution,
    global_shape,
    modes,
    dtype,
):
    """Pack descriptor arguments that do not require local storage."""
    block_sizes = [
        0 if block is None else int(block)
        for block in distribution.block_sizes
    ]
    return {
        "num_modes": len(modes),
        "extents": list(global_shape),
        "element_strides": 0,
        "block_sizes": block_sizes,
        "block_strides": 0,
        "nranks_per_mode": list(distribution._process_grid.shape),
        "mode_labels": list(modes),
        "data_type": typemaps.NAME_TO_DATA_TYPE[dtype_name(dtype)],
    }


def pack_distributed_descriptor_args(tensor, modes, wrapped):
    """Pack a ``DistributedTensor``'s layout into ``create_distributed_tensor_descriptor``
    keyword arguments.

    Compact Fortran-order element strides are passed using the null-strides
    default. A zero-sized local shard has no meaningful strides and is
    normalized the same way. Other layouts are currently rejected here with
    an actionable error.
    """
    shape = local_shape(tensor.local)
    if not is_canonical_local(shape, wrapped.strides):
        raise ValueError(
            f"the local array (shape {tuple(shape)}, element strides "
            f"{tuple(wrapped.strides)}) is not compact Fortran-order "
            f"(expected strides "
            f"{tuple(canonical_element_strides(shape))}); allocate it "
            "with empty_local(...) or convert it with asfortranarray"
        )
    return pack_distributed_descriptor_spec_args(
        tensor.distribution,
        tensor.global_shape,
        modes,
        wrapped.dtype,
    )


def _create_distributed_tensor_descriptor(handle, args):
    return cutn.create_distributed_tensor_descriptor(
        handle,
        args["num_modes"],
        args["extents"],
        args["element_strides"],
        args["block_sizes"],
        args["block_strides"],
        args["nranks_per_mode"],
        args["mode_labels"],
        args["data_type"],
    )


def create_distributed_tensor_descriptor(handle, tensor, modes, wrapped):
    args = pack_distributed_descriptor_args(tensor, modes, wrapped)
    return _create_distributed_tensor_descriptor(handle, args)


def create_distributed_tensor_descriptor_from_spec(
    handle,
    distribution,
    global_shape,
    modes,
    dtype,
):
    """Create a distributed descriptor without allocating local storage."""
    args = pack_distributed_descriptor_spec_args(
        distribution,
        global_shape,
        modes,
        dtype,
    )
    return _create_distributed_tensor_descriptor(handle, args)
