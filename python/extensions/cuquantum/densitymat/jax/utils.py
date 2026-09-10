# Copyright (c) 2025-2026, NVIDIA CORPORATION & AFFILIATES
#
# SPDX-License-Identifier: BSD-3-Clause

"""
Utility functions for cuQuantum Python JAX.
"""

import ctypes
import math
from collections.abc import Sequence
from dataclasses import dataclass, field

import cupy as cp
import jax
import jax.numpy as jnp

from cuquantum.bindings import cudensitymat as cudm


@dataclass
class BufferMetadata:
    """
    Metadata for a buffer.
    """
    indices: list[int] = field(default_factory=list)
    types: list[int] = field(default_factory=list)
    ptrs: list[int] = field(default_factory=list)
    shape_dtypes: list[jax.ShapeDtypeStruct] = field(default_factory=list)

    def __add__(self, other: "BufferMetadata") -> "BufferMetadata":
        """
        Add two BufferMetadata objects.
        """
        return BufferMetadata(
            indices=self.indices + other.indices,
            types=self.types + other.types,
            ptrs=self.ptrs + other.ptrs,    
            shape_dtypes=self.shape_dtypes + other.shape_dtypes,
        )
    
    def __len__(self) -> int:
        """
        Return the number of buffers.
        """
        return len(self.indices)


def is_vmap_traced(x: jax.Array | jax.core.Tracer) -> bool:
    """
    Check if a buffer is traced by jax.vmap.
    """
    return get_vmap_depth(x) > 0


def get_original_shape(x: jax.Array | jax.core.Tracer) -> tuple:
    """
    Return the shape of x with all vmap batch levels fused into one leading dim.

    For nested vmap the product of every BatchTrace size becomes a single leading
    batch dimension, matching the fused backend batch (see fuse_batched_inputs and
    get_state_batch_size_and_purity). For a single vmap level this reduces to the
    original (size, *shape) behavior.
    """
    shape = x.shape
    if isinstance(x, jax.core.Tracer):
        trace = x._trace
        batch_size = 1
        is_batched = False
        while hasattr(trace, "parent_trace"):
            if type(trace).__name__ == "BatchTrace":
                # NOTE: This assumes the batch dimensions are always the most major axes,
                # which is not the general case.
                batch_size *= trace.axis_data.size
                is_batched = True
            trace = trace.parent_trace
        if is_batched:
            shape = (batch_size, *shape)
    return shape


def get_batch_size(x: jax.Array | jax.core.Tracer) -> int:
    """
    Return the vmap batch size of x: the leading batch dimension inserted by jax.vmap,
    or 1 if x is not vmap-traced (including concrete arrays and non-vmap tracers like JVP).
    """
    if is_vmap_traced(x):
        return get_original_shape(x)[0]
    return 1


def get_vmap_depth(x: jax.Array | jax.core.Tracer) -> int:
    """
    Return the number of nested vmap transformations active on a traced object.
    """
    if not isinstance(x, jax.core.Tracer):
        return 0
    depth = 0
    trace = x._trace
    while hasattr(trace, "parent_trace"):
        if type(trace).__name__ == "BatchTrace":
            depth += 1
        trace = trace.parent_trace
    return depth


def maybe_expand_dim(bufs, has_explicit_batch: bool = False):
    """
    Expand to a leading batch=1 dimension when the state has no batch dim.

    Returns (bufs, did_expand). A vmap-traced state carries its batch on the trace, and a
    state with an explicit leading batch axis (has_explicit_batch, as reported by
    get_state_batch_size_and_purity) already has the dimension; neither is expanded.
    Everything else is a single unbatched state of rank ndim or 2*ndim and gets a leading 1.
    The caller must pass did_expand to maybe_squeeze_dim so the squeeze only happens when
    this function actually added the dimension.
    """
    if has_explicit_batch or is_vmap_traced(bufs[0]):
        return bufs, False
    return tuple([buf.reshape((1, *buf.shape)) for buf in bufs]), True


def maybe_squeeze_dim(bufs, did_expand):
    """
    Remove the leading batch=1 dimension that maybe_expand_dim added.

    Only squeezes when did_expand=True (i.e. maybe_expand_dim actually added the dim).
    """
    if did_expand:
        return tuple([buf.reshape(buf.shape[1:]) for buf in bufs])
    return bufs


def get_state_batch_size_and_purity(state_in_bufs: tuple[jax.Array, ...],
                                    dims: Sequence[int],
                                    ) -> tuple[int, cudm.StatePurity, bool]:
    """
    Get the batch size, purity, and explicit-batch flag of the state.

    The flag reports whether the batch is carried by an explicit leading axis, as opposed to
    a vmap trace or being absent; the caller passes it to maybe_expand_dim so an already
    batched state is not given a second, spurious batch axis.
    """
    # First check all state buffer shapes.
    shape = state_in_bufs[0].shape
    for buf in state_in_bufs[1:]:
        if buf.shape != shape:
            raise ValueError("All input state buffers must have the same shape.")

    dims = tuple(dims)
    ndim = len(dims)

    # Extract batch size. Only vmap contributes a batch; the product over all
    # (possibly nested) vmap levels becomes the fused backend batch.
    if is_vmap_traced(state_in_bufs[0]):
        # Detect batch size by traversing the trace stack.
        trace = state_in_bufs[0]._trace
        batch_size = 1
        while hasattr(trace, "parent_trace"):
            if type(trace).__name__ == "BatchTrace":
                batch_size *= trace.axis_data.size
            trace = trace.parent_trace
        has_explicit_batch = False
    elif ndim == 1:
        # `state.ndim % ndim == 1` below is `x % 1`, identically 0 for every state, so an
        # explicitly batched single-mode state was always misread as unbatched. Fixing this
        # needs the actual mode extent, not just the mode count: an explicitly batched pure
        # state (batch, d) and an unbatched mixed state (d, d) have the same rank whenever
        # ndim == 1, so a rank-only check can't tell them apart -- only the shape values can,
        # unless batch_size happens to equal d, which is an irreducible ambiguity resolved here
        # (as before this fix) by preferring the unbatched-mixed interpretation.
        mixed_dims = dims + dims
        if shape == dims:
            has_explicit_batch, batch_size, purity = False, 1, cudm.StatePurity.PURE
        elif shape == mixed_dims:
            has_explicit_batch, batch_size, purity = False, 1, cudm.StatePurity.MIXED
        elif shape[1:] == dims:
            has_explicit_batch, batch_size, purity = True, shape[0], cudm.StatePurity.PURE
        elif shape[1:] == mixed_dims:
            has_explicit_batch, batch_size, purity = True, shape[0], cudm.StatePurity.MIXED
        else:
            raise ValueError(
                f"State shape {shape} is not compatible with dims {dims}: expected a leading "
                f"batch axis (optional) followed by either {dims} (pure) or {mixed_dims} (mixed)."
            )
        return batch_size, purity, has_explicit_batch
    else:
        has_explicit_batch = state_in_bufs[0].ndim % ndim == 1
        batch_size = state_in_bufs[0].shape[0] if has_explicit_batch else 1

    # Extract purity: rank ndim -> pure, rank 2 * ndim -> mixed (per vmap element).
    if len(shape) // ndim == 1:
        purity = cudm.StatePurity.PURE
    else:  # 2 * ndim
        purity = cudm.StatePurity.MIXED

    return batch_size, purity, has_explicit_batch


def check_and_return_final_batch_size(state_batch_size, op_batch_size):
    """
    Check the state and operator batch sizes match (uniform batching) and return it.
    """
    if state_batch_size != op_batch_size:
        raise ValueError("The batch size of the input state does not match the batch size of the operator.")
    return state_batch_size


def check_and_return_device(op: "Operator",
                            state_in_bufs: tuple[jax.Array | jax.core.Tracer, ...],
                            ) -> jax.Device | None:
    """
    Check if all operator and state objects are on the same GPU device and return the device.
    """
    device = None

    def _check_device(x):
        nonlocal device
        if not isinstance(x, jax.core.Tracer):  # only check object device if it is not traced
            # Check if the object device is a GPU.
            if x.device.platform != 'gpu':
                raise ValueError("cuQuantum Python JAX only supports GPU devices.")

            # If device is already set, check it against the current object's device.
            # Otherwise, set it to the current object's device.
            if device is not None:
                if x.device != device:
                    raise ValueError("All objects must be on the same device.")
            else:
                device = x.device

    for op_term, op_term_coeff in zip(op.op_terms, op.coeffs):
        _check_device(op_term_coeff)

        for op_prod, op_prod_coeff in zip(op_term.op_prods, op_term.coeffs):
            _check_device(op_prod_coeff)

            for base_op in op_prod:
                _check_device(base_op.data)

    for buf in state_in_bufs:
        _check_device(buf)

    return device


def detect_ad_traced_object(x: jax.Array | jax.core.Tracer) -> bool:
    """
    Detect if an object has been AD-traced.
    """
    if isinstance(x, jax.core.Tracer):
        trace = x._trace
        while hasattr(trace, "parent_trace"):
            if type(trace).__name__ in ["LinearizeTrace", "JVPTrace"]:
                return True
            trace = trace.parent_trace
    return False


def get_empty_scalar_callback() -> cudm.WrappedScalarCallback:
    """
    Return an empty scalar callback for gradient attachment.
    """
    def f(*args, **kwargs):
        return
    return cudm.WrappedScalarCallback(f, cudm.CallbackDevice.GPU)


def get_scalar_assignment_callback(dtype: jnp.dtype) -> cudm.WrappedScalarCallback:
    """
    Return a scalar assignment callback.
    """
    def f(t, args, storage):
        storage[:] = f.coeff[0]
    f.coeff = cp.zeros((1,), dtype=dtype)
    return cudm.WrappedScalarCallback(f, cudm.CallbackDevice.GPU)


def get_empty_tensor_callback() -> cudm.WrappedTensorCallback:
    """
    Return an empty tensor callback for gradient attachment.
    """
    def f(*args, **kwargs):
        return
    return cudm.WrappedTensorCallback(f, cudm.CallbackDevice.GPU)


def get_scalar_gradient_attachment_callback(shape: tuple[int, ...],
                                            dtype: jnp.dtype,
                                            ) -> cudm.WrappedScalarGradientCallback:
    """
    Return a scalar gradient attachment callback.
    """
    def f(t, args, scalar_grad, params_grad):
        f.scalar_grad += scalar_grad.reshape(f.scalar_grad.shape)

    # f.scalar_grad is created as a CuPy array to escape JAX tracing.
    f.scalar_grad = cp.zeros(shape, dtype=dtype)

    grad_callback = cudm.WrappedScalarGradientCallback(f, cudm.CallbackDevice.GPU)
    return grad_callback


def get_tensor_gradient_attachment_callback(shape: tuple[int, ...],
                                            dtype: jnp.dtype,
                                            ) -> cudm.WrappedTensorGradientCallback:
    """
    Return a tensor gradient attachment callback.
    """
    def f(t, args, tensor_grad, params_grad):
        # Transpose tensor_grad so that the batch dimension is the first dimension.
        transpose_inds = (tensor_grad.ndim - 1, *range(tensor_grad.ndim - 1))
        f.tensor_grad += tensor_grad.transpose(transpose_inds).reshape(f.tensor_grad.shape)

    # f.tensor_grad is created as a CuPy array to escape JAX tracing.
    f.tensor_grad = cp.zeros(shape, dtype=dtype)

    grad_callback = cudm.WrappedTensorGradientCallback(f, cudm.CallbackDevice.GPU)
    return grad_callback


def get_random_odd_pointer_and_object() -> tuple[int, ctypes.c_short]:
    """
    Return a random odd pointer and its ctypes object.

    Returns:
        tuple: (pointer, obj) where obj must be kept alive by the caller
               to ensure the pointer remains valid.
    """
    obj = ctypes.c_short()
    ptr = ctypes.addressof(obj) + 1
    assert ptr % 2 == 1, "Temporary pointer must be an odd number."
    return ptr, obj


def fuse_batched_inputs(
    unfused_inputs: tuple[jax.Array, ...],
    batch_axes: tuple[int | None, ...],
    physical_state_ndim: int,
) -> tuple[tuple[jax.Array, ...], tuple[int, ...], tuple[int, ...], tuple[bool, ...]]:
    """
    Fuse the current batch axis with any already-accumulated inner batch dimensions.

    Supports arbitrary nesting depth: at each batcher call the state buffers may
    carry a fused batch dim from inner vmap levels plus the current batch axis.
    Both are collapsed into one leading axis for the backend.

    Args:
        unfused_inputs: The input tensors.
        batch_axes: The batch axes as reported by JAX's batcher (None if unbatched).
        physical_state_ndim: Number of physical (non-batch) dimensions in the state,
            i.e. ``len(state_shape) - 1`` where ``state_shape`` is the kwarg that
            always has exactly one leading batch dimension.

    Returns:
        fused_inputs: The fused input tensors.
        batch_sizes_new: The size of the current batch axis for each input; 1 if unbatched.
        batch_sizes_accumulated: The size of the already-fused inner batch for each
            input; 1 when there are no prior vmap levels or the input is unbatched.
        accumulated_axes_present: Whether an already-fused inner batch axis is present
            for each input, independent of its extent — an accumulated axis of extent 1
            is still present and must be collapsed, unlike a genuinely absent one.
    """
    assert len(batch_axes) == len(unfused_inputs)

    # Derive batch_size_accumulated from the first state buffer. JAX drives one vmap level
    # per batcher call, and the fuse/unfuse round-trip below collapses all inner levels
    # into a single leading batch axis, so at this point the state carries at most one
    # already-fused "accumulated" batch dim: shape
    # [batch_size_new, batch_size_accumulated, *physical].
    state_arr_0 = unfused_inputs[0]
    batch_axis_0 = batch_axes[0]
    if batch_axis_0 is not None:
        # Move the current batch axis to 0 first (as the loop does), so shape[1] reads the
        # already-fused inner batch regardless of which axis JAX reports.
        state_arr_0_moved = jnp.moveaxis(state_arr_0, batch_axis_0, 0)
        num_extra = state_arr_0_moved.ndim - 1 - physical_state_ndim
        assert num_extra <= 1, (
            f"fuse_batched_inputs expects at most one already-fused inner batch axis, got "
            f"{num_extra} (ndim={state_arr_0_moved.ndim}, physical_state_ndim={physical_state_ndim})."
        )
        # Presence of the accumulated axis is a rank question (num_extra == 1); its extent
        # (batch_size_accumulated) is a separate question that can legitimately be 1.
        accumulated_axis_present = num_extra == 1
        batch_size_accumulated = state_arr_0_moved.shape[1] if accumulated_axis_present else 1
    else:
        accumulated_axis_present = False
        batch_size_accumulated = 1

    fused_inputs = []
    batch_sizes_new = []
    batch_sizes_accumulated = []
    accumulated_axes_present = []

    for arr, batch_axis in zip(unfused_inputs, batch_axes):
        if batch_axis is not None:
            arr = jnp.moveaxis(arr, batch_axis, 0)  # ensure new batch axis is at dim 0
            if accumulated_axis_present:
                # Flatten [batch_size_new, batch_size_accumulated, *rest] → [batch_size_new * batch_size_accumulated, *rest]
                batch_size_new = arr.shape[0]
                arr = arr.reshape(batch_size_new * batch_size_accumulated, *arr.shape[2:])
            batch_sizes_new.append(arr.shape[0] // batch_size_accumulated)
            batch_sizes_accumulated.append(batch_size_accumulated)
            accumulated_axes_present.append(accumulated_axis_present)
        else:
            batch_sizes_new.append(1)
            batch_sizes_accumulated.append(1)
            accumulated_axes_present.append(False)
        fused_inputs.append(arr)

    return tuple(fused_inputs), tuple(batch_sizes_new), tuple(batch_sizes_accumulated), tuple(accumulated_axes_present)


def unfuse_batched_outputs(
    fused_outputs: tuple[jax.Array, ...],
    batch_sizes_new: tuple[int, ...],
    batch_sizes_accumulated: tuple[int, ...],
    accumulated_axes_present: tuple[bool, ...],
) -> list[jax.Array]:
    """
    Unfuse the flat batch dimension produced by fuse_batched_inputs.

    For single vmap (no accumulated axis present) the output keeps shape
    ``[batch_size_new, *physical]``.  For nested vmap with an accumulated axis present
    (at any extent, including 1) it becomes ``[batch_size_new, batch_size_accumulated,
    *physical]``.  In both cases the current batch axis is at position 0, which is what
    JAX's batcher expects.

    Args:
        fused_outputs: Flat-batch output tensors from the primitive.
        batch_sizes_new: Size of the current batch axis per output.
        batch_sizes_accumulated: Size of the already-fused inner batch per output.
        accumulated_axes_present: Whether an already-fused inner batch axis is present
            per output, independent of its extent (see fuse_batched_inputs).

    Returns:
        List of unfused tensors with the new batch axis at dim 0.
    """
    unfused_outputs = []
    for out, batch_size_new, batch_size_accumulated, accumulated_present in zip(
        fused_outputs, batch_sizes_new, batch_sizes_accumulated, accumulated_axes_present
    ):
        # out.shape == [batch_size_new * batch_size_accumulated, *physical]
        # The fused leading dim is always present: operator_action guarantees the state
        # carries exactly one leading batch axis regardless of its extent.
        physical_shape = out.shape[1:]
        if accumulated_present:
            # Nested: split fused batch back into [batch_size_new, batch_size_accumulated, *physical]
            out = out.reshape(batch_size_new, batch_size_accumulated, *physical_shape)
        else:
            # Single level: [batch_size_new, *physical] — reshape is a no-op but kept explicit
            out = out.reshape(batch_size_new, *physical_shape)
        # batch axis is already at dim 0
        unfused_outputs.append(out)

    return unfused_outputs


def multidiagonal_to_dense(dia_data: jax.Array, diag_offsets: tuple[int, ...]) -> jax.Array:
    """
    Convert a multidiagonal data buffer to dense form.

    Args:
        dia_data: Array of shape ``(batch, mode_dim, num_diag)`` where each slice along
            the last axis stores one diagonal's values, padded at the end for off-diagonals.
        diag_offsets: Diagonal offsets; positive = superdiagonal, negative = subdiagonal.

    Returns:
        Dense array of shape ``(batch, mode_dim, mode_dim)``.
    """
    batched = dia_data.ndim == 3
    if not batched:
        dia_data = dia_data[jnp.newaxis]
    batch_size, mode_dim, _ = dia_data.shape
    dense_data = jnp.zeros((batch_size, mode_dim, mode_dim), dtype=dia_data.dtype)
    for i, offset in enumerate(diag_offsets):
        diag_len = mode_dim - abs(offset)
        if offset >= 0:
            rows = jnp.arange(diag_len)
        else:
            rows = jnp.arange(-offset, mode_dim)
        dense_data = dense_data.at[:, rows, rows + offset].set(dia_data[:, :diag_len, i])
    return dense_data if batched else dense_data[0]


def multidiagonal_add(data_a: jax.Array,
                      diag_offsets_a: tuple[int, ...],
                      data_b: jax.Array,
                      diag_offsets_b: tuple[int, ...]
                      ) -> tuple[jax.Array, tuple[int, ...]]:
    """
    Add two multidiagonal data buffers, producing a buffer spanning the union of offsets.

    Args:
        data_a: Array of shape ``(batch, mode_dim, num_diag_a)``.
        diag_offsets_a: Diagonal offsets of ``data_a``.
        data_b: Array of shape ``(batch, mode_dim, num_diag_b)``.
        diag_offsets_b: Diagonal offsets of ``data_b``.

    Returns:
        Tuple of ``(data, diag_offsets)`` where ``data`` has shape
        ``(batch, mode_dim, num_diag_union)`` and ``diag_offsets`` is the union of offsets.
    """
    # Batchedness is judged per operand: an operand may carry a leading batch axis while the
    # other does not, and the result is batched if either is.
    batched_a, batched_b = data_a.ndim == 3, data_b.ndim == 3
    if not batched_a:
        data_a = data_a[jnp.newaxis]
    if not batched_b:
        data_b = data_b[jnp.newaxis]
    batched = batched_a or batched_b
    diag_offsets = tuple(sorted(set(diag_offsets_a) | set(diag_offsets_b)))
    mode_dim = data_a.shape[1]
    batch_size = max(data_a.shape[0], data_b.shape[0])
    data = jnp.zeros(
        (batch_size, mode_dim, len(diag_offsets)),
        dtype=jnp.promote_types(data_a.dtype, data_b.dtype),  # dense arithmetic is implicitly promoting dtypes
    )
    for kc_idx, kc in enumerate(diag_offsets):
        if kc in diag_offsets_a:
            data = data.at[:, :, kc_idx].add(data_a[:, :, diag_offsets_a.index(kc)])
        if kc in diag_offsets_b:
            data = data.at[:, :, kc_idx].add(data_b[:, :, diag_offsets_b.index(kc)])
    return (data if batched else data[0]), diag_offsets


def multidiagonal_matmul(data_a: jax.Array,
                         diag_offsets_a: tuple[int, ...],
                         data_b: jax.Array,
                         diag_offsets_b: tuple[int, ...]
                         ) -> tuple[jax.Array, tuple[int, ...]]:
    """
    Multiply two DIA data buffers and return the result in DIA format.

    Args:
        data_a: Array of shape ``(batch, mode_dim, num_diag_a)``.
        diag_offsets_a: Diagonal offsets of ``data_a``.
        data_b: Array of shape ``(batch, mode_dim, num_diag_b)``.
        diag_offsets_b: Diagonal offsets of ``data_b``.
    """
    # Batchedness is judged per operand: an operand may carry a leading batch axis while the
    # other does not, and the result is batched if either is.
    batched_a, batched_b = data_a.ndim == 3, data_b.ndim == 3
    if not batched_a:
        data_a = data_a[jnp.newaxis]
    if not batched_b:
        data_b = data_b[jnp.newaxis]
    batched = batched_a or batched_b
    batch_size = max(data_a.shape[0], data_b.shape[0])
    mode_dim = data_a.shape[1]

    # Obtain the result diagonal offsets and create result data buffer.
    diag_offsets = tuple(sorted(
        {ka + kb for ka in diag_offsets_a for kb in diag_offsets_b if abs(ka + kb) < mode_dim}
    ))
    data = jnp.zeros(
        (batch_size, mode_dim, len(diag_offsets)),
        dtype=jnp.promote_types(data_a.dtype, data_b.dtype),
    )

    # Broadcast the input data buffers to batch size.
    data_a = jnp.broadcast_to(data_a, (batch_size, *data_a.shape[1:]))
    data_b = jnp.broadcast_to(data_b, (batch_size, *data_b.shape[1:]))

    for ika, ka in enumerate(diag_offsets_a):
        for ikb, kb in enumerate(diag_offsets_b):
            # Obtain result diagonal offset and its index.
            kc = ka + kb
            if abs(kc) >= mode_dim:
                continue
            ikc = diag_offsets.index(kc)

            # Row indices used in this iteration, as the intersection of two ranges:
            #   - rows valid for a: if ka >= 0, 0 to mode_dim - ka; if ka < 0, -ka to mode_dim.
            #   - rows whose image under kc stays in range, i.e. 0 <= rows + kc < mode_dim.
            # Both bounds are Python ints, so the arange has a statically known length. Filtering
            # with a boolean mask instead would make the length data-dependent, which is not
            # traceable (NonConcreteBooleanIndexError under jit).
            low = max(max(0, -ka), max(0, -kc))
            high = min(mode_dim - max(0, ka), mode_dim - max(0, kc))
            if high <= low:  # no rows contribute to this diagonal pair
                continue
            rows = jnp.arange(low, high)

            # Obtain the positions of the elements in each column in the DIA representation. 
            # Note that max(0, -ka) is the start row index of a, so to obtain the position in the
            # DIA representation we need to subtract that offset.
            pos_a = rows - max(0, -ka)
            pos_b = rows + ka - max(0, -kb)  # rows for b is rows + ka
            pos_c = rows - max(0, -kc)

            # Add product of the used diagonal elements to data buffer for c.
            data = data.at[:, pos_c, ikc].add(data_a[:, pos_a, ika] * data_b[:, pos_b, ikb])

    return (data if batched else data[0]), diag_offsets


def multidiagonal_kron(data_a: jax.Array,
                       diag_offsets_a: tuple[int, ...],
                       data_b: jax.Array,
                       diag_offsets_b: tuple[int, ...]
                       ) -> tuple[jax.Array, tuple[int, ...]]:
    """
    Kronecker product of two DIA data buffers and return the result in DIA format.

    For ``C = A ⊗ B``, ``C[i*mode_dim_b+k, j*mode_dim_b+l] = A[i,j] * B[k,l]``, so diagonals
    ``ka`` of A and ``kb`` of B contribute to diagonal ``kc = ka*mode_dim_b + kb`` of C.

    Args:
        data_a: Array of shape ``(batch, mode_dim_a, num_diag_a)``.
        diag_offsets_a: Diagonal offsets of ``data_a``.
        data_b: Array of shape ``(batch, mode_dim_b, num_diag_b)``.
        diag_offsets_b: Diagonal offsets of ``data_b``.
    """
    # Batchedness is judged per operand: an operand may carry a leading batch axis while the
    # other does not, and the result is batched if either is.
    batched_a, batched_b = data_a.ndim == 3, data_b.ndim == 3
    if not batched_a:
        data_a = data_a[jnp.newaxis]
    if not batched_b:
        data_b = data_b[jnp.newaxis]
    batched = batched_a or batched_b
    batch_size = max(data_a.shape[0], data_b.shape[0])
    mode_dim_a, mode_dim_b = data_a.shape[1], data_b.shape[1]
    mode_dim = mode_dim_a * mode_dim_b

    # Obtain the result diagonal offsets and create result data buffer.
    diag_offsets = tuple(sorted(
        {ka * mode_dim_b + kb for ka in diag_offsets_a for kb in diag_offsets_b
        if abs(ka * mode_dim_b + kb) < mode_dim}
    ))
    data = jnp.zeros(
        (batch_size, mode_dim, len(diag_offsets)),
        dtype=jnp.promote_types(data_a.dtype, data_b.dtype),
    )

    # Broadcast the input data buffers to batch size.
    data_a = jnp.broadcast_to(data_a, (batch_size, mode_dim_a, len(diag_offsets_a)))
    data_b = jnp.broadcast_to(data_b, (batch_size, mode_dim_b, len(diag_offsets_b)))

    for ika, ka in enumerate(diag_offsets_a):
        for ikb, kb in enumerate(diag_offsets_b):
            # Obtain result diagonal offset and its index.
            kc = ka * mode_dim_b + kb
            if abs(kc) >= mode_dim:
                continue
            ikc = diag_offsets.index(kc)

            # Row indices within A and B for their respective diagonals.
            rows_a = jnp.arange(max(0, -ka), mode_dim_a - max(0, ka))
            rows_b = jnp.arange(max(0, -kb), mode_dim_b - max(0, kb))

            # Enumerate all (rows_a, rows_b) pairs; C's row is rows_a*mode_dim_b + rows_b.
            rows_a, rows_b = jnp.meshgrid(rows_a, rows_b, indexing='ij')
            rows_a, rows_b = rows_a.ravel(), rows_b.ravel()
            rows = rows_a * mode_dim_b + rows_b

            # Obtain the positions of the elements in each column in the DIA representation.
            # max(0, -ka) and max(0, -kb) are the start row indices of A and B's diagonals,
            # so subtract them to obtain DIA positions; max(0, -kc) is the start row of C's diagonal.
            pos_a = rows_a - max(0, -ka)
            pos_b = rows_b - max(0, -kb)
            pos_c = rows - max(0, -kc)

            data = data.at[:, pos_c, ikc].add(data_a[:, pos_a, ika] * data_b[:, pos_b, ikb])

    return (data if batched else data[0]), diag_offsets


def dense_data_dag(data: jax.Array) -> jax.Array:
    """
    Conjugate-transpose a dense operator data tensor of shape ``[*mode_extents, *mode_extents]``
    or ``[batch, *mode_extents, *mode_extents]`` by swapping bra and ket axes and applying
    ``jnp.conj``. Batchedness is inferred from ndim parity: odd ndim means batched.
    """
    batched = data.ndim % 2 == 1
    offset = 1 if batched else 0
    num_modes = (data.ndim - offset) // 2
    perm = (
        ((0,) if batched else ())
        + tuple(range(offset + num_modes, offset + 2 * num_modes))
        + tuple(range(offset, offset + num_modes))
    )
    return jnp.conj(jnp.transpose(data, perm))


def dense_batched_matmul(left: jax.Array, right: jax.Array) -> jax.Array:
    """
    Batched matrix multiplication of two dense operator data tensors with shape
    ``[*mode_extents, *mode_extents]`` or ``[batch, *mode_extents, *mode_extents]``.
    Batchedness is inferred from ndim parity: odd ndim means batched.
    Batch dimensions are broadcast: ``batch=1`` is compatible with ``batch=N``.
    """
    # Batchedness is judged per operand: an operand may carry a leading batch axis while the
    # other does not, and the result is batched if either is.
    batched_left = left.ndim % 2 == 1
    batched_right = right.ndim % 2 == 1
    offset = 1 if batched_left else 0
    num_modes = (left.ndim - offset) // 2
    # Both operands share mode_extents for matmul, so the left-derived matrix_dim applies to both.
    dims = left.shape[offset:offset + num_modes]
    matrix_dim = math.prod(dims)

    # Reshaping to a leading axis of -1 gives batch 1 for an unbatched operand, which matmul
    # then broadcasts against a batch of N.
    left_flat = left.reshape(-1, matrix_dim, matrix_dim)
    right_flat = right.reshape(-1, matrix_dim, matrix_dim)
    out = left_flat @ right_flat
    if batched_left or batched_right:
        return out.reshape(-1, *dims, *dims)
    else:
        return out.reshape(*dims, *dims)


def dense_batched_kron(left: jax.Array, right: jax.Array) -> jax.Array:
    """
    Batched Kronecker product of two dense operator data tensors with shape
    ``[*mode_extents, *mode_extents]`` or ``[batch, *mode_extents, *mode_extents]``.
    Batchedness is inferred from ndim parity: odd ndim means batched.
    Batch dimensions are broadcast: ``batch=1`` is compatible with ``batch=N``. The result has
    shape ``[*dims_left, *dims_right, *dims_left, *dims_right]`` (non-batched) or
    ``[batch, *dims_left, *dims_right, *dims_left, *dims_right]`` (batched).
    """
    # Batchedness is judged per operand: an operand may carry a leading batch axis while the
    # other does not, and the result is batched if either is. Unlike matmul, the two operands
    # act on different subspaces, so each operand's dims must be read through its own offset.
    batched_left = left.ndim % 2 == 1
    batched_right = right.ndim % 2 == 1
    offset_left = 1 if batched_left else 0
    offset_right = 1 if batched_right else 0
    num_modes_left = (left.ndim - offset_left) // 2
    num_modes_right = (right.ndim - offset_right) // 2
    dims_left = left.shape[offset_left:offset_left + num_modes_left]
    dims_right = right.shape[offset_right:offset_right + num_modes_right]
    matrix_dim_left = math.prod(dims_left)
    matrix_dim_right = math.prod(dims_right)

    left_flat = left.reshape(-1, matrix_dim_left, matrix_dim_left)
    right_flat = right.reshape(-1, matrix_dim_right, matrix_dim_right)
    batch_dim = max(left_flat.shape[0], right_flat.shape[0])
    left_flat = jnp.broadcast_to(left_flat, (batch_dim, matrix_dim_left, matrix_dim_left))
    right_flat = jnp.broadcast_to(right_flat, (batch_dim, matrix_dim_right, matrix_dim_right))
    out = jnp.einsum('bij,bkl->bikjl', left_flat, right_flat)
    out = out.reshape(-1, *dims_left, *dims_right, *dims_left, *dims_right)
    if batched_left or batched_right:
        return out
    else:
        return out[0]


def pad_with_identities(data: jax.Array,
                        modes: tuple[int, ...],
                        modes_target: tuple[int, ...],
                        dims: tuple[int, ...]
                        ) -> jax.Array:
    """
    Pad dense operator data with identities so that it acts on the target modes.

    Batchedness is inferred from ndim parity, consistently with dense_batched_kron.

    Args:
        data: Dense operator data with shape ``[*mode_extents, *mode_extents]`` or
            ``[batch, *mode_extents, *mode_extents]``.
        modes: Modes the operator data acts on.
        modes_target: Modes of the target Hilbert space.
        dims: Dimensions indexed by mode number.

    Returns:
        The padded operator data, acting on ``modes_target`` in order.
    """
    modes_set = set(modes)
    missing = [m for m in modes_target if m not in modes_set]

    batched = data.ndim % 2 == 1
    batch_size = data.shape[0] if batched else 1

    # Build: data ⊗ I_{missing[0]} ⊗ I_{missing[1]} ⊗ ...
    result = data
    for m in missing:
        eye = jnp.eye(dims[m], dtype=data.dtype)
        if batched:
            eye = jnp.broadcast_to(eye, (batch_size, dims[m], dims[m]))
        result = dense_batched_kron(result, eye)

    # result now acts on extended_modes = list(modes) + missing, in that order.
    # Permute axes so the final data acts on modes_target in order.
    extended_modes = list(modes) + missing

    # A duplicate mode makes extended_modes.index(m) below ambiguous (it silently picks the
    # first match), which would build a malformed, non-bijective perm rather than raise --
    # e.g. two base operators claiming the same mode via different dualities produce a
    # modes_target with a repeated entry. Catch that here with a clear error instead.
    if len(set(modes)) != len(modes):
        raise ValueError(f"Duplicate modes in modes={modes}.")
    if len(set(modes_target)) != len(modes_target):
        raise ValueError(f"Duplicate modes in modes_target={modes_target}.")
    if not set(modes_target) <= set(extended_modes):
        raise ValueError(
            f"modes_target={modes_target} is not a subset of extended_modes={extended_modes} "
            f"(modes={modes}, missing={missing})."
        )
    # Padding only ever embeds outward, so modes must be covered by modes_target. Without this
    # the `missing` loop above finds nothing to tensor in and the operator is returned on its
    # original, larger mode space -- silently via the identity-permutation early return below.
    if not set(modes) <= set(modes_target):
        raise ValueError(
            f"modes={modes} is not a subset of modes_target={modes_target}; "
            f"pad_with_identities cannot drop modes."
        )

    perm = [extended_modes.index(m) for m in modes_target]

    if perm == list(range(len(modes_target))):
        return result

    n = len(modes_target)
    if not batched:
        result = result[jnp.newaxis]
    perm_full = (0,) + tuple(1 + p for p in perm) + tuple(1 + n + p for p in perm)
    result = jnp.transpose(result, perm_full)
    return result if batched else result[0]


def padded_matrix_product(data: tuple[jax.Array, ...],
                          modes: tuple[tuple[int, ...], ...],
                          modes_target: tuple[int, ...],
                          dims: tuple[int, ...],
                          dual: bool = False,
                          ) -> jax.Array:
    """
    Compute the matrix product of dense operator data embedded into the target Hilbert space.

    Args:
        data: Dense operator data of the base operators to take the padded matrix product of.
        modes: Modes acted on by each base operator, one tuple per entry of ``data``.
        modes_target: Modes of the target Hilbert space.
        dims: Dimensions indexed by mode number.
        dual: If False (ket), compute data[-1] @ ... @ data[0] (data[0] applied first).
              If True (bra), reverse the product order: data[0] @ ... @ data[-1].

    Returns:
        The matrix product of the operator data embedded into the target Hilbert space.
    """
    assert {m for modes_base_op in modes for m in modes_base_op} <= set(modes_target)
    assert set(modes_target) <= set(range(len(dims)))

    # TODO: Currently, this function only support a single duality. In the future, we should support
    # mixed dualities. When duality is True, we should reverse the order of matrix multiplications.
    if dual:
        data = data[::-1]
        modes = modes[::-1]

    result = pad_with_identities(data[0], modes[0], modes_target, dims)
    for i in range(1, len(data)):
        padded = pad_with_identities(data[i], modes[i], modes_target, dims)
        result = dense_batched_matmul(padded, result)

    return result
