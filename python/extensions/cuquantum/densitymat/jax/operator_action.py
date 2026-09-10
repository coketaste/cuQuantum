# Copyright (c) 2025-2026, NVIDIA CORPORATION & AFFILIATES
#
# SPDX-License-Identifier: BSD-3-Clause

import logging
from functools import partial
from collections.abc import Sequence

import jax

from .pysrc.context import CudensitymatContext
from .pysrc.operator import Operator
from .pysrc.operator_action_prim import (
    operator_action_prim,
    operator_action_backward_diff_prim,
)
from .utils import (
    BufferMetadata,
    maybe_expand_dim,
    maybe_squeeze_dim,
    get_state_batch_size_and_purity,
    check_and_return_final_batch_size,
    check_and_return_device,
    get_original_shape,
    is_vmap_traced,
)


logger = logging.getLogger("cudensitymat-jax.operator_action")


_ffi_registered = False
_InputType = None


def _register_ffi_targets():
    global _ffi_registered, _InputType
    if _ffi_registered:
        return

    from cuquantum.bindings._internal import cudensitymat as _cudm
    _cudm._inspect_function_pointers()  # for loading libcudensitymat.so

    from cuquantum.lib import cudensitymat_jax
    _InputType = cudensitymat_jax.InputType

    for _name, _value in cudensitymat_jax.registrations().items():
        jax.ffi.register_ffi_target(_name, _value, platform="CUDA")
    _ffi_registered = True


def operator_action(op: Operator,
                    state_in_bufs: jax.Array | Sequence[jax.Array],
                    device: jax.Device | None = None,
                    ) -> jax.Array | list[jax.Array]:
    """
    Compute the action of an operator on a state.

    Args:
        op: Operator to compute the action of.
        state_in_bufs: Buffers of the input state components.
        device: Device to use for the operator action.

    Returns:
        Buffers of the output state components.
    """
    logger.info("Calling operator_action")

    _register_ffi_targets()

    if not jax.config.jax_enable_x64:
        raise RuntimeError("jax_enable_x64 must be set to True to use cuQuantum Python JAX")

    # Process input arguments.
    if isinstance(state_in_bufs, jax.Array):
        state_in_bufs = (state_in_bufs,)
    else:
        state_in_bufs = tuple(state_in_bufs)

    # Check and set device from op and state.
    device = check_and_return_device(op, state_in_bufs)

    # If device is None, as in the case of tracing, set it to the first GPU device.
    if device is None:
        devices = jax.devices('gpu')
        if len(devices) == 0:
            raise ValueError("No GPU devices found.")
        device = devices[0]
        logger.info("No device specified, using the first GPU device.")

    if op._ptr is None:
        op._simplify()
    op._update_metadata()  # update batch sizes

    # Check state shape and maybe expand to a leading batch dimension.
    state_batch_size, purity, has_explicit_batch = get_state_batch_size_and_purity(
        state_in_bufs, op.dims)
    batch_size = check_and_return_final_batch_size(state_batch_size, op._batch_size)

    state_in_bufs, did_expand = maybe_expand_dim(state_in_bufs, has_explicit_batch)
    # For nested vmap, get_original_shape would return a multi-dim batch prefix
    # (outer_B, inner_B, *physical), which breaks maybe_create_state_context's
    # single-leading-batch assumption. Fuse all batch levels into one leading dim.
    if is_vmap_traced(state_in_bufs[0]):
        state_shape = (batch_size, *state_in_bufs[0].shape)
    else:
        state_shape = tuple(state_in_bufs[0].shape)  # already has batch from maybe_expand_dim

    # Prepare library context for forward operator action.
    # NOTE: Assuming a single state component.
    if len(state_in_bufs) > 1:
        raise NotImplementedError("More than one state component is not implemented.")

    CudensitymatContext.maybe_create_operator_context(op, batch_size)
    CudensitymatContext.maybe_create_state_context(purity, state_shape, batch_size, state_in_bufs[0].dtype)

    # Create metadata objects for the other inputs.
    op_term_coeff_metadata = BufferMetadata()
    op_prod_coeff_metadata = BufferMetadata()
    base_op_metadata = BufferMetadata()

    # Create metadata objects for the gradient outputs.
    op_term_coeff_grad_metadata = BufferMetadata()
    op_prod_coeff_grad_metadata = BufferMetadata()
    base_op_grad_metadata = BufferMetadata()

    # Extract temporary batched coefficient buffers for operator terms.
    for i, (
        op_term,
        op_term_coeff,
        op_term_coeff_ptr,
        op_term_coeff_grad_ptr,
        op_term_total_coeffs_ptr,
    ) in enumerate(zip(
        op.op_terms,
        op.coeffs,
        op._coeff_ptrs,
        op._coeff_grad_ptrs,
        op._total_coeffs_ptrs,
        strict=True
    )):
        is_op_term_coeff_batched = op._op_term_batch_sizes[i] != 1
        if is_op_term_coeff_batched:
            dynamic_ptr = op_term_total_coeffs_ptr
            dynamic_type = _InputType.OPERATOR_TERM_BATCHED_COEFFS.value
            if dynamic_ptr == 0:
                raise RuntimeError("Missing total coefficient pointer for batched operator term coefficient.")
        else:
            dynamic_ptr = op_term_coeff_ptr
            dynamic_type = _InputType.NON_BATCHED_COEFFS.value

        if dynamic_ptr not in op_term_coeff_metadata.ptrs:
            op_term_coeff_metadata.indices.append(i)
            op_term_coeff_metadata.types.append(dynamic_type)
            op_term_coeff_metadata.ptrs.append(dynamic_ptr)

        if op_term_coeff_grad_ptr != 0 and op_term_coeff_grad_ptr not in op_term_coeff_grad_metadata.ptrs:
            op_term_coeff_grad_metadata.indices.append(i)
            op_term_coeff_grad_metadata.ptrs.append(op_term_coeff_grad_ptr)
            # The gradient buffer is always sized (batch_size,) regardless of whether the
            # coefficient is batched, so JAX must read batch_size elements from the ptr.
            shape_dtype = jax.ShapeDtypeStruct((batch_size,), op_term_coeff.dtype)
            op_term_coeff_grad_metadata.shape_dtypes.append(shape_dtype)

        # Extract temporary batched coefficient buffers for operator products.
        for j, (
            op_prod,
            op_prod_coeff,
            op_prod_coeff_ptr,
            op_prod_coeff_grad_ptr,
            op_prod_total_coeffs_ptr,
        ) in enumerate(zip(
            op_term.op_prods,
            op_term.coeffs,
            op_term._coeff_ptrs,
            op_term._coeff_grad_ptrs,
            op_term._total_coeffs_ptrs,
            strict=True,
        )):
            is_op_prod_coeff_batched = op_term._op_prod_batch_sizes[j] != 1
            if is_op_prod_coeff_batched:
                dynamic_ptr = op_prod_total_coeffs_ptr
                dynamic_type = _InputType.OPERATOR_PRODUCT_BATCHED_COEFFS.value
                if dynamic_ptr == 0:
                    raise RuntimeError("Missing total coefficient pointer for batched operator product coefficient.")
            else:
                dynamic_ptr = op_prod_coeff_ptr
                dynamic_type = _InputType.NON_BATCHED_COEFFS.value

            if dynamic_ptr not in op_prod_coeff_metadata.ptrs:
                op_prod_coeff_metadata.indices.append((i, j))
                op_prod_coeff_metadata.types.append(dynamic_type)
                op_prod_coeff_metadata.ptrs.append(dynamic_ptr)

            if op_prod_coeff_grad_ptr != 0 and op_prod_coeff_grad_ptr not in op_prod_coeff_grad_metadata.ptrs:
                op_prod_coeff_grad_metadata.indices.append((i, j))
                op_prod_coeff_grad_metadata.ptrs.append(op_prod_coeff_grad_ptr)
                # The gradient buffer is always sized (batch_size,) regardless of whether the
                # coefficient is batched, so JAX must read batch_size elements from the ptr.
                shape_dtype = jax.ShapeDtypeStruct((batch_size,), op_prod_coeff.dtype)
                op_prod_coeff_grad_metadata.shape_dtypes.append(shape_dtype)

            # Assign certain static attributes to the operator action context.
            for k, base_op in enumerate(op_prod):
                # Checking whether base_op._ptr is None since only the unique base operators need
                # to be buffer-attached.
                if base_op._ptr is not None and base_op._ptr not in base_op_metadata.ptrs:
                    base_op_metadata.indices.append((i, j, k))
                    base_op_metadata.types.append(
                        _InputType.ELEMENTARY_OPERATOR.value if base_op._is_elementary
                        else _InputType.MATRIX_OPERATOR.value
                    )
                    base_op_metadata.ptrs.append(base_op._ptr)

                if base_op._grad_ptr != 0 and base_op._grad_ptr not in base_op_grad_metadata.ptrs:
                    base_op_grad_metadata.indices.append((i, j, k))
                    base_op_grad_metadata.ptrs.append(base_op._grad_ptr)
                    data_shape = get_original_shape(base_op.data)
                    if is_vmap_traced(base_op.data) or base_op._batch_size > 1:
                        # data_shape already includes the batch dim: either fused in by
                        # get_original_shape for a vmap trace (at any extent, including a
                        # size-1 vmap axis), or baked into the concrete array's own shape
                        # for materialized (non-vmap) batching, where _batch_size > 1 is a
                        # rank-verified presence signal (elementary_operator.py's
                        # `ndim % 2 == 1` check), not an extent guess.
                        grad_shape = data_shape
                    else:
                        # Genuinely unbatched: cudensitymat writes batch_size gradient tensors
                        # with batch as the last dimension, so buffer has (batch_size, *data_shape).
                        grad_shape = (batch_size, *data_shape)
                    shape_dtype = jax.ShapeDtypeStruct(grad_shape, base_op.data.dtype)
                    base_op_grad_metadata.shape_dtypes.append(shape_dtype)

    # Combine types and pointers in the same order as they are unpacked in primitive wrappers.
    other_in_metadata = op_term_coeff_metadata + op_prod_coeff_metadata + base_op_metadata
    other_out_metadata = op_term_coeff_grad_metadata + op_prod_coeff_grad_metadata + base_op_grad_metadata

    num_state_components = len(state_in_bufs)

    # Invoke operator action.
    state_out_bufs = _operator_action(
        op,
        state_in_bufs,
        device,
        batch_size,
        num_state_components,
        purity,
        tuple(other_in_metadata.types),
        tuple(other_in_metadata.ptrs),
        tuple(other_out_metadata.shape_dtypes),
        tuple(other_out_metadata.ptrs),
        tuple(op_term_coeff_metadata.indices),
        tuple(op_prod_coeff_metadata.indices),
        tuple(base_op_metadata.indices),
        tuple(op_term_coeff_grad_metadata.indices),
        tuple(op_prod_coeff_grad_metadata.indices),
        tuple(base_op_grad_metadata.indices),
        state_shape,
    )

    # Undo the leading batch dim when it was added by maybe_expand_dim (single-state, non-vmap).
    state_out_bufs = maybe_squeeze_dim(state_out_bufs, did_expand)

    # Process output argument.
    if len(state_out_bufs) == 1:
        state_out_bufs = state_out_bufs[0]

    return state_out_bufs


@partial(jax.custom_vjp, nondiff_argnums=(2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16))
def _operator_action(op: Operator,
                     state_in_bufs: tuple[jax.Array, ...],
                     device: jax.Device,
                     batch_size: int,
                     num_state_components: int,
                     purity,
                     other_in_types: tuple[int, ...],
                     other_in_ptrs: tuple[int, ...],
                     other_out_shape_dtypes: tuple[jax.ShapeDtypeStruct, ...],
                     other_out_ptrs: tuple[int, ...],
                     op_term_coeffs_indices: tuple[int, ...],
                     op_prod_coeffs_indices: tuple[int, ...],
                     base_op_indices: tuple[int, ...],
                     op_term_coeff_grad_indices: tuple[int, ...],
                     op_prod_coeff_grad_indices: tuple[tuple[int, int], ...],
                     base_op_grad_indices: tuple[tuple[int, int, int], ...],
                     state_shape: tuple[int, ...],
                     ) -> list[jax.Array]:
    """
    Custom VJP rule for operator_action.
    """
    logger.info("Calling _operator_action")
    state_out_bufs, _ = _operator_action_fwd(
        op,
        state_in_bufs,
        device,
        batch_size,
        num_state_components,
        purity,
        other_in_types,
        other_in_ptrs,
        other_out_shape_dtypes,
        other_out_ptrs,
        op_term_coeffs_indices,
        op_prod_coeffs_indices,
        base_op_indices,
        op_term_coeff_grad_indices,
        op_prod_coeff_grad_indices,
        base_op_grad_indices,
        state_shape,
    )
    return state_out_bufs


def _operator_action_fwd(op: Operator,
                         state_in_bufs: tuple[jax.Array, ...],
                         device: jax.Device,
                         batch_size: int,
                         num_state_components: int,
                         purity,
                         other_in_types: tuple[int, ...],
                         other_in_ptrs: tuple[int, ...],
                         other_out_shape_dtypes: tuple[jax.ShapeDtypeStruct, ...],
                         other_out_ptrs: tuple[int, ...],
                         op_term_coeffs_indices: tuple[int, ...],
                         op_prod_coeffs_indices: tuple[int, ...],
                         base_op_indices: tuple[int, ...],
                         op_term_coeff_grad_indices: tuple[int, ...],
                         op_prod_coeff_grad_indices: tuple[tuple[int, int], ...],
                         base_op_grad_indices: tuple[tuple[int, int, int], ...],
                         state_shape: tuple[int, ...],
                         ) -> tuple[list[jax.Array], tuple[Operator, tuple[jax.Array, ...]]]:
    """
    Forward rule for operator_action.
    """
    logger.info("Calling _operator_action_fwd")
    state_out_bufs = operator_action_prim(
        op,
        state_in_bufs,
        device,
        batch_size,
        num_state_components,
        purity,
        other_in_types,
        other_in_ptrs,
        op_term_coeffs_indices,
        op_prod_coeffs_indices,
        base_op_indices,
        state_shape,
    )
    return state_out_bufs, (op, state_in_bufs)


def _operator_action_bwd(device: jax.Device,
                         batch_size: int,
                         num_state_components: int,
                         purity,
                         other_in_types: tuple[int, ...],
                         other_in_ptrs: tuple[int, ...],
                         other_out_shape_dtypes: tuple[jax.ShapeDtypeStruct, ...],
                         other_out_ptrs: tuple[int, ...],
                         op_term_coeffs_indices: tuple[int, ...],
                         op_prod_coeffs_indices: tuple[int, ...],
                         base_op_indices: tuple[int, ...],
                         op_term_coeff_grad_indices: tuple[int, ...],
                         op_prod_coeff_grad_indices: tuple[tuple[int, int], ...],
                         base_op_grad_indices: tuple[tuple[int, int, int], ...],
                         state_shape: tuple[int, ...],
                         res: tuple[Operator, tuple[jax.Array, ...]],
                         state_out_adj_bufs: jax.Array | Sequence[jax.Array]
                         ) -> tuple[Operator, tuple[jax.Array, ...]]:
    """
    Backward rule for operator_action.

    Args:
        device: Device for the operator action.
        batch_size: Batch size of the operator action.
        num_state_components: Number of state components.
        other_in_types: Non-differentiable argument from forward.
        other_in_ptrs: Non-differentiable argument from forward.
        op_term_coeffs_indices: Non-differentiable argument from forward.
        op_prod_coeffs_indices: Non-differentiable argument from forward.
        base_op_indices: Non-differentiable argument from forward.
        op_term_coeff_grad_indices: Non-differentiable argument from forward.
        op_prod_coeff_grad_indices: Non-differentiable argument from forward.
        base_op_grad_indices: Non-differentiable argument from forward.
        res: Residuals from forward pass.
        state_out_adj_bufs: Data buffers of the output state adjoint.
    """
    logger.info("Calling _operator_action_bwd")

    op, state_in_bufs = res

    # Prepare library context for backward operator action
    state_ctx = CudensitymatContext.get_state_context(purity, state_shape, batch_size, state_in_bufs[0].dtype)
    state_ctx.create_adjoint_buffers()

    # Process input argument.
    if isinstance(state_out_adj_bufs, jax.Array):
        state_out_adj_bufs = (state_out_adj_bufs,)
    else:
        state_out_adj_bufs = tuple(state_out_adj_bufs)

    if len(state_in_bufs) != len(state_out_adj_bufs):
        raise ValueError("state_in_bufs and state_out_adj_bufs must have the same number of components.")

    op_grad, state_in_adj_bufs = operator_action_backward_diff_prim(
        op,
        state_in_bufs,
        state_out_adj_bufs,
        device,
        batch_size,
        num_state_components,
        state_shape,
        purity,
        other_in_types,
        other_in_ptrs,
        other_out_shape_dtypes,
        other_out_ptrs,
        op_term_coeffs_indices,
        op_prod_coeffs_indices,
        base_op_indices,
        op_term_coeff_grad_indices,
        op_prod_coeff_grad_indices,
        base_op_grad_indices,
    )

    if len(state_in_adj_bufs) == 1:
        state_in_adj_bufs = state_in_adj_bufs[0]

    return op_grad, state_in_adj_bufs


_operator_action.defvjp(_operator_action_fwd, _operator_action_bwd)
