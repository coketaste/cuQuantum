# Copyright (c) 2025-2026, NVIDIA CORPORATION & AFFILIATES
#
# SPDX-License-Identifier: BSD-3-Clause

"""
Operator class in cuDensityMat.
"""

import ctypes
import logging
from collections.abc import Sequence
import math

import cupy as cp
import jax
import jax.numpy as jnp

from cuquantum.bindings import cudensitymat as cudm

from .operator_term import OperatorTerm
from .simplifier_config import SimplifierConfig
from ..utils import (
    get_batch_size,
    is_vmap_traced,
    get_scalar_assignment_callback,
    get_empty_scalar_callback,
    get_scalar_gradient_attachment_callback,
    get_random_odd_pointer_and_object,
    detect_ad_traced_object,
)


@jax.tree_util.register_pytree_node_class
class Operator:
    """
    PyTree class for cuDensityMat's operator.
    """

    logger = logging.getLogger("cudensitymat-jax.Operator")

    def __init__(self,
                 dims: Sequence[int],
                 simplify: bool | SimplifierConfig = False,
                 ) -> None:
        """
        Initialize an Operator object.

        Args:
            dims: Hilbert space dimensions.
            simplify: Controls operator-product simplification at compile time. ``True``
                uses the default passes; ``False`` disables all passes; a
                :class:`SimplifierConfig` object selects custom conditions. Opt-in for now:
                the passes are newly added, their failure mode is a silently different
                result rather than an error, and their benefit has not been quantified.
        """
        # Attributes set from constructor.
        self.dims: tuple[int, ...] = tuple(dims)
        if simplify is True:
            self._simplifier_config = SimplifierConfig()
        elif simplify is False:
            self._simplifier_config = SimplifierConfig(kron_cond=(), sum_cond=())
        elif isinstance(simplify, SimplifierConfig):
            self._simplifier_config = simplify
        else:
            # The branches above test identity, so truthy stand-ins for the booleans (1, 0, and
            # numpy's np.True_/np.False_, none of which are the bool singletons) would otherwise
            # fall through and only surface as an AttributeError on kron_cond deep inside
            # operator_action, far from the call that caused it.
            raise TypeError(
                f"simplify must be a bool or a SimplifierConfig, got {type(simplify)}."
            )

        # Attributes for arguments in append.
        self.op_terms: list[OperatorTerm] = []
        self.duals: list[bool] = []
        self.coeffs: list[jax.Array] = []

        # Attributes inferred from multiple append calls.
        self._op_term_batch_sizes: list[int] = []  # keep track of batch sizes of all operator terms
        self._batch_size: int = 1
        self.dtype: jnp.dtype | None = None

        # Internal attributes from interfacing to cuDensityMat.
        self._ptr: int | None = None

        # Attributes for handling JIT tracing.
        self._coeff_ptrs: list[int] = []
        self._coeff_ptr_objs: list[ctypes.c_short | None] = []  # Keep ctypes objects alive

        self._coeff_grad_ptrs: list[int] = []
        self._coeff_grad_ptr_objs: list[ctypes.c_short | None] = []  # Keep ctypes objects alive

        # Attributes for handling gradients.
        self._coeff_requires_grads: list[bool] = []
        self._coeff_callbacks = []
        self._coeff_grad_callbacks = []
        self._op_term_ids: list[int] = []
        self._total_coeffs_ptrs: list[int] = []
        self._total_coeffs_ptr_objs: list[ctypes.c_short | None] = []  # Keep ctypes objects alive

        # Whether shape-derived metadata (num_modes, mode_extents, ...) reflects the current
        # data view. False until _update_metadata runs.
        self._is_metadata_updated: bool = False

    def tree_flatten(self):
        """
        Flatten the operator PyTree.
        """
        children = (self.op_terms, self.coeffs)
        aux_data = (
            self.dims,
            self._simplifier_config,
            self.duals,
            self._batch_size,
            self._op_term_batch_sizes,
            self.dtype,
            self._ptr,
            self._coeff_ptrs,
            self._coeff_ptr_objs,
            self._coeff_grad_ptrs,
            self._coeff_grad_ptr_objs,
            self._coeff_requires_grads,
            self._coeff_callbacks,
            self._coeff_grad_callbacks,
            self._op_term_ids,
            self._total_coeffs_ptrs,
            self._total_coeffs_ptr_objs,
            self._is_metadata_updated,
        )
        return children, aux_data

    @classmethod
    def tree_unflatten(cls, aux_data, children):
        """
        Unflatten the operator PyTree.
        """
        inst = cls.__new__(cls)
        inst.op_terms, inst.coeffs = children
        (
            inst.dims,
            inst._simplifier_config,
            inst.duals,
            inst._batch_size,
            inst._op_term_batch_sizes,
            inst.dtype,
            inst._ptr,
            inst._coeff_ptrs,
            inst._coeff_ptr_objs,
            inst._coeff_grad_ptrs,
            inst._coeff_grad_ptr_objs,
            inst._coeff_requires_grads,
            inst._coeff_callbacks,
            inst._coeff_grad_callbacks,
            inst._op_term_ids,
            inst._total_coeffs_ptrs,
            inst._total_coeffs_ptr_objs,
            inst._is_metadata_updated,
        ) = aux_data
        return inst

    @property
    def in_axes(self) -> "Operator":
        """
        Return the in_axes PyTree spec for vmapping over the batch dimension.
        """
        in_axes_op_terms = [op_term.in_axes for op_term in self.op_terms]
        # Batched operator-level coefficients (size > 1) map their leading axis; scalar
        # coefficients (shape (1,)) are shared across all vmap instances and are not mapped.
        in_axes_coeffs = [0 if c.shape[0] > 1 else None for c in self.coeffs]

        _, aux_data = self.tree_flatten()
        return type(self).tree_unflatten(aux_data, (in_axes_op_terms, in_axes_coeffs))
    
    def _update_metadata(self) -> None:
        """
        Recompute coeff batch sizes, propagate the batch size update to all leaf base operators,
        and validate a uniform batch size across coefficients and operator terms.
        """
        # Under the uniform-batch contract, every coefficient and operator term shares a single
        # batch size; collect them and require they are equal (size-1 coefficients broadcast).
        coeff_batch_sizes = []
        op_term_batch_sizes_tmp = []
        for coeff, op_term in zip(self.coeffs, self.op_terms):
            coeff_batch_sizes.append(
                get_batch_size(coeff) if is_vmap_traced(coeff) else math.prod(coeff.shape)
            )
            op_term._update_metadata()
            op_term_batch_sizes_tmp.append(op_term._batch_size)
        all_batch_sizes = [b for b in coeff_batch_sizes + op_term_batch_sizes_tmp if b != 1]
        if len(set(all_batch_sizes)) > 1:
            raise ValueError("All coefficients and operator terms in an operator must have the same batch size.")
        self._batch_size = all_batch_sizes[0] if all_batch_sizes else 1

        for i, (coeff, op_term) in enumerate(zip(self.coeffs, self.op_terms)):
            self._op_term_batch_sizes[i] = coeff_batch_sizes[i]
            self._coeff_requires_grads[i] = detect_ad_traced_object(coeff)

        self._is_metadata_updated = True

    def _copy(self) -> "Operator":
        """
        Internal method to copy the operator for VJP backward pass.
        """
        op = type(self).__new__(type(self))

        op.op_terms = [op_term._copy() for op_term in self.op_terms]
        op.coeffs = [jnp.copy(c) for c in self.coeffs]
        op.dims = self.dims
        op._simplifier_config = self._simplifier_config
        op.duals = self.duals.copy()
        op._batch_size = self._batch_size
        op._op_term_batch_sizes = self._op_term_batch_sizes.copy()
        op.dtype = self.dtype
        op._ptr = self._ptr
        op._coeff_ptrs = self._coeff_ptrs.copy()
        op._coeff_ptr_objs = self._coeff_ptr_objs.copy()
        op._coeff_grad_ptrs = self._coeff_grad_ptrs.copy()
        op._coeff_grad_ptr_objs = self._coeff_grad_ptr_objs.copy()
        op._coeff_requires_grads = self._coeff_requires_grads.copy()
        op._coeff_callbacks = self._coeff_callbacks.copy()
        op._coeff_grad_callbacks = self._coeff_grad_callbacks.copy()
        op._op_term_ids = self._op_term_ids.copy()
        op._total_coeffs_ptrs = self._total_coeffs_ptrs.copy()
        op._total_coeffs_ptr_objs = self._total_coeffs_ptr_objs.copy()
        op._is_metadata_updated = self._is_metadata_updated
        return op

    def _check_and_set_dtype(self, op_term: OperatorTerm) -> None:
        """
        Check if the operator term has the same data type as the operator.
        """
        # Skip the check for an operator term whose dtype was never inferred from appended base
        # operators, i.e. one holding only empty operator products. Such a term carries no data
        # and must not constrain the operator's dtype. Its `dtype` attribute defaults to
        # complex128 rather than None, so `_dtype_overwritten` is what marks a real dtype.
        if op_term._dtype_overwritten:
            if self.dtype is None:
                # If the data type is not set, set it to the data type of the first operator term.
                self.dtype = op_term.dtype
            else:
                # If the data type is set, check if the operator term has the same data type as the operator.
                if op_term.dtype != self.dtype:
                    raise ValueError("All operator terms must have the same data type.")

    def append(self,
               op_term: OperatorTerm,
               dual: bool = False,
               coeff: float | complex | jax.Array = 1.0,
               ) -> None:
        """
        Append an operator term to an operator.

        Args:
            op_term: Operator term to be appended.
            dual: Duality of the operator term.
            coeff: Non-batched coefficient or batched coefficients of the operator term.
        """
        if self._ptr is not None:
            raise RuntimeError("Cannot modify operator after it has been used in an operator action.")

        # coeff is converted to a length-1 array if it is a Python scalar.
        if (
            isinstance(coeff, (float, complex)) or
            (isinstance(coeff, jax.Array) and coeff.ndim == 0)  # scalar but traced
        ):
            coeff = jnp.array([coeff], dtype=jnp.complex128)
        elif isinstance(coeff, jax.Array) and coeff.ndim > 0:
            if coeff.dtype != jnp.complex128:
                raise ValueError("Coefficient must be of type complex128.")
        else:
            raise ValueError("Coefficient must be a float, complex, or jax.Array.")

        # Attributes from function arguments.
        self.op_terms.append(op_term)
        self.duals.append(dual)
        self.coeffs.append(coeff)

        self._op_term_batch_sizes.append(1)  # updated by _update_metadata() inside operator_action
        self._check_and_set_dtype(op_term)

        # Internal attributes.
        self._coeff_callbacks.append(None)
        self._coeff_ptrs.append(0)
        self._coeff_ptr_objs.append(None)

        self._coeff_requires_grads.append(None)
        self._coeff_grad_callbacks.append(None)
        self._coeff_grad_ptrs.append(0)
        self._coeff_grad_ptr_objs.append(None)
        self._total_coeffs_ptrs.append(0)
        self._total_coeffs_ptr_objs.append(None)
        self._op_term_ids.append(id(op_term))

    def __getitem__(self, index: int) -> OperatorTerm:
        """
        Get an operator term from the operator.
        """
        return self.op_terms[index]

    def _simplify(self) -> None:
        """
        Apply the simplification passes to the operator terms.

        The two passes are applied interleaved by layer, i.e. kron_cond[0], sum_cond[0],
        kron_cond[1], sum_cond[1], ..., with the shorter tuple padded with None. Each
        unique operator term is simplified once and the result written back to every
        index sharing it, so that operator terms appended more than once stay shared.
        """
        config = self._simplifier_config
        n_layers = max(len(config.kron_cond), len(config.sum_cond))

        simplified = {}  # original op_term_id -> simplified operator term
        for i, op_term_id in enumerate(self._op_term_ids):
            if op_term_id not in simplified:
                op_term = self.op_terms[i]
                for layer in range(n_layers):
                    kron_cond = config.kron_cond[layer] if layer < len(config.kron_cond) else None
                    sum_cond = config.sum_cond[layer] if layer < len(config.sum_cond) else None
                    if kron_cond is not None:
                        op_term = op_term._kron_simplify(kron_cond)
                    if sum_cond is not None:
                        op_term = op_term._sum_simplify(sum_cond)
                simplified[op_term_id] = op_term
            self.op_terms[i] = simplified[op_term_id]

    def _reset_handles(self) -> None:
        """
        Clear the cuDensityMat handle state (own and cascading to op terms) so _create rebuilds
        it for a new batch size. The coefficient-pointer lists are reassigned to fresh placeholder
        lists (not mutated in place): pytree copies share these list objects, so in-place mutation
        by a later _create would corrupt another copy's handles.
        """
        self._ptr = None
        n = len(self.coeffs)
        self._coeff_ptrs = [0] * n
        self._coeff_ptr_objs = [None] * n
        self._coeff_grad_ptrs = [0] * n
        self._coeff_grad_ptr_objs = [None] * n
        self._coeff_requires_grads = [None] * n
        self._coeff_callbacks = [None] * n
        self._coeff_grad_callbacks = [None] * n
        self._total_coeffs_ptrs = [0] * n
        self._total_coeffs_ptr_objs = [None] * n
        # _coeff_requires_grads above was just wiped to None; force _create's guard to
        # recompute it via _update_metadata instead of skipping on a stale True.
        self._is_metadata_updated = False
        # Reset the unique op terms; duplicate op terms (same _op_term_id, e.g. one OperatorTerm
        # appended with different duals) re-adopt the first occurrence's freshly-reset handle
        # lists, mirroring the pytree list-sharing that _create's op-term dedup relies on.
        id_to_first_index = {}
        for i, op_term_id in enumerate(self._op_term_ids):
            if op_term_id not in id_to_first_index:
                self.op_terms[i]._reset_handles()
                id_to_first_index[op_term_id] = i
            else:
                self.op_terms[i]._adopt_handles(self.op_terms[id_to_first_index[op_term_id]])

    def _create(self, handle, batch_size: int = 1):
        """
        Create opaque handle to the operator.

        Simplification happens in operator_action (the only caller reachable via
        maybe_create_operator_context), not here, so that the metadata computed there
        reflects the simplified terms rather than the pre-simplify structure.
        """
        # Ensure metadata (own batch sizes + cascaded op-term/base-op shape attributes) is
        # populated for direct callers; operator_action refreshes it beforehand, so this is a
        # no-op there. The cascade sets descendants' flags too, so their _create guards then skip.
        if not self._is_metadata_updated:
            self._update_metadata()

        # Create a dictionary to map from the original op_term_id to the first index of
        # the op_term in the operator. The original op_term_id need to be used since id(op_term)
        # changes when JAX flattens and unflattens the PyTrees.
        id_to_first_index = {}
        for i, op_term_id in enumerate(self._op_term_ids):
            if op_term_id not in id_to_first_index:
                id_to_first_index[op_term_id] = i

        # Only create the opaque handles to the unique operator terms.
        for i in id_to_first_index.values():
            self.op_terms[i]._create(handle, batch_size)

        # Create the current operator.
        if self._ptr is None:
            self._ptr = cudm.create_operator(handle, len(self.dims), self.dims)
            self.logger.debug(f"Created operator at {hex(self._ptr)}")

            for i in range(len(self.op_terms)):
                # _coeff_requires_grads is set by _update_metadata; assign callback, gradient
                # callback, temporary coefficient pointer and object.
                if self._op_term_batch_sizes[i] == 1:
                    # Traced scalars need to be passed through an intermediate memory slot.
                    self._coeff_callbacks[i] = get_scalar_assignment_callback(self.coeffs[i].dtype)
                    self._coeff_ptrs[i] = self._coeff_callbacks[i].callback.coeff.data.ptr

                else:
                    static_coeff_buf = cp.ones(
                        (self._op_term_batch_sizes[i], *self.coeffs[i].shape), dtype=self.coeffs[i].dtype)
                    self._coeff_ptrs[i] = static_coeff_buf.data.ptr
                    self._coeff_ptr_objs[i] = static_coeff_buf

                    # For batched coefficients, callback is required by the API but is a no-op.
                    self._coeff_callbacks[i] = get_empty_scalar_callback()
                    self._total_coeffs_ptrs[i], self._total_coeffs_ptr_objs[i] = get_random_odd_pointer_and_object()

                # If gradient is computed on the coefficient, assign gradient callback and pointer.
                # The gradient buffer is sized to batch_size so cudensitymat can write one value
                # per batch element regardless of whether the coefficient itself is batched.
                if self._coeff_requires_grads[i]:
                    self._coeff_grad_callbacks[i] = get_scalar_gradient_attachment_callback(
                        (batch_size,), self.coeffs[i].dtype)
                    self._coeff_grad_ptrs[i] = self._coeff_grad_callbacks[i].callback.scalar_grad.data.ptr

                if self._op_term_batch_sizes[i] == 1:
                    cudm.operator_append_term(
                        handle,
                        self._ptr,
                        self.op_terms[id_to_first_index[self._op_term_ids[i]]]._ptr,
                        self.duals[i],
                        1.0,  # coefficient, to be updated to the real coefficient by callback
                        self._coeff_callbacks[i],
                        self._coeff_grad_callbacks[i],
                    )
                else:
                    # This is only needed when _op_term_batch_sizes[i] != 1
                    # and when _self._coeff_requires_grads[i] is True.
                    cudm.operator_append_term_batch(
                        handle,
                        self._ptr,
                        self.op_terms[id_to_first_index[self._op_term_ids[i]]]._ptr,
                        self.duals[i],
                        self._op_term_batch_sizes[i],
                        self._coeff_ptrs[i],
                        self._total_coeffs_ptrs[i],
                        self._coeff_callbacks[i],
                        self._coeff_grad_callbacks[i],
                    )
            self.logger.debug(f"Appended operator terms to operator at {hex(self._ptr)}")
    
    def _destroy(self):
        """
        Destroy opaque handle to the operator.
        """
        if self._ptr is not None:
            # Destroy the current operator.
            cudm.destroy_operator(self._ptr)
            self.logger.debug(f"Destroyed operator at {hex(self._ptr)}")
            self._ptr = None
