# Copyright (c) 2025-2026, NVIDIA CORPORATION & AFFILIATES
#
# SPDX-License-Identifier: BSD-3-Clause

"""
Elementary operator class in cuDensityMat.
"""

import logging
from collections.abc import Sequence
from numbers import Number
from typing import Callable

import jax
import jax.numpy as jnp

from cuquantum.bindings import cudensitymat as cudm
from nvmath.internal import typemaps

from ..utils import (
    dense_data_dag,
    dense_batched_kron,
    dense_batched_matmul,
    get_batch_size,
    is_vmap_traced,
    get_empty_tensor_callback,
    get_tensor_gradient_attachment_callback,
    detect_ad_traced_object,
    multidiagonal_add,
    multidiagonal_matmul,
    multidiagonal_to_dense,
)


@jax.tree_util.register_pytree_node_class
class ElementaryOperator:
    """
    PyTree class for cuDensityMat's elementary operator.
    """

    logger = logging.getLogger("cudensitymat-jax.ElementaryOperator")

    def __init__(self, data: jax.Array, diag_offsets: Sequence[int] = ()) -> None:
        """
        Initialize an ElementaryOperator object.

        Args:
            data: Data buffer of the elementary operator.
            diag_offsets: Diagonal offsets of the elementary operator.
        """
        self.data = data
        self.diag_offsets: tuple[int, ...] = tuple(diag_offsets)
        self.sparsity = (
            cudm.ElementaryOperatorSparsity.OPERATOR_SPARSITY_NONE if len(self.diag_offsets) == 0
            else cudm.ElementaryOperatorSparsity.OPERATOR_SPARSITY_MULTIDIAGONAL
        )
        self.dtype: jnp.dtype = self.data.dtype if isinstance(data, jax.Array) else jnp.dtype(float)

        # Shape-related attributes. Updated by _update_metadata.
        self._batch_size: int = 1
        self._num_modes: int = 0
        self._mode_extents: tuple[int, ...] = ()

        # Callback-related attributes.
        self._requires_grad: bool | None = None
        self._callback: Callable | None = None
        self._grad_callback: Callable | None = None
        self._grad_ptr: int = 0

        self._ptr: int | None = None
        self._is_elementary: bool = True

        # Whether shape-derived metadata (num_modes, mode_extents, ...) reflects the current
        # data view. False until _update_metadata runs.
        self._is_metadata_updated: bool = False

    def tree_flatten(self):
        """
        Flatten the elementary operator PyTree.
        """
        children = (self.data,)
        aux_data = (
            self.diag_offsets,
            self._batch_size,
            self.sparsity,
            self._num_modes,
            self._mode_extents,
            self.dtype,
            self._requires_grad,
            self._callback,
            self._grad_callback,
            self._grad_ptr,
            self._is_elementary,
            self._ptr,
            self._is_metadata_updated,
        )
        return children, aux_data

    @classmethod
    def tree_unflatten(cls, aux_data, children):
        """
        Unflatten the elementary operator PyTree.
        """
        inst = cls.__new__(cls)
        inst.data = children[0]
        (
            inst.diag_offsets,
            inst._batch_size,
            inst.sparsity,
            inst._num_modes,
            inst._mode_extents,
            inst.dtype,
            inst._requires_grad,
            inst._callback,
            inst._grad_callback,
            inst._grad_ptr,
            inst._is_elementary,
            inst._ptr,
            inst._is_metadata_updated,
        ) = aux_data
        return inst

    @property
    def in_axes(self) -> "ElementaryOperator":
        """
        Return the in_axes PyTree spec for vmapping the operator: the data leaf always maps
        its leading batch axis (0), since every operator leaf must carry a batch dimension
        matching the state's. A nested vmap peels one leading batch axis per level, so the
        same spec applies at every level.
        """
        _, aux_data = self.tree_flatten()
        return type(self).tree_unflatten(aux_data, (0,))

    def _update_metadata(self) -> None:
        """
        Refresh the requires-grad flag, batch size, and shape-derived attributes (num_modes,
        mode_extents) from the (possibly vmapped) data view, whose leading batch axis is peeled
        off relative to the construction-time view. Called inside operator_action.
        """
        if not isinstance(self.data, jax.Array):
            return

        if is_vmap_traced(self.data):
            self._batch_size = get_batch_size(self.data)
        else:
            self._batch_size = self.data.shape[0] if self.data.ndim % 2 == 1 else 1
        self._num_modes = self.data.ndim // 2  # with leading batch axis or not

        if len(self.diag_offsets) == 0:  # dense elementary operator
            # Check that bra and ket modes have the same shape.
            bra_modes = self.data.shape[-2 * self._num_modes:-self._num_modes]
            ket_modes = self.data.shape[-self._num_modes:]
            if bra_modes != ket_modes:
                raise ValueError("Dense elementary operator data must have the same shape on the bra and ket modes.")
            self._mode_extents = self.data.shape[-self._num_modes:]
        else:  # multidiagonal elementary operator
            # Multidiagonal data is (*batch, mode_extent, num_diagonals): a single mode
            # plus a trailing diagonal-columns axis, so num_modes == 1.
            if self._num_modes != 1:
                raise ValueError("Only single-mode multidiagonal elementary operator is supported.")
            if len(self.diag_offsets) != len(set(self.diag_offsets)):
                raise ValueError("Diagonal offsets cannot contain duplicate elements.")
            if self.data.shape[-1] != len(self.diag_offsets):
                raise ValueError("Number of columns in data does not match length of diagonal offsets.")
            # Single mode: its extent is the axis before the trailing diagonal-columns
            # axis (which also skips any leading batch axis).
            self._mode_extents = (self.data.shape[-2],)

        self._requires_grad = detect_ad_traced_object(self.data)
        self._is_metadata_updated = True

    @property
    def batch_size(self) -> int:
        if not self._is_metadata_updated:
            self._update_metadata()
        return self._batch_size

    @property
    def num_modes(self) -> int:
        if not self._is_metadata_updated:
            self._update_metadata()
        return self._num_modes

    @property
    def mode_extents(self) -> tuple:
        if not self._is_metadata_updated:
            self._update_metadata()
        return self._mode_extents

    def _copy(self) -> "ElementaryOperator":
        """
        Copy the elementary operator.
        """
        elem_op = type(self).__new__(type(self))

        elem_op.data = jnp.copy(self.data)
        elem_op.diag_offsets = self.diag_offsets
        elem_op._batch_size = self._batch_size
        elem_op.sparsity = self.sparsity
        elem_op._num_modes = self._num_modes
        elem_op._mode_extents = self._mode_extents
        elem_op.dtype = self.dtype
        elem_op._requires_grad = self._requires_grad
        elem_op._callback = self._callback
        elem_op._grad_callback = self._grad_callback
        elem_op._grad_ptr = self._grad_ptr
        elem_op._is_elementary = self._is_elementary
        elem_op._ptr = self._ptr
        elem_op._is_metadata_updated = self._is_metadata_updated

        return elem_op

    def _check_scalar_operation_compatibility(self, other: "Number | jax.Array", operation: str = "") -> None:
        """
        Check that ``other`` is a scalar or a jax.Array with compatible shape.
        """
        if isinstance(other, jax.Array):
            if other.ndim > 1:  # ndim = 0 or 1 is allowed
                raise ValueError(
                    f"jax.Array operand must be a scalar or 1-D array in {operation}, "
                    f"got shape {other.shape}."
                )
            if other.ndim == 1 and other.shape[0] not in (1, self.batch_size):
                raise ValueError(
                    f"jax.Array operand must have shape (1,) or ({self.batch_size},) in {operation}, "
                    f"got {other.shape}."
                )

    def _check_binary_operation_compatibility(self,
                                              other: "ElementaryOperator",
                                              operation: str = "",
                                              check_dims: bool = True,
                                              ) -> None:
        """
        Check that ``other`` is an ``ElementaryOperator`` and compatible for the given operation.
        """
        if not isinstance(other, ElementaryOperator):
            raise TypeError(
                f"Cannot perform {operation} between ElementaryOperator and {type(other).__name__}."
            )

        if check_dims and self.mode_extents != other.mode_extents:
            raise ValueError(
                f"Incompatible mode_extents for {operation}: {self.mode_extents} vs {other.mode_extents}."
            )

        if self.batch_size != 1 and other.batch_size != 1 and self.batch_size != other.batch_size:
            raise ValueError(
                f"Incompatible batch sizes for {operation}: {self.batch_size} vs {other.batch_size}."
            )

    def __add__(self, other: "ElementaryOperator") -> "ElementaryOperator":
        """
        Sum of an elementary operator on the left with another elementary operator.

        Adding a scalar is not supported: it is ambiguous between elementwise addition and
        adding ``scalar * identity``, and the two disagree for multidiagonal operators, where
        only the stored diagonals would be updated.
        """
        if isinstance(other, ElementaryOperator):
            self._check_binary_operation_compatibility(other, operation="addition")

            if self.diag_offsets != () and other.diag_offsets != ():  # both are multidiagonal
                data, diag_offsets = multidiagonal_add(self.data, self.diag_offsets, other.data, other.diag_offsets)
                return ElementaryOperator(data, diag_offsets=diag_offsets)
            else:  # one of the two is dense
                return ElementaryOperator(self.to_dense().data + other.to_dense().data)
        elif isinstance(other, (Number, jax.Array)):
            raise TypeError(
                "Adding a scalar to an ElementaryOperator is ambiguous. Add the scalar to the data buffer directly "
                "or multiply it with identity and then add to the ElementaryOperator."
            )
        else:
            raise TypeError(f"Cannot perform addition between ElementaryOperator and {type(other).__name__}.")

    def __radd__(self, other) -> "ElementaryOperator":
        """
        Sum of an elementary operator on the right with a non-ElementaryOperator operand.
        """
        # Only reached when other is not an ElementaryOperator: otherwise other.__add__(self)
        # would have handled it. Delegating raises the TypeError from __add__, so accumulating
        # with sum() reports the operand type rather than Python's bare operator message.
        return self + other

    def __neg__(self) -> "ElementaryOperator":
        """
        Negation of an elementary operator.
        """
        return ElementaryOperator(-self.data, diag_offsets=self.diag_offsets)

    def __sub__(self, other: "ElementaryOperator") -> "ElementaryOperator":
        """
        Difference of an elementary operator on the left with another elementary operator.

        Subtracting a scalar is not supported, for the same reason as in ``__add__``.
        """
        if isinstance(other, ElementaryOperator):
            self._check_binary_operation_compatibility(other, operation="subtraction")

            if self.diag_offsets != () and other.diag_offsets != ():  # both are multidiagonal
                data, diag_offsets = multidiagonal_add(self.data, self.diag_offsets, -other.data, other.diag_offsets)
                return ElementaryOperator(data, diag_offsets=diag_offsets)
            else:  # one of self or other is dense
                return ElementaryOperator(self.to_dense().data - other.to_dense().data)
        elif isinstance(other, (Number, jax.Array)):
            raise TypeError(
                "Subtracting a scalar from an ElementaryOperator is ambiguous. Subtract the scalar from the data "
                "buffer directly or multiply it with identity and then subtract from the ElementaryOperator."
            )
        else:
            raise TypeError(f"Cannot perform subtraction between ElementaryOperator and {type(other).__name__}.")

    def __rsub__(self, other) -> "ElementaryOperator":
        """
        Difference of a non-ElementaryOperator operand on the left with an elementary operator
        on the right.
        """
        # Only reached when other is not an ElementaryOperator: otherwise other.__sub__(self)
        # would have handled it. Delegating raises the TypeError from __add__.
        return -self + other

    def __mul__(self, other: "Number | jax.Array") -> "ElementaryOperator":
        """
        Scalar multiplication.
        """
        if isinstance(other, (Number, jax.Array)):
            self._check_scalar_operation_compatibility(other, operation="scalar multiplication")

            if isinstance(other, jax.Array) and other.ndim == 1 and other.shape[0] != 1:
                # Reshape (batch_size,) to (batch_size, 1, 1, ...) so it broadcasts
                # along the batch dimension of self.data.
                other = other.reshape(-1, *([1] * (self.data.ndim - 1)))

            return ElementaryOperator(self.data * other, diag_offsets=self.diag_offsets)
        else:
            raise TypeError(f"Cannot perform scalar multiplication between ElementaryOperator and {type(other).__name__}.")

    def __rmul__(self, other: "Number | jax.Array") -> "ElementaryOperator":
        """
        Right scalar multiplication.
        """
        return self * other

    def __matmul__(self, other: "ElementaryOperator") -> "ElementaryOperator":
        """
        Matrix multiplication of two elementary operators.
        """
        if isinstance(other, ElementaryOperator):
            self._check_binary_operation_compatibility(other, operation="matrix multiplication")
            if self.diag_offsets != () and other.diag_offsets != ():  # both are multidiagonal
                data, diag_offsets = multidiagonal_matmul(self.data, self.diag_offsets, other.data, other.diag_offsets)
                if not diag_offsets:  # every diagonal sum clipped — result is the zero operator
                    return ElementaryOperator(jnp.zeros_like(self.to_dense().data))
                return ElementaryOperator(data, diag_offsets=diag_offsets)
            else:  # one of self or other is dense
                return ElementaryOperator(dense_batched_matmul(self.to_dense().data, other.to_dense().data))
        else:
            raise TypeError(f"Cannot perform matrix multiplication between ElementaryOperator and {type(other).__name__}.")

    def __and__(self, other: "ElementaryOperator") -> "ElementaryOperator":
        """
        Tensor product of two elementary operators.

        Not supported when both operands are multidiagonal. cuDensityMat restricts
        multidiagonal storage to 1-body operators, so a tensor product — which is inherently
        2-body — cannot be represented in that format. A multidiagonal data buffer is always
        of rank 2, leaving nowhere to record a second mode extent, so the result would be
        silently reinterpreted as a single fused mode of extent ``mode_dim_a * mode_dim_b``
        and could no longer be appended to an OperatorTerm over the original two modes.
        """
        if isinstance(other, ElementaryOperator):
            self._check_binary_operation_compatibility(other, operation="tensor product", check_dims=False)
            if self.diag_offsets != () and other.diag_offsets != ():  # both are multidiagonal
                raise ValueError(
                    "Tensor product of two multidiagonal elementary operators is not supported, "
                    "since cuDensityMat only supports multidiagonal 1-body operators "
                    "while the result is 2-body. Take the tensor product at the OperatorTerm "
                    "level instead, or append both operators to the same operator product with "
                    "the modes they act on; either preserves the mode structure and keeps both "
                    "operands multidiagonal."
                )
            # At least one operand is dense, so the result is a genuine multi-mode dense
            # operator whose rank encodes the combined mode structure.
            return ElementaryOperator(dense_batched_kron(self.to_dense().data, other.to_dense().data))
        else:
            raise TypeError(f"Cannot perform tensor product between ElementaryOperator and {type(other).__name__}.")

    def dag(self) -> "ElementaryOperator":
        """
        Conjugate transpose of an elementary operator.
        """
        if self.diag_offsets != ():
            return ElementaryOperator(self.data.conj(), diag_offsets=tuple(-k for k in self.diag_offsets))
        else:
            return ElementaryOperator(dense_data_dag(self.data))

    def to_dense(self) -> "ElementaryOperator":
        """
        Return the dense form of an elementary operator.
        """
        if self.diag_offsets == ():  # already a dense elementary operator
            return self
        else:
            return ElementaryOperator(multidiagonal_to_dense(self.data, self.diag_offsets))

    def _reset_handles(self) -> None:
        """
        Clear the cuDensityMat handle state so _create rebuilds it. Used when the same operator
        is reused at a different batch size (handles are batch-config-specific).
        """
        self._ptr = None
        self._callback = None
        self._grad_callback = None
        self._grad_ptr = 0

    def _create(self, handle, batch_size: int = 1):
        """
        Create opaque handle to the elementary operator.
        """
        # TODO: The API now requires uniform batch size. The batch_size argument can be removed.

        # Ensure shape metadata (num_modes, mode_extents, ...) is populated for direct callers;
        # operator_action refreshes it beforehand, so the guard makes this a no-op there.
        if not self._is_metadata_updated:
            self._update_metadata()

        # Create opaque handle to the elementary operator
        if self._ptr is None:

            # _requires_grad is set by _update_metadata; assign callback + gradient buffers if needed.
            if self._requires_grad:
                self._callback = get_empty_tensor_callback()
                grad_shape = (batch_size, *self.data.shape) if is_vmap_traced(self.data) else self.data.shape
                self._grad_callback = get_tensor_gradient_attachment_callback(grad_shape, self.dtype)
                self._grad_ptr = self._grad_callback.callback.tensor_grad.data.ptr

            if self._batch_size == 1:
                self._ptr = cudm.create_elementary_operator(
                    handle,
                    self._num_modes,
                    self._mode_extents,
                    self.sparsity,
                    len(self.diag_offsets),
                    self.diag_offsets,
                    typemaps.NAME_TO_DATA_TYPE[self.dtype.name],
                    0,  # buffer pointer to be attached in the XLA layer
                    self._callback,
                    self._grad_callback
                )
            else:
                self._ptr = cudm.create_elementary_operator_batch(
                    handle,
                    self._num_modes,
                    self._mode_extents,
                    self._batch_size,
                    self.sparsity,
                    len(self.diag_offsets),
                    self.diag_offsets,
                    typemaps.NAME_TO_DATA_TYPE[self.dtype.name],
                    0,  # buffer pointer to be attached in the XLA layer
                    self._callback,
                    self._grad_callback,
                )

            self.logger.debug(f"Created elementary operator at {hex(self._ptr)}")

    def _destroy(self):
        """
        Destroy opaque handle to the elementary operator.
        """
        if self._ptr is not None:
            cudm.destroy_elementary_operator(self._ptr)
            self.logger.debug(f"Destroyed elementary operator at {hex(self._ptr)}")
            self._ptr = None
