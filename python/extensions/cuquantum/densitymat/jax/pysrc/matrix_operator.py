# Copyright (c) 2025-2026, NVIDIA CORPORATION & AFFILIATES
#
# SPDX-License-Identifier: BSD-3-Clause

"""
Matrix operator class in cuDensityMat.
"""

import logging
from numbers import Number
from typing import Any, Callable

import jax
import jax.numpy as jnp

from cuquantum.bindings import cudensitymat as cudm
from nvmath.internal import typemaps

from ..utils import (
    dense_data_dag,
    dense_batched_matmul,
    get_batch_size,
    is_vmap_traced,
    get_empty_tensor_callback,
    get_tensor_gradient_attachment_callback,
    detect_ad_traced_object,
)


@jax.tree_util.register_pytree_node_class
class MatrixOperator:
    """
    PyTree class for cuDensityMat's matrix operator.
    """

    logger = logging.getLogger("cudensitymat-jax.MatrixOperator")

    def __init__(self, data: jax.Array) -> None:
        """
        Initialize a MatrixOperator object.

        Args:
            data: Data buffer of the matrix operator.
        """
        self.data = data
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
        self._is_elementary: bool = False

        # Whether shape-derived metadata (num_modes, mode_extents, ...) reflects the current
        # data view. False until _update_metadata runs.
        self._is_metadata_updated: bool = False

    def tree_flatten(self):
        """
        Flatten the matrix operator PyTree.
        """
        children = (self.data,)
        aux_data = (
            self._batch_size,
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
        Unflatten the matrix operator PyTree.
        """
        inst = cls.__new__(cls)
        inst.data = children[0]
        (
            inst._batch_size,
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
    def in_axes(self) -> "MatrixOperator":
        """
        Return the in_axes PyTree spec for vmapping the operator: the data leaf always maps
        its leading batch axis (0), since every operator leaf must carry a batch dimension
        matching the state's. A nested vmap peels one leading batch axis per level, so the
        same spec applies at every level.
        """
        _, aux_data = self.tree_flatten()
        in_axes_data = 0
        return type(self).tree_unflatten(aux_data, (in_axes_data,))

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
        if self.data.shape[-2 * self._num_modes:-self._num_modes] != self.data.shape[-self._num_modes:]:
            raise ValueError("Data must have the same shape on the bra and ket modes.")
        self._mode_extents = self.data.shape[-self._num_modes:]

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

    def _copy(self) -> "MatrixOperator":
        """
        Copy the matrix operator.
        """
        mat_op = type(self).__new__(type(self))

        mat_op.data = jnp.copy(self.data)

        mat_op._batch_size = self._batch_size
        mat_op._num_modes = self._num_modes
        mat_op._mode_extents = self._mode_extents
        mat_op.dtype = self.dtype
        mat_op._requires_grad = self._requires_grad
        mat_op._callback = self._callback
        mat_op._grad_callback = self._grad_callback
        mat_op._grad_ptr = self._grad_ptr
        mat_op._is_elementary = self._is_elementary
        mat_op._ptr = self._ptr
        mat_op._is_metadata_updated = self._is_metadata_updated

        return mat_op

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

    def _check_binary_operation_compatibility(self, other: "MatrixOperator", operation: str = "") -> None:
        """
        Check that ``other`` is a ``MatrixOperator`` with matching ``mode_extents``
        and broadcast-compatible ``batch_size``.
        """
        if not isinstance(other, MatrixOperator):
            raise TypeError(
                f"Cannot perform {operation} between MatrixOperator and {type(other).__name__}."
            )

        if self.mode_extents != other.mode_extents:
            raise ValueError(
                f"Incompatible mode_extents for {operation}: {self.mode_extents} vs {other.mode_extents}."
            )

        if self.batch_size != 1 and other.batch_size != 1 and self.batch_size != other.batch_size:
            raise ValueError(
                f"Incompatible batch sizes for {operation}: {self.batch_size} vs {other.batch_size}."
            )

    def __add__(self, other: "MatrixOperator") -> "MatrixOperator":
        """
        Sum of a matrix operator on the left with another matrix operator.

        Adding a scalar is not supported: it is ambiguous between elementwise addition and
        adding ``scalar * identity``. Rejected here too so that the operator classes agree.
        """
        if isinstance(other, MatrixOperator):
            self._check_binary_operation_compatibility(other, operation="addition")
            return MatrixOperator(self.data + other.data)
        elif isinstance(other, (Number, jax.Array)):
            raise TypeError(
                "Adding a scalar to a MatrixOperator is ambiguous. Add the scalar to the data buffer directly "
                "or multiply it with identity and then add to the MatrixOperator."
            )
        else:
            raise TypeError(f"Cannot perform addition between MatrixOperator and {type(other).__name__}.")

    def __radd__(self, other) -> "MatrixOperator":
        """
        Sum of a matrix operator on the right with a non-MatrixOperator operand.
        """
        # Only reached when other is not a MatrixOperator: otherwise other.__add__(self) would
        # have handled it. Delegating raises the TypeError from __add__, so accumulating with
        # sum() reports the operand type rather than Python's bare operator message.
        return self + other

    def __neg__(self) -> "MatrixOperator":
        """
        Negation of a matrix operator.
        """
        return MatrixOperator(-self.data)

    def __sub__(self, other: "MatrixOperator") -> "MatrixOperator":
        """
        Difference of a matrix operator on the left with another matrix operator.

        Subtracting a scalar is not supported, for the same reason as in ``__add__``.
        """
        if isinstance(other, MatrixOperator):
            self._check_binary_operation_compatibility(other, operation="subtraction")
            return MatrixOperator(self.data - other.data)
        elif isinstance(other, (Number, jax.Array)):
            raise TypeError(
                "Subtracting a scalar from a MatrixOperator is ambiguous. Subtract the scalar from the data "
                "buffer directly or multiply it with identity and then subtract from the MatrixOperator."
            )
        else:
            raise TypeError(f"Cannot perform subtraction between MatrixOperator and {type(other).__name__}.")

    def __rsub__(self, other) -> "MatrixOperator":
        """
        Difference of a non-MatrixOperator operand on the left with a matrix operator on the right.
        """
        # Only reached when other is not a MatrixOperator: otherwise other.__sub__(self) would
        # have handled it. Delegating raises the TypeError from __add__.
        return -self + other

    def __mul__(self, other: "Number | jax.Array") -> "MatrixOperator":
        """
        Scalar multiplication.
        """
        if isinstance(other, (Number, jax.Array)):
            self._check_scalar_operation_compatibility(other, operation="scalar multiplication")

            if isinstance(other, jax.Array) and other.ndim == 1 and other.shape[0] != 1:
                # Reshape (batch_size,) to (batch_size, 1, 1, ...) so it broadcasts
                # along the batch dimension of self.data.
                other = other.reshape(-1, *([1] * (self.data.ndim - 1)))

            return MatrixOperator(self.data * other)
        else:
            raise TypeError(f"Cannot perform scalar multiplication between MatrixOperator and {type(other).__name__}.")

    def __rmul__(self, other: "Number | jax.Array") -> "MatrixOperator":
        """
        Right scalar multiplication.
        """
        return self * other

    def __matmul__(self, other: "MatrixOperator") -> "MatrixOperator":
        """
        Matrix multiplication of two matrix operators acting on the same modes.
        """
        if isinstance(other, MatrixOperator):
            self._check_binary_operation_compatibility(other, operation="matrix multiplication")
            return MatrixOperator(dense_batched_matmul(self.data, other.data))
        else:
            raise TypeError(f"Cannot perform matrix multiplication between MatrixOperator and {type(other).__name__}.")

    def __and__(self, other: Any) -> None:
        """
        Tensor product is not supported for MatrixOperator.
        """
        raise TypeError(
            "Tensor product is not supported for MatrixOperator. MatrixOperator implicitly "
            "acts on the full Hilbert space; use ElementaryOperator for tensor products."
        )

    def dag(self) -> "MatrixOperator":
        """
        Conjugate transpose of a matrix operator.
        """
        # TODO: We should not do conjugation eagerly but instead use a flag to keep track of it.
        # This is of minor importance since OperatorTerm.dag is not using MatrixOperator.dag,
        # and external libraries are not likely to use MatrixOperator.
        return MatrixOperator(dense_data_dag(self.data))

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
        Create opaque handle to the matrix operator.
        """
        # TODO: The API now requires uniform batch size. The batch_size argument can be removed.

        # Ensure shape metadata (num_modes, mode_extents, ...) is populated for direct callers;
        # operator_action refreshes it beforehand, so the guard makes this a no-op there.
        if not self._is_metadata_updated:
            self._update_metadata()

        # Create opaque handle to the matrix operator.
        if self._ptr is None:

            # _requires_grad is set by _update_metadata; assign callback + gradient buffers if needed.
            if self._requires_grad:
                self._callback = get_empty_tensor_callback()
                grad_shape = (batch_size, *self.data.shape) if is_vmap_traced(self.data) else self.data.shape
                self._grad_callback = get_tensor_gradient_attachment_callback(grad_shape, self.dtype)
                self._grad_ptr = self._grad_callback.callback.tensor_grad.data.ptr

            if self._batch_size == 1:
                self._ptr = cudm.create_matrix_operator_dense_local(
                    handle,
                    self._num_modes,
                    self._mode_extents,
                    typemaps.NAME_TO_DATA_TYPE[self.dtype.name],
                    0,  # buffer pointer to be attached in the XLA layer
                    self._callback,
                    self._grad_callback,
                )
            else:
                self._ptr = cudm.create_matrix_operator_dense_local_batch(
                    handle,
                    self._num_modes,
                    self._mode_extents,
                    self._batch_size,
                    typemaps.NAME_TO_DATA_TYPE[self.dtype.name],
                    0,  # buffer pointer to be attached in the XLA layer
                    self._callback,
                    self._grad_callback,
                )

            self.logger.debug(f"Created matrix operator at {hex(self._ptr)}")

    def _destroy(self):
        """
        Destroy opaque handle to the matrix operator.
        """
        if self._ptr is not None:
            cudm.destroy_matrix_operator(self._ptr)
            self.logger.debug(f"Destroyed matrix operator at {hex(self._ptr)}")
            self._ptr = None
