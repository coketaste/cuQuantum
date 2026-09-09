# Copyright (c) 2025-2026, NVIDIA CORPORATION & AFFILIATES
#
# SPDX-License-Identifier: BSD-3-Clause

"""
Operator term class in cuDensityMat.
"""

import ctypes
import functools
import logging
from collections.abc import Callable, Sequence
from numbers import Number
from enum import Enum
import math

import cupy as cp
import jax
import jax.numpy as jnp

from cuquantum.bindings import cudensitymat as cudm

from .elementary_operator import ElementaryOperator
from .matrix_operator import MatrixOperator
from ..utils import (
    padded_matrix_product,
    get_batch_size,
    is_vmap_traced,
    get_scalar_assignment_callback,
    get_empty_scalar_callback,
    get_scalar_gradient_attachment_callback,
    get_random_odd_pointer_and_object,
    detect_ad_traced_object,
)


class OperatorProductType(Enum):
    """
    Type of operator product.
    """
    IDENTITY = 0
    ELEMENTARY = 1
    MATRIX = 2

    @classmethod
    def from_base_op_type(cls, base_op_type: type[ElementaryOperator] | type[MatrixOperator]) -> "OperatorProductType":
        """
        Convert a base operator type to an operator product type.
        """
        if base_op_type is ElementaryOperator:
            return cls.ELEMENTARY
        elif base_op_type is MatrixOperator:
            return cls.MATRIX
        else:
            raise ValueError(f"Unknown base operator type: {base_op_type}")


@jax.tree_util.register_pytree_node_class
class OperatorTerm:
    """
    PyTree class for cuDensityMat's operator term.
    """

    logger = logging.getLogger("cudensitymat-jax.OperatorTerm")

    def __init__(self, dims: Sequence[int]) -> None:
        """
        Initialize an OperatorTerm object.

        Args:
            dims: Hilbert space dimensions.
        """
        # Input argument.
        self.dims: tuple[int, ...] = tuple(dims)

        # Attributes for handling arguments in append.
        self.op_prods: list[tuple[ElementaryOperator | MatrixOperator, ...]] = []
        self.modes: list[tuple[int, ...]] = []
        self.conjs: list[tuple[bool, ...]] = []
        self.duals: list[tuple[bool, ...]] = []

        # Used for operator term and product simplification passes.
        self._modes_duals: list[tuple[tuple[tuple[int, bool], ...], ...]] = []

        # When coeff is stored inside, it is always a jax.Array.
        self.coeffs: list[jax.Array] = []

        # Attributes inferred from multiple append calls.
        self._op_prod_batch_sizes: list[int] = []  # keep track of batch sizes of all operator products
        self.dtype: jnp.dtype = jnp.dtype(jnp.complex128)
        self._dtype_overwritten: bool = False
        self._batch_size: int = 1

        # Internal attributes from interfacing to cuDensityMat.
        self._op_prod_types: list[OperatorProductType] = []
        self._ptr: int | None = None
        # Set when simplification supersedes this term with a rebuilt one, which leaves _ptr
        # None here (the handle is created on the replacement) and would otherwise let append
        # mutate a term the operator no longer refers to. See the guard in append.
        self._consumed: bool = False

        # Attributes for handling gradients.
        self._coeff_requires_grads = []
        self._coeff_callbacks = []
        self._coeff_grad_callbacks = []
        self._coeff_ptrs: list[int] = []
        self._coeff_ptr_objs: list[ctypes.c_short | None] = []  # Keep ctypes objects alive
        self._coeff_grad_ptrs: list[int] = []
        self._coeff_grad_ptr_objs: list[ctypes.c_short | None] = []  # Keep ctypes objects alive
        self._total_coeffs_ptrs: list[int] = []
        self._total_coeffs_ptr_objs: list[ctypes.c_short | None] = []  # Keep ctypes objects alive

        # Whether shape-derived metadata (num_modes, mode_extents, ...) reflects the current
        # data view. False until _update_metadata runs.
        self._is_metadata_updated: bool = False

    def tree_flatten(self):
        """
        Flatten the operator term into a PyTree.
        """
        children = (self.op_prods, self.coeffs)
        aux_data = (
            self.dims,
            self.modes,
            self.conjs,
            self.duals,
            self._modes_duals,
            self._batch_size,
            self._op_prod_batch_sizes,
            self.dtype,
            self._dtype_overwritten,
            self._op_prod_types,
            self._ptr,
            self._consumed,
            self._coeff_ptrs,
            self._coeff_ptr_objs,
            self._coeff_grad_ptrs,
            self._coeff_requires_grads,
            self._coeff_callbacks,
            self._coeff_grad_callbacks,
            self._coeff_grad_ptr_objs,
            self._total_coeffs_ptrs,
            self._total_coeffs_ptr_objs,
            self._is_metadata_updated,
        )
        return children, aux_data

    @classmethod
    def tree_unflatten(cls, aux_data, children):
        """
        Unflatten the operator term from a PyTree.
        """
        inst = cls.__new__(cls)
        inst.op_prods, inst.coeffs = children
        (
            inst.dims,
            inst.modes,
            inst.conjs,
            inst.duals,
            inst._modes_duals,
            inst._batch_size,
            inst._op_prod_batch_sizes,
            inst.dtype,
            inst._dtype_overwritten,
            inst._op_prod_types,
            inst._ptr,
            inst._consumed,
            inst._coeff_ptrs,
            inst._coeff_ptr_objs,
            inst._coeff_grad_ptrs,
            inst._coeff_requires_grads,
            inst._coeff_callbacks,
            inst._coeff_grad_callbacks,
            inst._coeff_grad_ptr_objs,
            inst._total_coeffs_ptrs,
            inst._total_coeffs_ptr_objs,
            inst._is_metadata_updated,
        ) = aux_data
        return inst

    @property
    def in_axes(self) -> "OperatorTerm":
        """
        Return the in_axes PyTree spec for vmapping over the batch dimension.
        """
        in_axes_op_prods = [tuple(op.in_axes for op in op_prod) for op_prod in self.op_prods]
        # A coefficient is a scalar per instance, so its whole (possibly nested-batched) array
        # maps along the leading axis; a nested vmap peels one level per application.
        in_axes_coeffs = [0] * len(self.coeffs)

        _, aux_data = self.tree_flatten()
        return type(self).tree_unflatten(aux_data, (in_axes_op_prods, in_axes_coeffs))

    def _update_metadata(self) -> None:
        """
        Recompute coeff batch sizes, propagate the batch size / shape update to all leaf base
        operators, validate a uniform batch size across coefficients and base operators, and run
        the base-op shape-dependent consistency checks (deferred from append, since base-op
        num_modes and mode_extents are only populated by base_op._update_metadata()).
        """
        # Under the uniform-batch contract, every coefficient and (non-dummy) operator data
        # buffer shares a single batch size; seed from the first coefficient and check inline.
        if len(self.coeffs) > 0:
            self._batch_size = get_batch_size(self.coeffs[0]) if is_vmap_traced(self.coeffs[0]) \
                else math.prod(self.coeffs[0].shape)

        for i, (coeff, op_prod) in enumerate(zip(self.coeffs, self.op_prods)):
            coeff_batch_size = get_batch_size(coeff) if is_vmap_traced(coeff) \
                else math.prod(coeff.shape)
            self._op_prod_batch_sizes[i] = coeff_batch_size
            self._coeff_requires_grads[i] = detect_ad_traced_object(coeff)
            if coeff_batch_size != self._batch_size:
                raise ValueError("All coefficients and base operators in an operator term must have the same batch size.")
            for base_op in op_prod:
                base_op._update_metadata()
                # Dummy AD-placeholder ops keep the default _batch_size; skip them (like the
                # shape checks below), since their data is not a real array view.
                if isinstance(base_op.data, jax.Array) and base_op._batch_size != self._batch_size:
                    raise ValueError("All coefficients and base operators in an operator term must have the same batch size.")

            # Skip shape checks for AD-tracing dummy operators (placeholder shape attributes).
            # TODO: This may not be necessary since dummy operators never reach here.
            if any(not isinstance(base_op.data, jax.Array) for base_op in op_prod):
                continue

            if self._op_prod_types[i] == OperatorProductType.ELEMENTARY:
                modes = self.modes[i]
                # Check length of modes acted on match the combined number of modes in the product.
                if len(modes) != sum(elem_op._num_modes for elem_op in op_prod):
                    raise ValueError(f"Number of modes acted on {len(modes)} does not match combined number of modes in the operator product.")

                # Check mode extents of each elementary operator match corresponding qubit dimensions.
                modes_index = 0
                for elem_op in op_prod:
                    if elem_op._mode_extents != tuple(
                        self.dims[modes[j]] for j in range(modes_index, modes_index + elem_op._num_modes)
                    ):
                        raise ValueError("Mode extents of each elementary operator must match corresponding qubit dimensions.")
                    modes_index += elem_op._num_modes
            else:  # matrix operator product
                # Check that mode extents match Hilbert space dimensions.
                for matrix_op in op_prod:
                    if matrix_op._mode_extents != self.dims:
                        raise ValueError("Mode extents must match Hilbert space dimensions for matrix operators.")

        self._is_metadata_updated = True

    def _copy(self) -> "OperatorTerm":
        """
        Internal method to copy the operator term for VJP backward pass.
        """
        op_term = type(self).__new__(type(self))

        op_term.op_prods = [tuple(base_op._copy() for base_op in op_prod) for op_prod in self.op_prods]
        op_term.coeffs = [jnp.copy(c) for c in self.coeffs]
        op_term.dims = self.dims
        op_term.modes = self.modes.copy()
        op_term.conjs = self.conjs.copy()
        op_term.duals = self.duals.copy()
        op_term._modes_duals = self._modes_duals.copy()
        op_term._batch_size = self._batch_size
        op_term._op_prod_batch_sizes = self._op_prod_batch_sizes.copy()
        op_term.dtype = self.dtype
        op_term._dtype_overwritten = self._dtype_overwritten
        op_term._op_prod_types = self._op_prod_types.copy()
        op_term._ptr = self._ptr
        op_term._consumed = self._consumed
        op_term._coeff_ptrs = self._coeff_ptrs.copy()
        op_term._coeff_ptr_objs = self._coeff_ptr_objs.copy()
        op_term._coeff_grad_ptrs = self._coeff_grad_ptrs.copy()
        op_term._coeff_grad_ptr_objs = self._coeff_grad_ptr_objs.copy()
        op_term._coeff_requires_grads = self._coeff_requires_grads.copy()
        op_term._coeff_callbacks = self._coeff_callbacks.copy()
        op_term._coeff_grad_callbacks = self._coeff_grad_callbacks.copy()
        op_term._total_coeffs_ptrs = self._total_coeffs_ptrs.copy()
        op_term._total_coeffs_ptr_objs = self._total_coeffs_ptr_objs.copy()
        op_term._is_metadata_updated = self._is_metadata_updated

        return op_term

    def _check_and_set_dtype(self, op_prod: Sequence[ElementaryOperator | MatrixOperator]) -> None:
        """
        Check if the operator product has the same data type as the operator term.
        """
        if not self._dtype_overwritten:
            # If the default dtype has not been overwritten, set it to the dtype of the first base operator.
            # Ignore empty operator products, where the dtype will be kept as the default complex128.
            if len(op_prod) > 0:
                self.dtype = op_prod[0].dtype
                self._dtype_overwritten = True
                for base_op in op_prod[1:]:
                    if base_op.dtype != self.dtype:
                        raise ValueError("All elementary or matrix operators must have the same data type.")
        else:
            # If the default dtype has been overwritten, check if all base operators match the
            # dtype already set.
            for base_op in op_prod:
                if base_op.dtype != self.dtype:
                    raise ValueError("All elementary or matrix operators must have the same data type.")

    def _check_and_append_op_prod_type(self, op_prod: Sequence[ElementaryOperator | MatrixOperator]) -> None:
        """
        Check if all terms in an operator product are of the same type.
        """
        if len(op_prod) > 0:
            base_op_type = type(op_prod[0])
            for base_op in op_prod[1:]:
                if not isinstance(base_op, base_op_type):
                    raise ValueError("All terms in an operator product must be of the same type.")
            op_prod_type = OperatorProductType.from_base_op_type(base_op_type)

        else:  # empty operator product, treated as identity
            op_prod_type = OperatorProductType.IDENTITY

        self._op_prod_types.append(op_prod_type)

    def append(self,
               op_prod: Sequence[ElementaryOperator | MatrixOperator],
               modes: Sequence[int] | None = None,
               conjs: Sequence[bool] | None = None,
               duals: Sequence[bool] | None = None,
               coeff: Number | jax.Array = 1.0,
               ) -> None:
        """
        Append an elementary or matrix product to an operator term.

        Args:
            op_prod: Product of elementary or matrix operators to be appended.
            modes: Modes acted on by the operator product.
            conjs: Conjugations in the operator product. Only applies to MatrixOperators.
            duals: Dualities of the operator product.
            coeff: Non-batched coefficient or batched coefficients of the operator product.
        """
        if self._ptr is not None or self._consumed:
            raise RuntimeError("Cannot modify operator term after it has been used in an operator action.")

        # coeff is converted to a length-1 array if it is a Python scalar.
        if (
            isinstance(coeff, Number) or
            (isinstance(coeff, jax.Array) and coeff.ndim == 0)  # scalar but traced
        ):
            coeff = jnp.array([coeff], dtype=jnp.complex128)
        elif isinstance(coeff, jax.Array) and coeff.ndim > 0:
            if coeff.dtype != jnp.complex128:
                raise ValueError("Coefficient must be of type complex128.")
        else:
            raise ValueError("Coefficient must be a numeric scalar or jax.Array.")

        # Check if all elementary or matrix operators have the same data type.
        self._check_and_set_dtype(op_prod)
        self._check_and_append_op_prod_type(op_prod)

        # Check consistency and append modes, conjs and duals.
        if self._op_prod_types[-1] == OperatorProductType.ELEMENTARY:
            # Modes have to specified for elementary operators.
            if modes is None:
                raise ValueError("Modes acted on must be specified for elementary operators.")

            # Check all modes are in Hilbert space.
            if not set(modes) <= set(range(len(self.dims))):
                raise ValueError("Modes acted on must be in the Hilbert space, i.e. between 0 and len(self.dims) - 1")

            # NOTE: Shape-dependent checks (modes count vs combined num_modes, and mode extents
            # vs qubit dimensions) are deferred to _update_metadata, since base-op num_modes / mode_extents
            # are only populated by base_op._update_metadata() inside operator_action.

            # Check that matrix conjugations cannot be specified for elementary operators.
            if conjs is not None:
                raise ValueError("Matrix conjugations cannot be specified for elementary operators.")

            # Check that number of duals matches number of modes.
            if duals is None:
                duals = (False,) * len(modes)
            else:
                if len(duals) != len(modes):
                    raise ValueError("Number of duals must match number of modes acted on for elementary operator product.")

            # For elementary operator product, we only need modes and duals.
            self.modes.append(tuple(modes))
            self.conjs.append(())  # empty tuple is appended here to preserve length
            self.duals.append(tuple(duals))

        elif self._op_prod_types[-1] == OperatorProductType.MATRIX:
            # NOTE: The mode-extents-vs-Hilbert-space check is deferred to _update_metadata, since
            # base-op mode_extents are only populated by base_op._update_metadata() inside operator_action.

            # Check that modes acted on cannot be specified for matrix operators.
            if modes is not None:
                raise ValueError("Modes acted on cannot be specified for matrix operators.")

            # Check consistency of conjs.
            if conjs is None:
                conjs = (False,) * len(op_prod)
            else:
                if len(conjs) != len(op_prod):
                    raise ValueError("Number of matrix conjugations must match number of operator products.")

            # Check that number of duals matches number of matrix operators.
            if duals is None:
                duals = (False,) * len(op_prod)
            else:
                if len(duals) != len(op_prod):
                    raise ValueError("Number of duals must match number of matrix operators.")

            # For matrix operator product, we only need conjs and duals.
            self.modes.append(tuple(range(len(self.dims))))  # used in reference implementation during testing
            self.conjs.append(tuple(conjs))
            self.duals.append(tuple(duals))

        else:  # identity operator product
            self.modes.append(())
            self.conjs.append(())
            self.duals.append(())

        if self._op_prod_types[-1] == OperatorProductType.ELEMENTARY:
            modes_duals_all = []
            running_index = 0
            for base_op in op_prod:
                # Multidiagonal is always exactly 1 mode by construction (enforced in
                # _update_metadata), so it's read directly off diag_offsets rather than through
                # base_op.num_modes -- that property calls _update_metadata(), whose bra/ket shape
                # validation assumes at most one leading batch axis and would misfire here on
                # concrete (pre-vmap) multi-batch-axis data, e.g. base_op_batch_sizes=(5, 6) in
                # test_different_buffers_nested_vmap.
                num_modes = 1 if base_op.diag_offsets else base_op.data.ndim // 2
                modes_ = modes[running_index:running_index + num_modes]
                duals_ = duals[running_index:running_index + num_modes]
                modes_duals_single = tuple(zip(modes_, duals_, strict=True))
                modes_duals_all.append(modes_duals_single)
                running_index += num_modes
            self._modes_duals.append(tuple(modes_duals_all))
        else:
            self._modes_duals.append(None)

        # Attributes from function arguments.
        self.op_prods.append(tuple(op_prod))
        self.coeffs.append(coeff)

        self._op_prod_batch_sizes.append(1)  # updated by _update_metadata() inside operator_action

        # Attributes for handling gradients. None is appended here to preserve length, which is then
        # updated in the _create method.
        self._coeff_requires_grads.append(None)
        self._coeff_ptrs.append(0)
        self._coeff_ptr_objs.append(None)
        self._coeff_grad_ptrs.append(0)
        self._coeff_callbacks.append(None)
        self._coeff_grad_callbacks.append(None)
        self._coeff_grad_ptr_objs.append(None)
        self._total_coeffs_ptrs.append(0)
        self._total_coeffs_ptr_objs.append(None)

    def __len__(self) -> int:
        """
        Return the number of operator products in the operator term.
        """
        return len(self.op_prods)

    def __getitem__(self, index: int) -> tuple[ElementaryOperator | MatrixOperator, ...]:
        """
        Get an operator product from the operator term.
        """
        return self.op_prods[index]

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
            if other.ndim == 1 and other.shape[0] not in (1, self._batch_size):
                raise ValueError(
                    f"jax.Array operand must have shape (1,) or ({self._batch_size},) in {operation}, "
                    f"got {other.shape}. For per-element scaling use jax.vmap, where the factor "
                    f"arrives as a 0-d tracer."
                )

    def _check_binary_operation_compatibility(self, other: "OperatorTerm", operation: str = "") -> None:
        """
        Check that ``other`` is an ``OperatorTerm`` with matching Hilbert space dims.
        """
        if not isinstance(other, OperatorTerm):
            raise TypeError(f"Cannot perform {operation} between OperatorTerm and {type(other).__name__}.")

        if self.dims != other.dims:
            raise ValueError(f"Incompatible dims for {operation}: {self.dims} vs {other.dims}.")

        if self._dtype_overwritten and other._dtype_overwritten and self.dtype != other.dtype:
            raise ValueError(f"Incompatible dtypes for {operation}: {self.dtype} vs {other.dtype}.")

    def __add__(self, other: "OperatorTerm") -> "OperatorTerm":
        """
        Sum of two operator terms.
        """
        if not isinstance(other, OperatorTerm):
            raise TypeError(f"Cannot perform addition between OperatorTerm and {type(other).__name__}.")

        self._check_binary_operation_compatibility(other, operation="addition")

        op_term_new = OperatorTerm(self.dims)
        for op_term in (self, other):
            for op_prod, modes, conjs, duals, coeff, op_prod_type in zip(
                op_term.op_prods, op_term.modes, op_term.conjs,
                op_term.duals, op_term.coeffs, op_term._op_prod_types,
            ):
                op_prod_ = tuple(base_op._copy() for base_op in op_prod)
                if op_prod_type == OperatorProductType.ELEMENTARY:
                    op_term_new.append(op_prod_, modes=modes, duals=duals, coeff=coeff)
                elif op_prod_type == OperatorProductType.MATRIX:
                    op_term_new.append(op_prod_, conjs=conjs, duals=duals, coeff=coeff)
                else:  # identity operator product
                    op_term_new.append(op_prod_, coeff=coeff)

        return op_term_new

    def __neg__(self) -> "OperatorTerm":
        """
        Negation of an operator term.
        """
        return self * -1

    def __sub__(self, other: "OperatorTerm") -> "OperatorTerm":
        """
        Difference of two operator terms.
        """
        return self + (-other)

    def __mul__(self, other: "Number | jax.Array") -> "OperatorTerm":
        """
        Scalar multiplication on an operator term.
        """
        if not isinstance(other, (Number, jax.Array)):
            raise TypeError(f"Cannot perform scalar multiplication between OperatorTerm and {type(other).__name__}.")

        self._check_scalar_operation_compatibility(other, operation="scalar multiplication")

        op_term_new = OperatorTerm(self.dims)
        for op_prod, modes, conjs, duals, coeff, op_prod_type in zip(
            self.op_prods, self.modes, self.conjs, self.duals, self.coeffs, self._op_prod_types
        ):
            op_prod_ = tuple(base_op._copy() for base_op in op_prod)
            if op_prod_type == OperatorProductType.ELEMENTARY:
                op_term_new.append(op_prod_, modes=modes, duals=duals, coeff=coeff * other)
            elif op_prod_type == OperatorProductType.MATRIX:
                op_term_new.append(op_prod_, conjs=conjs, duals=duals, coeff=coeff * other)
            else:  # identity operator product, only coefficient needs to be multiplied
                op_term_new.append(op_prod_, coeff=coeff * other)

        return op_term_new

    def __rmul__(self, other: "Number | jax.Array") -> "OperatorTerm":
        """
        Right scalar multiplication on an operator term.
        """
        return self * other

    def __matmul__(self, other: "OperatorTerm") -> "OperatorTerm":
        """
        Composition of two operator terms.
        """
        if not isinstance(other, OperatorTerm):
            raise TypeError(f"Cannot perform composition between OperatorTerm and {type(other).__name__}.")

        self._check_binary_operation_compatibility(other, operation="multiplication")

        op_term_new = OperatorTerm(self.dims)

        for op_prod_left, modes_left, conjs_left, duals_left, coeff_left, op_prod_type_left in zip(
            self.op_prods, self.modes, self.conjs, self.duals, self.coeffs, self._op_prod_types
        ):
            for op_prod_right, modes_right, conjs_right, duals_right, coeff_right, op_prod_type_right in zip(
                other.op_prods, other.modes, other.conjs, other.duals, other.coeffs, other._op_prod_types
            ):
                if {op_prod_type_left, op_prod_type_right} == {OperatorProductType.ELEMENTARY, OperatorProductType.MATRIX}:
                    raise TypeError(
                        "Cannot matmul OperatorTerms with mixed ElementaryOperator and MatrixOperator products."
                    )

                op_prod_ = tuple(base_op._copy() for base_op in (*op_prod_right, *op_prod_left))
                if op_prod_type_left == OperatorProductType.ELEMENTARY or op_prod_type_right == OperatorProductType.ELEMENTARY:
                    op_term_new.append(
                        op_prod_,
                        modes=(*modes_right, *modes_left),
                        duals=(*duals_right, *duals_left),
                        coeff=coeff_right * coeff_left,
                    )
                elif op_prod_type_left == OperatorProductType.MATRIX or op_prod_type_right == OperatorProductType.MATRIX:
                    op_term_new.append(
                        op_prod_,
                        conjs=(*conjs_right, *conjs_left),
                        duals=(*duals_right, *duals_left),
                        coeff=coeff_right * coeff_left,
                    )
                else:  # both identity operator products, only coefficients need to be multiplied
                    op_term_new.append(op_prod_, coeff=coeff_right * coeff_left)

        return op_term_new

    def __and__(self, other: "OperatorTerm") -> "OperatorTerm":
        """
        Tensor product of two operator terms.
        """
        if not isinstance(other, OperatorTerm):
            raise TypeError(
                f"Cannot perform tensor product between OperatorTerm and {type(other).__name__}."
            )

        num_modes = len(self.dims)
        dims_combined = self.dims + other.dims

        op_term_left = OperatorTerm(dims_combined)
        for (op_prod, modes, duals, coeff, op_prod_type) in zip(
            self.op_prods, self.modes, self.duals, self.coeffs, self._op_prod_types
        ):
            if op_prod_type == OperatorProductType.MATRIX:
                raise NotImplementedError("Tensor product is not supported on MatrixOperator products.")
            op_prod_ = tuple(base_op._copy() for base_op in op_prod)
            op_term_left.append(op_prod_, modes=modes, duals=duals, coeff=coeff)

        op_term_right = OperatorTerm(dims_combined)
        for (op_prod, modes, duals, coeff, op_prod_type) in zip(
            other.op_prods, other.modes, other.duals, other.coeffs, other._op_prod_types
        ):
            if op_prod_type == OperatorProductType.MATRIX:
                raise NotImplementedError("Tensor product is not supported on MatrixOperator products.")
            op_prod_ = tuple(base_op._copy() for base_op in op_prod)
            op_term_right.append(op_prod_, modes=tuple(m + num_modes for m in modes), duals=duals, coeff=coeff)

        return op_term_left @ op_term_right

    def dag(self) -> "OperatorTerm":
        """
        Conjugate transpose of the operator term.
        """
        for duals in self.duals:
            if len(set(duals)) > 1:
                raise NotImplementedError(
                    "OperatorTerm.dag() is not supported if any product contains operators "
                    "acting on both bra and ket modes at the same time."
                )

        op_term_new = OperatorTerm(self.dims)
        for op_prod, modes, conjs, duals, coeff, op_prod_type in zip(
            self.op_prods, self.modes, self.conjs, self.duals, self.coeffs, self._op_prod_types
        ):
            if op_prod_type == OperatorProductType.ELEMENTARY:
                # Reconstruct per-operator mode and dual blocks, then reverse as blocks
                # to preserve each operator's internal mode ordering under reversal.
                mode_blocks, dual_blocks = [], []
                idx = 0
                for base_op in op_prod:
                    mode_blocks.append(modes[idx:idx + base_op.num_modes])
                    dual_blocks.append(duals[idx:idx + base_op.num_modes])
                    idx += base_op.num_modes

                op_term_new.append(
                    tuple(base_op.dag() for base_op in reversed(op_prod)),
                    modes=tuple(m for block in reversed(mode_blocks) for m in block),
                    duals=tuple(d for block in reversed(dual_blocks) for d in block),
                    coeff=coeff.conj(),
                )
            elif op_prod_type == OperatorProductType.MATRIX:
                op_term_new.append(
                    tuple(mat_op._copy() for mat_op in reversed(op_prod)),
                    conjs=tuple(not c for c in reversed(conjs)),
                    duals=tuple(reversed(duals)),
                    coeff=coeff.conj(),
                )
            else:  # identity operator product, only coefficient needs to be conjugated
                op_term_new.append(op_prod, modes=modes, duals=duals, coeff=coeff.conj())

        return op_term_new

    def _rebuild(self,
                 op_prods: list,
                 modes: list,
                 conjs: list,
                 duals: list,
                 modes_duals: list,
                 coeffs: list,
                 op_prod_types: list,
                 op_prod_batch_sizes: list,
                 ) -> "OperatorTerm":
        """
        Build a fresh operator term from already-simplified per-operator-product lists.

        The gradient and pointer attributes are re-initialized rather than carried over,
        since simplification runs before ``_create`` populates them.
        """
        op_term = type(self).__new__(type(self))

        op_term.op_prods = list(op_prods)
        op_term.modes = list(modes)
        op_term.conjs = list(conjs)
        op_term.duals = list(duals)
        op_term._modes_duals = list(modes_duals)
        op_term.coeffs = list(coeffs)
        op_term._op_prod_types = list(op_prod_types)
        op_term._op_prod_batch_sizes = list(op_prod_batch_sizes)

        op_term.dims = self.dims
        op_term._batch_size = self._batch_size
        op_term.dtype = self.dtype
        op_term._dtype_overwritten = self._dtype_overwritten

        # The simplified term is a new object and needs its own cuDensityMat handle.
        op_term._ptr = None
        op_term._consumed = False
        op_term._is_metadata_updated = False

        # `self` has been superseded by the term returned here, so the operator will refer to
        # the replacement from now on. Mark it consumed: it never receives a _ptr of its own,
        # so without this an append on it would silently succeed and be silently ignored.
        self._consumed = True

        n = len(op_term.op_prods)
        op_term._coeff_requires_grads = [None] * n
        op_term._coeff_callbacks = [None] * n
        op_term._coeff_grad_callbacks = [None] * n
        op_term._coeff_ptrs = [0] * n
        op_term._coeff_ptr_objs = [None] * n
        op_term._coeff_grad_ptrs = [0] * n
        op_term._coeff_grad_ptr_objs = [None] * n
        op_term._total_coeffs_ptrs = [0] * n
        op_term._total_coeffs_ptr_objs = [None] * n

        return op_term

    def _kron_simplify(self, kron_cond: Callable) -> "OperatorTerm":
        """
        Merge base operators within each operator product of the term.

        Args:
            kron_cond: Condition with contract ``kron_cond(modes_duals) -> bool``,
                where ``modes_duals`` is a tuple of per-base-op (mode, dual) pair tuples
                spanning the base operators at ``i`` and ``j`` and everything in between.

        Returns:
            A fresh operator term, or ``self`` when nothing was merged.
        """
        op_prods_new, modes_new, duals_new, modes_duals_new = [], [], [], []
        merged_any = False

        for op_prod, modes_op_prod, duals_op_prod, modes_duals_op_prod, op_prod_type in zip(
            self.op_prods, self.modes, self.duals, self._modes_duals, self._op_prod_types
        ):
            # Only elementary operator products contain base operators that can be kron-merged.
            if op_prod_type != OperatorProductType.ELEMENTARY:
                op_prods_new.append(op_prod)
                modes_new.append(modes_op_prod)
                duals_new.append(duals_op_prod)
                modes_duals_new.append(modes_duals_op_prod)
                continue

            # Work with mutable local copies: merged base ops are replaced in-place,
            # and eliminated base ops are set to None so that indices stay valid mid-scan.
            op_prod = list(op_prod)
            num_base_ops = len(op_prod)
            modes_duals_op_prod = list(modes_duals_op_prod)


            for i in range(num_base_ops):
                # Skip already-merged (None) and diagonal base ops.
                if op_prod[i] is None or op_prod[i].diag_offsets != ():
                    continue

                for j in range(i + 1, num_base_ops):
                    if op_prod[j] is None or op_prod[j].diag_offsets != ():
                        continue

                    # TODO: padded_matrix_product only supports a single duality. This can be
                    # relaxed in the future.
                    combined_duals = {dual for _, dual in modes_duals_op_prod[i] + modes_duals_op_prod[j]}
                    if len(combined_duals) > 1:
                        continue

                    # kron_cond receives the (mode, dual) pairs for base ops i through j
                    # inclusive, so it can inspect both the candidates and any operators
                    # sitting between them. Entries in between may have been set to None by
                    # an earlier merge in this sweep -- drop those here. A merge deposits its
                    # result at the lower of the two indices and the outer loop ascends, so
                    # whatever such an entry held now lives at an index <= i rather than
                    # between i and j: there is no operator left in between for the merge to
                    # reorder past, and kron_cond's contract promises a tuple of real
                    # per-base-op (mode, dual) pairs.
                    if kron_cond(tuple(md for md in modes_duals_op_prod[i:j + 1] if md is not None)):
                        # Mode indices per base operator, then the deduplicated sorted target
                        # (mode, dual) pairs.
                        modes_per_base_op = tuple(
                            tuple(md[0] for md in modes_duals_op_prod[k]) for k in (i, j))
                        modes_duals_target = tuple(sorted(set(modes_duals_op_prod[i]) | set(modes_duals_op_prod[j])))
                        modes_target = tuple(md[0] for md in modes_duals_target)

                        # padded_matrix_product operates on dense data, so unwrap the base
                        # operators here and rewrap the merged result. Both are already dense,
                        # since multidiagonal base operators are skipped above.
                        base_op_merged = ElementaryOperator(padded_matrix_product(
                            (op_prod[i].data, op_prod[j].data),
                            modes_per_base_op,
                            modes_target,
                            self.dims,
                            dual=modes_duals_op_prod[i][0][1],
                        ))

                        # Absorb j into i; mark j as empty.
                        op_prod[i] = base_op_merged
                        op_prod[j] = None
                        modes_duals_op_prod[i] = modes_duals_target
                        modes_duals_op_prod[j] = None

            # Compact out empty base operators.
            op_prod = tuple(base_op for base_op in op_prod if base_op is not None)
            modes_duals_op_prod = tuple(
                modes_duals_base_op for modes_duals_base_op in modes_duals_op_prod
                if modes_duals_base_op is not None)
            modes_op_prod = tuple(
                mode_dual[0] for modes_duals_base_op in modes_duals_op_prod
                for mode_dual in modes_duals_base_op)
            duals_op_prod = tuple(
                mode_dual[1] for modes_duals_base_op in modes_duals_op_prod
                for mode_dual in modes_duals_base_op)

            merged_any |= len(op_prod) < num_base_ops
            op_prods_new.append(op_prod)
            modes_duals_new.append(modes_duals_op_prod)
            modes_new.append(modes_op_prod)
            duals_new.append(duals_op_prod)

        if not merged_any:
            return self
        else:
            return self._rebuild(
                op_prods_new, modes_new, self.conjs, duals_new, modes_duals_new,
                self.coeffs, self._op_prod_types, self._op_prod_batch_sizes,
            )

    def _sum_simplify(self, sum_cond: Callable) -> "OperatorTerm":
        """
        Merge operator products of the term into each other.

        Args:
            sum_cond: Condition with contract ``sum_cond(modes_duals_src, modes_duals_dst)
                -> bool``, where each argument is a tuple of per-base-op (mode, dual) pair
                tuples, true when the source operator product can be summed into the
                destination operator product.

        Returns:
            A fresh operator term, or ``self`` when nothing was merged.
        """
        # Work with mutable local copies: containers are replaced in-place, and absorbed
        # operator products are set to None so that indices stay valid mid-scan.
        op_prods = list(self.op_prods)
        modes = list(self.modes)
        conjs = list(self.conjs)
        duals = list(self.duals)
        modes_duals = list(self._modes_duals)
        coeffs = list(self.coeffs)
        op_prod_types = list(self._op_prod_types)
        num_op_prods = len(op_prods)

        for i in range(num_op_prods):
            if op_prods[i] is None or op_prod_types[i] != OperatorProductType.ELEMENTARY:
                continue

            for j in range(i + 1, num_op_prods):
                if op_prods[j] is None or op_prod_types[j] != OperatorProductType.ELEMENTARY:
                    continue

                # sum_cond receives the (mode, dual) pairs of the source and destination
                # operator products, the destination being the container that absorbs the other.
                if sum_cond(modes_duals[j], modes_duals[i]):
                    src, dst = j, i
                elif sum_cond(modes_duals[i], modes_duals[j]):
                    src, dst = i, j
                else:
                    continue

                # Coefficients are always stored as complex128 (OperatorTerm.append enforces
                # this), but the fold below casts them to self.dtype before combining with
                # the (possibly real) base-operator data. For a real-dtype term this would
                # silently truncate the coefficient's imaginary part, changing the term's
                # value (e.g. a -1j coefficient on a real Hamiltonian would vanish instead of
                # contributing) -- skip the merge rather than risk that. sum_cond is still
                # consulted above regardless of dtype, so call-order/layering behavior is
                # unaffected; only the data fold itself is skipped here.
                if not jnp.issubdtype(self.dtype, jnp.complexfloating):
                    continue

                # padded_matrix_product only supports a single duality, and below we take
                # dual=modes_duals[src][0][0][1] -- the first source base operator's duality --
                # and apply it to the whole padded product. Mirror the kron pass's
                # combined_duals guard: skip the merge when the source's own base operators
                # don't all share that duality, rather than silently composing in the wrong
                # order for the ones that don't.
                src_duals = {dual for modes_duals_base_op in modes_duals[src]
                             for _, dual in modes_duals_base_op}
                if len(src_duals) > 1:
                    continue

                # The fold below assumes a single-base-op destination and only reads
                # op_prods[dst][0]; a custom sum_cond that allows len(modes_duals[dst]) > 1
                # would otherwise silently discard op_prods[dst][1:] instead of raising.
                # default_sum_cond enforces this by construction, but SimplifierConfig.sum_cond
                # is user-overridable, so check it explicitly here.
                if len(modes_duals[dst]) > 1:
                    raise ValueError(
                        "sum_cond selected a destination with more than one base operator "
                        f"(len(modes_duals[{dst}]) == {len(modes_duals[dst])}), but the merge "
                        "only folds into the first base operator (op_prods[dst][0]) and would "
                        "silently discard the rest. A conforming sum_cond must only merge into "
                        "a destination that already carries exactly one base operator."
                    )

                # Pad the absorbed product onto the container's modes, then sum the two with
                # their coefficients folded in, which leaves the container at unit coefficient.
                # padded_matrix_product operates on dense data, so unwrap the base operators
                # here, densifying any multidiagonal ones, and rewrap the padded result.
                op_prod_padded = ElementaryOperator(padded_matrix_product(
                    tuple(base_op.to_dense().data for base_op in op_prods[src]),
                    tuple(tuple(md[0] for md in modes_duals_base_op)
                          for modes_duals_base_op in modes_duals[src]),
                    modes[dst],
                    self.dims,
                    dual=modes_duals[src][0][0][1],
                ))

                op_prods[dst] = (
                    coeffs[dst].astype(self.dtype) * op_prods[dst][0]
                    + coeffs[src].astype(self.dtype) * op_prod_padded,)
                # The coefficients are now folded into the data, so the container carries a unit 
                # coefficient.
                coeffs[dst] = coeffs[dst] * 0 + 1

                # Absorb into the container; mark the absorbed product as empty. The container
                # keeps its own mode-dual pairs, since sum_cond requires them to already cover
                # the absorbed product.
                op_prods[src] = None
                modes[src] = None
                conjs[src] = None
                duals[src] = None
                modes_duals[src] = None
                coeffs[src] = None
                op_prod_types[src] = None

                # Operator product i was the one absorbed. Break out the inner loop
                # and move to the next i.
                if src == i:
                    break

        # Compact out empty operator products. Index every list by the surviving positions in
        # op_prods rather than filtering each on ``is not None``: op_prods is the only list
        # whose None entries always mean "absorbed", while _modes_duals also holds None for
        # non-elementary operator products, which must survive compaction.
        keep = [p for p in range(num_op_prods) if op_prods[p] is not None]

        op_prods_new = [op_prods[p] for p in keep]
        modes_new = [modes[p] for p in keep]
        conjs_new = [conjs[p] for p in keep]
        duals_new = [duals[p] for p in keep]
        modes_duals_new = [modes_duals[p] for p in keep]
        coeffs_new = [coeffs[p] for p in keep]
        op_prod_types_new = [op_prod_types[p] for p in keep]
        # Derive batch sizes the same way _update_metadata does, rather than with len(): a
        # coefficient is 0-d on the implicit-batching path, where len() raises.
        op_prod_batch_sizes_new = [
            get_batch_size(coeff) if is_vmap_traced(coeff) else math.prod(coeff.shape)
            for coeff in coeffs_new]

        merged_any = len(op_prods_new) < num_op_prods

        if not merged_any:
            return self
        else:
            return self._rebuild(
                op_prods_new, modes_new, conjs_new, duals_new, modes_duals_new,
                coeffs_new, op_prod_types_new, op_prod_batch_sizes_new,
            )

    def _reset_handles(self) -> None:
        """
        Clear the cuDensityMat handle state (own and cascading to base ops) so _create rebuilds
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
        for op_prod in self.op_prods:
            for base_op in op_prod:
                base_op._reset_handles()

    def _adopt_handles(self, other: "OperatorTerm") -> None:
        """
        Reset own handle state, then share the coefficient-pointer lists with a duplicate op term
        (same _op_term_id) so both observe the handles populated by _create on `other`. Mirrors
        the pytree list-sharing that Operator._create's op-term dedup relies on. Own base ops are
        reset (skipped in the action; the term handle is shared from `other` via that dedup), and
        `_ptr` stays None.
        """
        self._reset_handles()
        self._coeff_ptrs = other._coeff_ptrs
        self._coeff_ptr_objs = other._coeff_ptr_objs
        self._coeff_grad_ptrs = other._coeff_grad_ptrs
        self._coeff_grad_ptr_objs = other._coeff_grad_ptr_objs
        self._coeff_requires_grads = other._coeff_requires_grads
        self._coeff_callbacks = other._coeff_callbacks
        self._coeff_grad_callbacks = other._coeff_grad_callbacks
        self._total_coeffs_ptrs = other._total_coeffs_ptrs
        self._total_coeffs_ptr_objs = other._total_coeffs_ptr_objs

    def _create(self, handle, batch_size: int = 1):
        """
        Create opaque handle to the operator term.
        """
        # Ensure metadata (own batch sizes + cascaded base-op shape attributes) is populated for
        # direct callers; operator_action refreshes it beforehand, so this is a no-op there. The
        # cascade sets base-op flags too, so their own _create guards then skip.
        if not self._is_metadata_updated:
            self._update_metadata()

        # Create opaque handle to dependent elementary or matrix operators.
        # XXX: We are creating redundant elementary operators here.
        # When JAX flattens and unflattens the PyTrees, it creates new Python objects.
        for op_prod in self.op_prods:
            for elem_op in op_prod:
                elem_op._create(handle, batch_size)

        # Create the current operator term.
        if self._ptr is None:
            self._ptr = cudm.create_operator_term(handle, len(self.dims), self.dims)
            self.logger.debug(f"Created operator term at {hex(self._ptr)}")

            for i in range(len(self.op_prods)):

                if self._op_prod_batch_sizes[i] == 1:  # non-batched coefficient
                    # Traced scalars need to be passed through an intermediate memory slot.
                    self._coeff_callbacks[i] = get_scalar_assignment_callback(self.coeffs[i].dtype)
                    self._coeff_ptrs[i] = self._coeff_callbacks[i].callback.coeff.data.ptr

                else:  # batched coefficients
                    static_coeff_buf = cp.ones(
                        (self._op_prod_batch_sizes[i], *self.coeffs[i].shape), dtype=self.coeffs[i].dtype)
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

                if self._op_prod_types[i] == OperatorProductType.ELEMENTARY:
                    if self._op_prod_batch_sizes[i] == 1:
                        cudm.operator_term_append_elementary_product(
                            handle,
                            self._ptr,
                            len(self.op_prods[i]),
                            [elem_op._ptr for elem_op in self.op_prods[i]],
                            self.modes[i],
                            self.duals[i],
                            1.0,  # coefficient, to be updated to the real coefficient by callback
                            self._coeff_callbacks[i],
                            self._coeff_grad_callbacks[i],
                        )
                    else:
                        cudm.operator_term_append_elementary_product_batch(
                            handle,
                            self._ptr,
                            len(self.op_prods[i]),
                            [elem_op._ptr for elem_op in self.op_prods[i]],
                            self.modes[i],
                            self.duals[i],
                            self._op_prod_batch_sizes[i],
                            self._coeff_ptrs[i],
                            self._total_coeffs_ptrs[i],
                            self._coeff_callbacks[i],
                            self._coeff_grad_callbacks[i],
                        )
                else:  # MatrixOperator
                    if self._op_prod_batch_sizes[i] == 1:
                        cudm.operator_term_append_matrix_product(
                            handle,
                            self._ptr,
                            len(self.op_prods[i]),
                            [mat_op._ptr for mat_op in self.op_prods[i]],
                            self.conjs[i],
                            self.duals[i],
                            1.0,  # coefficient, to be updated to the real coefficient by callback
                            self._coeff_callbacks[i],
                            self._coeff_grad_callbacks[i],
                        )
                    else:
                        cudm.operator_term_append_matrix_product_batch(
                            handle,
                            self._ptr,
                            len(self.op_prods[i]),
                            [mat_op._ptr for mat_op in self.op_prods[i]],
                            self.conjs[i],
                            self.duals[i],
                            self._op_prod_batch_sizes[i],
                            self._coeff_ptrs[i],
                            self._total_coeffs_ptrs[i],
                            self._coeff_callbacks[i],
                            self._coeff_grad_callbacks[i],
                        )

            self.logger.debug(f"Appended operator products to operator term at {hex(self._ptr)}")

    def _destroy(self):
        """
        Destroy opaque handle to the operator term.
        """
        if self._ptr is not None:
            cudm.destroy_operator_term(self._ptr)
            self.logger.debug(f"Destroyed operator term at {hex(self._ptr)}")
            self._ptr = None
