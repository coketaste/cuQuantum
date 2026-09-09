# Copyright (c) 2025-2026, NVIDIA CORPORATION & AFFILIATES
#
# SPDX-License-Identifier: BSD-3-Clause

"""
Tests for the ElementaryOperator class.
"""

from itertools import product

import pytest
import jax
import jax.numpy as jnp

jax.config.update("jax_enable_x64", True)

from cuquantum.bindings import cudensitymat as cudm
from cuquantum.densitymat.jax import ElementaryOperator


key = jax.random.key(0)


@pytest.fixture(scope="class")
def handle():
    """
    Fixture to create a cuDensityMat handle.
    """
    handle_ = cudm.create()
    yield handle_
    cudm.destroy(handle_)


@pytest.fixture(scope="class")
def data(request):
    """
    Fixture to create a random tensor data buffer.
    """
    shape, dtype = request.param
    return jax.random.normal(key, shape, dtype=dtype)


data_ = data


class TestElementaryOperator:
    """
    Test the cuDensityMat ElementaryOperator class.
    """

    @pytest.mark.parametrize(
        "data",
        list(product(
            [(4, 4), (3, 5, 3, 5)],
            [jnp.float32, jnp.float64, jnp.complex64, jnp.complex128],
        )),
        indirect=True,
    )
    @pytest.mark.parametrize("batch_size", [1, 2])
    def test_init_dense(self, data, batch_size):
        """
        Test initializing dense elementary operator.
        """
        # TODO: This should be done in a better way, by creating random buffer rather than broadcasting.
        data_batched = jnp.broadcast_to(data, (batch_size, *data.shape))
        elem_op = ElementaryOperator(data_batched)
        elem_op._update_metadata()  # shape attrs (num_modes, mode_extents, sparsity) are set in _update_metadata

        assert elem_op._num_modes == len(data.shape) // 2
        assert elem_op._mode_extents == data.shape[:len(data.shape) // 2]
        assert elem_op.data.shape == (batch_size, *data.shape)
        assert elem_op.sparsity == cudm.ElementaryOperatorSparsity.OPERATOR_SPARSITY_NONE

    @pytest.mark.parametrize(
        "data",
        [
            ((3, 5, 3), jnp.complex128),
            ((3, 5, 3, 4), jnp.complex128),
        ],
        indirect=True,
    )
    def test_init_dense_fail(self, data):
        """
        Test initializing dense elementary operator with invalid data.
        """
        with pytest.raises(ValueError):
            ElementaryOperator(data)._update_metadata()  # shape validation is deferred to _update_metadata

    @pytest.mark.parametrize(
        "data",
        list(product(
            [(4, 2)],
            [jnp.float32, jnp.float64, jnp.complex64, jnp.complex128],
        )),
        indirect=True,
    )
    @pytest.mark.parametrize("diag_offsets", [(0, 1)])
    def test_init_multidiagonal(self, data, diag_offsets):
        """ 
        Test initializing multidiagonal elementary operator.
        """
        elem_op = ElementaryOperator(data, diag_offsets=diag_offsets)
        elem_op._update_metadata()  # shape attrs (num_modes, mode_extents, sparsity) are set in _update_metadata

        assert elem_op._num_modes == len(data.shape) // 2
        assert elem_op._mode_extents == data.shape[:len(data.shape) // 2]
        assert elem_op.data.shape == data.shape
        assert elem_op.sparsity == cudm.ElementaryOperatorSparsity.OPERATOR_SPARSITY_MULTIDIAGONAL
        assert elem_op.diag_offsets == (0, 1)

    @pytest.mark.parametrize(
        "data",
        list(product(
            [(2, 4, 2), (3, 6, 2)],
            [jnp.float32, jnp.float64, jnp.complex64, jnp.complex128],
        )),
        indirect=True,
    )
    @pytest.mark.parametrize("diag_offsets", [(0, 1)])
    def test_init_multidiagonal_batched(self, data, diag_offsets):
        """
        Test initializing explicitly batched multidiagonal elementary operator (3D data).
        Verifies that mode_extents reports the mode dimension, not the batch dimension.
        """
        batch_size = data.shape[0]
        extent = data.shape[1]
        elem_op = ElementaryOperator(data, diag_offsets=diag_offsets)
        elem_op._update_metadata()  # shape attrs (num_modes, mode_extents, sparsity) are set in _update_metadata

        assert elem_op._num_modes == 1
        assert elem_op._mode_extents == (extent,)
        assert elem_op.data.shape == data.shape
        assert elem_op.sparsity == cudm.ElementaryOperatorSparsity.OPERATOR_SPARSITY_MULTIDIAGONAL

    @pytest.mark.parametrize(
        "data,diag_offsets",
        [
            (((3, 5, 3, 5), jnp.complex128), (0, 1)),
            (((4, 3), jnp.complex128), (0, 1)),
            (((4, 3), jnp.complex128), (0, 1, 0)),
        ],
        indirect=["data"],
    )
    def test_init_multidiagonal_fail(self, data, diag_offsets):
        """
        Test initializing multidiagonal elementary operator with invalid diag_offsets.
        """
        with pytest.raises(ValueError):
            ElementaryOperator(data, diag_offsets=diag_offsets)._update_metadata()  # validation deferred to _update_metadata

    @pytest.mark.parametrize(
        "data",
        list(product(
            [(4, 4), (3, 5, 3, 5)],
            [jnp.float32, jnp.float64, jnp.complex64, jnp.complex128],
        )),
        indirect=True,
    )
    def test_create(self, data, handle):
        """
        Test elementary operator opaque handle creation.
        """
        elem_op = ElementaryOperator(data)

        # Test that _ptr is None before _create.
        assert elem_op._ptr is None

        # Test that _create sets the pointer.
        elem_op._create(handle)
        assert elem_op._ptr is not None

        # Test that calling _create again won't overwrite the pointer.
        ptr = elem_op._ptr
        elem_op._create(handle)
        assert elem_op._ptr == ptr

        elem_op._destroy()

    @pytest.mark.parametrize(
        "data",
        list(product(
            [(4, 4), (3, 5, 3, 5)],
            [jnp.float32, jnp.float64, jnp.complex64, jnp.complex128],
        )),
        indirect=True,
    )
    def test_destroy(self, data, handle):
        """
        Test elementary operator opaque handle destruction.
        """
        elem_op = ElementaryOperator(data)

        # Test that _create sets the pointer.
        elem_op._create(handle)
        assert elem_op._ptr is not None

        # Test that _destroy sets the pointer to None.
        elem_op._destroy()
        assert elem_op._ptr is None

        # Test that calling _destroy again has no effect.
        elem_op._destroy()
        assert elem_op._ptr is None

    @pytest.mark.parametrize("dtype", [jnp.float32, jnp.float64, jnp.complex64, jnp.complex128])
    @pytest.mark.parametrize("batch_size", [1, 3])
    def test_to_dense_multidiagonal(self, dtype, batch_size):
        """
        Test converting a multidiagonal elementary operator to dense form.
        """
        n = 4
        offsets = (-1, 0, 1)

        # column 0 (offset=-1): subdiagonal, last element is padding
        # column 1 (offset= 0): main diagonal
        # column 2 (offset= 1): superdiagonal, last element is padding
        sparse_2d = jnp.stack([
            jnp.array([8., 9., 10., 0.], dtype=dtype),
            jnp.array([1., 2.,  3., 4.], dtype=dtype),
            jnp.array([5., 6.,  7., 0.], dtype=dtype),
        ], axis=-1)  # (4, 3)

        if batch_size == 1:
            elem_op = ElementaryOperator(sparse_2d, diag_offsets=offsets)
        else:
            elem_op = ElementaryOperator(jnp.stack([sparse_2d] * batch_size), diag_offsets=offsets)

        dense_op = elem_op.to_dense()

        expected = jnp.array([
            [1, 5,  0,  0],
            [8, 2,  6,  0],
            [0, 9,  3,  7],
            [0, 0, 10,  4],
        ], dtype=dtype)

        assert dense_op.diag_offsets == ()
        assert dense_op.sparsity == cudm.ElementaryOperatorSparsity.OPERATOR_SPARSITY_NONE
        if batch_size == 1:  # non-batched 2D input keeps a batch-less data buffer
            assert dense_op.data.shape == (n, n)
            assert jnp.allclose(dense_op.data, expected)
        else:
            assert dense_op.data.shape == (batch_size, n, n)
            assert jnp.allclose(dense_op.data, jnp.broadcast_to(expected, (batch_size, n, n)))

    def test_to_dense_already_dense(self):
        """
        Test that to_dense on a dense elementary operator returns self.
        """
        elem_op = ElementaryOperator(jnp.eye(4, dtype=jnp.float64))
        assert elem_op.to_dense() is elem_op

    @pytest.mark.parametrize("offsets_a,offsets_b", [
        ((-1, 0, 1), (-1, 0, 1)),  # same offsets
        ((0,), (0,)),
        ((0,), (1,)),              # disjoint offsets
        ((-1, 0), (1,)),           # partial overlap
        ((-1,), (0, 1)),
    ])
    @pytest.mark.parametrize("dtype", [jnp.float32, jnp.float64, jnp.complex64, jnp.complex128])
    def test_dia_dia_addition(self, dtype, offsets_a, offsets_b):
        """
        Test DIA + DIA: result stays in DIA format with the union of offsets.
        """
        n = 4
        key_a, key_b = jax.random.split(key)
        data_a = jax.random.normal(key_a, (n, len(offsets_a)), dtype=dtype)
        data_b = jax.random.normal(key_b, (n, len(offsets_b)), dtype=dtype)

        op_a = ElementaryOperator(data_a, diag_offsets=offsets_a)
        op_b = ElementaryOperator(data_b, diag_offsets=offsets_b)
        result = op_a + op_b

        assert set(result.diag_offsets) == set(offsets_a) | set(offsets_b)
        assert result.sparsity == cudm.ElementaryOperatorSparsity.OPERATOR_SPARSITY_MULTIDIAGONAL
        assert jnp.allclose(result.to_dense().data, op_a.to_dense().data + op_b.to_dense().data)

    @pytest.mark.parametrize("offsets_a,offsets_b", [
        ((-1, 0, 1), (-1, 0, 1)),  # same offsets
        ((0,), (0,)),
        ((0,), (1,)),              # disjoint offsets
        ((-1, 0), (1,)),           # partial overlap
        ((-1,), (0, 1)),
    ])
    @pytest.mark.parametrize("dtype", [jnp.float32, jnp.float64, jnp.complex64, jnp.complex128])
    def test_dia_dia_subtraction(self, dtype, offsets_a, offsets_b):
        """
        Test DIA - DIA: result stays in DIA format with the union of offsets.
        """
        n = 4
        key_a, key_b = jax.random.split(key)
        data_a = jax.random.normal(key_a, (n, len(offsets_a)), dtype=dtype)
        data_b = jax.random.normal(key_b, (n, len(offsets_b)), dtype=dtype)

        op_a = ElementaryOperator(data_a, diag_offsets=offsets_a)
        op_b = ElementaryOperator(data_b, diag_offsets=offsets_b)
        result = op_a - op_b

        assert set(result.diag_offsets) == set(offsets_a) | set(offsets_b)
        assert result.sparsity == cudm.ElementaryOperatorSparsity.OPERATOR_SPARSITY_MULTIDIAGONAL
        assert jnp.allclose(result.to_dense().data, op_a.to_dense().data - op_b.to_dense().data)

    @pytest.mark.parametrize("offsets_a,offsets_b", [
        ((-1, 0, 1), (-1, 0, 1)),   # same offsets — product is pentadiagonal
        ((0,), (0,)),               # diagonal × diagonal — stays diagonal
        ((0,), (1,)),               # diagonal × superdiagonal
        ((-1, 0), (0, 1)),          # partial overlap
    ])
    @pytest.mark.parametrize("dtype", [jnp.float32, jnp.float64, jnp.complex64, jnp.complex128])
    def test_dia_dia_matrix_multiplication(self, dtype, offsets_a, offsets_b):
        """
        Test DIA @ DIA: result stays in DIA format with offsets equal to the pairwise sums.
        """
        n = 4
        key_a, key_b = jax.random.split(key)
        data_a = jax.random.normal(key_a, (n, len(offsets_a)), dtype=dtype)
        data_b = jax.random.normal(key_b, (n, len(offsets_b)), dtype=dtype)

        op_a = ElementaryOperator(data_a, diag_offsets=offsets_a)
        op_b = ElementaryOperator(data_b, diag_offsets=offsets_b)
        result = op_a @ op_b

        expected_offsets = {a + b for a in offsets_a for b in offsets_b if abs(a + b) < n}
        assert set(result.diag_offsets) == expected_offsets
        assert result.sparsity == cudm.ElementaryOperatorSparsity.OPERATOR_SPARSITY_MULTIDIAGONAL
        rtol = 1e-3 if dtype in (jnp.float32, jnp.complex64) else 1e-5
        atol = 1e-5 if dtype in (jnp.float32, jnp.complex64) else 1e-8
        assert jnp.allclose(result.to_dense().data, op_a.to_dense().data @ op_b.to_dense().data, rtol=rtol, atol=atol)

    @pytest.mark.parametrize("dtype", [jnp.float32, jnp.float64, jnp.complex64, jnp.complex128])
    def test_mixed_addition(self, dtype):
        """
        Test addition between dense and DIA elementary operators.
        """
        n = 4
        dense_data = jax.random.normal(key, (n, n), dtype=dtype)
        sparse_2d = jnp.stack([
            jnp.array([8., 9., 10., 0.], dtype=dtype),
            jnp.array([1., 2.,  3., 4.], dtype=dtype),
            jnp.array([5., 6.,  7., 0.], dtype=dtype),
        ], axis=-1)

        dense_op = ElementaryOperator(dense_data)
        dia_op = ElementaryOperator(sparse_2d, diag_offsets=(-1, 0, 1))

        a = dense_op.data  # (1, 4, 4)
        b = jnp.array([    # (1, 4, 4) — exact dense form of the DIA operator
            [1, 5,  0,  0],
            [8, 2,  6,  0],
            [0, 9,  3,  7],
            [0, 0, 10,  4],
        ], dtype=dtype)[None]

        assert jnp.allclose((dense_op + dia_op).data, a + b)
        assert jnp.allclose((dia_op + dense_op).data, b + a)

    @pytest.mark.parametrize("dtype", [jnp.float32, jnp.float64, jnp.complex64, jnp.complex128])
    def test_mixed_subtraction(self, dtype):
        """
        Test subtraction between dense and DIA elementary operators.
        """
        n = 4
        dense_data = jax.random.normal(key, (n, n), dtype=dtype)
        sparse_2d = jnp.stack([
            jnp.array([8., 9., 10., 0.], dtype=dtype),
            jnp.array([1., 2.,  3., 4.], dtype=dtype),
            jnp.array([5., 6.,  7., 0.], dtype=dtype),
        ], axis=-1)

        dense_op = ElementaryOperator(dense_data)
        dia_op = ElementaryOperator(sparse_2d, diag_offsets=(-1, 0, 1))

        a = dense_op.data  # (1, 4, 4)
        b = jnp.array([    # (1, 4, 4) — exact dense form of the DIA operator
            [1, 5,  0,  0],
            [8, 2,  6,  0],
            [0, 9,  3,  7],
            [0, 0, 10,  4],
        ], dtype=dtype)[None]

        assert jnp.allclose((dense_op - dia_op).data, a - b)
        assert jnp.allclose((dia_op - dense_op).data, b - a)

    @pytest.mark.parametrize("dtype", [jnp.float32, jnp.float64, jnp.complex64, jnp.complex128])
    def test_mixed_matrix_multiplication(self, dtype):
        """
        Test matrix multiplication between dense and DIA elementary operators.
        """
        n = 4
        dense_data = jax.random.normal(key, (n, n), dtype=dtype)
        sparse_2d = jnp.stack([
            jnp.array([8., 9., 10., 0.], dtype=dtype),
            jnp.array([1., 2.,  3., 4.], dtype=dtype),
            jnp.array([5., 6.,  7., 0.], dtype=dtype),
        ], axis=-1)

        dense_op = ElementaryOperator(dense_data)
        dia_op = ElementaryOperator(sparse_2d, diag_offsets=(-1, 0, 1))

        a = dense_op.data  # (1, 4, 4)
        b = jnp.array([    # (1, 4, 4) — exact dense form of the DIA operator
            [1, 5,  0,  0],
            [8, 2,  6,  0],
            [0, 9,  3,  7],
            [0, 0, 10,  4],
        ], dtype=dtype)[None]

        assert jnp.allclose((dense_op @ dia_op).data, a @ b)
        assert jnp.allclose((dia_op @ dense_op).data, b @ a)

    @pytest.mark.parametrize("dtype", [jnp.float32, jnp.float64, jnp.complex64, jnp.complex128])
    def test_mixed_tensor_product(self, dtype):
        """
        Test tensor product between dense and DIA elementary operators.
        """
        dense_data = jax.random.normal(key, (3, 3), dtype=dtype)
        sparse_2d = jnp.stack([
            jnp.array([8., 9., 10., 0.], dtype=dtype),
            jnp.array([1., 2.,  3., 4.], dtype=dtype),
            jnp.array([5., 6.,  7., 0.], dtype=dtype),
        ], axis=-1)

        dense_op = ElementaryOperator(dense_data)
        dia_op = ElementaryOperator(sparse_2d, diag_offsets=(-1, 0, 1))

        a = dense_op.data  # (3, 3)
        b = jnp.array([    # (4, 4) — exact dense form of the DIA operator
            [1, 5,  0,  0],
            [8, 2,  6,  0],
            [0, 9,  3,  7],
            [0, 0, 10,  4],
        ], dtype=dtype)

        # dense_batched_kron produces (m, n, m, n) for non-batched single-mode operators
        ref = jnp.einsum('ij,kl->ikjl', a, b)
        assert jnp.allclose((dense_op & dia_op).data, ref)
