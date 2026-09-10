# Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES
#
# SPDX-License-Identifier: BSD-3-Clause

"""Numerical assertions for the cuStateVec Ex examples."""

import numpy as np
from numpy.typing import ArrayLike

from .sv_tools import (
    ComplexDType,
    MPIContext,
    StateVectorDescriptorT,
    StateVectorFixtureMeta,
    accessible_amplitude_array_index,
    get_accessible_amplitudes,
)

###############################################################################
# Numerical tolerances
###############################################################################
#
# Generic comparisons use fixed relative and absolute tolerances. Values
# expected to be zero use an absolute-only tolerance.

ALLCLOSE_RTOL = 1e-5
ALLCLOSE_ATOL = 1e-8
NEAR_ZERO_ATOL = 1e-8
NEAR_ZERO_EPS_SCALE = 10


def near_zero_atol(dtype: ComplexDType) -> float:
    """Absolute tolerance scaled by precision."""
    return max(NEAR_ZERO_ATOL, NEAR_ZERO_EPS_SCALE * np.finfo(dtype).eps)


def _scaled_atol(*values: ArrayLike) -> float:
    """Absolute tolerance for the least precise value taking part in a comparison."""
    dtypes = [np.asarray(value).dtype for value in values]
    return max(
        (near_zero_atol(dtype) for dtype in dtypes if np.issubdtype(dtype, np.inexact)),
        default=NEAR_ZERO_ATOL,
    )


# These test assertions compare small host-resident example results with NumPy.
# Device-to-host copies may be expensive for large vectors.


def assert_allclose(
    actual: ArrayLike,
    desired: ArrayLike,
    rtol: float = ALLCLOSE_RTOL,
    atol: float | None = None,
    err_msg: str = "",
) -> None:
    np.testing.assert_allclose(
        np.asarray(actual),
        np.asarray(desired),
        rtol=rtol,
        atol=_scaled_atol(actual, desired) if atol is None else atol,
        err_msg=err_msg,
    )


def assert_near_zero(actual: ArrayLike, atol: float | None = None, err_msg: str = "") -> None:
    values = np.asarray(actual)
    np.testing.assert_allclose(
        values,
        np.zeros_like(values),
        rtol=0.0,
        atol=_scaled_atol(values) if atol is None else atol,
        err_msg=err_msg,
    )


###############################################################################
# State assertions
###############################################################################


def assert_basis_state(
    sv_descriptor: StateVectorDescriptorT,
    sv_meta: StateVectorFixtureMeta,
    index: int,
) -> None:
    """Assert that accessible amplitudes agree with the logical basis state ``|index>``.

    The process that owns ``index`` must see amplitude one there and zero
    elsewhere. Processes that do not own it must see only zero amplitudes.
    """
    amplitudes = get_accessible_amplitudes(sv_descriptor, sv_meta)
    array_index = accessible_amplitude_array_index(sv_descriptor, index)
    if array_index is None:
        assert_near_zero(
            amplitudes,
            err_msg=f"basis index {index} is not accessible to this process; accessible amplitudes must be zero",
        )
        return

    assert_allclose(
        amplitudes[array_index],
        1.0,
        err_msg=f"amplitude at basis index {index} != 1",
    )
    other_amplitudes_mask = np.ones(amplitudes.shape[0], dtype=bool)
    other_amplitudes_mask[array_index] = False
    assert_near_zero(
        amplitudes[other_amplitudes_mask],
        err_msg=f"nonzero amplitude away from basis index {index}",
    )


def assert_normalized(
    sv_descriptor: StateVectorDescriptorT,
    sv_meta: StateVectorFixtureMeta,
) -> None:
    """Assert that the complete logical state vector has unit norm.

    Each process computes the norm of its accessible amplitudes. Distributed
    state vectors sum those contributions across the MPI communicator.
    """
    amplitudes = get_accessible_amplitudes(sv_descriptor, sv_meta)
    local_norm_squared = float(np.sum(np.abs(amplitudes.astype(np.complex128)) ** 2))
    norm_squared = (
        sv_meta.mpi_context.comm.allreduce(local_norm_squared) if sv_meta.mpi_context is not None else local_norm_squared
    )
    assert_allclose(norm_squared, 1.0, err_msg="state is not normalized")


def assert_collective_result_consistent(
    mpi_context: MPIContext | None,
    value: ArrayLike,
    *,
    err_msg: str = "",
) -> None:
    """Assert that every MPI rank received a consistent collective result.

    Single-process calls are no-ops. Under MPI, ``allgather`` requires every
    rank to call this helper in the same order. Floating-point and complex
    results use the default tolerances; all other data types compare exactly.
    """
    if mpi_context is None:
        return

    local = np.asarray(value)
    gathered = mpi_context.comm.allgather(local)
    reference = gathered[0]
    prefix = err_msg or "collective result differed across ranks"
    for rank, result in enumerate(gathered[1:], start=1):
        if result.shape != reference.shape or result.dtype != reference.dtype:
            raise AssertionError(
                f"{prefix}: rank {rank} has shape {result.shape}, dtype {result.dtype}; "
                f"rank 0 has shape {reference.shape}, dtype {reference.dtype}"
            )
        if np.issubdtype(reference.dtype, np.inexact):
            assert_allclose(
                result,
                reference,
                err_msg=f"{prefix}: rank {rank}",
            )
        else:
            np.testing.assert_array_equal(result, reference, err_msg=f"{prefix}: rank {rank}")
