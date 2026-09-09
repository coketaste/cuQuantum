# Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES
#
# SPDX-License-Identifier: BSD-3-Clause

"""Demonstrate queued state-vector updates and stochastic channels.

SVUpdater copies matrices when they are enqueued, then applies queued operations together.
These examples cover composition, ordering, branch selection, and configuration on a
single-device state vector.
"""

import numpy as np
import pytest

from cuquantum.bindings import custatevec as cusv
from cuquantum.bindings import custatevecEx as cusvex

from . import assertions, dtype_to_data_type
from .conftest import over_dtypes, sv_on_single
from .sv_tools import (
    GATE_H,
    GATE_I,
    GATE_X,
    GATE_Z,
    MatrixData,
    WiresData,
    enqueue_matrix,
    get_accessible_amplitudes,
)


@sv_on_single
@over_dtypes
def test_all_hadamard_uniform(sv_factory, sv_updater_factory, dtype):
    """Create a uniform state with commuting queued Hadamard gates."""
    num_wires = 3
    sv_descriptor, sv_meta = sv_factory(dtype=dtype, num_wires=num_wires)
    updater = sv_updater_factory(dtype=dtype)

    for wire in range(num_wires):
        enqueue_matrix(
            updater,
            MatrixData(GATE_H, dtype=dtype),
            WiresData([wire]),
        )

    assert cusvex.sv_updater_get_max_num_required_randnums(updater) == 0
    cusvex.sv_updater_apply(
        sv_updater=updater,
        state_vector=sv_descriptor,
        randnums=[],  # Gate-only queue draws no random numbers
        num_randnums=0,
    )

    expected = np.full(
        sv_meta.created_num_elements,
        1.0 / np.sqrt(sv_meta.created_num_elements),
        dtype=dtype,
    )
    assertions.assert_allclose(
        get_accessible_amplitudes(sv_descriptor, sv_meta),
        expected,
        err_msg="queued Hadamards did not produce the uniform state",
    )
    assertions.assert_normalized(sv_descriptor, sv_meta)


@sv_on_single
@over_dtypes
def test_ghz_composition_and_ordering(sv_factory, sv_updater_factory, dtype):
    """Create a GHZ state while preserving queue order."""
    num_wires = 3
    sv_descriptor, sv_meta = sv_factory(dtype=dtype, num_wires=num_wires)
    updater = sv_updater_factory(dtype=dtype)

    enqueue_matrix(
        updater,
        MatrixData(GATE_H, dtype=dtype),
        WiresData([0]),
    )
    for control in range(num_wires - 1):  # CX gates down the chain
        enqueue_matrix(
            updater,
            MatrixData(GATE_X, dtype=dtype),
            WiresData(targets=[control + 1], controls=[control]),
        )

    assert cusvex.sv_updater_get_max_num_required_randnums(updater) == 0
    cusvex.sv_updater_apply(
        sv_updater=updater,
        state_vector=sv_descriptor,
        randnums=[],
        num_randnums=0,
    )

    state = get_accessible_amplitudes(sv_descriptor, sv_meta)
    inv_sqrt2 = 1.0 / np.sqrt(2.0)

    # GHZ components: |0...0> (0), |1...1> (dim-1)
    assertions.assert_allclose(state[0], inv_sqrt2, err_msg="GHZ amplitude at |0...0> != 1/sqrt(2)")
    assertions.assert_allclose(state[-1], inv_sqrt2, err_msg="GHZ amplitude at |1...1> != 1/sqrt(2)")

    interior = state[1:-1]  # every mixed bit string must be empty
    assertions.assert_near_zero(interior, err_msg="GHZ has amplitude outside |0...0> / |1...1>")
    assertions.assert_normalized(sv_descriptor, sv_meta)


@sv_on_single
@over_dtypes
def test_clear_discards_queued_ops(sv_factory, sv_updater_factory, dtype):
    """Discard queued operations before a later application.

    Enqueue X on wire 0, clear, then enqueue H and apply.
    """
    num_wires = 3
    sv_descriptor, sv_meta = sv_factory(dtype=dtype, num_wires=num_wires)
    updater = sv_updater_factory(dtype=dtype)

    enqueue_matrix(
        updater,
        MatrixData(GATE_X, dtype=dtype),
        WiresData([0]),
    )
    cusvex.sv_updater_clear(updater)

    # The same updater is then reused: only the post-clear op takes effect.
    enqueue_matrix(
        updater,
        MatrixData(GATE_H, dtype=dtype),
        WiresData([0]),
    )
    cusvex.sv_updater_apply(
        sv_updater=updater,
        state_vector=sv_descriptor,
        randnums=[],  # a gate-only queue needs no random numbers
        num_randnums=0,
    )

    expected = np.zeros(sv_meta.created_num_elements, dtype=dtype)
    expected[0] = 1.0 / np.sqrt(2.0)  # |0...00>
    expected[1] = 1.0 / np.sqrt(2.0)  # |0...01>
    assertions.assert_allclose(
        get_accessible_amplitudes(sv_descriptor, sv_meta),
        expected,
        err_msg="post-clear application does not equal H-only result",
    )
    assertions.assert_normalized(sv_descriptor, sv_meta)


@over_dtypes
def test_max_num_required_randnums_counts_channels(sv_updater_factory, dtype):
    """Count one required random number per queued channel."""
    # (a) gates alone
    updater = sv_updater_factory(dtype=dtype)
    enqueue_matrix(updater, MatrixData(GATE_X, dtype=dtype), WiresData([0]))
    enqueue_matrix(updater, MatrixData(GATE_H, dtype=dtype), WiresData([1]))
    assert cusvex.sv_updater_get_max_num_required_randnums(updater) == 0

    # (b) one unitary channel
    updater = sv_updater_factory(dtype=dtype)
    matrices = [np.ascontiguousarray(gate, dtype=dtype) for gate in (GATE_I, GATE_X)]
    cusvex.sv_updater_enqueue_unitary_channel(
        sv_updater=updater,
        unitaries=[int(matrix.ctypes.data) for matrix in matrices],  # host addresses
        unitaries_data_type=dtype_to_data_type[dtype],
        ex_matrix_types=[cusvex.MatrixType.DENSE] * len(matrices),
        num_unitaries=len(matrices),
        layout=cusv.MatrixLayout.ROW,
        probabilities=[0.5, 0.5],
        channel_wires=[0],
        num_channel_wires=1,
    )
    assert cusvex.sv_updater_get_max_num_required_randnums(updater) == 1

    # (c) clear resets
    cusvex.sv_updater_clear(updater)
    assert cusvex.sv_updater_get_max_num_required_randnums(updater) == 0


###############################################################################
# Unitary-channel branch selection
###############################################################################


@sv_on_single
@over_dtypes
def test_unitary_channel_branch_selection(sv_factory, sv_updater_factory, dtype):
    """Select unitary-channel branches from cumulative probabilities."""
    num_wires = 3
    data_type = dtype_to_data_type[dtype]

    def apply_branch(randnum):
        sv_descriptor, sv_meta = sv_factory(dtype=dtype, num_wires=num_wires)  # fresh |0...0>
        updater = sv_updater_factory(dtype=dtype)
        matrices = [
            np.ascontiguousarray(GATE_I, dtype=dtype),
            np.ascontiguousarray(GATE_X, dtype=dtype),
        ]
        cusvex.sv_updater_enqueue_unitary_channel(
            sv_updater=updater,
            unitaries=[int(matrix.ctypes.data) for matrix in matrices],  # host addresses
            unitaries_data_type=data_type,
            ex_matrix_types=[cusvex.MatrixType.DENSE] * len(matrices),  # one per unitary
            num_unitaries=len(matrices),
            layout=cusv.MatrixLayout.ROW,
            probabilities=[0.5, 0.5],
            channel_wires=[0],
            num_channel_wires=1,
        )
        # A channel draws exactly one random number, whichever branch it selects.
        assert cusvex.sv_updater_get_max_num_required_randnums(updater) == 1
        cusvex.sv_updater_apply(
            sv_updater=updater,
            state_vector=sv_descriptor,
            randnums=[randnum],
            num_randnums=1,
        )
        return sv_descriptor, sv_meta

    for randnum, expected_index in ((0.0, 0), (np.nextafter(1.0, 0.0), 1)):
        sv_descriptor, sv_meta = apply_branch(randnum)
        assertions.assert_basis_state(sv_descriptor, sv_meta, expected_index)
        assertions.assert_normalized(sv_descriptor, sv_meta)


###############################################################################
# General channel with one Kraus operator
###############################################################################


@sv_on_single
@over_dtypes
def test_general_channel_single_kraus_flip(sv_factory, sv_updater_factory, dtype):
    """Flip the zero state with a single-Kraus general channel."""
    num_wires = 3
    data_type = dtype_to_data_type[dtype]

    sv_descriptor, sv_meta = sv_factory(dtype=dtype, num_wires=num_wires)  # |0...0>
    updater = sv_updater_factory(dtype=dtype)

    matrices = [np.ascontiguousarray(GATE_X, dtype=dtype)]  # single Kraus operator
    cusvex.sv_updater_enqueue_general_channel(
        sv_updater=updater,
        matrices=[int(matrix.ctypes.data) for matrix in matrices],  # host addresses
        matrix_data_type=data_type,
        ex_matrix_types=[cusvex.MatrixType.DENSE] * len(matrices),
        num_matrices=len(matrices),
        layout=cusv.MatrixLayout.ROW,
        channel_wires=[0],
        num_channel_wires=1,
    )

    # A channel draws a random number even when its outcome is fixed.
    assert cusvex.sv_updater_get_max_num_required_randnums(updater) == 1
    cusvex.sv_updater_apply(
        sv_updater=updater,
        state_vector=sv_descriptor,
        randnums=[0.5],  # any value selects the sole Kraus operator
        num_randnums=1,
    )

    assertions.assert_basis_state(sv_descriptor, sv_meta, 1)
    assertions.assert_normalized(sv_descriptor, sv_meta)


###############################################################################
# Sequence and pointer-array channel arguments
###############################################################################


@sv_on_single
@over_dtypes
def test_channel_nptr_sequence_and_pointer_array_forms_equivalent(sv_factory, sv_updater_factory, dtype):
    """Produce the same channel result with both supported pointer forms.

    Each matrix list can be given either as a sequence of per-matrix host addresses or
    as the single address of a contiguous array holding those addresses. The array must
    stay alive until the enqueue call has copied the matrices.
    """
    num_wires = 3
    data_type = dtype_to_data_type[dtype]

    def unitary_state(as_pointer_array):
        sv_descriptor, sv_meta = sv_factory(dtype=dtype, num_wires=num_wires)
        updater = sv_updater_factory(dtype=dtype)
        enqueue_matrix(updater, MatrixData(GATE_H, dtype=dtype), WiresData([0]))
        matrices = [np.ascontiguousarray(gate, dtype=dtype) for gate in (GATE_I, GATE_Z)]
        addresses = np.array([int(matrix.ctypes.data) for matrix in matrices], dtype=np.uintp)
        cusvex.sv_updater_enqueue_unitary_channel(
            sv_updater=updater,
            # Either pointer to array of addresses or list of addresses themselves.
            unitaries=int(addresses.ctypes.data) if as_pointer_array else addresses.tolist(),
            unitaries_data_type=data_type,
            ex_matrix_types=[cusvex.MatrixType.DENSE] * len(matrices),
            num_unitaries=len(matrices),
            layout=cusv.MatrixLayout.ROW,
            probabilities=[0.5, 0.5],
            channel_wires=[0],
            num_channel_wires=1,
        )
        cusvex.sv_updater_apply(
            sv_updater=updater,
            state_vector=sv_descriptor,
            randnums=[np.nextafter(1.0, 0.0)],  # selects the last branch, Z
            num_randnums=1,
        )
        return get_accessible_amplitudes(sv_descriptor, sv_meta)

    def general_state(as_pointer_array):
        sv_descriptor, sv_meta = sv_factory(dtype=dtype, num_wires=num_wires)
        updater = sv_updater_factory(dtype=dtype)
        enqueue_matrix(updater, MatrixData(GATE_H, dtype=dtype), WiresData([0]))
        # X/sqrt(2) and Z/sqrt(2) are a complete Kraus set with distinct actions.
        kraus_operators = [GATE_X / np.sqrt(2.0), GATE_Z / np.sqrt(2.0)]
        matrices = [np.ascontiguousarray(gate, dtype=dtype) for gate in kraus_operators]
        addresses = np.array([int(matrix.ctypes.data) for matrix in matrices], dtype=np.uintp)
        cusvex.sv_updater_enqueue_general_channel(
            sv_updater=updater,
            matrices=int(addresses.ctypes.data) if as_pointer_array else addresses.tolist(),
            matrix_data_type=data_type,
            ex_matrix_types=[cusvex.MatrixType.DENSE] * len(matrices),
            num_matrices=len(matrices),
            layout=cusv.MatrixLayout.ROW,
            channel_wires=[0],
            num_channel_wires=1,
        )
        cusvex.sv_updater_apply(
            sv_updater=updater,
            state_vector=sv_descriptor,
            randnums=[np.nextafter(1.0, 0.0)],  # selects the last branch, Z/sqrt(2)
            num_randnums=1,
        )
        return get_accessible_amplitudes(sv_descriptor, sv_meta)

    # Result is a superposition that changes if the matrix addresses are misread.
    expected = np.zeros(1 << num_wires, dtype=dtype)
    expected[0] = 1.0 / np.sqrt(2.0)
    expected[1] = -1.0 / np.sqrt(2.0)

    unitary_pointer_array_state = unitary_state(as_pointer_array=True)
    assertions.assert_allclose(
        unitary_pointer_array_state,
        expected,
        err_msg="unitary-channel pointer-array form did not apply Z to the superposition",
    )
    assertions.assert_allclose(
        unitary_state(as_pointer_array=False),
        unitary_pointer_array_state,
        err_msg="unitary-channel pointer-array and sequence forms disagreed",
    )

    general_pointer_array_state = general_state(as_pointer_array=True)
    assertions.assert_allclose(
        general_pointer_array_state,
        expected,
        err_msg="general-channel pointer-array form did not apply Z to the superposition",
    )
    assertions.assert_allclose(
        general_state(as_pointer_array=False),
        general_pointer_array_state,
        err_msg="general-channel pointer-array and sequence forms disagreed",
    )


###############################################################################
# Error translation
###############################################################################


def test_enqueue_matrix_precision_error_translates_to_cuStateVecError(sv_updater_factory):
    """Reject a matrix with lower precision than the updater.

    The check is against the updater's dtype and happens at enqueue.
    """
    updater = sv_updater_factory(dtype=np.complex128)
    with pytest.raises(cusv.cuStateVecError) as excinfo:
        enqueue_matrix(
            updater,
            MatrixData(GATE_X, dtype=np.complex64),
            WiresData([0]),
        )
    assert excinfo.value.status == cusv.Status.INVALID_VALUE


@sv_on_single
def test_apply_state_vector_dtype_error_translates_to_cuStateVecError(sv_factory, sv_updater_factory):
    """Reject a state vector whose dtype differs from the updater.

    An updater accepts only state vectors of its own dtype, so this is caught at apply.
    """
    sv_descriptor, _ = sv_factory(dtype=np.complex128, num_wires=3)
    updater = sv_updater_factory(dtype=np.complex64)
    enqueue_matrix(
        updater,
        MatrixData(GATE_X, dtype=np.complex64),
        WiresData([0]),
    )

    with pytest.raises(cusv.cuStateVecError) as excinfo:
        cusvex.sv_updater_apply(
            sv_updater=updater,
            state_vector=sv_descriptor,
            randnums=[],
            num_randnums=0,
        )
    assert excinfo.value.status == cusv.Status.INVALID_VALUE


###############################################################################
# Configure a working updater with SVUpdaterConfigItem
###############################################################################


@sv_on_single
@over_dtypes
def test_config_item_configures_working_updater(sv_factory, sv_updater_factory, dtype):
    """Configure a working updater from one Python configuration value.

    A configuration item tunes host threading or gate fusion, but does not change the result. Check basic validation.
    """
    num_wires = 3
    item = cusvex.SVUpdaterConfigItem(
        name=cusvex.SVUpdaterConfigName.MAX_NUM_HOST_THREADS,
        value=8,  # acceptable range [1, 32]
    )
    updater = sv_updater_factory(dtype=dtype, config_items=[item])
    sv_descriptor, sv_meta = sv_factory(dtype=dtype, num_wires=num_wires)

    for wire in range(num_wires):
        enqueue_matrix(
            updater,
            MatrixData(GATE_H, dtype=dtype),
            WiresData([wire]),
        )
    cusvex.sv_updater_apply(
        sv_updater=updater,
        state_vector=sv_descriptor,
        randnums=[],
        num_randnums=0,
    )

    expected = np.full(
        sv_meta.created_num_elements,
        1.0 / np.sqrt(sv_meta.created_num_elements),
        dtype=dtype,
    )
    assertions.assert_allclose(
        get_accessible_amplitudes(sv_descriptor, sv_meta),
        expected,
        err_msg="configured updater did not produce the uniform state",
    )
    assertions.assert_normalized(sv_descriptor, sv_meta)


@sv_on_single
@over_dtypes
def test_multiple_config_items_configure_working_updater(sv_factory, sv_updater_factory, dtype):
    """Marshal multiple Python configuration values into one native C array."""
    num_wires = 3
    items = [
        cusvex.SVUpdaterConfigItem(
            name=cusvex.SVUpdaterConfigName.MAX_NUM_HOST_THREADS,
            value=8,  # acceptable range [1, 32]
        ),
        cusvex.SVUpdaterConfigItem(
            name=cusvex.SVUpdaterConfigName.DENSE_FUSION_SIZE,
            value=4,  # acceptable range [1, 10]
        ),
    ]

    sv_descriptor, sv_meta = sv_factory(dtype=dtype, num_wires=num_wires)
    updater = sv_updater_factory(dtype=dtype, config_items=items)
    for wire in range(num_wires):
        enqueue_matrix(
            updater,
            MatrixData(GATE_H, dtype=dtype),
            WiresData([wire]),
        )
    cusvex.sv_updater_apply(
        sv_updater=updater,
        state_vector=sv_descriptor,
        randnums=[],
        num_randnums=0,
    )

    expected = np.full(
        sv_meta.created_num_elements,
        1.0 / np.sqrt(sv_meta.created_num_elements),
        dtype=dtype,
    )
    assertions.assert_allclose(
        get_accessible_amplitudes(sv_descriptor, sv_meta),
        expected,
        err_msg="updater configured from multiple items did not produce the uniform state",
    )
    assertions.assert_normalized(sv_descriptor, sv_meta)
