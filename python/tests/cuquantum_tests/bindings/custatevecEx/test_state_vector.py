# Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES
#
# SPDX-License-Identifier: BSD-3-Clause

"""Demonstrate state-vector state access, properties, synchronization, and growth."""

import numpy as np
import pytest
from cuda.bindings import runtime as cudart

from cuquantum.bindings import custatevec as cusv
from cuquantum.bindings import custatevecEx as cusvex

from . import assertions
from .conftest import (
    over_dtypes,
    sv_across_distributions,
    sv_on_multi_device,
    sv_on_multi_process,
    sv_on_single,
    sv_on_single_hostmem,
)
from .sv_tools import (
    GATE_X,
    MULTI_DEVICE,
    SINGLE,
    SINGLE_HOSTMEM,
    MatrixData,
    WiresData,
    apply_matrix,
    get_accessible_amplitudes,
    select_accessible_amplitudes,
    set_accessible_amplitudes_from_full_state,
    state_vector_get_property_value,
)

###############################################################################
# State vector creation and state set/get/zero
###############################################################################

@sv_across_distributions
@over_dtypes
def test_factory_creates_zero_state(sv_factory, dtype):
    """sv_factory returns a new state vector in the zero state."""
    sv_descriptor, sv_meta = sv_factory(dtype=dtype, num_wires=3)
    assertions.assert_basis_state(sv_descriptor, sv_meta, 0)


@sv_across_distributions
@over_dtypes
def test_set_state_and_zero(sv_factory, dtype):
    """Reset a mutated state vector back to the zero state."""
    sv_descriptor, sv_meta = sv_factory(dtype=dtype, num_wires=3)

    psi = np.ones(sv_meta.created_num_elements, dtype=dtype) / np.sqrt(sv_meta.created_num_elements)
    set_accessible_amplitudes_from_full_state(sv_descriptor, sv_meta, psi)

    cusvex.state_vector_set_zero_state(sv_descriptor)
    cusvex.state_vector_synchronize(sv_descriptor)

    assertions.assert_basis_state(sv_descriptor, sv_meta, 0)


@sv_across_distributions
@over_dtypes
def test_set_get_state_roundtrip(sv_factory, dtype):
    """Round-trip a generic normalised state."""
    sv_descriptor, sv_meta = sv_factory(dtype=dtype, num_wires=3)

    # host-resident array psi
    rng = np.random.default_rng(0) # match seed on all processes
    psi = (
        rng.standard_normal(sv_meta.created_num_elements)
        + 1j * rng.standard_normal(sv_meta.created_num_elements)
    ).astype(dtype)
    psi /= np.linalg.norm(psi)

    set_accessible_amplitudes_from_full_state(sv_descriptor, sv_meta, psi)

    # get the amplitudes accessible to local process back from state vec
    accessible_amplitudes = get_accessible_amplitudes(sv_descriptor, sv_meta)

    # filter the psi amplitudes to the ones accessible to the state vector
    select_psi_amplitudes = select_accessible_amplitudes(sv_descriptor, psi)

    assertions.assert_normalized(sv_descriptor, sv_meta)
    assertions.assert_allclose(
        accessible_amplitudes,
        select_psi_amplitudes,
        err_msg="arbitrary state did not round-trip through set/get",
    )


@sv_across_distributions
@over_dtypes
def test_set_math_mode(sv_factory, dtype):
    """Set the compute precision mode for a state vector.

    Math mode changes only the internal working precision of compute kernels,
    not the state vector's storage precision.
    """
    sv_descriptor, sv_meta = sv_factory(dtype=dtype, num_wires=3)

    cusvex.state_vector_set_math_mode(state_vector=sv_descriptor, mode=cusv.MathMode.DEFAULT)

    apply_matrix(sv_descriptor, MatrixData(GATE_X, dtype=sv_meta.dtype), WiresData([0]))
    assertions.assert_basis_state(sv_descriptor, sv_meta, 1)


@sv_across_distributions
@over_dtypes
def test_get_property_value(sv_factory, dtype):
    """Access scalar and array properties."""
    expected_num_wires = 3
    sv_descriptor, sv_meta = sv_factory(dtype=dtype, num_wires=expected_num_wires)

    # state_vector_get_property_value is a local helper; see sv_tools.py.
    num_wires = state_vector_get_property_value(
        sv_descriptor,
        cusvex.StateVectorProperty.NUM_WIRES,
    )
    assert num_wires == expected_num_wires

    wire_ordering = state_vector_get_property_value(
        sv_descriptor,
        cusvex.StateVectorProperty.WIRE_ORDERING,
    )
    np.testing.assert_array_equal(wire_ordering, np.arange(expected_num_wires, dtype=np.int32))

    distribution = state_vector_get_property_value(
        sv_descriptor,
        cusvex.StateVectorProperty.DISTRIBUTION_TYPE,
    )
    if sv_meta.distribution in (SINGLE, SINGLE_HOSTMEM):
        expected_distribution = cusvex.StateVectorDistributionType.SINGLE_DEVICE
    elif sv_meta.distribution == MULTI_DEVICE:
        expected_distribution = cusvex.StateVectorDistributionType.MULTI_DEVICE
    else:
        expected_distribution = cusvex.StateVectorDistributionType.MULTI_PROCESS
    assert distribution == expected_distribution


###############################################################################
# Synchronization
###############################################################################


@sv_across_distributions
@over_dtypes
def test_synchronize(sv_factory, dtype):
    """Wait for prior operations to complete; global synchronization."""
    sv_descriptor, sv_meta = sv_factory(dtype=dtype, num_wires=3)

    apply_matrix(sv_descriptor, MatrixData(GATE_X, dtype=dtype), WiresData([0]))
    cusvex.state_vector_synchronize(sv_descriptor)

    # The assertion synchronizes too, so it would also pass without the call above.
    assertions.assert_basis_state(sv_descriptor, sv_meta, 1)


@sv_on_multi_device
@over_dtypes
def test_synchronize_process_local(sv_factory, dtype):
    """PROCESS_LOCAL synchronizes every CUDA stream within the current process."""
    sv_descriptor, sv_meta = sv_factory(dtype=dtype, num_wires=3)
    apply_matrix(sv_descriptor, MatrixData(GATE_X, dtype=dtype), WiresData([0]))

    cusvex.state_vector_synchronize_scoped(
        state_vector=sv_descriptor,
        scope=cusvex.SynchronizationScope.PROCESS_LOCAL,
    )

    # The assertion synchronizes too, so it would also pass without the call above.
    assertions.assert_basis_state(sv_descriptor, sv_meta, 1)


@sv_on_multi_process
@over_dtypes
def test_synchronize_global(sv_factory, dtype):
    """GLOBAL synchronizes local streams; includes an inter-process barrier."""
    sv_descriptor, sv_meta = sv_factory(dtype=dtype, num_wires=3)
    apply_matrix(sv_descriptor, MatrixData(GATE_X, dtype=dtype), WiresData([0]))

    cusvex.state_vector_synchronize_scoped(
        state_vector=sv_descriptor,
        scope=cusvex.SynchronizationScope.GLOBAL,
    )

    # The assertion synchronizes too, so it would also pass without the call above.
    assertions.assert_basis_state(sv_descriptor, sv_meta, 1)


###############################################################################
# Wire layout and index permutation
###############################################################################


@sv_on_single
@over_dtypes
def test_reassign_wire_ordering(sv_factory, dtype):
    """Reassigning the wire ordering changes index bits."""
    num_wires = 3
    to_order = [1, 2, 0]
    sv_descriptor, sv_meta = sv_factory(dtype=dtype, num_wires=num_wires)

    cusvex.state_vector_reassign_wire_ordering(
        state_vector=sv_descriptor, wire_ordering=to_order, wire_ordering_len=len(to_order)
    )
    reported_order = state_vector_get_property_value(sv_descriptor, cusvex.StateVectorProperty.WIRE_ORDERING)
    np.testing.assert_array_equal(reported_order, to_order)

    apply_matrix(sv_descriptor, MatrixData(GATE_X, dtype=sv_meta.dtype), WiresData([0]))
    assertions.assert_basis_state(sv_descriptor, sv_meta, 4)


@pytest.mark.parametrize(
    "permutation_type,expected_index",
    [(cusvex.PermutationType.SCATTER, 2), (cusvex.PermutationType.GATHER, 4)],
)
@sv_on_single
@over_dtypes
def test_permute_index_bits(sv_factory, dtype, permutation_type, expected_index):
    """Permuting index bits moves amplitudes; SCATTER and GATHER apply
    permutation in opposite directions."""

    num_wires = 3
    sv_descriptor, sv_meta = sv_factory(dtype=dtype, num_wires=num_wires)

    # Array for state |001>
    psi = np.zeros(sv_meta.created_num_elements, dtype=sv_meta.dtype)
    psi[1] = 1.0

    set_accessible_amplitudes_from_full_state(sv_descriptor, sv_meta, psi)

    permutation = [1, 2, 0]
    # With Gather Wires Reorder: [0, 1, 2] -> [1, 2, 0]
    # With Scatter Wires Reorder: [0, 1, 2] -> [2, 0, 1]
    
    cusvex.state_vector_permute_index_bits(
        state_vector=sv_descriptor,
        permutation=permutation,
        permutation_len=len(permutation),
        permutation_type=permutation_type,
    )
    cusvex.state_vector_synchronize(sv_descriptor)

    assertions.assert_basis_state(sv_descriptor, sv_meta, expected_index)


###############################################################################
# Device sub-SV resources: handing the pointer and handle to plain cuStateVec
###############################################################################


@sv_on_single
@over_dtypes
def test_device_sub_sv_resources(sv_factory, dtype):
    """Query a device sub-SV's resources, then call cuStateVec on them directly.

    The returned device pointer and handle allow operations directly on the state vector's
    own memory. A host-memory state vector must call state_vector_expose_resources before
    querying; see test_host_memory_staging_workflow.
    """
    num_wires = 3
    sv_descriptor, sv_meta = sv_factory(dtype=dtype, num_wires=num_wires)
    (err,) = cudart.cudaSetDevice(sv_meta.device_ids[0])
    assert err == cudart.cudaError_t.cudaSuccess

    # Prepare (|001> + |100>)/sqrt(2) 
    psi = np.zeros(sv_meta.created_num_elements, dtype=sv_meta.dtype)
    psi[1] = 1.0 / np.sqrt(2)
    psi[4] = 1.0 / np.sqrt(2)
    set_accessible_amplitudes_from_full_state(sv_descriptor, sv_meta, psi)
    cusvex.state_vector_synchronize(sv_descriptor)

    # Query resources after the Ex API work: earlier results can be invalidated
    # by later calls, so a caller should not hold them across Ex operations.
    # A single-device state vector has one device sub-SV
    device_resources = cusvex.state_vector_get_resources_from_device_sub_sv(state_vector=sv_descriptor, sub_sv_index=0)
    device_id, d_sub_sv, _stream, handle = device_resources
    assert device_id == sv_meta.device_ids[0]
    assert d_sub_sv != 0, "device sub-SV pointer is null"
    assert handle != 0, "no cuStateVec handle for the device sub-SV"

    # The "_view" getter reports the identical resources, read-only.
    device_view_resources = cusvex.state_vector_get_resources_from_device_sub_sv_view(
        state_vector=sv_descriptor, sub_sv_index=0
    )
    assert device_view_resources == device_resources

    # Read the probabilities back with cuStateVec on the exposed pointer.
    abs2sum_buffer = np.zeros(2**num_wires, dtype=np.float64)
    cusv.abs2sum_array(
        handle=handle,
        sv=d_sub_sv,
        sv_data_type=sv_meta.sv_data_type,
        n_index_bits=num_wires,
        abs2sum=abs2sum_buffer.ctypes.data,
        bit_ordering=list(range(num_wires)),
        bit_ordering_len=num_wires,
        mask_bit_string=0,
        mask_ordering=0,
        mask_len=0,
    )
    cusvex.state_vector_synchronize(sv_descriptor)

    assertions.assert_allclose(abs2sum_buffer, np.abs(psi) ** 2)


###############################################################################
# Host-memory resource-access workflow: expose_resources, stage_sub_sv, and the
# writable/view resource getters for unstaged (host) and device sub-state vectors.
###############################################################################


@sv_on_single_hostmem
@over_dtypes
def test_host_memory_staging_workflow(sv_factory, dtype):
    """Read a host-resident sub-SV's resources, then stage it onto device.

    A host-memory state vector keeps its migration-bit sub-SV unstaged in host
    memory until an Ex API call needs it there. ``expose_resources`` makes
    resources queryable; ``stage_sub_sv`` then migrates one onto device.
    """
    sv_descriptor, sv_meta = sv_factory(dtype=dtype, num_wires=3)
    (err,) = cudart.cudaSetDevice(sv_meta.device_ids[0])
    assert err == cudart.cudaError_t.cudaSuccess

    rng = np.random.default_rng(0)
    psi = (
        rng.standard_normal(sv_meta.created_num_elements) + 1j * rng.standard_normal(sv_meta.created_num_elements)
    ).astype(sv_meta.dtype)
    psi /= np.linalg.norm(psi)

    set_accessible_amplitudes_from_full_state(sv_descriptor, sv_meta, psi)

    # Resources must be exposed before every query: staging can invalidate.
    cusvex.state_vector_expose_resources(state_vector=sv_descriptor, expose_resources=cusvex.ExposeResources.ACCESSIBLE)

    unstaged_indices = state_vector_get_property_value(
        sv_descriptor, cusvex.StateVectorProperty.UNSTAGED_SUBSV_INDICES
    )
    sub_sv_index = int(unstaged_indices[0])
    staged_indices = state_vector_get_property_value(sv_descriptor, cusvex.StateVectorProperty.DEVICE_SUBSV_INDICES)
    assert sub_sv_index not in staged_indices, "sub-SV is already staged on device"

    # The unstaged getters return (sub_sv_slice, placement, device_id, stream).
    host_resources = cusvex.state_vector_get_resources_from_unstaged_sub_sv_slice(
        state_vector=sv_descriptor, sub_sv_index=sub_sv_index, slice_index=0
    )
    host_slice, placement, device_id, _stream = host_resources
    assert host_slice != 0, "unstaged sub-SV slice pointer is null"
    assert placement == cusvex.MemoryPlacement.ON_HOST
    assert device_id == sv_meta.device_ids[0]

    # The "_view" getter reports the identical resources, read-only.
    host_view_resources = cusvex.state_vector_get_resources_from_unstaged_sub_sv_slice_view(
        state_vector=sv_descriptor, sub_sv_index=sub_sv_index, slice_index=0
    )
    assert host_view_resources == host_resources

    cusvex.state_vector_stage_sub_sv(state_vector=sv_descriptor, sub_sv_index=sub_sv_index)
    cusvex.state_vector_synchronize(sv_descriptor)
    cusvex.state_vector_expose_resources(state_vector=sv_descriptor, expose_resources=cusvex.ExposeResources.ACCESSIBLE)

    # This configuration holds one sub-SV on device, so staging swaps: the target
    # moves on and the previously staged sub-SV is evicted to host memory.
    staged_after = state_vector_get_property_value(sv_descriptor, cusvex.StateVectorProperty.DEVICE_SUBSV_INDICES)
    unstaged_after = state_vector_get_property_value(sv_descriptor, cusvex.StateVectorProperty.UNSTAGED_SUBSV_INDICES)
    assert staged_after.tolist() == [sub_sv_index]
    assert unstaged_after.tolist() == staged_indices.tolist()

    # The device getters use tuple order: (device_id, d_sub_sv, stream, handle).
    device_resources = cusvex.state_vector_get_resources_from_device_sub_sv(
        state_vector=sv_descriptor, sub_sv_index=sub_sv_index
    )
    _device_id, d_sub_sv, _stream, handle = device_resources
    assert d_sub_sv != 0, "staged sub-SV has no device pointer"
    assert handle != 0, "no cuStateVec handle for the staged sub-SV"

    device_view_resources = cusvex.state_vector_get_resources_from_device_sub_sv_view(
        state_vector=sv_descriptor, sub_sv_index=sub_sv_index
    )
    assert device_view_resources == device_resources

    # Migrating host-resident amplitudes onto device leaves the state unchanged.
    assertions.assert_allclose(
        get_accessible_amplitudes(sv_descriptor, sv_meta),
        select_accessible_amplitudes(sv_descriptor, psi),
        err_msg="staging did not preserve the state",
    )


###############################################################################
# Wire growth
###############################################################################

@sv_across_distributions
@over_dtypes
def test_add_wires(sv_factory, dtype):
    """Grow a resizable state vector; new wires start at zero and prior state is unaffected.

    Growing ``LOCAL`` works on every distribution; 5 wires cover the widest layout.
    """
    sv_descriptor, sv_meta = sv_factory(
        dtype=dtype,
        num_wires=5,
        capability=cusvex.StateVectorCapability.RESIZABLE,
    )
    assert state_vector_get_property_value(sv_descriptor, cusvex.StateVectorProperty.NUM_WIRES) == 0

    buffer = np.empty(2, dtype=np.int32)
    cusvex.state_vector_add_wires(
        state_vector=sv_descriptor,
        index_bit_domain=cusvex.IndexBitDomain.LOCAL,
        num_wires_to_add=2,
        wire_init_mode=cusvex.WireInitMode.ZERO,
        wires_added=buffer.ctypes.data,
    )
    cusvex.state_vector_synchronize(sv_descriptor)
    # Growth is collective, so the wire IDs it hands back must agree everywhere. Checking
    # that first makes the rank-local assertions below deterministic across ranks.
    assertions.assert_collective_result_consistent(
        sv_meta.mpi_context, buffer, err_msg="add_wires returned different wire IDs across ranks"
    )
    assert buffer.tolist() == [0, 1]
    assert state_vector_get_property_value(sv_descriptor, cusvex.StateVectorProperty.NUM_WIRES) == 2

    # Make the state non-trivial
    apply_matrix(sv_descriptor, MatrixData(GATE_X, dtype=dtype), WiresData([0]))
    assertions.assert_normalized(sv_descriptor, sv_meta)
    assertions.assert_basis_state(sv_descriptor, sv_meta, 1)

    buffer = np.empty(1, dtype=np.int32)
    cusvex.state_vector_add_wires(
        state_vector=sv_descriptor,
        index_bit_domain=cusvex.IndexBitDomain.LOCAL,
        num_wires_to_add=1,
        wire_init_mode=cusvex.WireInitMode.ZERO,
        wires_added=buffer.ctypes.data,
    )
    cusvex.state_vector_synchronize(sv_descriptor)
    assertions.assert_collective_result_consistent(
        sv_meta.mpi_context, buffer, err_msg="add_wires returned different wire IDs across ranks"
    )
    assert buffer.tolist() == [2]
    assert state_vector_get_property_value(sv_descriptor, cusvex.StateVectorProperty.NUM_WIRES) == 3

    # Amplitude at index 1 is unchanged and new wire at bit 2 is zero.
    assertions.assert_normalized(sv_descriptor, sv_meta)
    assertions.assert_basis_state(sv_descriptor, sv_meta, 1)


@sv_on_single_hostmem
@over_dtypes
def test_add_wires_migration(sv_factory, dtype):
    """Grow a host-memory state vector through the ``MIGRATION`` domain."""
    sv_descriptor, sv_meta = sv_factory(
        dtype=dtype,
        num_wires=3,
        capability=cusvex.StateVectorCapability.RESIZABLE,
    )

    buffer = np.empty(2, dtype=np.int32)
    cusvex.state_vector_add_wires(
        state_vector=sv_descriptor,
        index_bit_domain=cusvex.IndexBitDomain.LOCAL,
        num_wires_to_add=2,
        wire_init_mode=cusvex.WireInitMode.ZERO,
        wires_added=buffer.ctypes.data,
    )
    cusvex.state_vector_synchronize(sv_descriptor)
    assert buffer.tolist() == [0, 1]
    assert state_vector_get_property_value(sv_descriptor, cusvex.StateVectorProperty.NUM_MIGRATION_WIRES) == 0

    apply_matrix(sv_descriptor, MatrixData(GATE_X, dtype=dtype), WiresData([0]))
    assertions.assert_basis_state(sv_descriptor, sv_meta, 1)

    buffer = np.empty(1, dtype=np.int32)
    cusvex.state_vector_add_wires(
        state_vector=sv_descriptor,
        index_bit_domain=cusvex.IndexBitDomain.MIGRATION,
        num_wires_to_add=1,
        wire_init_mode=cusvex.WireInitMode.ZERO,
        wires_added=buffer.ctypes.data,
    )
    cusvex.state_vector_synchronize(sv_descriptor)
    assert buffer.tolist() == [2]
    assert state_vector_get_property_value(sv_descriptor, cusvex.StateVectorProperty.NUM_WIRES) == 3
    assert state_vector_get_property_value(sv_descriptor, cusvex.StateVectorProperty.NUM_MIGRATION_WIRES) == 1

    # A migration wire sits above the local index bits, so the populated index is unchanged.
    assertions.assert_basis_state(sv_descriptor, sv_meta, 1)
    assertions.assert_normalized(sv_descriptor, sv_meta)


@sv_on_multi_device
@over_dtypes
def test_add_wires_global_device(sv_factory, dtype):
    """Grow a multi-device state vector through the ``GLOBAL_DEVICE`` domain."""
    sv_descriptor, sv_meta = sv_factory(
        dtype=dtype,
        num_wires=3,
        capability=cusvex.StateVectorCapability.RESIZABLE,
    )

    buffer = np.empty(2, dtype=np.int32)
    cusvex.state_vector_add_wires(
        state_vector=sv_descriptor,
        index_bit_domain=cusvex.IndexBitDomain.LOCAL,
        num_wires_to_add=2,
        wire_init_mode=cusvex.WireInitMode.ZERO,
        wires_added=buffer.ctypes.data,
    )
    cusvex.state_vector_synchronize(sv_descriptor)
    assert buffer.tolist() == [0, 1]
    assert state_vector_get_property_value(sv_descriptor, cusvex.StateVectorProperty.NUM_DEVICE_SUBSVS) == 1

    apply_matrix(sv_descriptor, MatrixData(GATE_X, dtype=dtype), WiresData([0]))
    assertions.assert_basis_state(sv_descriptor, sv_meta, 1)

    buffer = np.empty(1, dtype=np.int32)
    cusvex.state_vector_add_wires(
        state_vector=sv_descriptor,
        index_bit_domain=cusvex.IndexBitDomain.GLOBAL_DEVICE,
        num_wires_to_add=1,
        wire_init_mode=cusvex.WireInitMode.ZERO,
        wires_added=buffer.ctypes.data,
    )
    cusvex.state_vector_synchronize(sv_descriptor)
    assert buffer.tolist() == [2]
    assert state_vector_get_property_value(sv_descriptor, cusvex.StateVectorProperty.NUM_WIRES) == 3
    assert state_vector_get_property_value(sv_descriptor, cusvex.StateVectorProperty.NUM_DEVICE_SUBSVS) == 2

    # A global-device wire sits above the local index bits, so the populated index is unchanged.
    assertions.assert_basis_state(sv_descriptor, sv_meta, 1)
    assertions.assert_normalized(sv_descriptor, sv_meta)
