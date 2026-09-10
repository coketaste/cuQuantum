# Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES
#
# SPDX-License-Identifier: BSD-3-Clause

"""Reusable cuStateVec Ex state-vector workflows.

Pytest fixtures and assertions live in ``conftest.py`` and ``assertions.py``. This
module keeps reusable construction, state access, property decoding, and
argument preparation separate from test verification.
"""

from dataclasses import dataclass
from typing import TYPE_CHECKING, Sequence, TypeAlias

import numpy as np
from cuquantum.bindings import custatevec as cusv
from cuquantum.bindings import custatevecEx as cusvex
from numpy.typing import ArrayLike

from . import cudaDataType, dtype_to_data_type

if TYPE_CHECKING:
    from mpi4py import MPI


###############################################################################
# Data types and distributions
###############################################################################

ComplexDType = type[np.complex64] | type[np.complex128]
DTYPES: tuple[ComplexDType, ...] = (np.complex64, np.complex128)
StateVectorDescriptorT: TypeAlias = int  # ``custatevecExStateVectorDescriptor_t``
SVUpdaterDescriptorT: TypeAlias = int  # ``custatevecExSVUpdaterDescriptor_t``
CommunicatorDescriptorT: TypeAlias = int  # ``custatevecExCommunicatorDescriptor_t``

SINGLE = "single"
SINGLE_HOSTMEM = "single_hostmem"
MULTI_DEVICE = "multi_device"
MULTI_PROCESS = "multi_process"
MULTI_PROCESS_HOSTMEM = "multi_process_hostmem"


# Row-major single-qubit matrices.
GATE_X = np.array([[0, 1], [1, 0]])
GATE_H = np.array([[1, 1], [1, -1]]) / np.sqrt(2)
GATE_I = np.array([[1, 0], [0, 1]])
GATE_Z = np.array([[1, 0], [0, -1]])


###############################################################################
# MPI helpers
###############################################################################


@dataclass(frozen=True, slots=True)
class MPIContext:
    """Pair of mpi4py communicator and its corresponding cuStateVec Ex descriptor."""

    comm: "MPI.Comm"
    ex_communicator: CommunicatorDescriptorT


def initialize_communicator() -> CommunicatorDescriptorT:
    """Initialize MPI for cuStateVec Ex and create its communicator.

    MPI itself must be first initialized externally by mpi4py. For these tests,
    we only initialize the communicator once per session (see conftest.py). Calling
    it again returns ALREADY_INITIALIZED, which we accept.

    Raises ``cuStateVecError`` for other failures.

    .. seealso:: `custatevecExCommunicatorInitialize`
    """
    from mpi4py import MPI

    assert MPI.Is_initialized(), "mpi4py MPI initialization failed"

    vendor = MPI.get_vendor()[0]
    if vendor == "Open MPI":
        comm_type = cusv.CommunicatorType.OPENMPI
    elif vendor == "MPICH":
        comm_type = cusv.CommunicatorType.MPICH
    else:
        raise RuntimeError(f"unsupported MPI vendor {vendor!r}; expected Open MPI or MPICH")

    try:
        # Since MPI_Init is called by mpi4py, this call is trivial and ex_comm_status is always a SUCCESS.
        # mpi4py has also already loaded MPI, so the library is resolved from the symbols present in
        # this process. The optional library_path argument names a library to dlopen only when no MPI
        # library is loaded yet, and is therefore ignored here.
        ex_comm_status = cusvex.communicator_initialize(comm_type)


    except cusv.cuStateVecError as exc:
        if getattr(exc, "status", None) != cusv.Status.ALREADY_INITIALIZED:
            raise
    else:
        assert ex_comm_status == cusvex.CommunicatorStatus.SUCCESS

    return cusvex.communicator_create()


def finalize_communicator(ex_communicator: CommunicatorDescriptorT) -> None:
    """Destroy the Ex communicator and finalize.

    The caller must ensure every rank is done using ex_communicator before calling.

    If MPI_Init was not called by cuStateVec Ex, finalization does not call ``MPI_Finalize``.
    This is the case here as mpi4py initialized MPI, so cusvex.communicator_finalize() is trivial.

    .. seealso:: `custatevecExCommunicatorDestroy`, `custatevecExCommunicatorFinalize`
    """
    try:
        cusvex.communicator_destroy(ex_communicator)
    finally:
        ex_comm_status = cusvex.communicator_finalize()  # No MPI_Finalize with mpi4py
        assert ex_comm_status == cusvex.CommunicatorStatus.SUCCESS


###############################################################################
# State-vector distribution builders
###############################################################################


@dataclass(frozen=True, slots=True)
class StateVectorFixtureMeta:
    """Python metadata retained alongside a state-vector descriptor in tests."""

    dtype: ComplexDType
    """NumPy dtype of elements."""

    sv_data_type: cudaDataType
    """CUDA data type corresponding to NumPy dtype."""

    created_num_wires: int
    """Logical wire count immediately after creation; 0 if created with ``RESIZABLE`` capability."""

    distribution: str
    """Distribution identifier."""

    device_ids: tuple[int, ...] | None
    """Configured device IDs, or ``None`` when assignment is dynamic."""

    mpi_context: MPIContext | None = None
    """MPI resources for a multi-process state vector."""

    @property
    def created_num_elements(self) -> int:
        """Number of state-vector elements immediately after creation."""
        return 1 << self.created_num_wires


def _num_global_index_bits(num_sub_svs: int) -> int:
    """Global index bits needed to address ``num_sub_svs`` sub-state vectors.

    Each global index bit doubles the number of addressable sub-state vectors, so
    ``num_sub_svs = 2**num_global_index_bits``.
    """
    if num_sub_svs < 1 or num_sub_svs & (num_sub_svs - 1):
        raise ValueError("num_sub_svs must be a positive power of two")
    return num_sub_svs.bit_length() - 1


def create_state_vector_single_device(
    dtype: ComplexDType,
    num_wires: int,
    *,
    device_id: int,
    num_migration_bits: int = 0,
    capability: cusvex.StateVectorCapability = cusvex.StateVectorCapability.NONE,
) -> tuple[StateVectorDescriptorT, StateVectorFixtureMeta]:
    """Create a state vector on one device with optional host-memory migration.

    Without migration bits (``SINGLE``), every index bit is local to
    ``device_id``. With migration bits (``SINGLE_HOSTMEM``), we can select
    sub-state vectors to stage from host memory; ``num_wires - num_migration_bits``
    wires remain device-local. The API permits at most three migration bits per state vector.

    Successful creation initializes the zero state and does not synchronize.

    .. seealso:: `custatevecExConfigureStateVectorSingleDevice`,
        `custatevecExStateVectorCreateSingleProcess`, `custatevecExDictionaryDestroy`
    """
    # At least one wire must stay device-local, and the API permits at most three
    # migration wires.
    if not 0 <= num_migration_bits <= min(3, num_wires - 1):
        raise ValueError("num_migration_bits must be between zero and min(3, num_wires - 1)")

    distribution = SINGLE_HOSTMEM if num_migration_bits else SINGLE
    sv_data_type = dtype_to_data_type[dtype]
    resizable = (capability & cusvex.StateVectorCapability.RESIZABLE) != 0
    created_num_wires = 0 if resizable else num_wires

    sv_config = cusvex.configure_state_vector_single_device(
        sv_data_type=sv_data_type,
        num_wires=num_wires,
        num_device_wires=num_wires - num_migration_bits,
        device_id=device_id,
        capability=capability,
    )
    try:
        sv_descriptor = cusvex.state_vector_create_single_process(
            sv_config=sv_config,
            streams=0,  # library-managed streams
            num_streams=0,
            resource_manager=0,  # default resource manager
        )
    finally:
        cusvex.dictionary_destroy(sv_config)  # SV create copies config, now destroy

    meta = StateVectorFixtureMeta(
        dtype=dtype,
        sv_data_type=sv_data_type,
        created_num_wires=created_num_wires,
        distribution=distribution,
        device_ids=(device_id,),
    )
    return sv_descriptor, meta


def create_state_vector_multi_device(
    dtype: ComplexDType,
    num_wires: int,
    *,
    device_ids: tuple[int, ...],
    network_type: cusv.DeviceNetworkType = cusv.DeviceNetworkType.SWITCH,
    capability: cusvex.StateVectorCapability = cusvex.StateVectorCapability.NONE,
) -> tuple[StateVectorDescriptorT, StateVectorFixtureMeta]:
    """Create one state vector distributed across devices in one process.

    The device count must be a power of two. Its ``log2(num_devices)`` global index bits
    select a device sub-state vector; the remaining index bits are local within it.

    Successful creation initializes the zero state and does not synchronize.

    .. seealso:: `custatevecExConfigureStateVectorMultiDevice`,
        `custatevecExStateVectorCreateSingleProcess`, `custatevecExDictionaryDestroy`
    """
    num_devices = len(device_ids)
    if num_devices < 2 or num_devices & (num_devices - 1):
        raise ValueError("device_ids must contain a power-of-two number of at least two devices")
    if len(set(device_ids)) != num_devices:
        raise ValueError("device_ids must be unique")

    num_global_device_bits = _num_global_index_bits(num_devices)
    if num_global_device_bits > num_wires:
        raise ValueError("num_wires is too small for the requested devices")

    sv_data_type = dtype_to_data_type[dtype]
    resizable = (capability & cusvex.StateVectorCapability.RESIZABLE) != 0
    created_num_wires = 0 if resizable else num_wires

    sv_config = cusvex.configure_state_vector_multi_device(
        sv_data_type=sv_data_type,
        num_wires=num_wires,
        num_device_wires=num_wires - num_global_device_bits,
        device_ids=device_ids,
        num_devices=num_devices,
        network_type=network_type,
        capability=capability,
    )
    try:
        # Multiple devices still belong to one process.
        sv_descriptor = cusvex.state_vector_create_single_process(
            sv_config=sv_config,
            streams=0,  # library-managed streams
            num_streams=0,
            resource_manager=0,  # default resource manager
        )
    finally:
        cusvex.dictionary_destroy(sv_config)  # SV create copies config, now destroy

    meta = StateVectorFixtureMeta(
        dtype=dtype,
        sv_data_type=sv_data_type,
        created_num_wires=created_num_wires,
        distribution=MULTI_DEVICE,
        device_ids=device_ids,
    )
    return sv_descriptor, meta


def create_state_vector_multi_process(
    dtype: ComplexDType,
    num_wires: int,
    *,
    mpi_context: MPIContext,
    num_migration_bits: int = 0,
    transfer_workspace_size_in_bytes: int = 1 << 24,
    capability: cusvex.StateVectorCapability = cusvex.StateVectorCapability.NONE,
) -> tuple[StateVectorDescriptorT, StateVectorFixtureMeta]:
    """Create a distributed state vector with one process per device.

    ``log2(num_processes)`` communicator bits select the process sub-state vector.
    Optional migration bits select sub-state vectors that can be staged from host
    memory; all remaining index bits are device-local. ``mpi_context`` must
    contain an initialized Ex communicator whose lifetime is managed by the caller.

    Successful creation initializes the zero state and does not synchronize.

    .. seealso:: `custatevecExConfigureStateVectorMultiProcess`,
        `custatevecExStateVectorCreateMultiProcess`, `custatevecExDictionaryDestroy`
    """
    if num_migration_bits < 0:
        raise ValueError("num_migration_bits must be non-negative")

    distribution = MULTI_PROCESS_HOSTMEM if num_migration_bits else MULTI_PROCESS
    num_processes, _rank, _status = cusvex.communicator_get_size_and_rank()
    num_interprocess_global_index_bits = _num_global_index_bits(num_processes)

    if distribution == MULTI_PROCESS:
        global_index_bit_classes = [cusvex.GlobalIndexBitClass.COMMUNICATOR]
        num_global_index_bits_per_layer = [num_interprocess_global_index_bits]
    else:  # MULTI_PROCESS_HOSTMEM
        global_index_bit_classes = [cusvex.GlobalIndexBitClass.MIGRATION, cusvex.GlobalIndexBitClass.COMMUNICATOR]
        num_global_index_bits_per_layer = [num_migration_bits, num_interprocess_global_index_bits]

    num_global_index_bits = sum(num_global_index_bits_per_layer)
    if num_global_index_bits > num_wires:
        raise ValueError("num_wires is too small for the requested distribution")

    sv_data_type = dtype_to_data_type[dtype]
    resizable = (capability & cusvex.StateVectorCapability.RESIZABLE) != 0
    created_num_wires = 0 if resizable else num_wires

    sv_config = cusvex.configure_state_vector_multi_process(
        sv_data_type=sv_data_type,
        num_wires=num_wires,
        num_device_wires=num_wires - num_global_index_bits,
        device_id=-1,  # dynamic rank-to-device assignment
        memory_sharing_method=cusvex.MemorySharingMethod.NONE,
        global_index_bit_classes=global_index_bit_classes,
        num_global_index_bits_per_layer=num_global_index_bits_per_layer,
        num_global_index_bit_layers=len(global_index_bit_classes),
        transfer_workspace_size_in_bytes=transfer_workspace_size_in_bytes,
        aux_config=0,
        capability=capability,
    )
    try:
        sv_descriptor = cusvex.state_vector_create_multi_process(
            sv_config=sv_config,
            stream=0,  # library-managed stream
            ex_communicator=mpi_context.ex_communicator,
            resource_manager=0,  # default resource manager
        )
    finally:
        cusvex.dictionary_destroy(sv_config)  # SV create copies config, now destroy

    meta = StateVectorFixtureMeta(
        dtype=dtype,
        sv_data_type=sv_data_type,
        created_num_wires=created_num_wires,
        distribution=distribution,
        device_ids=None,
        mpi_context=mpi_context,
    )
    return sv_descriptor, meta


###############################################################################
# SVUpdater builder
###############################################################################


def create_sv_updater(
    dtype: ComplexDType,
    *,
    config_items: Sequence[cusvex.SVUpdaterConfigItem] = (),
) -> SVUpdaterDescriptorT:
    """Create an SVUpdater for state vectors of the given dtype.

    The updater's dtype is fixed: it applies only to state vectors of the
    same dtype, and accepts enqueued matrices of the same or higher precision.

    .. seealso:: `custatevecExConfigureSVUpdater`, `custatevecExSVUpdaterCreate`,
        `custatevecExDictionaryDestroy`, `custatevecExSVUpdaterDestroy`
    """
    sv_updater_config = cusvex.configure_sv_updater(
        data_type=dtype_to_data_type[dtype],
        config_items=config_items,
    )
    try:
        return cusvex.sv_updater_create(
            sv_updater_config=sv_updater_config,
            resource_manager=0,  # default resource manager
        )
    finally:
        cusvex.dictionary_destroy(sv_updater_config)  # updater create copies config, now destroy


###############################################################################
# State property query helper
###############################################################################

_ARRAY_PROPERTY_COUNTS = {
    cusvex.StateVectorProperty.WIRE_ORDERING: cusvex.StateVectorProperty.NUM_WIRES,
    cusvex.StateVectorProperty.DEVICE_SUBSV_INDICES: cusvex.StateVectorProperty.NUM_DEVICE_SUBSVS,
    cusvex.StateVectorProperty.SUBSV_INDICES: cusvex.StateVectorProperty.NUM_SUBSVS,
    cusvex.StateVectorProperty.UNSTAGED_SUBSV_INDICES: cusvex.StateVectorProperty.NUM_UNSTAGED_SUBSVS,
}
_SCALAR_PROPERTIES = {
    # DISTRIBUTION_TYPE and DATA_TYPE are not int32_t but correspond to the C enums
    # custatevecExStateVectorDistributionType_t, cudaDataType_t. We can decode them
    # directly as plain int32 on LP64-LE.
    cusvex.StateVectorProperty.DISTRIBUTION_TYPE,
    cusvex.StateVectorProperty.DATA_TYPE,
    cusvex.StateVectorProperty.NUM_WIRES,
    cusvex.StateVectorProperty.NUM_LOCAL_WIRES,
    cusvex.StateVectorProperty.NUM_DEVICE_SUBSVS,
    cusvex.StateVectorProperty.NUM_INTERPROC_DEVICE_WIRES,
    cusvex.StateVectorProperty.NUM_INPROC_DEVICE_WIRES,
    cusvex.StateVectorProperty.NUM_MIGRATION_WIRES,
    cusvex.StateVectorProperty.NUM_SUBSV_SLICES,
    cusvex.StateVectorProperty.NUM_SUBSVS,
    cusvex.StateVectorProperty.NUM_UNSTAGED_SUBSVS,
    cusvex.StateVectorProperty.MAX_NUM_WIRES,
    cusvex.StateVectorProperty.MAX_NUM_LOCAL_WIRES,
    cusvex.StateVectorProperty.MAX_NUM_INTERPROC_DEVICE_WIRES,
    cusvex.StateVectorProperty.MAX_NUM_INPROC_DEVICE_WIRES,
    cusvex.StateVectorProperty.MAX_NUM_MIGRATION_WIRES,
}


def state_vector_get_property_value(
    sv_descriptor: StateVectorDescriptorT,
    prop: cusvex.StateVectorProperty,
) -> int | np.ndarray:
    """Return a decoded state-vector property.

    This helper provides a higher-level interface over
    ``custatevecEx.state_vector_get_property``, which requires callers to
    allocate a correctly sized buffer and pass its address. Scalar and enum
    properties return a Python ``int``. Array properties return a
    one-dimensional NumPy ``int32`` array whose length is inferred from the
    corresponding count property.

    Errors for resource-sensitive properties propagate to the caller; this
    helper does not expose state-vector resources automatically.
    """
    if prop in _SCALAR_PROPERTIES:
        value = np.empty(1, dtype=np.int32)
        cusvex.state_vector_get_property(
            sv_descriptor,
            prop,
            value.ctypes.data,
            value.nbytes,
        )
        return int(value[0])

    if prop in _ARRAY_PROPERTY_COUNTS:
        count_property = _ARRAY_PROPERTY_COUNTS[prop]
        count = state_vector_get_property_value(sv_descriptor, count_property)
        value = np.empty(count, dtype=np.int32)
        if count:
            cusvex.state_vector_get_property(
                sv_descriptor,
                prop,
                value.ctypes.data,
                value.nbytes,
            )
        return value

    raise ValueError(f"unknown state vector property: {prop}")


###############################################################################
# State readback and assignment
###############################################################################

# These helpers are for the test suite only. They map wire assignments to state
# indices. Then they read or write state vector data with those indices. The
# mapping is not stable: operations on global wires can permute the wire ordering,
# so indices refer to different states. `state_vector_permute_index_bits` restores
# the original ordering if needed.


def accessible_state_index_ranges(
    sv_descriptor: StateVectorDescriptorT,
) -> list[tuple[int, int, int]]:
    """Return accessible state-index ranges sorted by sub-state-vector index.

    This function defines the canonical order for process-accessible state data. Helpers that
    produce aligned indices or amplitudes preserve this order rather than sorting
    independently. Each result is ``(sub_sv_index, begin, end)``; the queried
    ``NUM_LOCAL_WIRES`` and ``SUBSV_INDICES`` properties reflect the current layout,
    including after wires are added.
    """
    num_local_wires = state_vector_get_property_value(
        sv_descriptor,
        cusvex.StateVectorProperty.NUM_LOCAL_WIRES,
    )
    sub_sv_size = 1 << num_local_wires
    indices = state_vector_get_property_value(
        sv_descriptor,
        cusvex.StateVectorProperty.SUBSV_INDICES,
    )
    return [
        (sub_sv_index, sub_sv_index * sub_sv_size, (sub_sv_index + 1) * sub_sv_size)
        for sub_sv_index in sorted(int(index) for index in indices)
    ]


def select_accessible_amplitudes(
    sv_descriptor: StateVectorDescriptorT,
    full_state: ArrayLike,
) -> np.ndarray:
    """Copy this process's accessible amplitudes from a complete logical state vector.

    The result follows the same range order as :func:`get_accessible_amplitudes`.
    """
    src = np.asarray(full_state)
    num_wires = state_vector_get_property_value(
        sv_descriptor,
        cusvex.StateVectorProperty.NUM_WIRES,
    )
    expected_num_elements = 1 << num_wires
    if src.ndim != 1 or src.size != expected_num_elements:
        raise ValueError(f"full_state must be one-dimensional with {expected_num_elements} elements")

    ranges = accessible_state_index_ranges(sv_descriptor)
    if not ranges:
        return np.empty(0, dtype=src.dtype)
    return np.concatenate([src[begin:end] for _, begin, end in ranges])


def accessible_amplitude_array_index(
    sv_descriptor: StateVectorDescriptorT,
    state_index: int,
) -> int | None:
    """Map a logical state index to the accessible-amplitude array, or return ``None``."""
    num_wires = state_vector_get_property_value(
        sv_descriptor,
        cusvex.StateVectorProperty.NUM_WIRES,
    )
    num_elements = 1 << num_wires
    if not 0 <= state_index < num_elements:
        raise IndexError(f"state_index must be between zero and {num_elements - 1}")

    array_begin = 0
    for _sub_sv_index, begin, end in accessible_state_index_ranges(sv_descriptor):
        if begin <= state_index < end:
            return int(array_begin + state_index - begin)
        array_begin += end - begin
    return None


def get_accessible_amplitudes(
    sv_descriptor: StateVectorDescriptorT,
    sv_meta: StateVectorFixtureMeta,
    max_num_concurrent_copies: int = 1,
) -> np.ndarray:
    """Return this process's accessible amplitudes in ascending index order.

    Amplitudes are returned in the current wire ordering, which global wire operations
    can permute. To restore the original ordering, use `state_vector_permute_index_bits`.
    """
    sub_sv_ranges = accessible_state_index_ranges(sv_descriptor)
    total = sum(end - begin for _, begin, end in sub_sv_ranges)
    out = np.empty(total, dtype=sv_meta.dtype)
    offset = 0
    try:
        # GetState may be asynchronous, so keep the output buffer alive until synchronization.
        for _idx, begin, end in sub_sv_ranges:
            n = end - begin
            cusvex.state_vector_get_state(
                sv_descriptor,
                out[offset : (offset + n)].ctypes.data,
                sv_meta.sv_data_type,
                begin,
                end,
                max_num_concurrent_copies,
            )
            offset += n
    finally:
        cusvex.state_vector_synchronize(sv_descriptor)
    return out


def set_accessible_amplitudes_from_full_state(
    sv_descriptor: StateVectorDescriptorT,
    sv_meta: StateVectorFixtureMeta,
    full_state,
    max_num_concurrent_copies: int = 1,
) -> None:
    """Write this process's sub-state-vector ranges from a complete logical state vector.

    Each accessible range receives the matching slice of the host-resident ``full_state``.
    """
    src = np.asarray(full_state, dtype=sv_meta.dtype)
    num_wires = state_vector_get_property_value(
        sv_descriptor,
        cusvex.StateVectorProperty.NUM_WIRES,
    )
    expected_num_elements = 1 << num_wires
    if src.ndim != 1 or src.size != expected_num_elements:
        raise ValueError(f"full_state must be one-dimensional with {expected_num_elements} elements")
    src = np.ascontiguousarray(src)

    try:
        # SetState may be asynchronous, so keep the input buffer alive until synchronization.
        for _idx, begin, end in accessible_state_index_ranges(sv_descriptor):
            cusvex.state_vector_set_state(
                sv_descriptor,
                src.ctypes.data + begin * src.itemsize,
                sv_meta.sv_data_type,
                begin,
                end,
                max_num_concurrent_copies,
            )
    finally:
        cusvex.state_vector_synchronize(sv_descriptor)


def abs2sum_over_wires(
    sv_descriptor: StateVectorDescriptorT,
    output_ordering: Sequence[int] | None = None,
) -> np.ndarray:
    """Return probabilities for the requested output-wire ordering."""
    if output_ordering is None:
        num_wires = state_vector_get_property_value(
            sv_descriptor,
            cusvex.StateVectorProperty.NUM_WIRES,
        )
        output_ordering = tuple(range(num_wires))

    probabilities = np.empty(1 << len(output_ordering), dtype=np.float64)
    cusvex.abs2sum_array(
        state_vector=sv_descriptor,
        abs2sum=probabilities.ctypes.data,
        output_ordering=output_ordering,
        output_ordering_len=len(output_ordering),
        mask_bit_string=0,
        mask_wire_ordering=0,
        mask_len=0,
    )
    cusvex.state_vector_synchronize(sv_descriptor)
    return probabilities


###############################################################################
# Apply and enqueue matrix operations on state-vectors
###############################################################################


@dataclass(slots=True, init=False)
class MatrixData:
    """Contiguous host matrix and the metadata required by matrix operations.

    For ``n`` targets, dense data is a ``2**n`` square matrix; diagonal and
    anti-diagonal data contain ``2**n`` elements. Layout selects row- or
    column-major dense storage and anti-diagonal element order; it is ignored for
    diagonal data. This container does not validate those sizes. Host storage is
    required for distributed state vectors.
    """

    data: np.ndarray
    matrix_type: cusvex.MatrixType
    layout: cusv.MatrixLayout

    def __init__(
        self,
        data: ArrayLike,
        *,
        dtype: ComplexDType,
        matrix_type: cusvex.MatrixType = cusvex.MatrixType.DENSE,
        layout: cusv.MatrixLayout = cusv.MatrixLayout.ROW,
    ) -> None:
        """Convert the input to host storage with the requested dtype and layout."""
        if matrix_type == cusvex.MatrixType.DENSE:
            # DENSE: buffer is 2D (dim x dim).
            if layout == cusv.MatrixLayout.ROW:
                self.data = np.ascontiguousarray(data, dtype=dtype)
            elif layout == cusv.MatrixLayout.COL:
                self.data = np.asfortranarray(data, dtype=dtype)
            else:
                raise ValueError(f"unsupported matrix layout: {layout}")
        elif matrix_type in (
            cusvex.MatrixType.DIAGONAL,
            cusvex.MatrixType.ANTI_DIAGONAL,
        ):
            # DIAGONAL or ANTI_DIAGONAL: buffer is 1D (dim).
            self.data = np.ascontiguousarray(data, dtype=dtype)
        else:
            raise ValueError(f"unsupported matrix type: {matrix_type}")
        self.matrix_type = matrix_type
        self.layout = layout

    @property
    def data_type(self) -> cudaDataType:
        """CUDA data type of the stored matrix elements."""
        return dtype_to_data_type[self.data.dtype.type]


@dataclass(slots=True, init=False)
class WiresData:
    """Target wires and optional controls for one matrix operation.

    Each control value corresponds to the control at the same position. ``None``
    passes NULL, making every control activate on bit value 1. Wire indices must
    be in range, with no duplicates or overlap between targets and controls.
    """

    targets: Sequence[int]
    controls: Sequence[int]
    control_bit_values: Sequence[int] | None

    def __init__(
        self,
        targets: Sequence[int],
        *,
        controls: Sequence[int] = (),
        control_bit_values: Sequence[int] | None = None,
    ) -> None:
        """Retain the sequences and validate the parallel control arrays."""
        self.targets = targets
        self.controls = controls
        self.control_bit_values = control_bit_values

        if self.control_bit_values is not None and len(self.control_bit_values) != len(self.controls):
            raise ValueError("control_bit_values must contain one value per control wire")


def apply_matrix(
    sv_descriptor: StateVectorDescriptorT,
    matrix: MatrixData,
    wires: WiresData,
    *,
    adjoint: bool = False,
    wires_as_pointers: bool = False,
) -> None:
    """Apply a matrix using either supported wire-argument form.

    A complex128 state vector requires a complex128 matrix; a complex64 state
    vector accepts either matrix precision. ``adjoint=True`` applies the conjugate
    transpose. ``wires_as_pointers`` selects raw pointers to contiguous host
    ``int32`` arrays instead of Python sequences; the operation is otherwise
    identical. For distributed state vectors, dense and anti-diagonal operations
    cannot have more targets than device wires.
    """

    def _host_array_ptr(array: np.ndarray) -> int:
        """Return the array address, or NULL for an empty array."""
        return int(array.ctypes.data) if array.size else 0

    if wires_as_pointers:
        # Host arrays must remain alive until the binding call returns.
        targets_array = np.ascontiguousarray(wires.targets, dtype=np.int32)
        controls_array = np.ascontiguousarray(wires.controls, dtype=np.int32)

        if wires.control_bit_values is not None:
            control_bit_values_array = np.ascontiguousarray(wires.control_bit_values, dtype=np.int32)
        else:
            # A NULL control-bit-values pointer selects the C API default of all ones.
            control_bit_values_array = np.empty(0, dtype=np.int32)

        cusvex.apply_matrix(
            state_vector=sv_descriptor,
            matrix=_host_array_ptr(matrix.data),
            matrix_data_type=matrix.data_type,
            ex_matrix_type=matrix.matrix_type,
            layout=matrix.layout,
            adjoint=int(adjoint),
            targets=_host_array_ptr(targets_array),
            num_targets=targets_array.size,
            controls=_host_array_ptr(controls_array),
            control_bit_values=_host_array_ptr(control_bit_values_array),
            num_controls=controls_array.size,
        )
    else:
        # cybind marshals Python sequences to the C int32 arrays.
        cusvex.apply_matrix(
            state_vector=sv_descriptor,
            matrix=_host_array_ptr(matrix.data),
            matrix_data_type=matrix.data_type,
            ex_matrix_type=matrix.matrix_type,
            layout=matrix.layout,
            adjoint=int(adjoint),
            targets=wires.targets,
            num_targets=len(wires.targets),
            controls=wires.controls,
            control_bit_values=0 if wires.control_bit_values is None else wires.control_bit_values,
            num_controls=len(wires.controls),
        )


def enqueue_matrix(
    sv_updater: SVUpdaterDescriptorT,
    matrix: MatrixData,
    wires: WiresData,
    *,
    adjoint: bool = False,
    wires_as_pointers: bool = False,
) -> None:
    """Enqueue a matrix using either supported wire-argument form.

    The updater API requires host matrix data. A complex128 updater requires a
    complex128 matrix; a complex64 updater accepts either matrix precision.
    ``adjoint=True`` applies the conjugate transpose. ``wires_as_pointers`` changes
    only how the binding receives the wire arrays.

    For distributed state vectors, targets and controls together cannot exceed
    the device-wire count. This is checked by ``sv_updater_apply``.
    """

    def _host_array_ptr(array: np.ndarray) -> int:
        """Return the array address, or NULL for an empty array."""
        return int(array.ctypes.data) if array.size else 0

    if wires_as_pointers:
        # Host arrays must remain alive until the binding call returns.
        targets_array = np.ascontiguousarray(wires.targets, dtype=np.int32)
        controls_array = np.ascontiguousarray(wires.controls, dtype=np.int32)

        if wires.control_bit_values is not None:
            control_bit_values_array = np.ascontiguousarray(wires.control_bit_values, dtype=np.int32)
        else:
            # A NULL control-bit-values pointer selects the C API default of all ones.
            control_bit_values_array = np.empty(0, dtype=np.int32)

        cusvex.sv_updater_enqueue_matrix(
            sv_updater=sv_updater,
            matrix=_host_array_ptr(matrix.data),
            matrix_data_type=matrix.data_type,
            ex_matrix_type=matrix.matrix_type,
            layout=matrix.layout,
            adjoint=int(adjoint),
            targets=_host_array_ptr(targets_array),
            num_targets=targets_array.size,
            controls=_host_array_ptr(controls_array),
            control_bit_values=_host_array_ptr(control_bit_values_array),
            num_controls=controls_array.size,
        )
    else:
        # cybind marshals Python sequences to the C int32 arrays.
        cusvex.sv_updater_enqueue_matrix(
            sv_updater=sv_updater,
            matrix=_host_array_ptr(matrix.data),
            matrix_data_type=matrix.data_type,
            ex_matrix_type=matrix.matrix_type,
            layout=matrix.layout,
            adjoint=int(adjoint),
            targets=wires.targets,
            num_targets=len(wires.targets),
            controls=wires.controls,
            control_bit_values=0 if wires.control_bit_values is None else wires.control_bit_values,
            num_controls=len(wires.controls),
        )
