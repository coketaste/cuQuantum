# Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES
#
# SPDX-License-Identifier: BSD-3-Clause

"""Pytest configuration and object factories for cuStateVec Ex examples.

The distribution parameters exercise the same examples with single-device, host-memory,
multi-device, and multi-process state vectors. Construction and shared test data live in
``sv_tools.py``; numerical assertions live in ``assertions.py``.

The multi-device examples deliberately use exactly two GPUs so their smallest two-wire state
vectors remain valid. This is not an intrinsic cuStateVec Ex device limit; see
``sv_tools.create_state_vector_multi_device`` for details.
"""

import contextlib
import os
import sys
from collections.abc import Sequence
from pathlib import Path

import pytest
from cuda.bindings import runtime as cudart
from cuquantum.bindings import custatevec as cusv
from cuquantum.bindings import custatevecEx as cusvex

from .sv_tools import (
    DTYPES,
    MULTI_DEVICE,
    MULTI_PROCESS,
    MULTI_PROCESS_HOSTMEM,
    SINGLE,
    SINGLE_HOSTMEM,
    ComplexDType,
    MPIContext,
    StateVectorDescriptorT,
    StateVectorFixtureMeta,
    SVUpdaterDescriptorT,
    create_state_vector_multi_device,
    create_state_vector_multi_process,
    create_state_vector_single_device,
    create_sv_updater,
    finalize_communicator,
    initialize_communicator,
)

###############################################################################
# Pytest markers and distribution parameters
###############################################################################


def pytest_configure(config):
    """Register markers and, under Open MPI, set CUDA support before any mpi4py import."""
    config.addinivalue_line("markers", "custatevecEx: tests for cuStateVec Ex")
    config.addinivalue_line("markers", "mpi: tests that require an MPI launcher")
    if _detect_mpi_launcher() == "openmpi":
        os.environ.setdefault("OMPI_MCA_opal_cuda_support", "true")


DISTRIBUTIONS = [
    pytest.param(SINGLE, id=SINGLE),
    pytest.param(SINGLE_HOSTMEM, id=SINGLE_HOSTMEM),
    pytest.param(
        MULTI_DEVICE,
        id=MULTI_DEVICE,
        marks=pytest.mark.skipif(cudart.cudaGetDeviceCount()[1] < 2, reason="requires >= 2 GPUs"),
    ),
    pytest.param(MULTI_PROCESS, id=MULTI_PROCESS, marks=pytest.mark.mpi),
    pytest.param(MULTI_PROCESS_HOSTMEM, id=MULTI_PROCESS_HOSTMEM, marks=pytest.mark.mpi),
]

SINGLE_ONLY = [pytest.param(SINGLE, id=SINGLE)]
SINGLE_HOSTMEM_ONLY = [pytest.param(SINGLE_HOSTMEM, id=SINGLE_HOSTMEM)]

# Decorators for indirect parametrisation pass the given distribution to ``sv_factory``.
sv_across_distributions = pytest.mark.parametrize("sv_factory", DISTRIBUTIONS, indirect=True)
sv_on_single = pytest.mark.parametrize("sv_factory", SINGLE_ONLY, indirect=True)
sv_on_single_hostmem = pytest.mark.parametrize("sv_factory", SINGLE_HOSTMEM_ONLY, indirect=True)
sv_on_multi_device = pytest.mark.parametrize(
    "sv_factory",
    [
        pytest.param(
            MULTI_DEVICE,
            id=MULTI_DEVICE,
            marks=pytest.mark.skipif(cudart.cudaGetDeviceCount()[1] < 2, reason="requires >= 2 GPUs"),
        )
    ],
    indirect=True,
)
sv_on_multi_process = pytest.mark.parametrize(
    "sv_factory",
    [pytest.param(MULTI_PROCESS, id=MULTI_PROCESS, marks=pytest.mark.mpi)],
    indirect=True,
)

over_dtypes = pytest.mark.parametrize("dtype", DTYPES)


###############################################################################
# MPI setup
###############################################################################


def mpi_is_unsupported(mpi_ctx: MPIContext) -> str | None:
    """MPI distributed examples currently require exactly two MPI ranks."""
    if mpi_ctx.comm.Get_size() != 2:
        return f"exactly 2 ranks (got {mpi_ctx.comm.Get_size()})"
    return None


def _detect_mpi_launcher() -> str | None:
    """Return ``openmpi``, ``mpich``, or ``None`` from launcher variables.

    Launcher variables provide a proxy without importing ``mpi4py.MPI`` and initialising MPI
    while pytest collects the tests. pytest_configure uses this proxy to set required
    environment variables before any import mpi4py can trigger MPI_Init.
    """
    if "OMPI_COMM_WORLD_SIZE" in os.environ:
        return "openmpi"
    if any(name in os.environ for name in ("PMI_SIZE", "PMI_RANK")):
        return "mpich"
    return None


def pytest_collection_modifyitems(items):
    """Modify marks to skip MPI tests in this directory unless pytest is actually running under an MPI launcher.

    Pytest passes every collected item to this hook even though it is defined in
    a nested conftest, so only modify tests that belong to this directory.
    """
    if _detect_mpi_launcher() is not None:
        return

    current_test_dir = Path(__file__).parent
    skip_without_mpi = pytest.mark.skip(reason="requires an MPI launcher (mpirun -n 2)")
    for item in items:
        if Path(item.path).is_relative_to(current_test_dir) and item.get_closest_marker("mpi"):
            item.add_marker(skip_without_mpi)


@pytest.hookimpl(hookwrapper=True)
def pytest_runtest_makereport(item, call):
    """Abort all ranks when one fails instead of leaving peers in collectives."""
    outcome = yield
    report = outcome.get_result()
    if report.failed and _detect_mpi_launcher() is not None:
        from mpi4py import MPI

        # MPI_Abort pre-empts pytest's normal report, so preserve the traceback.
        sys.stderr.write(
            f"\n[custatevecEx MPI] rank {MPI.COMM_WORLD.Get_rank()}: "
            f"{item.nodeid} failed during {report.when}:\n{report.longreprtext}\n"
            f"[custatevecEx MPI] calling MPI_Abort to avoid a cross-rank deadlock.\n"
        )
        sys.stderr.flush()
        MPI.COMM_WORLD.Abort(1)


@pytest.fixture(scope="session")
def mpi_context():
    """Create one cuStateVec Ex communicator and select a GPU for each rank.

    Outside an MPI launcher, this fixture yields ``None`` without importing ``mpi4py``."""
    if _detect_mpi_launcher() is None:
        yield None
        return

    # pytest_configure already set OMPI_MCA_opal_cuda_support, if needed.
    from mpi4py import MPI

    ctx = MPIContext(comm=MPI.COMM_WORLD, ex_communicator=initialize_communicator())
    try:
        yield ctx
    finally:
        # No rank may destroy the communicator while another still uses it.
        with contextlib.suppress(Exception):
            ctx.comm.Barrier()
        finalize_communicator(ctx.ex_communicator)


@pytest.fixture()
def require_mpi(mpi_context):
    """Return a supported MPI context or skip on every rank."""
    if reason := mpi_is_unsupported(mpi_context):
        pytest.skip(f"multi_process tests require {reason}")
    return mpi_context


###############################################################################
# State-vector factory fixture
###############################################################################


@pytest.fixture()
def sv_factory(request, mpi_context):
    """Provide the complete creation and destruction lifecycle for state vectors.

    Indirect parametrisation selects the distribution and defaults to
    ``single``. Each call configures that distribution, creates a state vector,
    and returns it in the zero state::

        sv_descriptor, sv_meta = sv_factory(dtype=..., num_wires=...)

    With ``RESIZABLE`` capability, ``num_wires`` is the maximum wire count and
    the state vector initially has zero wires. The fixture owns every returned
    state vector and destroys them after the test.
    """
    distribution = getattr(request, "param", SINGLE)
    active_mpi_context = mpi_context if distribution in (MULTI_PROCESS, MULTI_PROCESS_HOSTMEM) else None

    # Distributed state vectors require the supported two-rank MPI context.
    if distribution in (MULTI_PROCESS, MULTI_PROCESS_HOSTMEM) and (reason := mpi_is_unsupported(active_mpi_context)):
        pytest.skip(f"{distribution} requires {reason}")

    owned_state_vectors: list[StateVectorDescriptorT] = []

    def create(
        *,
        dtype: ComplexDType,
        num_wires: int,
        capability: cusvex.StateVectorCapability = cusvex.StateVectorCapability.NONE,
    ) -> tuple[StateVectorDescriptorT, StateVectorFixtureMeta]:
        if distribution in (SINGLE, SINGLE_HOSTMEM):
            sv_descriptor, sv_meta = create_state_vector_single_device(
                dtype,
                num_wires,
                device_id=0,
                num_migration_bits=1 if distribution == SINGLE_HOSTMEM else 0,
                capability=capability,
            )
        elif distribution == MULTI_DEVICE:
            sv_descriptor, sv_meta = create_state_vector_multi_device(
                dtype,
                num_wires,
                device_ids=(0, 1),
                capability=capability,
            )
        elif active_mpi_context is not None:
            sv_descriptor, sv_meta = create_state_vector_multi_process(
                dtype,
                num_wires,
                mpi_context=active_mpi_context,
                num_migration_bits=1 if distribution == MULTI_PROCESS_HOSTMEM else 0,
                capability=capability,
            )
        else:
            raise ValueError(f"unknown distribution {distribution!r}")

        owned_state_vectors.append(sv_descriptor)

        # Return a ready-to-use descriptor with deterministic initial state.
        if sv_meta.created_num_wires > 0:
            cusvex.state_vector_set_zero_state(sv_descriptor)
            cusvex.state_vector_synchronize(sv_descriptor)
        return sv_descriptor, sv_meta

    yield create

    # Destroy in reverse creation order and attempt every cleanup.
    teardown_errors = []
    for sv_descriptor in reversed(owned_state_vectors):
        try:
            cusvex.state_vector_destroy(sv_descriptor)
        except cusv.cuStateVecError as exc:
            teardown_errors.append(exc)
    if teardown_errors:
        raise teardown_errors[0]


###############################################################################
# SVUpdater factory fixture
###############################################################################


@pytest.fixture()
def sv_updater_factory():
    """Return a factory that creates and owns SVUpdaters.

    Each call creates an updater from a temporary configuration dictionary.
    Pass a sequence of ``SVUpdaterConfigItem`` values, or omit it to select the
    defaults. The fixture destroys every returned updater after the test.
    """
    owned_updaters: list[SVUpdaterDescriptorT] = []

    def create(
        *,
        dtype: ComplexDType,
        config_items: Sequence[cusvex.SVUpdaterConfigItem] = (),
    ) -> SVUpdaterDescriptorT:
        updater = create_sv_updater(dtype, config_items=config_items)
        owned_updaters.append(updater)
        return updater

    yield create

    teardown_errors = []
    for updater in reversed(owned_updaters):
        try:
            cusvex.sv_updater_destroy(updater)
        except cusv.cuStateVecError as exc:
            teardown_errors.append(exc)
    if teardown_errors:
        raise teardown_errors[0]
