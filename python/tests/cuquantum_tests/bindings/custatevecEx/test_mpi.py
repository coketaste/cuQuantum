# Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES
#
# SPDX-License-Identifier: BSD-3-Clause

"""Exercise the cuStateVec Ex communicator query."""

import pytest

from cuquantum.bindings import custatevecEx as cusvex


@pytest.mark.mpi
def test_communicator_size_and_rank(require_mpi) -> None:
    """cuStateVec Ex communicator size and rank match mpi4py."""
    size, rank, status = cusvex.communicator_get_size_and_rank()
    assert status == cusvex.CommunicatorStatus.SUCCESS
    assert size == require_mpi.comm.Get_size(), "communicator size != mpi4py size"
    assert rank == require_mpi.comm.Get_rank(), "communicator rank != mpi4py rank"
