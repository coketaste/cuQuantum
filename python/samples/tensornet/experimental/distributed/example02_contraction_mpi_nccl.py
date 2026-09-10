# Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES
#
# SPDX-License-Identifier: BSD-3-Clause

"""
Distributed binary tensor contraction with MPI and NCCL.

This example uses a contraction from the anisotropic tensor renormalization
group (ATRG) [Adachi, Okubo, and Todo, Phys. Rev. B 102, 054432 (2020)]:

    Theta[u, v, a, x, y, b] = sum_i B[i, u, v, a] * C[i, x, y, b]

The inputs and output use independent distributions. Along each distributed
mode, an integer block size assigns consecutive blocks round-robin across
process-grid coordinates, while ``None`` selects one near-even contiguous
slab per coordinate. A process-grid extent of one leaves that mode
undistributed.

The sample also replaces the operands and executes the same plan again.

Run with::

    mpirun -n 4 python example02_contraction_mpi_nccl.py

Use one GPU per process.

Requirements:
    - mpi4py
    - cupy
    - nvmath-python with distributed support (nccl)
    - CUTENSORNET_COMM_LIB pointing at the MPI distributed interface
"""

from __future__ import annotations

import itertools

import cupy as cp
from mpi4py import MPI
import numpy as np
import nvmath.distributed

from cuquantum.bindings import cutensornet as cutn
from cuquantum.tensornet import get_mpi_comm_pointer
from cuquantum.tensornet.experimental.distributed import (
    BlockCyclic,
    DistributedBinaryContraction,
    DistributedContractionOptions,
    DistributedTensor,
    ProcessGrid,
    empty_local,
)

D = 7       # spatial degrees of freedom (u, v, x, y)
CHI = 13    # vertical bond dimension (a, b)
CHI_T = 6   # temporal bond dimension (i), contracted away
BS = 2      # block size of every block-cyclic (explicitly blocked) mode
SEED = 42


def owned_segments(layout):
    """Return each rank-owned global segment and its local offset.

    ``owned_global_segments`` lists the global ranges in local storage order,
    so accumulating their extents gives each range's local start.
    """
    per_mode = []
    for mode_segments in layout.owned_global_segments:
        triples, local_start = [], 0
        for global_start, extent in mode_segments:
            triples.append((global_start, local_start, extent))
            local_start += extent
        per_mode.append(triples)
    return per_mode


def make_distributed(
    global_host: np.ndarray,
    distribution: BlockCyclic,
) -> DistributedTensor:
    """Scatter a replicated host tensor into a compact Fortran-order shard."""
    global_shape = tuple(int(extent) for extent in global_host.shape)
    local = empty_local(
        distribution, global_shape, dtype=global_host.dtype, like=cp.empty(0)
    )
    tensor = DistributedTensor(local, distribution, global_shape)
    for combo in itertools.product(*owned_segments(tensor.layout())):
        global_slices = tuple(slice(g, g + e) for g, _, e in combo)
        local_slices = tuple(slice(l, l + e) for _, l, e in combo)
        local[local_slices] = cp.asarray(global_host[global_slices])
    return tensor


def gather(tensor: DistributedTensor, comm: MPI.Comm) -> np.ndarray:
    """Reassemble the global tensor on every rank (zero-fill + Allreduce)."""
    local_host = cp.asnumpy(tensor.local)
    contrib = np.zeros(tensor.global_shape, dtype=local_host.dtype)
    for combo in itertools.product(*owned_segments(tensor.layout())):
        global_slices = tuple(slice(g, g + e) for g, _, e in combo)
        local_slices = tuple(slice(l, l + e) for _, l, e in combo)
        contrib[global_slices] = local_host[local_slices]
    full = np.zeros_like(contrib)
    comm.Allreduce(contrib, full, op=MPI.SUM)
    return full


def describe_mode(layout, axis: int, mode_name: str) -> str:
    """Render one mode's owned global segments, e.g. ``u:[0:2)+[4:6)``."""
    segments = layout.owned_global_segments[axis]
    return f"{mode_name}:" + "+".join(f"[{s}:{s + e})" for s, e in segments)


def check(condition: bool, comm: MPI.Comm, message: str) -> None:
    """Abort all ranks when a verification check fails."""
    if not condition:
        print(f"ERROR: {message}", flush=True)
        comm.Abort(1)


comm = MPI.COMM_WORLD.Dup()
rank = comm.Get_rank()
nranks = comm.Get_size()

device = cp.cuda.Device(rank % cp.cuda.runtime.getDeviceCount())
device.use()

# Initialize the distributed runtime and configure cuTensorNet with the
# same communicator.
nvmath.distributed.initialize(
    device_id=device.id, process_group=comm, backends=["nccl"]
)

handle = cutn.create()
options = DistributedContractionOptions(handle=handle, device_id=device.id)
try:
    cutn.distributed_reset_configuration(handle, *get_mpi_comm_pointer(comm))

    # B is partitioned along u (round-robin blocks); C along x (slab).
    col_major = ProcessGrid.Layout.COL_MAJOR
    b_dist = BlockCyclic(
        ProcessGrid(shape=(1, nranks, 1, 1), layout=col_major),
        (None, BS, None, None),
    )
    c_dist = BlockCyclic(
        ProcessGrid(shape=(1, nranks, 1, 1), layout=col_major),
        (None, None, None, None),
    )
    if nranks == 4:
        theta_grid = (2, 1, 1, 2, 1, 1)                # u and x
        theta_blocks = (BS, None, None, BS, None, None)
    else:
        theta_grid = (nranks, 1, 1, 1, 1, 1)           # u only
        theta_blocks = (BS, None, None, None, None, None)
    theta_dist = BlockCyclic(
        ProcessGrid(shape=theta_grid, layout=col_major), theta_blocks
    )

    rng = np.random.RandomState(SEED)
    b_host = rng.randn(CHI_T, D, D, CHI)
    c_host = rng.randn(CHI_T, D, D, CHI)
    b_half = make_distributed(b_host, b_dist)
    c_half = make_distributed(c_host, c_dist)

    # Show the contrast between the two layout flavors: B.u wraps around in
    # blocks of BS while C.x is one contiguous slab.
    ownership = ", ".join((
        describe_mode(b_half.layout(), 1, "B.u(cyclic)"),
        describe_mode(c_half.layout(), 1, "C.x(slab)"),
    ))
    for r in range(nranks):
        if r == rank:
            print(f"rank {rank}: {ownership}", flush=True)
        comm.Barrier()
    if rank == 0:
        print(
            f"B{b_half.global_shape}, C{c_half.global_shape}, "
            f"Theta over grid {theta_grid} with blocks {theta_blocks}; "
            f"contracted bond i (extent {CHI_T}) stays whole",
            flush=True,
        )

    with DistributedBinaryContraction(
        "iuva,ixyb->uvaxyb",
        b_half,
        c_half,
        out=theta_dist,
        options=options,
    ) as contraction:
        contraction.plan()
        theta = contraction.execute()

        expected = np.einsum("iuva,ixyb->uvaxyb", b_host, c_host)
        error = float(np.max(np.abs(gather(theta, comm) - expected)))
        if rank == 0:
            print(f"step 1: max|Theta - reference| = {error:.3e}", flush=True)
        check(
            error < 1e-12 * float(np.max(np.abs(expected))),
            comm,
            f"step 1 deviates from host reference: {error:.3e}",
        )

        # Replace the operands with tensors of the same shape and layout,
        # then reuse the prepared plan.
        b2_host = rng.randn(CHI_T, D, D, CHI)
        c2_host = rng.randn(CHI_T, D, D, CHI)
        contraction.reset_operands(
            a=make_distributed(b2_host, b_dist),
            b=make_distributed(c2_host, c_dist),
        )
        theta2 = contraction.execute()

        expected2 = np.einsum("iuva,ixyb->uvaxyb", b2_host, c2_host)
        error2 = float(np.max(np.abs(gather(theta2, comm) - expected2)))
        if rank == 0:
            print(
                f"step 2 (plan reuse): max|Theta - reference| = {error2:.3e}",
                flush=True,
            )
        check(
            error2 < 1e-12 * float(np.max(np.abs(expected2))),
            comm,
            f"step 2 deviates from host reference: {error2:.3e}",
        )
        if rank == 0:
            print("Distributed contraction matches the host reference.", flush=True)
    comm.Barrier()
finally:
    cutn.destroy(handle)
    nvmath.distributed.finalize()
    comm.Free()
