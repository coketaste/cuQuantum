# Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES
#
# SPDX-License-Identifier: BSD-3-Clause

"""
Distributed truncated SVD of a six-mode tensor with MPI and NCCL.

This example demonstrates distributed_decompose (a distributed truncated
SVD) on the bond-swapping step of the anisotropic tensor renormalization
group (ATRG) [Adachi, Okubo, and Todo, Phys. Rev. B 102, 054432 (2020)],
where two half-tensors are contracted over ``i`` and the result is
decomposed:

    Theta[u, v, a, x, y, b] = sum_i B[i, u, v, a] * C[i, x, y, b]
    Theta ~= sum_g X[a, x, y, g] * s[g] * Y[g, u, v, b]

Theta is built with a replicated NumPy einsum on the host purely to
produce input data; the SVD below is the sample's only distributed
cuTensorNet call. See example02_contraction_mpi_nccl.py for distributed
contraction.

The input and output factors use independent block-cyclic distributions.
Along each distributed mode, an integer block size assigns consecutive
blocks round-robin across process-grid coordinates, while ``None`` selects
one near-even contiguous slab per coordinate. A process-grid extent of one
leaves that mode undistributed. The result is checked against a NumPy SVD.

Run with::

    mpirun -n 4 python example01_svd_mpi_nccl.py

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
    DistributedDecompositionOptions,
    DistributedTensor,
    ProcessGrid,
    distributed_decompose,
    empty_local,
    get_local_layout,
)
from cuquantum.tensornet.tensor import SVDMethod

D = 7       # spatial degrees of freedom (u, v, x, y) and the kept extent of g
CHI = 13    # vertical bond dimension (a, b), enlarged by bond overspanning
CHI_T = 6   # temporal bond dimension (i), contracted away below
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

# Initialize the distributed runtime and configure cuTensorNet with the same communicator.
nvmath.distributed.initialize(
    device_id=device.id, process_group=comm, backends=["nccl"]
)

handle = cutn.create()
options = DistributedDecompositionOptions(handle=handle, device_id=device.id)
try:
    cutn.distributed_reset_configuration(handle, *get_mpi_comm_pointer(comm))

    # Build Theta = sum_i B[i,u,v,a] C[i,x,y,b] (replicated on host)
    rng = np.random.RandomState(SEED)
    b_half = rng.randn(CHI_T, D, D, CHI)
    c_half = rng.randn(CHI_T, D, D, CHI)
    theta_host = np.einsum("iuva,ixyb->uvaxyb", b_half, c_half)

    # Integer entries select round-robin blocks; None selects contiguous slabs.
    col_major = ProcessGrid.Layout.COL_MAJOR
    if nranks == 4:
        # Split Theta over a 2x2 grid spanning modes u and x.
        theta_grid = (2, 1, 1, 2, 1, 1)                # u and x
        theta_blocks = (BS, None, None, BS, None, None)
    else:
        # Use a one-dimensional grid over mode u for other process counts.
        theta_grid = (nranks, 1, 1, 1, 1, 1)           # u only
        theta_blocks = (BS, None, None, None, None, None)
    theta_dist = BlockCyclic(
        ProcessGrid(shape=theta_grid, layout=col_major), theta_blocks
    )
    # Split X over x and Y over u; keep the shared mode g undistributed.
    x_dist = BlockCyclic(
        ProcessGrid(shape=(1, nranks, 1, 1), layout=col_major),
        (None, BS, None, None),
    )
    y_dist = BlockCyclic(
        ProcessGrid(shape=(1, nranks, 1, 1), layout=col_major),
        (None, BS, None, None),
    )

    theta = make_distributed(theta_host, theta_dist)

    # out_left and out_right describe placement; the API allocates storage.
    x_shape = (CHI, D, D, D)
    y_shape = (D, D, D, CHI)

    # Display the global blocks owned by selected local modes on each rank.
    ownership = ", ".join((
        describe_mode(theta.layout(), 0, "Theta.u"),
        describe_mode(get_local_layout(x_dist, x_shape), 1, "X.x"),
        describe_mode(get_local_layout(y_dist, y_shape), 1, "Y.u"),
    ))
    for r in range(nranks):
        if r == rank:
            print(f"rank {rank}: {ownership}", flush=True)
        comm.Barrier()
    if rank == 0:
        print(
            f"Theta{theta.global_shape} over grid {theta_grid}; "
            f"local shard {tuple(theta.local.shape)}; "
            f"bond swap keeps g = {D} of {D * D * CHI} singular values",
            flush=True,
        )

    # Compute the distributed truncated SVD.
    x_res, s_res, y_res, info = distributed_decompose(
        "uvaxyb->axyg,guvb",
        theta,
        out_left=x_dist,
        out_right=y_dist,
        # Distributed SVD requires algorithm='gesvdp' (SVDMethod defaults to 'gesvd').
        method=SVDMethod(algorithm="gesvdp", max_extent=D),
        options=options,
        return_info=True,
    )

    # Verify against a host reference.
    theta_mat = np.ascontiguousarray(theta_host.transpose(2, 3, 4, 0, 1, 5))
    theta_mat = theta_mat.reshape(CHI * D * D, D * D * CHI)
    s_ref = np.linalg.svd(theta_mat, compute_uv=False)

    s_host = cp.asnumpy(s_res)
    s_err = float(np.max(np.abs(s_host - s_ref[:D])))

    rec = np.einsum(
        "axyg,g,guvb->uvaxyb", gather(x_res, comm), s_host, gather(y_res, comm)
    )
    theta_norm2 = float(np.sum(theta_host**2))
    residual2 = float(np.sum((rec - theta_host) ** 2))
    # ||Theta - X s Y||^2 / ||Theta||^2 equals the relative weight of
    # the discarded singular values.
    expected_residual2 = float(np.sum(s_ref[D:] ** 2)) / theta_norm2

    if rank == 0:
        print(f"singular values: max|s - s_ref| = {s_err:.3e}", flush=True)
        print(
            f"truncation: relative residual^2 = {residual2 / theta_norm2:.6f}, "
            f"reference = {expected_residual2:.6f}, "
            f"reported discarded weight = {info.discarded_weight:.6f}",
            flush=True,
        )
    check(s_err < 1e-10 * s_ref[0], comm,
          f"singular values deviate from host reference: {s_err:.3e}")
    check(abs(residual2 / theta_norm2 - expected_residual2) < 1e-8, comm,
          "reconstruction residual inconsistent with discarded spectrum")
    if rank == 0:
        print("Distributed bond-swap SVD matches the host reference.", flush=True)

    # Value-based truncation: rel_cutoff discards sigma < rel_cutoff *
    # sigma_max. Outputs are freshly allocated each call, so no descriptor
    # rebuild is needed (see the bindings sample).
    #
    # The shared mode g is owned by a single rank here (process-grid extent
    # 1), which satisfies the owner-stability requirement for value-based
    # truncation. A near-even slab over more than one rank is rejected.
    k_target = 4
    rel_cutoff = 0.5 * (s_ref[k_target - 1] + s_ref[k_target]) / s_ref[0]
    x_cut, s_cut, y_cut, info_cut = distributed_decompose(
        "uvaxyb->axyg,guvb",
        theta,
        out_left=BlockCyclic(
            ProcessGrid(shape=(1, nranks, 1, 1), layout=col_major),
            (None, BS, None, None),
        ),
        out_right=BlockCyclic(
            ProcessGrid(shape=(1, nranks, 1, 1), layout=col_major),
            (None, BS, None, None),
        ),
        method=SVDMethod(algorithm="gesvdp", rel_cutoff=rel_cutoff),
        options=options,
        return_info=True,
    )
    kred = int(info_cut.reduced_extent)
    s_cut_host = cp.asnumpy(s_cut)[:kred]
    s_err_cut = float(np.max(np.abs(s_cut_host - s_ref[:kred])))
    rec_cut = np.einsum(
        "axyg,g,guvb->uvaxyb",
        gather(x_cut, comm), s_cut_host, gather(y_cut, comm),
    )
    residual2_cut = float(np.sum((rec_cut - theta_host) ** 2))
    expected_residual2_cut = float(np.sum(s_ref[kred:] ** 2)) / theta_norm2

    if rank == 0:
        print(
            f"value-based: rel_cutoff = {rel_cutoff:.3e} -> "
            f"reduced extent {kred} (cap {D})",
            flush=True,
        )
        print(
            f"value-based: relative residual^2 = "
            f"{residual2_cut / theta_norm2:.6f}, "
            f"reference = {expected_residual2_cut:.6f}, "
            f"reported discarded weight = {info_cut.discarded_weight:.6f}",
            flush=True,
        )
    check(kred == k_target, comm,
          f"value-based truncation kept {kred} values, expected {k_target}")
    check(s_err_cut < 1e-10 * s_ref[0], comm,
          f"retained singular values deviate from reference: {s_err_cut:.3e}")
    check(abs(residual2_cut / theta_norm2 - expected_residual2_cut) < 1e-8, comm,
          "value-based reconstruction inconsistent with discarded spectrum")
    if rank == 0:
        print("Distributed value-based truncation matches the host reference.",
              flush=True)
    comm.Barrier()
finally:
    cutn.destroy(handle)
    nvmath.distributed.finalize()
    comm.Free()
