# Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES
#
# SPDX-License-Identifier: BSD-3-Clause

"""
Distributed binary tensor contraction D = alpha*A*B + beta*C over
block-cyclic distributed operands, at the raw bindings level.

The tensors are the two halves of an ATRG coarse-graining step (Adachi,
Okubo & Todo, Phys. Rev. B 102, 054432 (2020)): upper half B[i, u, v, a]
and lower half C[i, x, y, b] joined by a temporal bond ``i``, contracted
into the 6-mode tensor

    Theta[u, v, a, x, y, b] = sum_i B[i, u, v, a] * C[i, x, y, b]

whose truncated SVD (the "bond swap") is the step's central approximation.
Each operand carries its own placement and the library reshuffles
internally.

The sample walks the block-cyclic layout by hand (see ``owned_segments``)
and showcases both flavors: B's split mode u is block-cyclic with an
explicit block size (blocks dealt round-robin, the general layout), while
C's split mode x uses block size 0, the library's slab default (one
contiguous near-even chunk per rank, the special case
``block size = ceil(extent / nranks)``).

The second Compute call exercises the full D = alpha*A*B + beta*C form:
with beta = 1 and the C pointer aliasing the output buffer (allowed), a
second product accumulates onto the first step's Theta in place.

Run with::

    mpirun -n 4 python example02_contraction_mpi_nccl.py

Use one GPU per process.

Requirements:
    - mpi4py
    - cupy
    - CUTENSORNET_COMM_LIB pointing at the MPI distributed interface
"""

import itertools

import cupy as cp
from mpi4py import MPI
import numpy as np

import cuquantum
from cuquantum.bindings import cutensornet as cutn

D = 7       # spatial degrees of freedom (u, v, x, y)
CHI = 13    # vertical bond dimension (a, b)
CHI_T = 6   # temporal bond dimension (i), contracted away
BS = 2      # block size of every block-cyclic (explicitly blocked) mode
SEED = 42

root = 0
comm = MPI.COMM_WORLD
rank, nranks = comm.Get_rank(), comm.Get_size()


def check(condition, message):
    """mpirun-safe verification: evaluated on EVERY rank (the checked
    values are replicated, so all ranks agree), not stripped by
    ``python -O`` like a bare assert, and aborting the whole job rather
    than stranding peer ranks in the next collective call."""
    if not condition:
        print(f"ERROR: {message}", flush=True)
        comm.Abort(1)


# One GPU per process; the device must be selected before cutn.create().
device_id = rank % cp.cuda.runtime.getDeviceCount()
dev = cp.cuda.Device(device_id)
dev.use()

if rank == root:
    print("cuTensorNet-vers:", cutn.get_version())


def owned_segments(extent, block_size, nranks_for_mode, r):
    """This rank's ``(global_start, local_start, count)`` runs of one mode.

    Block-cyclic ownership: a mode of global extent E with block size bs
    over p ranks is cut into ceil(E / bs) blocks of bs consecutive indices
    (the last possibly short), dealt round-robin -- block k lives on rank
    k mod p as that rank's local block k // p. Hence global index g sits on
    rank (g // bs) mod p at local offset (g // bs) // p * bs + g % bs.

    Block size 0 is the slab default the descriptor creation applies when
    no block sizes are given: ONE contiguous near-even chunk per rank
    (ranks r < E mod p own one extra element) -- equivalently a single
    block of size ceil(E / p) each. An undistributed mode (p == 1) is one
    full-extent segment.
    """
    r = r % nranks_for_mode
    if block_size == 0:
        base, rem = divmod(extent, nranks_for_mode)
        start = r * base + min(r, rem)
        count = base + (1 if r < rem else 0)
        return [(start, 0, count)] if count else []
    segments, local_start = [], 0
    num_blocks = (extent + block_size - 1) // block_size
    for k in range(r, num_blocks, nranks_for_mode):
        start = k * block_size
        count = min(block_size, extent - start)
        segments.append((start, local_start, count))
        local_start += count
    return segments


def local_shard(global_host, block_sizes, nranks_per_mode, r):
    """Fortran-ordered CuPy shard of a block-cyclic-distributed host tensor.

    Element ownership is the Cartesian product of the per-mode segments,
    so the scatter is a nested loop over one segment choice per mode,
    copying a contiguous hyperblock each. Fortran order is REQUIRED: the
    contraction backend reads every shard as compact Fortran-order and
    applies no staging for other layouts -- a row-major shard would
    silently produce wrong results.
    """
    per_mode = [
        owned_segments(e, bs, p, r)
        for e, bs, p in zip(global_host.shape, block_sizes, nranks_per_mode)
    ]
    shape = tuple(sum(c for _, _, c in mode) for mode in per_mode)
    local = cp.empty(shape, dtype=global_host.dtype, order="F")
    for combo in itertools.product(*per_mode):
        global_slices = tuple(slice(g, g + c) for g, _, c in combo)
        local_slices = tuple(slice(l, l + c) for _, l, c in combo)
        local[local_slices] = cp.asarray(global_host[tuple(global_slices)])
    return local


def gather(local, global_shape, block_sizes, nranks_per_mode, r):
    """Reassemble the global tensor on every rank (zero-fill + Allreduce)."""
    per_mode = [
        owned_segments(e, bs, p, r)
        for e, bs, p in zip(global_shape, block_sizes, nranks_per_mode)
    ]
    contrib = np.zeros(global_shape, dtype=local.dtype)
    local_host = cp.asnumpy(local)
    for combo in itertools.product(*per_mode):
        global_slices = tuple(slice(g, g + c) for g, _, c in combo)
        local_slices = tuple(slice(l, l + c) for _, l, c in combo)
        contrib[tuple(global_slices)] = local_host[local_slices]
    full = np.zeros_like(contrib)
    comm.Allreduce(contrib, full, op=MPI.SUM)
    return full


##################################################
# Half-tensors for two coarse-graining steps
##################################################

rng = np.random.RandomState(SEED)
b1_host = rng.randn(CHI_T, D, D, CHI)
c1_host = rng.randn(CHI_T, D, D, CHI)
b2_host = rng.randn(CHI_T, D, D, CHI)
c2_host = rng.randn(CHI_T, D, D, CHI)

data_type = cuquantum.cudaDataType.CUDA_R_64F
compute_type = cutn.ComputeType.COMPUTE_64F

modes_b = [ord(c) for c in "iuva"]
modes_c = [ord(c) for c in "ixyb"]
modes_theta = [ord(c) for c in "uvaxyb"]

extents_b = (CHI_T, D, D, CHI)
extents_c = (CHI_T, D, D, CHI)
extents_theta = (D, D, CHI, D, D, CHI)

# Per-mode process grids: B split over u (block-cyclic, block size BS),
# C split over x (slab default, block size 0), Theta over u
# (block-cyclic).
grid_b = [1, nranks, 1, 1]
grid_c = [1, nranks, 1, 1]
grid_theta = [nranks, 1, 1, 1, 1, 1]
blocks_b = [0, BS, 0, 0]
blocks_c = [0, 0, 0, 0]
blocks_theta = [BS, 0, 0, 0, 0, 0]

################################################################
# Bind the MPI communicator and create distributed descriptors
################################################################

handle = cutn.create()

cutn_comm = comm.Dup()
cutn.distributed_reset_configuration(
    handle, MPI._addressof(cutn_comm), MPI._sizeof(cutn_comm)
)

# Descriptor creation is collective: every rank passes the same global
# metadata in the same order. The block-size arrays make the split modes
# block-cyclic; zeros stand in for the optional element strides and block
# strides (compact Fortran-order local layout).
desc_b = cutn.create_distributed_tensor_descriptor(
    handle, len(modes_b), extents_b, 0, blocks_b, 0, grid_b, modes_b, data_type
)
desc_c = cutn.create_distributed_tensor_descriptor(
    handle, len(modes_c), extents_c, 0, blocks_c, 0, grid_c, modes_c, data_type
)
desc_theta = cutn.create_distributed_tensor_descriptor(
    handle, len(modes_theta), extents_theta, 0, blocks_theta, 0,
    grid_theta, modes_theta, data_type
)

#####################################################
# Allocate local shards sized by the descriptors
#####################################################

b1_d = local_shard(b1_host, blocks_b, grid_b, rank)
c1_d = local_shard(c1_host, blocks_c, grid_c, rank)
# Zero-filled (not uninitialized) output: step 2 reads it back as the C
# addend of D = alpha*A*B + beta*C.
theta_shape = tuple(
    sum(c for _, _, c in owned_segments(e, bs, p, rank))
    for e, bs, p in zip(extents_theta, blocks_theta, grid_theta)
)
theta_d = cp.zeros(theta_shape, dtype=np.float64, order="F")

# Cross-check the hand-computed shard size against the descriptor.
local_size = np.zeros(1, dtype=np.uint64)
cutn.tensor_descriptor_get_attribute(
    handle, desc_theta, cutn.TensorDescriptorAttribute.LOCAL_DATA_SIZE,
    local_size.ctypes.data, local_size.itemsize,
)
check(int(local_size[0]) == theta_d.nbytes,
      "hand-computed shard size disagrees with the descriptor's LOCAL_DATA_SIZE")

# Show the contrast between the two layout flavors: B.u wraps around in
# blocks of BS while C.x is one contiguous slab.
b_owned = "+".join(f"[{g}:{g + c})"
                   for g, _, c in owned_segments(D, BS, nranks, rank))
c_owned = "+".join(f"[{g}:{g + c})"
                   for g, _, c in owned_segments(D, 0, nranks, rank))
for r in range(nranks):
    if r == rank:
        print(f"rank {rank}: B.u(cyclic):{b_owned}, C.x(slab):{c_owned}",
              flush=True)
    comm.Barrier()
if rank == root:
    print(f"B{extents_b}, C{extents_c}, Theta{extents_theta} split over u "
          f"across {nranks} rank(s); contracted bond i (extent {CHI_T}) "
          f"stays whole")

######################################################
# Create and prepare the stateful binary contraction
######################################################

# The addend C of D = alpha*A*B + beta*C must match D's logical and
# distributed layout exactly; reusing the output descriptor expresses that,
# and below the C data pointer aliases the output buffer.
contraction = cutn.create_binary_tensor_contraction(
    handle, desc_b, desc_c, desc_theta, desc_theta, compute_type
)

# Prepare publishes the plan's exact device and host scratch requirements
# on the workspace descriptor (no device planning budget is declared here;
# set a real size via workspace_set_memory before Prepare to declare one).
work_desc = cutn.create_workspace_descriptor(handle)
cutn.binary_tensor_contraction_prepare(handle, contraction, work_desc)

device_work_size = cutn.workspace_get_memory_size(
    handle, work_desc, cutn.WorksizePref.MIN,
    cutn.Memspace.DEVICE, cutn.WorkspaceKind.SCRATCH,
)
host_work_size = cutn.workspace_get_memory_size(
    handle, work_desc, cutn.WorksizePref.MIN,
    cutn.Memspace.HOST, cutn.WorkspaceKind.SCRATCH,
)
device_work = cp.cuda.alloc(device_work_size)
cutn.workspace_set_memory(
    handle, work_desc, cutn.Memspace.DEVICE, cutn.WorkspaceKind.SCRATCH,
    device_work.ptr, device_work_size,
)
# A zero requirement must not be "set": a pointer paired with size 0 is
# rejected as INVALID_VALUE. Keep the buffer alive for the Compute calls.
if host_work_size:
    host_work = np.empty(host_work_size, dtype=np.int8)
    cutn.workspace_set_memory(
        handle, work_desc, cutn.Memspace.HOST, cutn.WorkspaceKind.SCRATCH,
        host_work.ctypes.data, host_work_size,
    )

##################################################################
# Step 1: Theta = B1 * C1 (beta = 0 ignores the C addend's data)
##################################################################

# alpha/beta live in host buffers whose pointers the call reads; their
# dtype follows the cuTENSOR scalar rules for the output data type.
alpha = np.ones(1, dtype=np.float64)
beta = np.zeros(1, dtype=np.float64)

stream = cp.cuda.Stream()
cutn.binary_tensor_contraction_compute(
    handle, contraction,
    alpha.ctypes.data, b1_d.data.ptr, c1_d.data.ptr,
    beta.ctypes.data, theta_d.data.ptr, theta_d.data.ptr,
    work_desc, stream.ptr,
)
stream.synchronize()

expected = np.einsum("iuva,ixyb->uvaxyb", b1_host, c1_host)
error = float(np.max(np.abs(
    gather(theta_d, extents_theta, blocks_theta, grid_theta, rank) - expected)))
if rank == root:
    print(f"step 1 (beta=0): max|Theta - reference| = {error:.3e}")
# Relative to the reference's scale, not an absolute pin.
check(error < 1e-12 * float(np.max(np.abs(expected))),
      f"step 1 deviates from host reference: {error:.3e}")

########################################################################
# Step 2: Theta = B2 * C2 + Theta, accumulating in place with beta = 1
########################################################################

b2_d = local_shard(b2_host, blocks_b, grid_b, rank)
c2_d = local_shard(c2_host, blocks_c, grid_c, rank)
beta[0] = 1.0

cutn.binary_tensor_contraction_compute(
    handle, contraction,
    alpha.ctypes.data, b2_d.data.ptr, c2_d.data.ptr,
    beta.ctypes.data, theta_d.data.ptr, theta_d.data.ptr,
    work_desc, stream.ptr,
)
stream.synchronize()

expected += np.einsum("iuva,ixyb->uvaxyb", b2_host, c2_host)
error = float(np.max(np.abs(
    gather(theta_d, extents_theta, blocks_theta, grid_theta, rank) - expected)))
if rank == root:
    print(f"step 2 (beta=1, in-place accumulate): max|Theta - reference| = {error:.3e}")
check(error < 1e-12 * float(np.max(np.abs(expected))),
      f"step 2 deviates from host reference: {error:.3e}")
if rank == root:
    print("Distributed binary contraction matches the host reference.")

#################
# Free resources
#################

cutn.destroy_binary_tensor_contraction(contraction)
cutn.destroy_tensor_descriptor(desc_b)
cutn.destroy_tensor_descriptor(desc_c)
cutn.destroy_tensor_descriptor(desc_theta)
cutn.destroy_workspace_descriptor(work_desc)
cutn.destroy(handle)
cutn_comm.Free()

if rank == root:
    print("Free resource and exit.")
