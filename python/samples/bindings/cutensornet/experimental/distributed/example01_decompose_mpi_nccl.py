# Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES
#
# SPDX-License-Identifier: BSD-3-Clause

"""
Distributed truncated SVD of a six-mode tensor with MPI and NCCL using the
raw cuTensorNet bindings.

This example demonstrates the distributed truncated SVD entry points on the
bond-swapping step of the anisotropic tensor renormalization group (ATRG)
[Adachi, Okubo, and Todo, Phys. Rev. B 102, 054432 (2020)], where two
half-tensors are contracted over ``i`` and the result is decomposed:

    Theta[u, v, a, x, y, b] = sum_i B[i, u, v, a] * C[i, x, y, b]
    Theta ~= sum_g X[a, x, y, g] * s[g] * Y[g, u, v, b]

Theta is built with a replicated NumPy einsum on every rank purely to
produce input data; the SVD below is the sample's only distributed
cuTensorNet call. See example02_contraction_mpi_nccl.py for distributed
contraction.

The input and output factors use independent block-cyclic distributions.
With an explicit block size, consecutive blocks are assigned round-robin
across process-grid coordinates; the final block may be shorter. Block size
zero selects one contiguous near-even slab per coordinate. The result is
checked against a NumPy SVD.

Run with::

    mpirun -n 4 python example01_decompose_mpi_nccl.py

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

D = 7       # spatial degrees of freedom (u, v, x, y) and the kept extent of g
CHI = 13    # vertical bond dimension (a, b), enlarged by bond overspanning
CHI_T = 6   # temporal bond dimension (i), contracted away on the host below
BS = 2      # block size of every block-cyclic (explicitly blocked) mode
SEED = 42

root = 0
comm = MPI.COMM_WORLD
rank, nranks = comm.Get_rank(), comm.Get_size()


def check(condition, message):
    """Abort all ranks when a verification check fails."""
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

    Block size 0 selects one contiguous near-even slab per rank; ranks
    ``r < E mod p`` own one extra element. An undistributed mode
    (``p == 1``) is one full-extent segment.
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


def local_shape_of(extents, block_sizes, nranks_per_mode, r):
    """Per-mode sum of owned segment counts = this rank's local shape."""
    return tuple(
        sum(count for _, _, count in owned_segments(e, bs, p, r))
        for e, bs, p in zip(extents, block_sizes, nranks_per_mode)
    )


def local_shard(global_host, block_sizes, nranks_per_mode, r):
    """Fortran-ordered CuPy shard of a block-cyclic-distributed host tensor.

    Element ownership is the Cartesian product of the per-mode segments,
    so the scatter is a nested loop over one segment choice per mode,
    copying a contiguous hyperblock each.
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


##########################################################
# Build Theta = sum_i B[i,u,v,a] C[i,x,y,b] on every rank
##########################################################

rng = np.random.RandomState(SEED)
b_half = rng.randn(CHI_T, D, D, CHI)
c_half = rng.randn(CHI_T, D, D, CHI)
theta_host = np.einsum("iuva,ixyb->uvaxyb", b_half, c_half)

data_type = cuquantum.cudaDataType.CUDA_R_64F

modes_theta = [ord(c) for c in "uvaxyb"]
modes_x = [ord(c) for c in "axyg"]
modes_y = [ord(c) for c in "guvb"]

extents_theta = (D, D, CHI, D, D, CHI)
# The shared extent of the outputs at creation time is the truncation cap:
# the bond swap keeps g = D of the D*D*CHI available singular values.
extents_x = (CHI, D, D, D)
extents_y = (D, D, D, CHI)

# Per-mode process grids: Theta is split over u, X over x, Y over u; the
# swapped bond g stays whole on every rank. Every split mode is
# block-cyclic with block size BS; 0 leaves a mode on the slab default.
grid_theta = [nranks, 1, 1, 1, 1, 1]
grid_x = [1, nranks, 1, 1]
grid_y = [1, nranks, 1, 1]
blocks_theta = [BS, 0, 0, 0, 0, 0]
blocks_x = [0, BS, 0, 0]
blocks_y = [0, BS, 0, 0]

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
desc_theta = cutn.create_distributed_tensor_descriptor(
    handle, len(modes_theta), extents_theta, 0, blocks_theta, 0,
    grid_theta, modes_theta, data_type
)
desc_x = cutn.create_distributed_tensor_descriptor(
    handle, len(modes_x), extents_x, 0, blocks_x, 0, grid_x, modes_x, data_type
)
desc_y = cutn.create_distributed_tensor_descriptor(
    handle, len(modes_y), extents_y, 0, blocks_y, 0, grid_y, modes_y, data_type
)

#####################################################
# Allocate local shards sized by the descriptors
#####################################################

theta_d = local_shard(theta_host, blocks_theta, grid_theta, rank)
# Random initial contents for the outputs (the SVD overwrites them) rather
# than uninitialized memory, so no stray bytes can ever be read back.
x_d = cp.asfortranarray(cp.random.standard_normal(
    local_shape_of(extents_x, blocks_x, grid_x, rank)))
y_d = cp.asfortranarray(cp.random.standard_normal(
    local_shape_of(extents_y, blocks_y, grid_y, rank)))
s_d = cp.zeros(D, dtype=np.float64)  # replicated: full length on every rank

# Cross-check the hand-computed shard size against the descriptor.
local_size = np.zeros(1, dtype=np.uint64)
cutn.tensor_descriptor_get_attribute(
    handle, desc_theta, cutn.TensorDescriptorAttribute.LOCAL_DATA_SIZE,
    local_size.ctypes.data, local_size.itemsize,
)
check(int(local_size[0]) == theta_d.nbytes,
      "hand-computed shard size disagrees with the descriptor's LOCAL_DATA_SIZE")

# Show the block-cyclic ownership this rank computed: with BS=2 and two
# ranks, mode u (extent 7) has blocks [0:2) [2:4) [4:6) [6:7); rank 0 owns
# blocks 0 and 2, rank 1 owns 1 and 3.
u_owned = "+".join(f"[{g}:{g + c})"
                   for g, _, c in owned_segments(D, BS, nranks, rank))
for r in range(nranks):
    if r == rank:
        print(f"rank {rank}: Theta.u(cyclic):{u_owned}, "
              f"local shard {tuple(theta_d.shape)}", flush=True)
    comm.Barrier()
if rank == root:
    print(f"Theta{extents_theta} split over u across {nranks} rank(s); "
          f"bond swap keeps g = {D} of {D * D * CHI} singular values")

###############################################################
# SVD config/info and workspace (same calls as the local path)
###############################################################

# Distributed SVD requires GESVDP; the config default (GESVD) returns
# NOT_SUPPORTED. This first pass truncates to a fixed extent -- the shared
# extent the output descriptors were created with.
svd_config = cutn.create_tensor_svd_config(handle)
algorithm = np.asarray(
    [cutn.TensorSVDAlgo.GESVDP],
    dtype=cutn.tensor_svd_config_get_attribute_dtype(
        cutn.TensorSVDConfigAttribute.ALGO),
)
cutn.tensor_svd_config_set_attribute(
    handle, svd_config, cutn.TensorSVDConfigAttribute.ALGO,
    algorithm.ctypes.data, algorithm.dtype.itemsize,
)
svd_info = cutn.create_tensor_svd_info(handle)

work_desc = cutn.create_workspace_descriptor(handle)
cutn.workspace_compute_svd_sizes(handle, desc_theta, desc_x, desc_y, svd_config, work_desc)

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
# rejected as INVALID_VALUE. Keep the buffer alive for the SVD call.
if host_work_size:
    host_work = np.empty(host_work_size, dtype=np.int8)
    cutn.workspace_set_memory(
        handle, work_desc, cutn.Memspace.HOST, cutn.WorkspaceKind.SCRATCH,
        host_work.ctypes.data, host_work_size,
    )

############################################
# Execution: the distributed bond-swap SVD
############################################

# Fixed-extent truncation: Kred == Kcap == D, so desc_x/desc_y stay at D.
stream = cp.cuda.Stream()
cutn.tensor_svd(
    handle, desc_theta, theta_d.data.ptr,
    desc_x, x_d.data.ptr,
    s_d.data.ptr,
    desc_y, y_d.data.ptr,
    svd_config, svd_info, work_desc, stream.ptr,
)
stream.synchronize()

reduced_extent = np.zeros(1, dtype=cutn.tensor_svd_info_get_attribute_dtype(
    cutn.TensorSVDInfoAttribute.REDUCED_EXTENT))
cutn.tensor_svd_info_get_attribute(
    handle, svd_info, cutn.TensorSVDInfoAttribute.REDUCED_EXTENT,
    reduced_extent.ctypes.data, reduced_extent.itemsize,
)
discarded_weight = np.zeros(1, dtype=cutn.tensor_svd_info_get_attribute_dtype(
    cutn.TensorSVDInfoAttribute.DISCARDED_WEIGHT))
cutn.tensor_svd_info_get_attribute(
    handle, svd_info, cutn.TensorSVDInfoAttribute.DISCARDED_WEIGHT,
    discarded_weight.ctypes.data, discarded_weight.itemsize,
)

#######################################
# Verify against a host reference
#######################################

theta_mat = np.ascontiguousarray(theta_host.transpose(2, 3, 4, 0, 1, 5))
theta_mat = theta_mat.reshape(CHI * D * D, D * D * CHI)
s_ref = np.linalg.svd(theta_mat, compute_uv=False)

s_host = cp.asnumpy(s_d)
s_err = float(np.max(np.abs(s_host - s_ref[:D])))

x_full = gather(x_d, extents_x, blocks_x, grid_x, rank)
y_full = gather(y_d, extents_y, blocks_y, grid_y, rank)
rec = np.einsum("axyg,g,guvb->uvaxyb", x_full, s_host, y_full)
theta_norm2 = float(np.sum(theta_host**2))
residual2 = float(np.sum((rec - theta_host) ** 2))
expected_residual2 = float(np.sum(s_ref[D:] ** 2)) / theta_norm2

if rank == root:
    print(f"reduced extent: {int(reduced_extent[0])} (cap {D})")
    print(f"singular values: max|s - s_ref| = {s_err:.3e}")
    print(f"truncation: relative residual^2 = {residual2 / theta_norm2:.6f}, "
          f"reference = {expected_residual2:.6f}, "
          f"reported discarded weight = {float(discarded_weight[0]):.6f}")
# Tolerances are relative to the problem's scale (leading singular value /
# unit-normalized residual ratio), not absolute last-digit pins.
check(s_err < 1e-10 * s_ref[0],
      f"singular values deviate from host reference: {s_err:.3e}")
check(abs(residual2 / theta_norm2 - expected_residual2) < 1e-8,
      "reconstruction residual inconsistent with discarded spectrum")
if rank == root:
    print("Distributed bond-swap SVD matches the host reference.")

##########################################################
# Second pass: value-based truncation
##########################################################

# rel_cutoff discards sigma < rel_cutoff * sigma_max, setting Kred from the
# spectrum (read back via SVDInfo). The shared bond mode on each output must
# be owned by a single rank (g here has process-grid extent 1) or explicitly
# block-cyclic (blockSizes > 0). A near-even slab over more than one rank is
# rejected.

# Pick a cutoff between singular values so Kred is predictable.
K_TARGET = 4
rel_cutoff_value = 0.5 * (s_ref[K_TARGET - 1] + s_ref[K_TARGET]) / s_ref[0]
rel_cutoff = np.asarray(
    [rel_cutoff_value],
    dtype=cutn.tensor_svd_config_get_attribute_dtype(
        cutn.TensorSVDConfigAttribute.REL_CUTOFF),
)
cutn.tensor_svd_config_set_attribute(
    handle, svd_config, cutn.TensorSVDConfigAttribute.REL_CUTOFF,
    rel_cutoff.ctypes.data, rel_cutoff.dtype.itemsize,
)

# After tensor_svd, desc_x/desc_y describe Kred, not Kcap. Recreate them at D
# before each call when truncation can shrink the bond.
cutn.destroy_tensor_descriptor(desc_x)
cutn.destroy_tensor_descriptor(desc_y)
desc_x = cutn.create_distributed_tensor_descriptor(
    handle, len(modes_x), extents_x, 0, blocks_x, 0, grid_x, modes_x, data_type
)
desc_y = cutn.create_distributed_tensor_descriptor(
    handle, len(modes_y), extents_y, 0, blocks_y, 0, grid_y, modes_y, data_type
)

# Workspace requirements may differ from the fixed-extent pass; re-query.
work_desc2 = cutn.create_workspace_descriptor(handle)
cutn.workspace_compute_svd_sizes(
    handle, desc_theta, desc_x, desc_y, svd_config, work_desc2)
device_work_size2 = cutn.workspace_get_memory_size(
    handle, work_desc2, cutn.WorksizePref.MIN,
    cutn.Memspace.DEVICE, cutn.WorkspaceKind.SCRATCH,
)
host_work_size2 = cutn.workspace_get_memory_size(
    handle, work_desc2, cutn.WorksizePref.MIN,
    cutn.Memspace.HOST, cutn.WorkspaceKind.SCRATCH,
)
device_work2 = cp.cuda.alloc(device_work_size2)
cutn.workspace_set_memory(
    handle, work_desc2, cutn.Memspace.DEVICE, cutn.WorkspaceKind.SCRATCH,
    device_work2.ptr, device_work_size2,
)
if host_work_size2:
    host_work2 = np.empty(host_work_size2, dtype=np.int8)
    cutn.workspace_set_memory(
        handle, work_desc2, cutn.Memspace.HOST, cutn.WorkspaceKind.SCRATCH,
        host_work2.ctypes.data, host_work_size2,
    )

# Allocate outputs at cap D; only the leading Kred entries are used.
x2_d = cp.asfortranarray(cp.random.standard_normal(
    local_shape_of(extents_x, blocks_x, grid_x, rank)))
y2_d = cp.asfortranarray(cp.random.standard_normal(
    local_shape_of(extents_y, blocks_y, grid_y, rank)))
s2_d = cp.zeros(D, dtype=np.float64)

cutn.tensor_svd(
    handle, desc_theta, theta_d.data.ptr,
    desc_x, x2_d.data.ptr,
    s2_d.data.ptr,
    desc_y, y2_d.data.ptr,
    svd_config, svd_info, work_desc2, stream.ptr,
)
stream.synchronize()

reduced_extent2 = np.zeros(1, dtype=cutn.tensor_svd_info_get_attribute_dtype(
    cutn.TensorSVDInfoAttribute.REDUCED_EXTENT))
cutn.tensor_svd_info_get_attribute(
    handle, svd_info, cutn.TensorSVDInfoAttribute.REDUCED_EXTENT,
    reduced_extent2.ctypes.data, reduced_extent2.itemsize,
)
discarded_weight2 = np.zeros(1, dtype=cutn.tensor_svd_info_get_attribute_dtype(
    cutn.TensorSVDInfoAttribute.DISCARDED_WEIGHT))
cutn.tensor_svd_info_get_attribute(
    handle, svd_info, cutn.TensorSVDInfoAttribute.DISCARDED_WEIGHT,
    discarded_weight2.ctypes.data, discarded_weight2.itemsize,
)
kred = int(reduced_extent2[0])

# Re-view the cap-sized buffers at extent Kred for verification.
extents_x_red = (CHI, D, D, kred)
extents_y_red = (kred, D, D, CHI)
x_red = cp.ndarray(local_shape_of(extents_x_red, blocks_x, grid_x, rank),
                   dtype=np.float64, memptr=x2_d.data, order="F")
y_red = cp.ndarray(local_shape_of(extents_y_red, blocks_y, grid_y, rank),
                   dtype=np.float64, memptr=y2_d.data, order="F")

s_host2 = cp.asnumpy(s2_d)[:kred]
s_err2 = float(np.max(np.abs(s_host2 - s_ref[:kred])))
x_full2 = gather(x_red, extents_x_red, blocks_x, grid_x, rank)
y_full2 = gather(y_red, extents_y_red, blocks_y, grid_y, rank)
rec2 = np.einsum("axyg,g,guvb->uvaxyb", x_full2, s_host2, y_full2)
residual2_val = float(np.sum((rec2 - theta_host) ** 2))
expected_residual2_val = float(np.sum(s_ref[kred:] ** 2)) / theta_norm2

if rank == root:
    print(f"value-based: rel_cutoff = {rel_cutoff_value:.3e} -> "
          f"reduced extent {kred} (cap {D})")
    print(f"value-based: max|s - s_ref| = {s_err2:.3e}")
    print(f"value-based: relative residual^2 = "
          f"{residual2_val / theta_norm2:.6f}, "
          f"reference = {expected_residual2_val:.6f}, "
          f"reported discarded weight = {float(discarded_weight2[0]):.6f}")

check(kred == K_TARGET,
      f"value-based truncation kept {kred} values, expected {K_TARGET}")
check(s_err2 < 1e-10 * s_ref[0],
      f"retained singular values deviate from host reference: {s_err2:.3e}")
check(abs(residual2_val / theta_norm2 - expected_residual2_val) < 1e-8,
      "value-based reconstruction inconsistent with discarded spectrum")
if rank == root:
    print("Distributed value-based truncation matches the host reference.")

#################
# Free resources
#################

cutn.destroy_tensor_descriptor(desc_theta)
cutn.destroy_tensor_descriptor(desc_x)
cutn.destroy_tensor_descriptor(desc_y)
cutn.destroy_workspace_descriptor(work_desc)
cutn.destroy_workspace_descriptor(work_desc2)
cutn.destroy_tensor_svd_config(svd_config)
cutn.destroy_tensor_svd_info(svd_info)
cutn.destroy(handle)
cutn_comm.Free()

if rank == root:
    print("Free resource and exit.")
