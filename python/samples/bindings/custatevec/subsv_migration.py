# Copyright (c) 2021-2026, NVIDIA CORPORATION & AFFILIATES
#
# SPDX-License-Identifier: BSD-3-Clause

import cupy as cp
import cupyx as cpx
import numpy as np

from cuquantum.bindings import custatevec as cusv
from cuquantum import cudaDataType

dtype = np.complex128
sv_data_type = cudaDataType.CUDA_C_64F

n_slice_local_index_bits = 3

sub_sv_size = 2 ** n_slice_local_index_bits

# allocate host sub state vectors
n_sub_svs = 2
sub_svs = [None] * n_sub_svs
sub_svs[0] = cpx.empty_pinned(sub_sv_size, dtype=dtype)
sub_svs[0][:] = 0.25 + 0.j
sub_svs[1] = cpx.zeros_pinned(sub_sv_size, dtype=dtype)

# allocate device slices

n_device_slices = 1
device_slices_size = sub_sv_size * n_device_slices
device_slices = cp.zeros([device_slices_size], dtype=dtype)

# initialize custatevec handle
handle = cusv.create()

# create migrator
migrator = cusv.sub_sv_migrator_create(handle, device_slices.data.ptr, sv_data_type,
                                       n_device_slices, n_slice_local_index_bits)

device_slice_index = 0
src_sub_sv_slice = sub_svs[0]
dst_sub_sv_slice = sub_svs[1]

# migrate sub_svs[0] into device_slices
cusv.sub_sv_migrator_migrate(handle, migrator, device_slice_index,
                             src_sub_sv_slice.ctypes.data, 0, 0, sub_sv_size)

# migrate device_slices into sub_svs[1]
cusv.sub_sv_migrator_migrate(handle, migrator, device_slice_index,
                             0, dst_sub_sv_slice.ctypes.data, 0, sub_sv_size)

# destroy migrator
cusv.sub_sv_migrator_destroy(handle, migrator)

# destroy custatevec handle
cusv.destroy(handle)

# check if sub_svs[1] has expected values
correct = np.all(sub_svs[1] == 0.25 + 0.j)

if correct:
    print('subsv_migration example PASSED')
else:
    raise RuntimeError('subsv_migration example FAILED: wrong result')
