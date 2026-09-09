# Copyright (c) 2021-2026, NVIDIA CORPORATION & AFFILIATES
#
# SPDX-License-Identifier: BSD-3-Clause

import os
import re

from packaging.version import Version
from setuptools.command.build_ext import build_ext as _build_ext


# Get __version__ variable
source_root = os.path.abspath(os.path.dirname(__file__))
with open(os.path.join(source_root, '..', 'cuquantum', '_version.py')) as f:
    exec(f.read())
cuqnt_py_ver = __version__
cuqnt_py_ver_obj = Version(cuqnt_py_ver)
cuqnt_ver_major_minor = f"{cuqnt_py_ver_obj.major}.{cuqnt_py_ver_obj.minor}"

del __version__, cuqnt_py_ver_obj, source_root


# We can't assume users to have CTK installed via pip, so we really need this...
# TODO(leofang): try /usr/local/cuda?
try:
    cuda_path = os.environ['CUDA_PATH']
except KeyError as e:
    raise RuntimeError('CUDA is not found, please set $CUDA_PATH') from e


def check_cuda_version():
    try:
        # We cannot do a dlopen and call cudaRuntimeGetVersion, because it
        # requires GPUs. We also do not want to rely on the compiler utility
        # provided in distutils (deprecated) or setuptools, as this is a very
        # simple string parsing task.
        # TODO: switch to cudaRuntimeGetVersion once it's fixed (nvbugs 3624208)
        cuda_h = os.path.join(cuda_path, 'include', 'cuda.h')
        with open(cuda_h, 'r') as f:
            cuda_h = f.read()
        m = re.search('#define CUDA_VERSION ([0-9]*)', cuda_h)
        if m:
            ver = int(m.group(1))
        else:
            raise RuntimeError("cannot parse CUDA_VERSION")
    except:
        raise
    else:
        # 12020 -> "12.2"
        return str(ver // 1000) + '.' + str((ver % 100) // 10)


# We support CUDA 12/13 starting 25.09
cuda_ver = check_cuda_version()
print("\n"+"*"*80)
print("CUDA version:", cuda_ver)
print("CUDA path:", cuda_path)
print("*"*80+"\n")

if '12.0' <= cuda_ver < '13.0':
    cuda_major_ver = '12'
elif '13.0' <= cuda_ver < '14.0':
    cuda_major_ver = '13'
else:
    raise RuntimeError(f"Unsupported CUDA version: {cuda_ver}")


class build_ext(_build_ext):

    def build_extension(self, ext):
        # The bindings do not link to any cuQuantum/CUDA DSO; the cuQuantum C
        # libraries are located and loaded at runtime via cuda-pathfinder (by
        # absolute path), and each of those libraries already carries its own
        # RUNPATH for its transitive dependencies. No rpath/link flags are
        # needed on the binding modules.
        ext.include_dirs = (os.path.join(cuda_path, 'include'),)
        super().build_extension(ext)

    def build_extensions(self):
        self.parallel = 4  # use 4 threads
        super().build_extensions()
