# Copyright (c) 2025, NVIDIA CORPORATION & AFFILIATES
#
# SPDX-License-Identifier: BSD-3-Clause

import glob
import os
import re

import pytest

from ..helpers import run_sample

sample_files = []
for sub_directory in ('pauliprop', 'bindings/cupauliprop'):
    samples_path = os.path.abspath(os.path.join(
        os.path.dirname(__file__), '..', '..', '..', 'samples', sub_directory))
    sample_files += glob.glob(samples_path+'*/*.py', recursive=True)

# Handle MPI and NCCL tests separately.
mpi_re = r".*_mpi[_]?.*\.py"
sample_files = list(filter(lambda f: not re.search(mpi_re, f), sample_files))

@pytest.mark.parametrize("sample", sample_files)
class TestcuPauliPropSamples:

    def test_sample(self, sample):
        run_sample(samples_path, sample)
