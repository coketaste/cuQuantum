# Copyright (c) 2025, NVIDIA CORPORATION & AFFILIATES
#
# SPDX-License-Identifier: BSD-3-Clause

try:
    import cupy as cp
except ImportError:
    cp = None

try:
    import torch
    if not torch.cuda.is_available():
        raise ImportError
except ImportError:
    torch = None

import pytest
from nvmath.internal import memory


# Free up cupy/torch memory between test files
@pytest.fixture(scope="module", autouse=True)
def cleanup_between_files():
    yield
    memory.free_reserved_memory()
    if cp is not None:
        cp.get_default_memory_pool().free_all_blocks()
    if torch is not None:
        torch.cuda.empty_cache()


def pytest_collection_modifyitems(items):
    # gradient-parametrized tests must run in the serial (exclusive-GPU) mode.
    import pytest as _pytest
    for item in items:
        callspec = getattr(item, "callspec", None)
        if callspec is not None and callspec.params.get("gradient") in ("all", "random"):
            item.add_marker(_pytest.mark.exclusive_gpu)
