# Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Process-global cuStabilizer library handle."""

from __future__ import annotations

import atexit
import threading

from cuquantum.bindings import custabilizer as _cust


_lock = threading.Lock()
_handle: int | None = None


def get_handle() -> int:
    """Return the process-global cuStabilizer handle, creating it on first use."""
    global _handle
    if _handle is None:
        with _lock:
            if _handle is None:
                _handle = _cust.create()
    return _handle


def _free_handle() -> None:
    global _handle
    if _handle is not None:
        _cust.destroy(_handle)
        _handle = None


atexit.register(_free_handle)
