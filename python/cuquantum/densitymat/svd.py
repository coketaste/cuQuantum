# Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES
#
# SPDX-License-Identifier: BSD-3-Clause

"""SVD truncation configuration shared by 2-site TDVP and 2-site DMRG."""

from dataclasses import dataclass
from typing import Optional

import numpy as np

from cuquantum.bindings import cudensitymat as cudm


__all__ = ["SVDConfig"]


@dataclass
class SVDConfig:
    """A data class for SVD truncation policy used by MPS algorithms.

    Attributes:
        abs_cutoff: Absolute cutoff for singular values. Defaults to ``1e-16``.
        rel_cutoff: Relative cutoff. Defaults to ``1e-14``.
        discarded_weight_cutoff: Discarded-weight cutoff. Defaults to ``0.0``.
        max_extent: Maximum number of singular values retained; ``0`` means no limit
            (truncation governed by the cutoffs). Defaults to ``0``.
    """
    abs_cutoff: Optional[float] = None
    rel_cutoff: Optional[float] = None
    discarded_weight_cutoff: Optional[float] = None
    max_extent: Optional[int] = None


_SVD_FIELDS = (
    ("abs_cutoff", cudm.SVDConfigAttribute.ABS_CUTOFF),
    ("rel_cutoff", cudm.SVDConfigAttribute.REL_CUTOFF),
    ("discarded_weight_cutoff", cudm.SVDConfigAttribute.DISCARDED_WEIGHT_CUTOFF),
    ("max_extent", cudm.SVDConfigAttribute.MAX_EXTENT),
)


def _build_svd_config_handle(handle: int, svd_config: "SVDConfig") -> int:
    """Materialize a temporary cudensitymatSVDConfig_t. Caller owns the handle."""
    ptr = cudm.create_svd_config(handle)
    try:
        for field_name, enum_val in _SVD_FIELDS:
            value = getattr(svd_config, field_name)
            if value is None:
                continue
            dtype = cudm.get_svd_config_attribute_dtype(enum_val)
            val_arr = np.array([value], dtype=dtype)
            cudm.svd_config_set_attribute(
                handle, ptr, enum_val,
                val_arr.ctypes.data, val_arr.dtype.itemsize,
            )
    except Exception:
        cudm.destroy_svd_config(ptr)
        raise
    return ptr
