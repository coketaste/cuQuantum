# Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES
#
# SPDX-License-Identifier: BSD-3-Clause

"""Configuration dataclasses for the State-Fitting (``OperatorAction``) family."""

from dataclasses import dataclass
from typing import Optional

from cuquantum.bindings import cudensitymat as cudm

from .svd import SVDConfig, _build_svd_config_handle
from ._internal.utils import set_config_attribute as _set_config_attribute


__all__ = [
    "ALSConfig",
    "StateFittingApproachLinSolveConfig",
]


@dataclass
class ALSConfig:
    """Configuration for ALS (Alternating Least Squares) split-scope state fitting.

    Args:
        num_sites: Number of sites swept per local update. ``1`` for 1-site
            ALS, ``2`` for 2-site ALS. Defaults to ``1`` when ``None``.
        svd_config: SVD truncation policy for the 2-site sweep
            (:class:`SVDConfig`).
        max_sweeps: Maximum number of full forward+backward sweeps. Defaults
            to ``20`` when ``None``.
        tolerance: Sweep-to-sweep convergence tolerance. Defaults to ``1e-10``
            when ``None``.
    """
    num_sites: Optional[int] = None
    svd_config: Optional[SVDConfig] = None
    max_sweeps: Optional[int] = None
    tolerance: Optional[float] = None


@dataclass
class StateFittingApproachLinSolveConfig:
    """Configuration for the LinSolve state-fitting approach.

    Args:
        tolerance: Orthonormality residual tolerance. Defaults to ``0`` when
            ``None``, resolved to machine epsilon of the compute precision.
    """
    tolerance: Optional[float] = None


def _build_als_config_handle(handle: int, als_config: "ALSConfig") -> int:
    """Materialize a temporary cudensitymatStateFittingScopeSplitALSConfig_t. Caller owns the handle."""
    ptr = cudm.create_state_fitting_scope_split_als_config(handle)
    try:
        _ALS_SCALAR_FIELDS = {
            "num_sites": cudm.StateFittingScopeSplitALSConfigAttribute.FITTING_SPLIT_SCOPE_ALS_NUM_SITES,
            "max_sweeps": cudm.StateFittingScopeSplitALSConfigAttribute.FITTING_SPLIT_SCOPE_ALS_MAX_SWEEPS,
            "tolerance": cudm.StateFittingScopeSplitALSConfigAttribute.FITTING_SPLIT_SCOPE_ALS_TOLERANCE,
        }
        for field_name, enum_val in _ALS_SCALAR_FIELDS.items():
            value = getattr(als_config, field_name)
            if value is None:
                continue
            _set_config_attribute(
                cudm.state_fitting_scope_split_als_config_set_attribute,
                cudm.get_state_fitting_scope_split_als_config_attribute_dtype,
                handle, ptr, enum_val, value,
            )
        if als_config.svd_config is not None:
            if not isinstance(als_config.svd_config, SVDConfig):
                raise TypeError(
                    f"ALSConfig.svd_config must be an SVDConfig instance, got "
                    f"{type(als_config.svd_config).__name__}."
                )
            svd_ptr = _build_svd_config_handle(handle, als_config.svd_config)
            try:
                _set_config_attribute(
                    cudm.state_fitting_scope_split_als_config_set_attribute,
                    cudm.get_state_fitting_scope_split_als_config_attribute_dtype,
                    handle, ptr,
                    cudm.StateFittingScopeSplitALSConfigAttribute.FITTING_SPLIT_SCOPE_ALS_SVD_CONFIG,
                    svd_ptr,
                )
            finally:
                cudm.destroy_svd_config(svd_ptr)
    except Exception:
        cudm.destroy_state_fitting_scope_split_als_config(ptr)
        raise
    return ptr


def _build_lin_solve_config_handle(handle: int, lin_solve_config: "StateFittingApproachLinSolveConfig") -> int:
    """Materialize a temporary cudensitymatStateFittingApproachLinSolveConfig_t. Caller owns the handle."""
    ptr = cudm.create_state_fitting_approach_lin_solve_config(handle)
    try:
        if lin_solve_config.tolerance is not None:
            _set_config_attribute(
                cudm.state_fitting_approach_lin_solve_config_set_attribute,
                cudm.get_state_fitting_approach_lin_solve_config_attribute_dtype,
                handle, ptr,
                cudm.StateFittingApproachLinSolveConfigAttribute.FITTING_APPROACH_LINSOLVE_TOLERANCE,
                lin_solve_config.tolerance,
            )
    except Exception:
        cudm.destroy_state_fitting_approach_lin_solve_config(ptr)
        raise
    return ptr
