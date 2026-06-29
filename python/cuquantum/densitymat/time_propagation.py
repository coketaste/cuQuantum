# Copyright (c) 2025-2026, NVIDIA CORPORATION & AFFILIATES
#
# SPDX-License-Identifier: BSD-3-Clause

"""Time propagation of quantum states."""

from typing import Optional, Sequence, Union
from dataclasses import dataclass
import weakref
import collections

import numpy as np
import cupy as cp

from nvmath.internal import utils as nvmath_utils
from nvmath.internal.tensor_wrapper import wrap_operand

from cuquantum.bindings import cudensitymat as cudm
from .operators import Operator, _handle_callback_params
from .state import State
from .svd import SVDConfig, _build_svd_config_handle
from .work_stream import WorkStream
from ._internal import utils
from ._internal.utils import (
    InvalidObjectState,
    check_and_get_batchsize,
    resolve_enum as _resolve_enum,
    set_config_attribute as _set_config_attribute,
)
from ._internal.typemaps import CUDENSITYMAT_COMPUTE_TYPE_MAP


__all__ = [
    "TimePropagationApproachKrylovConfig",
    "TDVPConfig",
    "TimePropagation",
]


SCOPE_KIND_MAP = {
    "split": cudm.TimePropagationScopeKind.PROPAGATION_SCOPE_SPLIT,
}

APPROACH_KIND_MAP = {
    "krylov": cudm.TimePropagationApproachKind.PROPAGATION_APPROACH_KRYLOV,
}


# ---------------------------------------------------------------------------
# Configuration dataclasses
# ---------------------------------------------------------------------------

@dataclass
class TimePropagationApproachKrylovConfig:
    """Configuration for the Krylov subspace time propagation approach.

    Args:
        tolerance: Convergence tolerance. Defaults to ``0`` when ``None``,
            resolved to machine epsilon of the compute precision.
        max_dim: Maximum Krylov subspace dimension. Defaults to ``30`` when ``None``.
        min_beta: Minimum beta to proceed with expansion. Defaults to ``0``
            when ``None``, resolved to machine epsilon of the compute
            precision.
        adaptive_step_size: Enable adaptive step size control (``0`` =
            disabled, ``1`` = enabled). Defaults to ``1`` (enabled) when
            ``None``.
    """
    tolerance: Optional[float] = None
    max_dim: Optional[int] = None
    min_beta: Optional[float] = None
    adaptive_step_size: Optional[int] = None


@dataclass
class TDVPConfig:
    """Configuration for TDVP (Time-Dependent Variational Principle) split propagation.

    Args:
        order: Order of TDVP sweeps (``2`` or ``4``). Defaults to ``2`` when
            ``None``.
        num_sites: Number of sites swept per local update. ``1`` for 1-site
            TDVP, ``2`` for 2-site TDVP. Defaults to ``1`` when ``None``.
        svd_config: SVD truncation policy for the 2-site sweep
            (:class:`SVDConfig`).
    """
    order: Optional[int] = None
    num_sites: Optional[int] = None
    svd_config: Optional[SVDConfig] = None


# ---------------------------------------------------------------------------
# TimePropagation
# ---------------------------------------------------------------------------

class TimePropagation:
    """
    TimePropagation(operator, is_hermitian=True, scope="split", approach="krylov", scope_config=None, approach_config=None)

    Time propagation of a quantum state under the action of an operator.

    The propagation advances pure states according to the Schrodinger equation and
    mixed states according to the Liouvillian equation.

    Args:
        operator: The :class:`Operator` governing the time evolution.
        is_hermitian: Whether the operator is Hermitian.
        scope: Propagation scope. Accepts ``"split"`` or
            a ``cudm.TimePropagationScopeKind`` enum value.
        approach: Propagation approach. Accepts ``"krylov"`` or
            a ``cudm.TimePropagationApproachKind`` enum value.
        scope_config: Optional scope-specific configuration
            (:class:`TDVPConfig`).
        approach_config: Optional approach-specific configuration
            (:class:`TimePropagationApproachKrylovConfig`).
    """

    def __init__(
        self,
        operator: Operator,
        is_hermitian: bool = True,
        scope: Union[str, "cudm.TimePropagationScopeKind"] = "split",
        approach: Union[str, "cudm.TimePropagationApproachKind"] = "krylov",
        scope_config: Union[TDVPConfig, None] = None,
        approach_config: Union[TimePropagationApproachKrylovConfig, None] = None,
    ) -> None:
        self._finalizer = weakref.finalize(self, lambda: None)
        self._finalizer.detach()

        self.operator = operator
        self._is_hermitian = is_hermitian
        self._scope_kind = _resolve_enum(scope, SCOPE_KIND_MAP, "scope")
        self._approach_kind = _resolve_enum(approach, APPROACH_KIND_MAP, "approach")

        self._scope_config = scope_config
        self._approach_config = approach_config

        self._ctx: Optional[WorkStream] = None
        self._ptr = None
        self._last_compute_event: Optional[cp.cuda.Event] = None
        self._upstream_finalizers = collections.OrderedDict()
        self._requires_configuration = scope_config is not None or approach_config is not None
        self._current_compute_type = None

    # ------------------------------------------------------------------
    # Valid-state checks
    # ------------------------------------------------------------------

    @property
    def _valid_state(self):
        return self._finalizer.alive

    def _check_valid_state(self, *args, **kwargs):
        if not self._valid_state:
            raise InvalidObjectState(
                "The TimePropagation instance cannot be used after resources are freed."
            )

    @property
    @nvmath_utils.precondition(_check_valid_state)
    def _validated_ptr(self):
        return self._ptr

    def _sync(self):
        if self._last_compute_event is not None:
            self._last_compute_event.synchronize()
            self._last_compute_event = None

    @property
    def _batch_size(self) -> int:
        return self.operator._batch_size

    # ------------------------------------------------------------------
    # Configuration
    # ------------------------------------------------------------------

    def configure(
        self,
        scope_config: Union[TDVPConfig, None] = None,
        approach_config: Union[TimePropagationApproachKrylovConfig, None] = None,
    ) -> None:
        """
        Update the propagation configuration.

        :meth:`prepare` must be invoked before the next :meth:`compute` call.

        Args:
            scope_config: Scope-specific configuration (:class:`TDVPConfig`).
            approach_config: Approach-specific configuration
                (:class:`TimePropagationApproachKrylovConfig`).
        """
        if scope_config is not None:
            self._scope_config = scope_config
        if approach_config is not None:
            self._approach_config = approach_config
        if scope_config is None and approach_config is None:
            return
        if self._valid_state:
            self._apply_configs()
        else:
            self._requires_configuration = True

    # ------------------------------------------------------------------
    # Internal: config application
    # ------------------------------------------------------------------

    def _configure_tp_attribute(self, attr_enum, config_ptr):
        """Pass a config handle to the time propagation object via time_propagation_configure."""
        handle = self._ctx._handle._validated_ptr
        dtype = cudm.get_time_propagation_attribute_dtype(attr_enum)
        val_arr = np.array([config_ptr], dtype=dtype)
        cudm.time_propagation_configure(
            handle, self._ptr, attr_enum, val_arr.ctypes.data, val_arr.dtype.itemsize
        )

    def _apply_configs(self):
        """Create temporary C config objects, apply them to the time propagation, and destroy them."""
        handle = self._ctx._handle._validated_ptr

        # --- Scope (TDVP) config ---
        if self._scope_config is not None:
            if not isinstance(self._scope_config, TDVPConfig):
                raise TypeError(
                    f"scope_config must be a TDVPConfig instance, got "
                    f"{type(self._scope_config).__name__}."
                )
            tdvp_ptr = cudm.create_time_propagation_scope_split_tdvp_config(handle)
            try:
                _TDVP_SCALAR_FIELDS = {
                    "order": cudm.TimePropagationScopeSplitTDVPConfigAttribute.PROPAGATION_SPLIT_SCOPE_TDVP_ORDER,
                    "num_sites": cudm.TimePropagationScopeSplitTDVPConfigAttribute.PROPAGATION_SPLIT_SCOPE_TDVP_NUM_SITES,
                }
                for field_name, enum_val in _TDVP_SCALAR_FIELDS.items():
                    value = getattr(self._scope_config, field_name)
                    if value is not None:
                        _set_config_attribute(
                            cudm.time_propagation_scope_split_tdvp_config_set_attribute,
                            cudm.get_time_propagation_scope_split_tdvp_config_attribute_dtype,
                            handle, tdvp_ptr, enum_val, value,
                        )
                # SVDConfig nested attach 
                if self._scope_config.svd_config is not None:
                    if not isinstance(self._scope_config.svd_config, SVDConfig):
                        raise TypeError(
                            f"TDVPConfig.svd_config must be an SVDConfig instance, got "
                            f"{type(self._scope_config.svd_config).__name__}."
                        )
                    svd_ptr = _build_svd_config_handle(handle, self._scope_config.svd_config)
                    try:
                        _set_config_attribute(
                            cudm.time_propagation_scope_split_tdvp_config_set_attribute,
                            cudm.get_time_propagation_scope_split_tdvp_config_attribute_dtype,
                            handle, tdvp_ptr,
                            cudm.TimePropagationScopeSplitTDVPConfigAttribute.PROPAGATION_SPLIT_SCOPE_TDVP_SVD_CONFIG,
                            svd_ptr,
                        )
                    finally:
                        cudm.destroy_svd_config(svd_ptr)

                self._configure_tp_attribute(
                    cudm.TimePropagationAttribute.PROPAGATION_SPLIT_SCOPE_TDVP_CONFIG, tdvp_ptr,
                )
            finally:
                cudm.destroy_time_propagation_scope_split_tdvp_config(tdvp_ptr)

        # --- Approach (Krylov) config ---
        if self._approach_config is not None:
            if not isinstance(self._approach_config, TimePropagationApproachKrylovConfig):
                raise TypeError(
                    f"approach_config must be a TimePropagationApproachKrylovConfig "
                    f"instance, got {type(self._approach_config).__name__}."
                )
            krylov_ptr = cudm.create_time_propagation_approach_krylov_config(handle)
            try:
                _KRYLOV_FIELDS = {
                    "tolerance": cudm.TimePropagationApproachKrylovConfigAttribute.PROPAGATION_APPROACH_KRYLOV_TOLERANCE,
                    "max_dim": cudm.TimePropagationApproachKrylovConfigAttribute.PROPAGATION_APPROACH_KRYLOV_MAX_DIM,
                    "min_beta": cudm.TimePropagationApproachKrylovConfigAttribute.PROPAGATION_APPROACH_KRYLOV_MIN_BETA,
                    "adaptive_step_size": cudm.TimePropagationApproachKrylovConfigAttribute.PROPAGATION_APPROACH_KRYLOV_ADAPTIVE_STEP_SIZE,
                }
                for field_name, enum_val in _KRYLOV_FIELDS.items():
                    value = getattr(self._approach_config, field_name)
                    if value is not None:
                        _set_config_attribute(
                            cudm.time_propagation_approach_krylov_config_set_attribute,
                            cudm.get_time_propagation_approach_krylov_config_attribute_dtype,
                            handle, krylov_ptr, enum_val, value,
                        )
                self._configure_tp_attribute(
                    cudm.TimePropagationAttribute.PROPAGATION_APPROACH_KRYLOV_CONFIG, krylov_ptr,
                )
            finally:
                cudm.destroy_time_propagation_approach_krylov_config(krylov_ptr)

        self._requires_configuration = False

    # ------------------------------------------------------------------
    # Instantiation
    # ------------------------------------------------------------------

    def _maybe_instantiate(self, ctx: WorkStream) -> None:
        if self._valid_state:
            if self._ctx != ctx:
                raise ValueError(
                    "TimePropagation objects can only be used with a single WorkStream. "
                    "Switching WorkStream is not supported."
                )
        else:
            self._ctx = ctx
            self.operator._maybe_instantiate(ctx)

            self._ptr = cudm.create_time_propagation(
                self._ctx._handle._validated_ptr,
                self.operator._validated_ptr,
                int(self._is_hermitian),
                self._scope_kind,
                self._approach_kind,
            )

            # Apply stored configs (creates temporary C config handles, applies, destroys)
            self._apply_configs()

            self._finalizer = weakref.finalize(
                self,
                utils.generic_finalizer,
                self._ctx.logger,
                self._upstream_finalizers,
                (cudm.destroy_time_propagation, self._ptr),
                msg=f"Destroying TimePropagation instance {self}, ptr: {self._ptr}",
            )
            utils.register_with(self, self._ctx, self._ctx.logger)
            utils.register_with(self, self.operator, self._ctx.logger)

    # ------------------------------------------------------------------
    # Prepare
    # ------------------------------------------------------------------

    def prepare(
        self,
        ctx: WorkStream,
        state_in: State,
        state_out: State,
        compute_type: Optional[str] = None,
    ) -> None:
        """
        Prepare the time propagation for computation.

        Args:
            ctx: Library context containing workspace, stream and other configuration.
            state_in: Representative input quantum state.
            state_out: Representative output quantum state.
            compute_type: CUDA compute type string (e.g. ``"complex128"``).
                Defaults to ``ctx.compute_type`` when set, otherwise to the
                operator's dtype.
        """
        if not self._valid_state:
            self._maybe_instantiate(ctx)
        else:
            if self._ctx != ctx:
                raise ValueError(
                    "TimePropagation objects can only be used with a single WorkStream. "
                    "Switching WorkStream is not supported."
                )

        if self._requires_configuration:
            self._apply_configs()

        if self.operator.hilbert_space_dims != state_in.hilbert_space_dims:
            raise ValueError(
                f"Hilbert space dimensions of Operator, {self.operator.hilbert_space_dims}, "
                f"and input State, {state_in.hilbert_space_dims}, do not match."
            )
        if self.operator.hilbert_space_dims != state_out.hilbert_space_dims:
            raise ValueError(
                f"Hilbert space dimensions of Operator, {self.operator.hilbert_space_dims}, "
                f"and output State, {state_out.hilbert_space_dims}, do not match."
            )

        default_compute_type = (
            self._ctx.compute_type if self._ctx.compute_type is not None else self.operator.dtype
        )
        self._current_compute_type = compute_type if compute_type else default_compute_type

        cudm.time_propagation_prepare(
            self._ctx._handle._validated_ptr,
            self._ptr,
            state_in._validated_ptr,
            state_out._validated_ptr,
            CUDENSITYMAT_COMPUTE_TYPE_MAP[self._current_compute_type],
            self._ctx._memory_limit,
            self._ctx._validated_ptr,
            0,
        )
        self._work_size, _ = self._ctx._update_required_size_upper_bound()

    # ------------------------------------------------------------------
    # Compute
    # ------------------------------------------------------------------

    def compute(
        self,
        dt: Union[float, complex, tuple],
        t: float,
        params: Union[cp.ndarray, np.ndarray, Sequence[float], None],
        state_in: State,
        state_out: State,
    ) -> None:
        """
        Compute the time propagation of a quantum state.

        Args:
            dt: Time step. Accepts a real ``float``, a ``complex`` number, or a
                ``(real, imag)`` tuple. The real and imaginary parts are passed
                separately to the C API.
            t: Current time value passed to all callback functions.
            params: Additional callback parameters. Element type must be ``float64``.
            state_in: Input quantum state.
            state_out: Output quantum state (will be overwritten with propagated result).
        """
        if self._ctx is None:
            raise RuntimeError(
                "This instance has not been used with a WorkStream. "
                "Call its ``prepare`` method once before calls to ``compute``."
            )
        _ = self._validated_ptr
        if self._ctx != state_in._ctx:
            raise ValueError(
                "This TimePropagation's WorkStream and the input State's WorkStream do not match."
            )
        if self._ctx != state_out._ctx:
            raise ValueError(
                "This TimePropagation's WorkStream and the output State's WorkStream do not match."
            )

        # Decompose dt into real and imaginary parts
        if isinstance(dt, tuple):
            dt_real, dt_imag = float(dt[0]), float(dt[1])
        elif isinstance(dt, complex):
            dt_real, dt_imag = dt.real, dt.imag
        else:
            dt_real, dt_imag = float(dt), 0.0

        # Re-prepare (like Operator.compute_action does)
        self.prepare(self._ctx, state_in, state_out, self._current_compute_type)

        batch_size = check_and_get_batchsize(self._batch_size, state_in.batch_size)
        batch_size = check_and_get_batchsize(batch_size, state_out.batch_size)

        self._ctx._maybe_allocate()

        with nvmath_utils.device_ctx(self._ctx.device_id), utils.cuda_call_ctx(self._ctx) as (
            self._last_compute_event,
            elapsed,
        ):
            params, num_params, _params_batch_size = _handle_callback_params(params, batch_size)
            params_ptr = wrap_operand(params).data_ptr

            # Update last event on downstream objects for proper synchronization
            self._ctx._last_compute_event = self._last_compute_event
            state_in._last_compute_event = self._last_compute_event
            state_out._last_compute_event = self._last_compute_event
            self.operator._update_last_compute_event_downstream(self._last_compute_event)

            cudm.time_propagation_compute(
                self._ctx._handle._validated_ptr,
                self._ptr,
                dt_real,
                dt_imag,
                t,
                batch_size,
                num_params,
                params_ptr,
                state_in._validated_ptr,
                state_out._validated_ptr,
                self._ctx._validated_ptr,
                self._ctx._stream_holder.ptr,
            )
