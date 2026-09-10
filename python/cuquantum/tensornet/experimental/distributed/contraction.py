# Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES
#
# SPDX-License-Identifier: BSD-3-Clause

"""Experimental distributed binary tensor contraction."""

from __future__ import annotations

import logging
import warnings
from dataclasses import dataclass, replace as _dataclass_replace

import numpy
from nvmath import memory
from nvmath.internal import typemaps, utils as nvmath_utils
from nvmath.internal.tensor_wrapper import wrap_operand

from cuquantum.bindings import cutensornet as cutn
from cuquantum.memory import MemoryLimitExceeded

from ..._internal import decomposition_utils
from ...configuration import NetworkOptions
from ._internal import contraction_utils as _contraction_utils
from ._internal import runtime as _runtime
from ._internal import tensor_utils as _tensor_utils
from ._internal.tensor_utils import (
    allocate_distributed as _allocate_distributed,
    create_distributed_tensor_descriptor as _create_distributed_tensor_descriptor,
    create_distributed_tensor_descriptor_from_spec as _create_distributed_tensor_descriptor_from_spec,
)
from .base import BlockCyclic
from .tensor import DistributedTensor


__all__ = ["DistributedBinaryContraction", "DistributedContractionOptions", "distributed_binary_contraction"]


@dataclass
class DistributedContractionOptions(NetworkOptions):
    """:class:`~cuquantum.tensornet.configuration.NetworkOptions`, plus
    options specific to :class:`DistributedBinaryContraction`.

    Attributes:
        collective_error_agreement: If ``True``, :meth:`DistributedBinaryContraction.plan`
            agrees rank-local pre-check failures across ranks before collective
            planning begins. Requires initialized ``nvmath.distributed``.
            When ``False`` (the default), rank-local failures raise immediately.
    """

    collective_error_agreement: bool = False


def _require_bound(tensor: DistributedTensor, name: str) -> tuple[int, ...]:
    if not isinstance(tensor, DistributedTensor):
        raise TypeError(f"{name} must be a DistributedTensor")
    return tensor.global_shape


def _operand_layout_fingerprint(tensor: DistributedTensor) -> dict:
    """Return the operand metadata that must remain fixed across resets."""
    wrapped = wrap_operand(tensor.local)
    return {
        "global_shape": tuple(tensor.global_shape),
        "local_shape": tuple(wrapped.shape),
        "element_strides": tuple(wrapped.strides),
        "dtype": wrapped.dtype,
        "distribution": tensor.distribution,
    }


class DistributedBinaryContraction:
    """Stateful distributed binary contraction :math:`D = \\alpha A B + \\beta C`.

    Construct the problem, call :meth:`plan`, then call :meth:`execute` one or
    more times. Release resources with :meth:`free` or a context manager.

    Inputs are :class:`DistributedTensor` objects. ``out`` is a
    :class:`BlockCyclic` distribution describing the output placement; output
    storage is allocated freshly by each :meth:`execute` call.

    Currently, this stateful form requires compact Fortran-order local input
    shards and does not stage incompatible layouts. Allocate shards with
    :func:`~cuquantum.tensornet.experimental.distributed.empty_local` or use
    ``asfortranarray``. The single-use :func:`distributed_binary_contraction`
    stages incompatible inputs automatically.

    Operand layouts are fixed at construction. Replacements supplied through
    :meth:`reset_operands` must have matching global and local shapes, dtype,
    element strides, and distribution.

    .. warning:: This API is experimental and subject to future changes.

    .. note::
        :meth:`plan` performs Python-side validation before entering collective operations.
        Pass consistent arguments on every rank, or set
        ``options.collective_error_agreement=True`` to make these pre-checks
        fail consistently across ranks.

    Args:
        expr: Explicit binary einsum expression with two inputs and one output.
        a: First distributed input tensor.
        b: Second distributed input tensor.
        c: Optional distributed addend with the output's global shape.
        out: Distribution describing the output placement.
        qualifiers: Reserved; currently unsupported.
        stream: Accepted for interface consistency; construction performs no
            device work. Pass a stream to :meth:`plan` and :meth:`execute`.
        options: A :class:`DistributedContractionOptions` object or ``dict``;
            ``options.handle`` is required by :meth:`plan`.
        execution: Reserved; currently unsupported.
    """

    def __init__(
        self,
        expr,
        a: DistributedTensor,
        b: DistributedTensor,
        *,
        c: DistributedTensor | None = None,
        out: BlockCyclic,
        qualifiers=None,
        stream=None,
        options=None,
        execution=None,
    ):
        warnings.warn(
            "DistributedBinaryContraction is an experimental API and subject to "
            "future changes",
            stacklevel=2,
        )
        if qualifiers is not None or execution is not None:
            raise NotImplementedError(
                "qualifiers and execution are not supported by "
                "DistributedBinaryContraction"
            )
        # Construction performs no device work.
        del stream

        # The output argument describes placement; execute() allocates its storage.
        if not isinstance(out, BlockCyclic):
            raise TypeError(
                f"out must be a BlockCyclic distribution (got "
                f"{type(out).__name__}); the output is always freshly "
                "allocated in a compact Fortran-order layout — pre-allocated "
                "output tensors are not accepted"
            )
        self._expr = expr
        self._a = a
        self._b = b
        self._c = c
        self._out_spec = out
        self._out: DistributedTensor | None = None
        # Preserve whether this problem includes an addend across releases.
        self._c_provided = c is not None
        self._planned = False
        self._operands_released = False
        # Resolve per-instance defaults without mutating caller-owned options.
        self._options = _dataclass_replace(
            nvmath_utils.check_or_create_options(
                DistributedContractionOptions, options, "distributed contraction options"
            )
        )

        self._a_global_shape = _require_bound(a, "a")
        self._b_global_shape = _require_bound(b, "b")
        if c is not None:
            c_shape = _require_bound(c, "c")
        else:
            c_shape = None

        self._input_modes, self._output_modes = _contraction_utils.parse_binary_expr(expr)
        self._output_global_shape = _contraction_utils.output_global_shape(
            self._input_modes, self._output_modes, self._a_global_shape, self._b_global_shape
        )
        if c_shape is not None and c_shape != self._output_global_shape:
            raise ValueError(
                f"c global_shape {c_shape} does not match output shape "
                f"{self._output_global_shape}"
            )

        # Native resources owned by the planned operation.
        self._contraction_ptr = None
        self._workspace_desc = None
        self._workspaces = {}
        self._stream_holder = None

        # Keep host scalar storage alive through asynchronous execution.
        self._alpha_buf = None
        self._beta_buf = None

        # Snapshot the layouts enforced by reset_operands().
        self._init_layouts = {
            "a": _operand_layout_fingerprint(a),
            "b": _operand_layout_fingerprint(b),
            "c": None if c is None else _operand_layout_fingerprint(c),
        }

    def _destroy_native(self):
        """Destroy native contraction/workspace resources, if any."""
        if self._stream_holder is not None:
            # Synchronize potentially asynchronous work before releasing resources.
            self._stream_holder.obj.sync()
            self._stream_holder = None
        if self._workspace_desc is not None:
            cutn.destroy_workspace_descriptor(self._workspace_desc)
            self._workspace_desc = None
        self._workspaces = {}
        if self._contraction_ptr is not None:
            cutn.destroy_binary_tensor_contraction(self._contraction_ptr)
            self._contraction_ptr = None
        self._planned = False

    def _prepare_workspace(self, handle, stream_holder):
        """Prepare within the device-memory budget and allocate scratch."""
        self._workspace_desc = cutn.create_workspace_descriptor(handle)
        memory_limit = nvmath_utils.get_memory_limit_from_device_id(
            self._options.memory_limit, self._options.device_id
        )
        cutn.workspace_set_memory(
            handle, self._workspace_desc, cutn.Memspace.DEVICE, cutn.WorkspaceKind.SCRATCH, 0, memory_limit
        )
        try:
            cutn.binary_tensor_contraction_prepare(handle, self._contraction_ptr, self._workspace_desc)
        except cutn.cuTensorNetError as exc:
            if exc.status == cutn.Status.INSUFFICIENT_WORKSPACE:
                required = cutn.workspace_get_memory_size(
                    handle, self._workspace_desc, cutn.WorksizePref.MIN,
                    cutn.Memspace.DEVICE, cutn.WorkspaceKind.SCRATCH,
                )
                raise MemoryLimitExceeded(memory_limit, required, self._options.device_id) from exc
            raise
        for mem_space in (cutn.Memspace.DEVICE, cutn.Memspace.HOST):
            self._workspaces[mem_space] = decomposition_utils.allocate_and_set_workspace(
                self._options,
                self._workspace_desc,
                cutn.WorksizePref.MIN,
                mem_space,
                cutn.WorkspaceKind.SCRATCH,
                stream_holder,
                task_name="distributed binary tensor contraction",
            )

    def plan(self, *, stream=None):
        """Plan the contraction and allocate workspace storage.

        When ``options.collective_error_agreement=True``, rank-local
        pre-checks are agreed across the initialized distributed process group
        before collective planning begins.

        Args:
            stream: CUDA stream used for workspace setup.
        """
        if self._operands_released or self._a is None or self._b is None:
            raise RuntimeError("operands have been released; call reset_operands first")

        # Replace any prior plan.
        self._destroy_native()
        self._out = None

        # Validate rank-local inputs before entering native collectives.
        try:
            handle = self._options.handle
            if handle is None:
                raise ValueError(
                    "DistributedBinaryContraction requires options.handle; create a "
                    "cutensornet handle and call cutensornet.distributed_reset_configuration "
                    "before calling plan()"
                )
            logger = logging.getLogger() if self._options.logger is None else self._options.logger

            a_w = wrap_operand(self._a.local)
            b_w = wrap_operand(self._b.local)
            for operand_w, name in ((a_w, "a"), (b_w, "b")):
                if operand_w.device != "cuda":
                    raise TypeError(
                        "DistributedBinaryContraction requires device arrays (CuPy) "
                        f"as local shards; {name}'s NumPy local operand is not "
                        "currently supported"
                    )
            if self._c is not None:
                c_w_check = wrap_operand(self._c.local)
                if c_w_check.device != "cuda":
                    raise TypeError(
                        "DistributedBinaryContraction requires device arrays (CuPy) "
                        "as local shards; c's NumPy local operand is not "
                        "currently supported"
                    )
                if c_w_check.dtype != a_w.dtype:
                    raise TypeError(
                        f"c dtype {c_w_check.dtype!r} does not match a's dtype {a_w.dtype!r}"
                    )
            if b_w.dtype != a_w.dtype:
                raise TypeError(f"b dtype {b_w.dtype!r} does not match a's dtype {a_w.dtype!r}")
            package = a_w.name
            dtype_name = a_w.dtype

            device_id = int(a_w.device_id)
            stream_holder = nvmath_utils.get_or_create_stream(device_id, stream, package)
            c_w = None if self._c is None else wrap_operand(self._c.local)

            compute_type = (
                self._options.compute_type
                if self._options.compute_type is not None
                else typemaps.NAME_TO_COMPUTE_TYPE[dtype_name]
            )
            self._scalar_dtype = _contraction_utils.scalar_dtype_for(dtype_name)
            self._alpha_buf = numpy.empty(1, dtype=self._scalar_dtype)
            self._beta_buf = numpy.empty(1, dtype=self._scalar_dtype)

            self._options.device_id = device_id
            self._options.logger = logger
            if self._options.allocator is None:
                self._options.allocator = memory._MEMORY_MANAGER[package](device_id, logger)
        except Exception as exc:
            if self._options.collective_error_agreement:
                _runtime.ballot(exc, what="DistributedBinaryContraction.plan()")
            raise
        if self._options.collective_error_agreement:
            _runtime.ballot(None, what="DistributedBinaryContraction.plan()")

        desc_a = desc_b = desc_c = desc_d = None
        try:
            desc_a = _create_distributed_tensor_descriptor(
                handle, self._a, self._input_modes[0], a_w
            )
            desc_b = _create_distributed_tensor_descriptor(
                handle, self._b, self._input_modes[1], b_w
            )
            desc_d = _create_distributed_tensor_descriptor_from_spec(
                handle,
                self._out_spec,
                self._output_global_shape,
                self._output_modes,
                dtype_name,
            )
            # Without an addend, use the output layout for C.
            desc_c = (
                desc_d
                if self._c is None
                else _create_distributed_tensor_descriptor(
                    handle, self._c, self._output_modes, c_w
                )
            )

            self._contraction_ptr = cutn.create_binary_tensor_contraction(
                handle, desc_a, desc_b, desc_c, desc_d, compute_type
            )
        finally:
            # Layout metadata is snapshotted at creation.
            seen = set()
            for desc in (desc_a, desc_b, desc_c, desc_d):
                if desc is not None and desc not in seen:
                    cutn.destroy_tensor_descriptor(desc)
                    seen.add(desc)

        try:
            self._prepare_workspace(handle, stream_holder)
        except Exception:
            self._destroy_native()
            raise

        self._stream_holder = stream_holder
        self._planned = True

    def execute(self, *, alpha=1.0, beta=None, release_workspace=False, stream=None):
        """Execute the planned contraction.

        Args:
            alpha: Scale factor for the ``A @ B`` product.
            beta: Scale factor for the addend ``c``. Must be set when ``c``
                was specified at construction, and must be left unset (the
                default) otherwise.
            release_workspace: If ``True``, release the workspace memory
                back to the allocator after this call; otherwise it is kept
                for a subsequent :meth:`execute` call.
            stream: CUDA stream used for output allocation and execution.

        Returns:
            The freshly allocated output :class:`DistributedTensor`.
        """
        if self._operands_released or self._a is None or self._b is None:
            raise RuntimeError("operands have been released; call reset_operands first")
        if not self._planned or self._contraction_ptr is None:
            raise RuntimeError("plan() must succeed before execute()")

        if beta is None:
            if self._c is not None:
                raise ValueError("beta must be set when c is specified")
            beta = 0.0
        elif self._c is None:
            raise ValueError("beta can only be set if c is specified")

        handle = self._options.handle
        logger = self._options.logger
        package = wrap_operand(self._a.local).name

        self._alpha_buf[0] = alpha
        self._beta_buf[0] = beta

        a_w = wrap_operand(self._a.local)
        b_w = wrap_operand(self._b.local)

        device_id = int(a_w.device_id)
        stream_holder = nvmath_utils.get_or_create_stream(device_id, stream, package)
        self._stream_holder = stream_holder
        self._out = _allocate_distributed(
            self._out_spec,
            self._output_global_shape,
            dtype=a_w.dtype,
            like=self._a.local,
            stream_holder=stream_holder,
        )
        c_operand = self._c if self._c is not None else self._out
        c_w = wrap_operand(c_operand.local)
        out_w = wrap_operand(self._out.local)
        blocking = self._options.blocking is True
        timing = bool(logger and logger.handlers)

        if self._workspace_desc is None:
            # Recreate workspace released by an earlier execution.
            self._prepare_workspace(handle, stream_holder)

        with nvmath_utils.cuda_call_ctx(stream_holder, blocking, timing):
            cutn.binary_tensor_contraction_compute(
                handle,
                self._contraction_ptr,
                self._alpha_buf.ctypes.data,
                a_w.data_ptr,
                b_w.data_ptr,
                self._beta_buf.ctypes.data,
                c_w.data_ptr,
                out_w.data_ptr,
                self._workspace_desc,
                stream_holder.ptr,
            )

        if release_workspace:
            if self._workspaces.get(cutn.Memspace.HOST) is not None:
                stream_holder.obj.sync()
            cutn.destroy_workspace_descriptor(self._workspace_desc)
            self._workspace_desc = None
            self._workspaces = {}

        return self._out

    def _operand_layout_changed(self, tensor, name: str) -> bool:
        """Return whether an operand differs from its construction-time layout."""
        snapshot = self._init_layouts[name]
        current = _operand_layout_fingerprint(tensor)
        return not _tensor_utils.same_distribution(
            current["distribution"], snapshot["distribution"]
        ) or any(
            current[key] != snapshot[key]
            for key in ("global_shape", "local_shape", "element_strides", "dtype")
        )

    def reset_operands(
        self,
        *,
        a: DistributedTensor | None = None,
        b: DistributedTensor | None = None,
        c: DistributedTensor | None = None,
    ):
        """Attach replacement operands without changing the problem layout.

        Each replacement must match its construction-time global and local
        shapes, dtype, element strides, and distribution. ``c`` may be
        replaced only when the problem was constructed with an addend.

        After :meth:`release_operands`, all originally provided operands must
        be supplied together. The output placement remains fixed.

        Args:
            a: Replacement first input tensor.
            b: Replacement second input tensor.
            c: Replacement addend tensor.
        """
        if self._operands_released:
            missing = a is None or b is None or (self._c_provided and c is None)
            if missing:
                raise ValueError(
                    "After release_operands(), all operands originally "
                    "provided must be provided again."
                )

        def _layout_error(name: str):
            return ValueError(
                f"The operand {name} must have the same layout (global "
                "shape, local shape, dtype, element strides, distribution) "
                "as the one specified during the initialization of the "
                "DistributedBinaryContraction object; create a new "
                "DistributedBinaryContraction for a different layout."
            )

        if a is not None:
            _require_bound(a, "a")
            if self._operand_layout_changed(a, "a"):
                raise _layout_error("a")
            self._a = a
        if b is not None:
            _require_bound(b, "b")
            if self._operand_layout_changed(b, "b"):
                raise _layout_error("b")
            self._b = b
        if c is not None:
            if not self._c_provided:
                raise ValueError(
                    "operand c was not specified during the initialization "
                    "of the DistributedBinaryContraction object and "
                    "therefore cannot be reset to a concrete tensor."
                )
            _require_bound(c, "c")
            if self._operand_layout_changed(c, "c"):
                raise _layout_error("c")
            self._c = c
        self._operands_released = False

    def release_operands(self):
        """Release operand and output references while retaining any prepared plan."""
        self._a = None
        self._b = None
        self._c = None
        self._out = None
        self._operands_released = True

    def free(self):
        """Release all resources associated with this operation."""
        self._destroy_native()
        self.release_operands()
        self._out_spec = None

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_value, traceback):
        self.free()


def distributed_binary_contraction(
    expr,
    a: DistributedTensor,
    b: DistributedTensor,
    *,
    c: DistributedTensor | None = None,
    alpha=1.0,
    beta=None,
    out: BlockCyclic,
    options=None,
    stream=None,
):
    """Evaluate a distributed binary contraction :math:`D = \\alpha A B + \\beta C`.

    This convenience function plans and executes one contraction. To amortize
    planning across repeated executions of the same problem, keep a planned
    :class:`DistributedBinaryContraction` alive and swap operands with
    :meth:`~DistributedBinaryContraction.reset_operands`.

    Currently, non-compact input shards are staged to compact Fortran-order
    copies. Staging performs one rank-local device copy per incompatible
    operand and does not modify caller storage. This matches
    :func:`~cuquantum.tensornet.experimental.distributed.distributed_decompose`'s
    input staging.

    Args:
        expr: The einsum expression (explicit form, exactly two inputs), as
            for :class:`DistributedBinaryContraction`.
        a: The first operand, a :class:`DistributedTensor` with a known
            ``global_shape`` and a device-resident local shard.
        b: The second operand, like ``a``.
        c: Optionally, the addend tensor, like ``a`` with the output's
            global shape.
        alpha: Scale factor for the ``A @ B`` product.
        beta: Scale factor for the addend ``c``. Must be set when ``c`` is
            provided, and must be left unset (the default) otherwise.
        out: The placement of the output, as a :class:`BlockCyclic`
            distribution. The output is always freshly allocated in the
            compact Fortran-order layout through the operands' array package;
            pre-allocated output buffers are not accepted.
        options: A :class:`DistributedContractionOptions` object (or a
            `dict`); ``options.handle`` is required.
        stream: The CUDA stream the operation is ordered on. If not
            provided, the package's current stream is used.

    Returns:
        The freshly allocated output :class:`DistributedTensor`.

    .. warning:: This API is experimental and subject to future changes.
    """

    # Stage inputs before the stateful object fixes their layouts.
    def _staged(tensor, name):
        if tensor is None:
            return None
        _require_bound(tensor, name)
        wrapped = wrap_operand(tensor.local)
        if _tensor_utils.is_canonical_local(tuple(wrapped.shape), wrapped.strides):
            return tensor
        stream_holder = None
        if wrapped.device == "cuda":
            stream_holder = nvmath_utils.get_or_create_stream(
                int(wrapped.device_id), stream, wrapped.name
            )
        staged_local = _tensor_utils.as_canonical_local(
            tensor.local, stream_holder=stream_holder
        )
        return DistributedTensor(
            staged_local, tensor.distribution, tensor.global_shape
        )

    a = _staged(a, "a")
    b = _staged(b, "b")
    c = _staged(c, "c")

    with DistributedBinaryContraction(
        expr, a, b, c=c, out=out, options=options, stream=stream
    ) as contraction:
        contraction.plan(stream=stream)
        return contraction.execute(alpha=alpha, beta=beta, stream=stream)
