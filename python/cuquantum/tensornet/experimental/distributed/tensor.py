# Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES
#
# SPDX-License-Identifier: BSD-3-Clause

"""Experimental distributed tensor operations."""

from __future__ import annotations

import dataclasses
import logging
import warnings
from collections.abc import Sequence

from nvmath import memory
from nvmath.internal import utils as nvmath_utils
from nvmath.internal.tensor_wrapper import wrap_operand

from cuquantum.bindings import cutensornet as cutn

from ..._internal import decomposition_utils
from ...tensor import DecompositionOptions, QRMethod, SVDInfo, SVDMethod
from ._internal import decompose_utils as _decompose_utils
from ._internal import runtime as _runtime
from ._internal import tensor_utils as _tensor_utils
from .base import BlockCyclic, LocalTensorLayout, get_local_layout


__all__ = [
    "DistributedDecompositionOptions",
    "DistributedTensor",
    "distributed_decompose",
    "empty_local",
]


#: SVD algorithm required by the distributed backend.
_DISTRIBUTED_SVD_ALGORITHM = "gesvdp"


class DistributedTensor:
    """Non-owning local shard bundled with a block-cyclic distribution and
    the tensor's global shape.

    The global shape is part of the tensor's specification: a single rank's
    local shard plus the distribution underdetermine it (e.g. a 4-row slab
    shard on rank 0 of 2 is consistent with global extent 7 and 8), so it is
    required at construction — either explicitly, or adopted from an
    already-bound distribution. The local shard is validated against it.

    .. warning:: This API is experimental and subject to future changes.
    """

    def __init__(
        self,
        local,
        distribution: BlockCyclic,
        global_shape: Sequence[int] | None = None,
    ):
        warnings.warn(
            "DistributedTensor is an experimental API and subject to future changes",
            stacklevel=2,
        )
        if not isinstance(distribution, BlockCyclic):
            raise TypeError(
                "distribution must be an experimental distributed.BlockCyclic"
            )
        self._local = local
        self._distribution = distribution.copy()
        if global_shape is None and self._distribution._bound:
            global_shape = tuple(self._distribution._data_global_shape)
        if global_shape is None:
            raise ValueError(
                "global_shape is required: a local shard and an unbound "
                "distribution underdetermine the global tensor"
            )
        global_shape = _tensor_utils.as_shape(global_shape, "global_shape")
        rank = _runtime.get_process_group().rank
        expected = self._distribution.shape(rank, global_shape)
        actual = _tensor_utils.local_shape(self._local)
        if actual != expected:
            raise ValueError(
                f"local shape {actual} on rank {rank} does not match expected "
                f"shape {expected} for global_shape {global_shape}"
            )
        if self._distribution._bound:
            if tuple(self._distribution._data_global_shape) != global_shape:
                raise ValueError(
                    f"distribution already bound to {self._distribution._data_global_shape}, "
                    f"cannot rebind to {global_shape}"
                )
        else:
            self._distribution._bind(global_shape, shape=actual)
        self._global_shape = global_shape

    @property
    def local(self):
        """Local array shard for the current process."""
        return self._local

    @property
    def distribution(self) -> BlockCyclic:
        """Private copy of the distribution associated with this tensor."""
        return self._distribution

    @property
    def global_shape(self) -> tuple[int, ...]:
        """Global logical shape."""
        return self._global_shape

    def layout(self, *, rank: int | None = None) -> LocalTensorLayout:
        """Return the compact local layout for this tensor's global shape."""
        return get_local_layout(
            self._distribution,
            self._global_shape,
            rank=rank,
        )


def empty_local(
    distribution: BlockCyclic,
    global_shape: Sequence[int],
    *,
    dtype,
    like,
    rank: int | None = None,
    stream=None,
):
    """Allocate this rank's local shard in a compact Fortran-order layout.

    .. warning:: This API is experimental and subject to future changes.

    This is the layout the native library currently selects for null element
    strides and accepts without staging. Arrays allocated here can be passed
    directly to :func:`distributed_decompose`; see :func:`get_local_layout`
    to inspect the expected local shape and element strides without allocating.

    Args:
        distribution: The ``BlockCyclic`` distribution of the tensor.
        global_shape: Global extents of the tensor.
        dtype: Element data type for the allocation.
        like: An array of the desired package/device (e.g. a CuPy array on
            the target device).
        rank: Process rank; defaults to the calling process.
        stream: Stream the allocation is ordered on (device arrays only);
            defaults to the package's current stream.
    """
    warnings.warn(
        "empty_local() is an experimental API and subject to future changes",
        stacklevel=2,
    )
    wrapped_like = wrap_operand(like)
    stream_holder = None
    if wrapped_like.device == "cuda":
        ctx = _runtime.get_distributed_context()
        if int(wrapped_like.device_id) != int(ctx.device_id):
            raise ValueError(
                f"like must live on device {ctx.device_id}, the device "
                "nvmath.distributed was initialized with; got device "
                f"{wrapped_like.device_id}"
            )
        stream_holder = nvmath_utils.get_or_create_stream(
            int(wrapped_like.device_id), stream, wrapped_like.name
        )
    return _tensor_utils.allocate_local(
        distribution,
        global_shape,
        dtype=dtype,
        like=like,
        rank=rank,
        stream_holder=stream_holder,
    )

@dataclasses.dataclass
class DistributedDecompositionOptions(DecompositionOptions):
    """:class:`~cuquantum.tensornet.tensor.DecompositionOptions`, plus
    options specific to :func:`distributed_decompose`.

    ``compute_type`` must be ``None``; decomposition precision follows the
    operand data type.

    Attributes:
        collective_error_agreement: If ``True``, :func:`distributed_decompose`
            ballots across ranks on its Python-side pre-checks (operand,
            dtype, layout, and output validation, staging, allocation)
            before any collective native call, so a rank-divergent failure
            raises consistently on every rank instead of stranding a peer
            in a collective call it will never reach. When ``False`` (the
            default), a rank-local failure simply raises immediately -- the
            standard SPMD/MPI contract, where it is the caller's
            responsibility to construct equivalent, valid arguments on
            every rank or to recover (e.g. ``MPI_Abort``).
    """

    collective_error_agreement: bool = False


def _validate_svd_algorithm(method):
    """Reject SVD algorithms other than the one the distributed backend supports."""
    if method.algorithm != _DISTRIBUTED_SVD_ALGORITHM:
        raise ValueError(
            f"distributed_decompose supports only "
            f"algorithm='{_DISTRIBUTED_SVD_ALGORITHM}' (got "
            f"'{method.algorithm}'). Pass SVDMethod(algorithm='{_DISTRIBUTED_SVD_ALGORITHM}', "
            f"...), or a dict without an 'algorithm' key to have it supplied "
            f"automatically."
        )


def distributed_decompose(
    subscripts,
    operand: DistributedTensor,
    *,
    out_left: BlockCyclic,
    out_right: BlockCyclic,
    method=None,
    options=None,
    stream=None,
    return_info=False,
):
    r"""
    Perform QR or SVD decomposition of a distributed tensor based on the expression
    described by ``subscripts``.

    The expression follows the same conventions as
    :func:`cuquantum.tensornet.tensor.decompose`: one input term and two output terms
    sharing exactly one mode. The operand and both factors are distributed: every
    process passes the local shard of the operand and receives the local shards of the
    factors, with each tensor's placement described by a :class:`BlockCyclic`
    distribution.

    Args:
        subscripts: The mode labels (subscripts) defining the decomposition, with the
            same conventions as :func:`cuquantum.tensornet.tensor.decompose`.
        operand: The tensor to decompose, as a :class:`DistributedTensor` with a
            device-resident local shard.
        out_left: The placement of the left factor, as a :class:`BlockCyclic`
            distribution. The output is always freshly allocated: its local shard is
            a compact Fortran-order arrangement of the requested distribution,
            allocated through the operand's array package. Pre-allocated output
            buffers are not accepted.
        out_right: Like ``out_left``, for the right factor. The singular values (SVD
            only) are likewise always freshly allocated.
        method: A :class:`QRMethod` or :class:`SVDMethod` object (or a `dict` of the
            corresponding constructor parameters) selecting the decomposition.
            Defaults to QR. For SVD, only ``"gesvdp"`` is supported. A `dict`
            without an ``algorithm`` key selects it automatically, so
            ``method={'max_extent': 8}`` works. An
            :class:`~cuquantum.tensornet.tensor.SVDMethod` instance must set
            ``algorithm='gesvdp'`` because its default is ``"gesvd"``. Any other
            algorithm raises `ValueError`.
        options: A :class:`DistributedDecompositionOptions` object (or a
            `dict`); ``options.handle`` is required (see the notes below).
        stream: The CUDA stream the operation is ordered on. If not provided, the
            package's current stream is used.
        return_info: If true, also return a :class:`SVDInfo` object describing the
            decomposition. Supported for SVD only.

    Returns:
        Depending on the decomposition method specified in ``method``:

            - QR returns ``(left, right)`` as :class:`DistributedTensor` objects.
            - SVD returns ``(left, s, right)``, where the factors are
              :class:`DistributedTensor` objects and ``s`` is ``None`` if partitioned
              via :attr:`~SVDMethod.partition`. When ``return_info`` is ``True``, an
              :class:`SVDInfo` object is appended to the tuple.

    .. note::
        When value-based truncation reduces the shared extent, the factors and ``s``
        carry the realized (reduced) extent; their local arrays are zero-copy views of
        the capacity-sized allocations (copy them to release the extra memory).

    .. note::
        With ``return_info=True``, :class:`~cuquantum.tensornet.tensor.SVDInfo`
        reports ``algorithm='gesvdp'`` and ``gesvdp_err_sigma=0.0``. On the
        distributed path, ``gesvdp_err_sigma`` is not measured and must not be
        used as a convergence signal; unlike the single-process path, ``0.0``
        does not imply the input was well conditioned.

    .. note::
        Prerequisites: initialize ``nvmath.distributed`` with an MPI process group,
        create a cutensornet handle, and configure it with the same MPI communicator
        via ``cutensornet.distributed_reset_configuration``. The world size and rank
        of the handle's communicator are validated against the process group.

    .. note::
        Local input shards are currently staged when they are not compact
        Fortran-order (allocate with :func:`empty_local`, inspect with
        :func:`get_local_layout`). Staging performs one device-side copy and leaves
        the caller's array unchanged. Outputs use compact Fortran-order layouts.

    .. note::
        With value-based SVD truncation (``abs_cutoff``, ``rel_cutoff``, or
        ``discarded_weight_cutoff``), the shared mode of each output must be
        owned by a single rank or explicitly block-cyclic (``blockSizes > 0``).
        A near-even slab over more than one rank is rejected.

    .. note::
        Validation is rank-local: pass consistent arguments on every process, as an
        argument rejected on only some ranks leaves the other ranks waiting in a
        collective. Set ``options.collective_error_agreement=True`` to instead
        ballot across ranks on the Python-side pre-checks, so a rank-divergent
        failure raises consistently on every rank.

    .. warning:: This API is experimental and subject to future changes.
    """
    warnings.warn(
        "distributed_decompose() is an experimental API and subject to future changes",
        stacklevel=2,
    )
    options = nvmath_utils.check_or_create_options(
        DistributedDecompositionOptions, options, "distributed decomposition options"
    )

    # Validate rank-local inputs before entering native collectives.
    try:
        if not isinstance(operand, DistributedTensor):
            raise TypeError("operand must be a DistributedTensor")

        for method_class in (QRMethod, SVDMethod):
            candidate = method
            if (
                method_class is SVDMethod
                and isinstance(method, dict)
                and "algorithm" not in method
            ):
                # Inject gesvdp only for dicts that omit algorithm, and only
                # after QRMethod has been ruled out (method={} means QR).
                candidate = {**method, "algorithm": _DISTRIBUTED_SVD_ALGORITHM}
            try:
                resolved = nvmath_utils.check_or_create_options(
                    method_class, candidate, method_class.__name__
                )
            except TypeError:
                continue
            else:
                method = resolved
                break
        else:
            raise ValueError(
                "method must be either a QRMethod/SVDMethod object or a dict that "
                "can be used to construct QRMethod/SVDMethod"
            )

        if isinstance(method, SVDMethod):
            _validate_svd_algorithm(method)

        if return_info and not isinstance(method, SVDMethod):
            raise ValueError("return_info is only supported for SVDMethod")

        # Output arguments specify layout; storage is allocated below.
        for arg_name, value in (("out_left", out_left), ("out_right", out_right)):
            if not isinstance(value, BlockCyclic):
                raise TypeError(
                    f"{arg_name} must be a BlockCyclic distribution (got "
                    f"{type(value).__name__}); outputs are always freshly "
                    "allocated by distributed_decompose in a compact Fortran-order layout — "
                    "pre-allocated output tensors are not accepted"
                )

        if options.handle is None:
            raise ValueError(
                "distributed_decompose requires options.handle; create a cutensornet "
                "handle and call cutensornet.distributed_reset_configuration before "
                "calling distributed_decompose"
            )

        inputs, outputs, size_dict, max_mid_extent = (
            _decompose_utils.parse_distributed_decompose(
                subscripts, operand.global_shape
            )
        )

        if isinstance(method, QRMethod):
            mid_extent = max_mid_extent
        else:
            mid_extent = (
                max_mid_extent
                if method.max_extent is None
                else min(max_mid_extent, method.max_extent)
            )

        left_shape, right_shape = _decompose_utils.output_global_shapes(
            outputs, size_dict, mid_extent
        )

        operand_w = wrap_operand(operand.local)
        if operand_w.device != "cuda":
            raise TypeError(
                "distributed_decompose requires device arrays (CuPy) as local "
                "shards; NumPy local operands are not currently supported"
            )
        package = operand_w.name
        if operand_w.dtype not in decomposition_utils.DECOMPOSITION_DTYPE_NAMES:
            raise ValueError(f"dtype {operand_w.dtype} not supported")
        dtype = operand_w.dtype
        device_id = int(operand_w.device_id)
        ctx = _runtime.get_distributed_context()
        if device_id != int(ctx.device_id):
            raise ValueError(
                f"the operand must live on device {ctx.device_id}, the device "
                "nvmath.distributed was initialized with; got device "
                f"{device_id}"
            )
        # Ensure the Python and native distributed runtimes address the same ranks.
        process_group = ctx.process_group
        handle_nranks = cutn.distributed_get_num_ranks(options.handle)
        handle_rank = cutn.distributed_get_proc_rank(options.handle)
        if handle_nranks != process_group.nranks or handle_rank != process_group.rank:
            raise ValueError(
                "the communicator configured on the handle does not match the "
                f"nvmath.distributed process group: the handle spans "
                f"{handle_nranks} rank(s) with this process at rank "
                f"{handle_rank}, the process group spans {process_group.nranks} "
                f"with this process at rank {process_group.rank}; configure the "
                "handle with the same MPI communicator via "
                "distributed_reset_configuration"
            )
        # Use one stream for staging, allocation, and native execution.
        stream_holder = nvmath_utils.get_or_create_stream(device_id, stream, package)
        # Stage noncanonical input without modifying the caller's array.
        if not _tensor_utils.is_canonical_local(
            _tensor_utils.local_shape(operand.local), operand_w.strides
        ):
            operand_w = wrap_operand(
                _tensor_utils.as_canonical_local(
                    operand.local, stream_holder=stream_holder
                )
            )
        # Allocate compact column-major output shards through the operand package.
        left = _tensor_utils.allocate_distributed(
            out_left,
            left_shape,
            dtype=dtype,
            like=operand_w.tensor,
            stream_holder=stream_holder,
        )
        right = _tensor_utils.allocate_distributed(
            out_right,
            right_shape,
            dtype=dtype,
            like=operand_w.tensor,
            stream_holder=stream_holder,
        )
        left_w = wrap_operand(left.local)
        right_w = wrap_operand(right.local)

        s = s_w = None
        if isinstance(method, SVDMethod):
            if method.partition is None:
                s_dtype = _decompose_utils.s_dtype_for(operand_w.dtype)
                s = nvmath_utils.create_empty_tensor(
                    type(operand_w),
                    (mid_extent,),
                    s_dtype,
                    device_id,
                    stream_holder,
                    verify_strides=False,
                ).tensor
                s_w = wrap_operand(s)

        handle = options.handle
        logger = logging.getLogger() if options.logger is None else options.logger
        # Resolve per-call options without mutating the caller's object.
        options = dataclasses.replace(options, device_id=device_id, logger=logger)
        if options.allocator is None:
            options.allocator = memory._MEMORY_MANAGER[package](device_id, logger)
    except Exception as exc:
        if options.collective_error_agreement:
            _runtime.ballot(exc, what="distributed_decompose()")
        raise
    if options.collective_error_agreement:
        _runtime.ballot(None, what="distributed_decompose()")

    input_desc = left_desc = right_desc = workspace_desc = None
    svd_config = svd_info = None
    workspaces = {}
    try:
        input_desc = _tensor_utils.create_distributed_tensor_descriptor(
            handle, operand, inputs[0], operand_w
        )
        left_desc = _tensor_utils.create_distributed_tensor_descriptor(
            handle, left, outputs[0], left_w
        )
        right_desc = _tensor_utils.create_distributed_tensor_descriptor(
            handle, right, outputs[1], right_w
        )

        workspace_desc = cutn.create_workspace_descriptor(handle)
        if isinstance(method, QRMethod):
            cutn.workspace_compute_qr_sizes(
                handle, input_desc, left_desc, right_desc, workspace_desc
            )
        else:
            svd_config = cutn.create_tensor_svd_config(handle)
            decomposition_utils.parse_svd_config(handle, svd_config, method, logger)
            cutn.workspace_compute_svd_sizes(
                handle,
                input_desc,
                left_desc,
                right_desc,
                svd_config,
                workspace_desc,
            )

        for mem_space in (cutn.Memspace.DEVICE, cutn.Memspace.HOST):
            workspaces[mem_space] = decomposition_utils.allocate_and_set_workspace(
                options,
                workspace_desc,
                cutn.WorksizePref.MIN,
                mem_space,
                cutn.WorkspaceKind.SCRATCH,
                stream_holder,
                task_name="distributed tensor decomposition",
            )

        stream_ptr = stream_holder.ptr
        blocking = options.blocking is True
        timing = bool(logger and logger.handlers)
        s_ptr = 0 if s_w is None else s_w.data_ptr

        if isinstance(method, QRMethod):
            with nvmath_utils.cuda_call_ctx(stream_holder, blocking, timing):
                cutn.tensor_qr(
                    handle,
                    input_desc,
                    operand_w.data_ptr,
                    left_desc,
                    left_w.data_ptr,
                    right_desc,
                    right_w.data_ptr,
                    workspace_desc,
                    stream_ptr,
                )
            result = (left, right)
        else:
            svd_info = cutn.create_tensor_svd_info(handle)
            with nvmath_utils.cuda_call_ctx(stream_holder, blocking, timing):
                cutn.tensor_svd(
                    handle,
                    input_desc,
                    operand_w.data_ptr,
                    left_desc,
                    left_w.data_ptr,
                    s_ptr,
                    right_desc,
                    right_w.data_ptr,
                    svd_config,
                    svd_info,
                    workspace_desc,
                    stream_ptr,
                )
            svd_info_obj = SVDInfo(
                **decomposition_utils.get_svd_info_dict(handle, svd_info)
            )
            reduced_extent = int(svd_info_obj.reduced_extent)
            if reduced_extent != mid_extent:
                # Value-based truncation shrank the bond below the planned
                # capacity: rebind outputs to the realized layout before the
                # descriptors are destroyed, and trim s to the kept values.
                left = _decompose_utils.rebind_realized(handle, left_desc, left)
                right = _decompose_utils.rebind_realized(handle, right_desc, right)
                if s is not None:
                    s = s[:reduced_extent]
            if return_info:
                result = (left, s, right, svd_info_obj)
            else:
                result = (left, s, right)
    finally:
        if workspaces.get(cutn.Memspace.HOST) is not None:
            stream_holder.obj.sync()
        if svd_config is not None:
            cutn.destroy_tensor_svd_config(svd_config)
        if svd_info is not None:
            cutn.destroy_tensor_svd_info(svd_info)
        for desc in (input_desc, left_desc, right_desc):
            if desc is not None:
                cutn.destroy_tensor_descriptor(desc)
        if workspace_desc is not None:
            cutn.destroy_workspace_descriptor(workspace_desc)

    return result
