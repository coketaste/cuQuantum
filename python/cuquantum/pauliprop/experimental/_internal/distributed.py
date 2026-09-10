# Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES
#
# SPDX-License-Identifier: BSD-3-Clause

"""Internal helpers for configuring distributed cuPauliProp execution."""

from collections.abc import Sequence
from typing import Literal

import cuquantum.bindings.cupauliprop as cupp


ProviderName = Literal["MPI", "NCCL"]


def is_mpi_comm(comm) -> bool:
    try:
        from mpi4py import MPI
    except ImportError:
        return False
    return isinstance(comm, MPI.Comm)


def is_nccl_comm(comm) -> bool:
    try:
        from nccl.core import Communicator
    except ImportError:
        return False
    return isinstance(comm, Communicator)


def is_nvmath_distributed_context(comm) -> bool:
    try:
        from nvmath.distributed import DistributedContext
    except ImportError:
        return False
    return isinstance(comm, DistributedContext)


def is_ptr_and_size(comm) -> bool:
    return (
        isinstance(comm, Sequence)
        and not isinstance(comm, (str, bytes, bytearray))
        and len(comm) == 2
        and all(isinstance(value, int) for value in comm)
    )


def _normalize_provider(provider) -> cupp.DistributedProvider | None:
    if provider is None:
        return None
    if isinstance(provider, cupp.DistributedProvider):
        return provider
    if isinstance(provider, str):
        normalized = provider.upper()
        if normalized == "MPI":
            return cupp.DistributedProvider.MPI
        if normalized == "NCCL":
            return cupp.DistributedProvider.NCCL
    raise ValueError(f"Unknown provider: {provider}. Supported: ['MPI', 'NCCL']")


def _infer_provider(comm) -> cupp.DistributedProvider | None:
    if comm is None:
        return cupp.DistributedProvider.NONE
    if is_mpi_comm(comm):
        return cupp.DistributedProvider.MPI
    if is_nccl_comm(comm):
        return cupp.DistributedProvider.NCCL
    # An nvmath.distributed context supplies an NCCL communicator; it exposes
    # no MPI communicator, so NCCL is the only provider it can serve.
    if is_nvmath_distributed_context(comm):
        return cupp.DistributedProvider.NCCL
    return None


def resolve_provider(comm, provider) -> cupp.DistributedProvider:
    """Resolve an explicit or inferred communicator provider."""
    inferred = _infer_provider(comm)
    resolved = _normalize_provider(provider)

    if resolved is None and inferred is None:
        raise ValueError(
            "provider is required when the communicator type cannot be inferred. "
            "Pass an mpi4py.MPI.Comm, an nccl.core.Communicator, or an "
            "nvmath.distributed context, or set provider='MPI' or 'NCCL' for a "
            "raw pointer / (ptr, size) pair."
        )
    if resolved is None:
        resolved = inferred
    if inferred is not None and inferred != resolved:
        raise ValueError(
            f"communicator type implies provider {inferred!r} but provider={resolved!r} "
            "was given."
        )

    # Both cannot be None after the checks above.
    assert resolved is not None
    return resolved
