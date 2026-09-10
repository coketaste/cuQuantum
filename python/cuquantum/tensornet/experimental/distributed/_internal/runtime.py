# Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES
#
# SPDX-License-Identifier: BSD-3-Clause

"""Access to the required nvmath.distributed runtime."""

from __future__ import annotations

from nvmath import distributed as _nvmath_distributed


def get_distributed_context():
    """Return the initialized nvmath.distributed context.

    The initialized runtime is the single source of truth for the process
    universe (rank/size for all layout math) and the per-process device. The
    process group must be MPI-backed: the cuTensorNet library handle is
    configured with an MPI communicator, and layout math computed against a
    different kind of process group could not be checked against it.
    """
    distributed_context = _nvmath_distributed.get_context()
    if distributed_context is None:
        raise RuntimeError(
            "nvmath.distributed must be initialized (with an MPI process group) "
            "before using the cuquantum distributed APIs. Refer to "
            "https://docs.nvidia.com/cuda/nvmath-python/latest/distributed-apis/"
            "runtime.html for more information."
        )
    from nvmath.distributed.process_group import MPIProcessGroup

    if not isinstance(distributed_context.process_group, MPIProcessGroup):
        raise RuntimeError(
            "the cuquantum distributed APIs require nvmath.distributed to be "
            "initialized with an MPI process group (an mpi4py communicator); "
            f"got {type(distributed_context.process_group).__name__}"
        )
    return distributed_context


def get_process_group():
    """Return the active process group for layout rank/size queries."""
    return get_distributed_context().process_group


def ballot(local_error: Exception | None, *, what: str = "the distributed operation"):
    """Agree collectively on whether every rank's rank-local pre-checks
    succeeded, so a failure strands no rank in a collective call its peer
    never reaches. Opt-in only
    (``DistributedContractionOptions.collective_error_agreement=True``) --
    see ``DistributedBinaryContraction``'s docstring for why this isn't the
    default.

    ``plan()`` calls several collective native functions (descriptor
    creation, Create, Prepare); the Python-side validation that runs before
    them (handle/device/dtype checks, output allocation, shape checks) is
    rank-local. Without this, a rank whose local operand happens to be
    malformed raises and never reaches those collective calls, while a peer
    with a well-formed operand proceeds into one and blocks forever waiting
    for the rank that already bailed out.

    Requires ``nvmath.distributed`` to be initialized, exactly like
    ``nvmath.distributed.linalg``'s own solvers (see e.g.
    ``DirectSolver``'s ``_verify_and_snapshot_context``, which raises the
    same way) -- unlike ``runtime.get_process_group()`` elsewhere in this
    package, there is deliberately NO fallback to ``MPI.COMM_WORLD`` here:
    that fallback can silently resolve a DIFFERENT (super/sub-)set of ranks
    than whatever communicator is actually configured on the cutensornet
    handle via ``distributed_reset_configuration``, which would make the
    ballot itself hang rather than the failure it's meant to prevent.

    If every rank is ok, returns normally. If this rank itself failed
    locally, re-raises its OWN exception (preserving its type/message,
    e.g. ValueError/TypeError) once every rank has agreed to stop -- a
    peer that only fails because ANOTHER rank failed has no original
    exception of its own, so it gets a generic collective RuntimeError
    instead.
    """
    distributed_ctx = _nvmath_distributed.get_context()
    if distributed_ctx is None:
        raise RuntimeError(
            "options.collective_error_agreement=True requires the nvmath.distributed "
            "runtime to be initialized; call nvmath.distributed.initialize(...) first, or "
            "leave collective_error_agreement=False (the default) and handle "
            "rank-divergent failures yourself."
        )
    payload = (local_error is None, str(local_error) if local_error is not None else None)

    def _first_failure(a, b):
        return a if not a[0] else b

    ok, message = distributed_ctx.process_group.allreduce_object(payload, op=_first_failure)
    if ok:
        return
    if local_error is not None:
        raise local_error
    raise RuntimeError(
        f"{what} failed on at least one peer rank "
        f"(collectively reported so every rank raises together): {message}"
    )
