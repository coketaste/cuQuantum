# Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES
#
# SPDX-License-Identifier: BSD-3-Clause

"""
Python wrappers for the cuStabilizer-JAX FFI handlers.
"""

import logging

import jax
import jax.numpy as jnp

from ._backend import get_handle


logger = logging.getLogger("custabilizer-jax._ffi")

_ffi_registered = False


def _register_ffi_targets():
    global _ffi_registered
    if _ffi_registered:
        return

    from cuquantum.bindings._internal import custabilizer as _cust
    _cust._inspect_function_pointers()  # for loading libcustabilizer.so

    from cuquantum.lib import custabilizer_jax
    for _name, _value in custabilizer_jax.registrations().items():
        jax.ffi.register_ffi_target(_name, _value, platform="CUDA")
    _ffi_registered = True


def matmul_gf2_spdn(
    A_rowoff: jax.Array,
    A_colidx: jax.Array,
    B_packed: jax.Array,
    *, m: int, n: int, k: int,
) -> jax.Array:
    """
    GF(2) sparse-dense matmul ``C = A @ B`` via cuStabilizer SpDn FFI.

    Args:
        A_rowoff: ``(m+1,)`` uint64 CSR row offsets of ``A`` (device).
        A_colidx: ``(>= rowoff[m],)`` uint64 CSR column indices of ``A``
            (device). nnz is read from ``A_rowoff[m]``; trailing entries are
            ignored.
        B_packed: ``(k, n // 32)`` uint32 bit-packed dense matrix (device).
        m: rows of ``A``/``C``.
        n: columns of ``B``/``C``; must be a multiple of 32.
        k: columns of ``A`` / rows of ``B``.

    Returns:
        ``(m, n // 32)`` uint32 bit-packed device tensor.

    See Also:
        For computing ``(A_rowoff, A_colidx)`` from a dense ``A``:

        * :func:`jax.experimental.sparse.CSR.fromdense` -- jax-native.
        * :func:`cupyx.scipy.sparse.csr_matrix`.
        * :mod:`nvmath.bindings.cusparse` ``cusparseDenseToSparse_*``.
    """
    logger.info("Calling matmul_gf2_spdn")
    if not jax.config.jax_enable_x64:
        raise RuntimeError("jax_enable_x64 must be set to True to use cuQuantum Python JAX")
    handle = get_handle()
    _register_ffi_targets()
    if n % 32 != 0:
        raise ValueError(f"n must be a multiple of 32, got {n}")
    n_words = n // 32
    return jax.ffi.ffi_call(
        "cust_spdn_matmul_gf2",
        jax.ShapeDtypeStruct((m, n_words), jnp.uint32),
        vmap_method="sequential",
    )(
        A_rowoff, A_colidx, B_packed,
        m=int(m), n=int(n), k=int(k),
        handle=int(handle),
    )
