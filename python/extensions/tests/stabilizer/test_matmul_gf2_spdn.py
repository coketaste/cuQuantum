# Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Correctness tests for cuquantum.stabilizer.jax.matmul_gf2_spdn.

The SpDn matmul computes ``C = A @ B`` over GF(2):

- ``A`` is sparse, supplied as CSR. Produced here via
  :func:`jax.experimental.sparse.CSR.fromdense` from an ``(m, k) uint8`` mask.
- ``B`` is bit-packed dense, supplied as ``(k, n // 32) uint32``. Produced
  here from ``(k, n) uint8`` via :func:`numpy.packbits` (little-endian) +
  a uint32 view.
- ``C`` is bit-packed ``(m, n // 32) uint32``; this test unpacks it to
  ``(m, n) uint8`` and compares against a float32-mod-2 reference.
"""

from __future__ import annotations

import subprocess
import sys
import textwrap

import numpy as np
import pytest

jax = pytest.importorskip("jax")
jax.config.update("jax_enable_x64", True)
import jax.numpy as jnp  # noqa: E402
from jax.experimental import sparse  # noqa: E402

from cuquantum.stabilizer.jax import matmul_gf2_spdn  # noqa: E402


def test_lazy_loading():
    """
    Test that cuStabilizer-JAX defers loading its native FFI extension until first use.

    Runs in a subprocess so that `sys.modules` starts clean, regardless of what other
    tests in the same pytest session may have already imported.
    """
    script = textwrap.dedent("""
        import sys
        import cuquantum.stabilizer.jax
        assert "cuquantum.lib.custabilizer_jax" not in sys.modules

        from cuquantum.stabilizer.jax.pysrc._ffi import _register_ffi_targets
        _register_ffi_targets()
        assert "cuquantum.lib.custabilizer_jax" in sys.modules
    """)
    result = subprocess.run([sys.executable, "-c", script], capture_output=True)
    assert result.returncode == 0, result.stderr.decode()


def _reference(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """Compute ``C = A @ B`` over GF(2) via float32 matmul.

    ``a (m, k) uint8``, ``b (k, n) uint8``; returns ``(m, n) uint8``.
    """
    s = a.astype(jnp.float32) @ b.astype(jnp.float32)
    return np.asarray((s % 2).astype(jnp.uint8))


def _rand(rng: np.random.Generator, shape: tuple[int, ...], density: float) -> np.ndarray:
    return (rng.random(shape) < density).astype(np.uint8)


def _unpack(C_packed: jax.Array, m: int, n: int) -> jax.Array:
    """Bit-packed ``(m, n_pad // 32) uint32`` -> ``(m, n) uint8``."""
    shifts = jnp.arange(32, dtype=jnp.uint32)
    expanded = ((C_packed[..., None] >> shifts) & jnp.uint32(1)).astype(jnp.uint8)
    return expanded.reshape(m, -1)[:, :n]


def _pack_a_csr(a_dense: jax.Array) -> tuple[jax.Array, jax.Array]:
    """Convert dense ``(m, k) uint8`` to ``(rowoff, colidx) uint64`` via jax sparse.

    ``sparse.CSR.fromdense`` accepts int8 (not uint8). Indices come out int32;
    the SpDn FFI ABI wants uint64.
    """
    m, k = int(a_dense.shape[0]), int(a_dense.shape[1])
    a_i8 = a_dense.view(jnp.int8) if a_dense.dtype == jnp.uint8 else a_dense.astype(jnp.int8)
    csr = sparse.CSR.fromdense(a_i8, nse=m * k)
    return csr.indptr.astype(jnp.uint64), csr.indices.astype(jnp.uint64)


def _pack_b_dense(b_dense: np.ndarray) -> tuple[np.ndarray, int]:
    """Bit-pack ``(k, n) uint8`` into ``(k, n_pad // 32) uint32``.

    ``n_pad`` is ``n`` rounded up to a multiple of 32; pad bits are zero
    (multiplicative identity in GF(2) matmul). The uint32 view reinterprets
    the little-endian-packed bytes — the FFI ABI wants uint32 words.
    """
    k, n = b_dense.shape
    n_pad = ((n + 31) // 32) * 32
    if n_pad > n:
        b_dense = np.concatenate(
            [b_dense, np.zeros((k, n_pad - n), dtype=np.uint8)], axis=1,
        )
    packed_u8 = np.packbits(np.ascontiguousarray(b_dense), axis=1, bitorder="little")
    return packed_u8.view(np.uint32), n_pad


@pytest.mark.parametrize(
    "name, n, k, m, b_density, a_density",
    [
        ("tiny",        6,    4,    8, 0.5,  0.5),    # n=6, pads to 32
        ("padded_n",    40,   8,   16, 0.3,  0.2),    # n=40, pads to 64
        ("empty_a",     64,   16,  32, 0.2,  0.0),    # a all zero, n divides 32
        ("large",       22528, 43, 1000, 0.12, 0.01), # realistic working point
    ],
)
def test_matmul_gf2_spdn_matches_reference(
    name: str, n: int, k: int, m: int, b_density: float, a_density: float,
) -> None:
    rng = np.random.default_rng(seed=0)
    b = _rand(rng, (k, n), b_density)
    a = _rand(rng, (m, k), a_density)

    packed_h, n_pad = _pack_b_dense(b)
    B_packed_d = jnp.asarray(packed_h, dtype=jnp.uint32)

    A_rowoff, A_colidx = _pack_a_csr(jnp.asarray(a))
    C_packed = matmul_gf2_spdn(
        A_rowoff, A_colidx, B_packed_d, m=m, n=n_pad, k=k,
    )
    got = np.asarray(_unpack(C_packed, m, n))

    expected = _reference(a, b)
    assert got.shape == expected.shape
    np.testing.assert_array_equal(got, expected)


def test_jit_chained_matmul_matches_reference() -> None:
    """Two FFI calls under one ``jax.jit``: ``C2 = A2 @ (A1 @ B)`` over GF(2).

    Checks that the FFI primitive lowers under ``jax.jit`` and that the
    bit-packed output is a valid XLA buffer the next FFI call can consume as
    its dense operand. Single-matmul correctness is already covered by
    :func:`test_matmul_gf2_spdn_matches_reference`; here we exercise the
    FFI-output-as-FFI-input dataflow and multi-call scheduling inside one
    compiled region.
    """
    rng = np.random.default_rng(seed=0)
    n, k, m1, m2 = 40, 8, 16, 12  # n=40 pads to 64
    b = _rand(rng, (k, n), 0.3)
    a1 = _rand(rng, (m1, k), 0.2)
    a2 = _rand(rng, (m2, m1), 0.2)

    packed_h, n_pad = _pack_b_dense(b)
    B_packed_d = jnp.asarray(packed_h, dtype=jnp.uint32)
    A1_rowoff, A1_colidx = _pack_a_csr(jnp.asarray(a1))
    A2_rowoff, A2_colidx = _pack_a_csr(jnp.asarray(a2))

    @jax.jit
    def chained(A1_ro, A1_ci, A2_ro, A2_ci, B):
        C1 = matmul_gf2_spdn(A1_ro, A1_ci, B, m=m1, n=n_pad, k=k)
        # C1 layout (m1, n_pad // 32) uint32 is exactly the FFI's B-operand
        # format for k' = m1, n' = n_pad.
        C2 = matmul_gf2_spdn(A2_ro, A2_ci, C1, m=m2, n=n_pad, k=m1)
        return C2

    C2_packed = chained(A1_rowoff, A1_colidx, A2_rowoff, A2_colidx, B_packed_d)
    got = np.asarray(_unpack(C2_packed, m2, n))

    expected = _reference(a2, _reference(a1, b))
    assert got.shape == expected.shape
    np.testing.assert_array_equal(got, expected)


def test_vmap_matmul_gf2_spdn_sequential() -> None:
    """``jax.vmap`` correctness against the sequential per-element reference.

    Each batch element must share the same static shapes (``m``, ``k``,
    ``n_pad``, and the CSR allocation ``nse = m*k``); the FFI handler reads
    ``nnz = rowoff[m]`` at runtime, so trailing entries in each ``colidx``
    slice are ignored.
    """
    rng = np.random.default_rng(seed=0)
    bsz, m, k, n = 3, 8, 4, 32  # n already a multiple of 32
    n_pad = 32

    a_list = [_rand(rng, (m, k), 0.4) for _ in range(bsz)]
    b_list = [_rand(rng, (k, n), 0.3) for _ in range(bsz)]

    rows, cols = zip(*[_pack_a_csr(jnp.asarray(a)) for a in a_list])
    A_rowoff = jnp.stack(rows)                                                # (bsz, m+1)  uint64
    A_colidx = jnp.stack(cols)                                                # (bsz, m*k)  uint64
    B_packed = jnp.stack([
        jnp.asarray(_pack_b_dense(b)[0], dtype=jnp.uint32) for b in b_list
    ])                                                                        # (bsz, k, n_pad//32) uint32

    batched = jax.vmap(
        lambda r, c, bp: matmul_gf2_spdn(r, c, bp, m=m, n=n_pad, k=k),
        in_axes=(0, 0, 0),
    )
    got_packed = batched(A_rowoff, A_colidx, B_packed)                        # (bsz, m, n_pad//32)

    expected = np.stack([_reference(a, b) for a, b in zip(a_list, b_list)])
    got = np.stack([np.asarray(_unpack(got_packed[i], m, n)) for i in range(bsz)])
    np.testing.assert_array_equal(got, expected)
