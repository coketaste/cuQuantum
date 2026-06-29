# Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
#
# SPDX-License-Identifier: BSD-3-Clause

"""GF(2) sparse-dense matmul ``C = A @ B`` via cuquantum.stabilizer.jax.

Operands:

* ``A``: ``(m, k) uint8`` mask, packed to CSR using
  :func:`jax.experimental.sparse.CSR.fromdense`. The FFI ABI takes
  ``rowoff`` and ``colidx`` as ``uint64`` device arrays.
* ``B``: ``(k, n) uint8`` mask, bit-packed to ``(k, n_pad // 32) uint32``
  via :func:`numpy.packbits` (little-endian) plus a uint32 view, where
  ``n_pad`` is ``n`` rounded up to a multiple of 32.
* ``C``: returned as bit-packed ``(m, n_pad // 32) uint32``; unpacked
  here for comparison against a float32-mod-2 reference.
"""

import numpy as np

import jax
import jax.numpy as jnp
from jax.experimental import sparse

jax.config.update("jax_enable_x64", True)

from cuquantum.stabilizer.jax import matmul_gf2_spdn


def pack_a_csr(a_dense: jax.Array) -> tuple[jax.Array, jax.Array]:
    """Dense ``(m, k) uint8`` -> ``(rowoff, colidx) uint64`` CSR."""
    m, k = int(a_dense.shape[0]), int(a_dense.shape[1])
    a_i8 = a_dense.view(jnp.int8) if a_dense.dtype == jnp.uint8 else a_dense.astype(jnp.int8)
    csr = sparse.CSR.fromdense(a_i8, nse=m * k)
    return csr.indptr.astype(jnp.uint64), csr.indices.astype(jnp.uint64)


def pack_b_dense(b_dense: np.ndarray) -> tuple[np.ndarray, int]:
    """Dense ``(k, n) uint8`` -> ``(k, n_pad // 32) uint32`` bit-packed."""
    k, n = b_dense.shape
    n_pad = ((n + 31) // 32) * 32
    if n_pad > n:
        b_dense = np.concatenate(
            [b_dense, np.zeros((k, n_pad - n), dtype=np.uint8)], axis=1,
        )
    packed_u8 = np.packbits(np.ascontiguousarray(b_dense), axis=1, bitorder="little")
    return packed_u8.view(np.uint32), n_pad


def unpack_c(c_packed: jax.Array, m: int, n: int) -> np.ndarray:
    """Bit-packed ``(m, n_pad // 32) uint32`` -> ``(m, n) uint8``."""
    shifts = jnp.arange(32, dtype=jnp.uint32)
    expanded = ((c_packed[..., None] >> shifts) & jnp.uint32(1)).astype(jnp.uint8)
    return np.asarray(expanded.reshape(m, -1)[:, :n])


def main() -> None:
    rng = np.random.default_rng(seed=0)
    m, k, n = 64, 16, 40           # n=40 will pad to 64
    a_density, b_density = 0.1, 0.3

    a = (rng.random((m, k)) < a_density).astype(np.uint8)
    b = (rng.random((k, n)) < b_density).astype(np.uint8)

    packed_b, n_pad = pack_b_dense(b)
    B_packed = jnp.asarray(packed_b, dtype=jnp.uint32)
    A_rowoff, A_colidx = pack_a_csr(jnp.asarray(a))

    C_packed = matmul_gf2_spdn(A_rowoff, A_colidx, B_packed, m=m, n=n_pad, k=k)
    c = unpack_c(C_packed, m, n)

    # Verify against a float32-mod-2 reference.
    expected = ((a.astype(np.float32) @ b.astype(np.float32)) % 2).astype(np.uint8)
    np.testing.assert_array_equal(c, expected)
    print(f"OK: matmul_gf2_spdn shape={c.shape} matches reference")


if __name__ == "__main__":
    main()
