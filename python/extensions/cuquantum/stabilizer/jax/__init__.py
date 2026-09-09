# Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
#
# SPDX-License-Identifier: BSD-3-Clause

"""JAX bindings for cuStabilizer linear-algebra primitives.

Public surface:
    matmul_gf2_spdn    GF(2) sparse-dense matmul on pre-packed CSR, returns
                       bit-packed (m, n // 32) uint32
"""

try:
    from ._build_info import check_jax_abi as _check_jax_abi
except ImportError:
    pass
else:
    _check_jax_abi()
    del _check_jax_abi


from .pysrc._ffi import matmul_gf2_spdn

__all__ = [
    "matmul_gf2_spdn",
]
