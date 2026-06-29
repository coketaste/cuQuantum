# Copyright (c) 2025-2026, NVIDIA CORPORATION & AFFILIATES
#
# SPDX-License-Identifier: BSD-3-Clause

from cuquantum.bindings._internal import cudensitymat as _cudm
_cudm._inspect_function_pointers()  # for loading libcudensitymat.so

import jax

try:
    from ._build_info import check_jax_abi as _check_jax_abi
except ImportError:
    pass
else:
    _check_jax_abi()
    del _check_jax_abi

if not jax.config.jax_enable_x64:
    raise RuntimeError("jax_enable_x64 must be set to True to use cuQuantum Python JAX")

from .operator_action import operator_action
from .pysrc import (
    ElementaryOperator,
    MatrixOperator,
    OperatorTerm,
    Operator
)
