# Copyright (c) 2021-2026, NVIDIA CORPORATION & AFFILIATES
#
# SPDX-License-Identifier: BSD-3-Clause

from cuquantum.bindings import custabilizer as custab

from typing import Union
import numpy as np

Array = Union[np.ndarray, "cupy.ndarray", "torch.Tensor"] #noqa: F821
Stream = Union[int, "cupy.cuda.Stream", "torch.cuda.Stream", "cuda.core.Stream"] #noqa: F821

from ._options import Options
from .simulator import FrameSimulator, LeakageFrameSimulator, Circuit
from .pauli_table import PauliTable, PauliFrame
from .bit_matrix import BitMatrixCSR
from .dem_sampling import (
    DEMSampler,
    BitMatrixSampler,
    BitMatrixSparseSampler,
)
from .circuit_converter import DeltakitParserOptions

__all__ = [
    "FrameSimulator",
    "LeakageFrameSimulator",
    "Circuit",
    "Options",
    "PauliTable",
    "PauliFrame",
    "BitMatrixCSR",
    "BitMatrixSampler",
    "BitMatrixSparseSampler",
    "DEMSampler",
    "DeltakitParserOptions",
]
