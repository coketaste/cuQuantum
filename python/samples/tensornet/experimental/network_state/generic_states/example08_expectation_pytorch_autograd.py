# Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES
#
# SPDX-License-Identifier: BSD-3-Clause

"""
PyTorch autograd through :meth:`NetworkState.compute_expectation`.

On a torch-backend state with gradient-registered gates whose operands have
``requires_grad=True``, :meth:`compute_expectation` returns 0-D ``torch.Tensor``
scalars on the autograd graph (unless :func:`torch.is_grad_enabled` is false).
Without trainable gates, it still returns 0-D tensors but they are not connected
to the graph.

``apply_tensor_operator`` applies a **bra/ket axis transpose** to operands (see
``state_operands_wrapper(..., transpose=True)``). Autograd tracks the **stored**
tensor view (e.g. ``ry_gate.mT`` for a 2×2 gate).

Circuit (2 qubits):
  - H on q1
  - RY(theta) on q0  (theta is a trainable ``torch`` scalar)
  - CNOT(0, 1)

Observable: ``ZI`` (Pauli Z on q0).

Loss: ``L = Re(E) ** 2`` (scalar real loss for ``.backward()``).
"""

from __future__ import annotations

import math

import torch

from cuquantum.tensornet.experimental import NetworkState


if not torch.cuda.is_available():
    raise RuntimeError("This example requires CUDA and a GPU-visible torch build.")

device = torch.device("cuda:0")
dtype_sv = "complex128"
torch_cdtype = torch.complex128

inv_sqrt2 = 1.0 / math.sqrt(2.0)
h_r = torch.tensor([[1.0, 1.0], [1.0, -1.0]], device=device, dtype=torch.float64) * inv_sqrt2
h_gate = torch.complex(h_r, torch.zeros_like(h_r))

theta = torch.tensor(math.pi / 4, device=device, dtype=torch.float64, requires_grad=True)
cy, sy = torch.cos(theta / 2), torch.sin(theta / 2)
ry_r = torch.stack([torch.stack([cy, -sy]), torch.stack([sy, cy])])
ry_gate = torch.complex(ry_r, torch.zeros_like(ry_r)).to(torch_cdtype)

cx = torch.zeros((2, 2, 2, 2), device=device, dtype=torch_cdtype)
cx[0, 0, 0, 0] = 1.0
cx[0, 1, 0, 1] = 1.0
cx[1, 0, 1, 1] = 1.0
cx[1, 1, 1, 0] = 1.0

with NetworkState((2, 2), dtype=dtype_sv) as state:
    state.apply_tensor_operator((1,), h_gate, unitary=True)
    # Gradient registration for Torch is inferred from ry_gate.requires_grad.
    state.apply_tensor_operator((0,), ry_gate, unitary=True)

    state.apply_tensor_operator((0, 1), cx, unitary=True)

    e = state.compute_expectation("ZI")
    
    loss = e.real**2
    loss.backward()

print(f"expectation <ZI> (0-D torch.Tensor): {e}")
print(f"loss Re(E)^2: {loss.item():.6f}")
print(f"d(loss)/d(theta) (leaf): {theta.grad.item():.6f}")
