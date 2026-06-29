# Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES
#
# SPDX-License-Identifier: BSD-3-Clause

import numpy as np
from nvmath.internal import utils as nvmath_utils

try:
    from cuda.core import Device
except ImportError:
    from cuda.core.experimental import Device


try:
    import torch
except ImportError:
    torch = None


if torch is not None:

    def _scalar_grad_to_numpy(grad, dtype, *, name):
        if grad is None:
            return np.array(0.0, dtype=np.dtype(dtype))
        if grad.numel() != 1:
            raise ValueError(f"gradient w.r.t. {name} must be a scalar (0-dim tensor).")
        return np.array(grad.detach().cpu().item(), dtype=np.dtype(dtype))

    class _TorchExpectation(torch.autograd.Function):

        @staticmethod
        def forward(context, state, operators, stream, release_workspace, return_norm, tensor_ids, *trainable_tensors):
            """CUTN ``tensor_ids`` align 1:1 with ``trainable_tensors`` (operand tensors from
            :meth:`NetworkState.compute_expectation`). May both be empty. Tensors must be passed as separate ``apply()`` arguments
            so PyTorch registers them as graph inputs;
            """
            if state.backend != "torch":
                raise TypeError(
                    "PyTorch expectation autograd requires NetworkState backend 'torch'; "
                    f"got {state.backend!r}."
                )
            if len(tensor_ids) != len(trainable_tensors):
                raise ValueError("tensor_ids and trainable_tensors must have the same length.")

            context.state = state
            context.operators = operators
            context.stream = stream
            context.release_workspace = release_workspace
            context.return_norm = return_norm
            context.tensor_ids = tensor_ids
            context.save_for_backward(*trainable_tensors)

            return state._compute_expectation(
                operators,
                return_norm=return_norm,
                stream=stream,
                release_workspace=release_workspace,
            )

        @staticmethod
        def backward(context, *grad_outputs):
            state = context.state
            stream = context.stream
            release_workspace = context.release_workspace
            return_norm = context.return_norm
            tensor_ids = context.tensor_ids
            trainable_tensors = context.saved_tensors

            stream_holder = nvmath_utils.get_or_create_stream(
                state.device_id, stream, state.internal_package
            )
            old_device = None
            try:
                # WAR: PyTorch backward can leave cuda.core without a current context.
                old_device = Device()
                Device(state.device_id).set_current()
                with stream_holder.ctx:
                    grad_e = grad_outputs[0]
                    grad_n = grad_outputs[1] if return_norm else None
                    adj_e_np = _scalar_grad_to_numpy(
                        grad_e, state.dtype, name="expectation"
                    )
                    adj_n_np = (
                        _scalar_grad_to_numpy(
                            grad_n, state.dtype, name="state norm"
                        )
                        if return_norm
                        else None
                    )
                    grads_dict = state.compute_expectation_with_gradients(
                        context.operators,
                        adj_e_np,
                        return_norm=return_norm,
                        state_norm_adjoint=adj_n_np,
                        stream=stream,
                        release_workspace=release_workspace,
                    )[-1]

                    input_grads = []
                    for tid, t in zip(tensor_ids, trainable_tensors, strict=True):
                        g_t = grads_dict.get(tid)
                        if g_t is None:
                            input_grads.append(None)
                        elif not isinstance(g_t, torch.Tensor):
                            raise RuntimeError(
                                "Internal error: expected torch.Tensor from "
                                "compute_expectation_with_gradients for torch backend."
                            )
                        else:
                            input_grads.append(g_t)
            finally:
                if old_device is not None:
                    old_device.set_current()

            # Grads in apply() order:
            # state, operators, stream, release_workspace, return_norm, tensor_ids
            # then one entry per *trainable_tensors.
            return (
                None,
                None,
                None,
                None,
                None,
                None,
                *input_grads,
            )

else:

    _TorchExpectation = None
