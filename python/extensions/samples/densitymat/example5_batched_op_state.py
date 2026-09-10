# Copyright (c) 2025-2026, NVIDIA CORPORATION & AFFILIATES
#
# SPDX-License-Identifier: BSD-3-Clause

"""
Operator action example with batched operator and state.
"""

import jax
import jax.numpy as jnp

jax.config.update("jax_enable_x64", True)
jax.config.update("jax_default_matmul_precision", "highest")

from cuquantum.bindings import cudensitymat as cudm
from cuquantum.densitymat.jax import (
    ElementaryOperator,
    OperatorTerm,
    Operator,
    operator_action
)

# Toggle logging from the cuQuantum Python JAX API.
ENABLE_LOGGING = False

if ENABLE_LOGGING:
    import logging
    logging.basicConfig(
        level=logging.INFO,  # logging level can be modified as well
        format='%(name)s [%(levelname)s] %(message)s'
    )


def print_device_info():
    """
    Print the information of the current device.
    """
    from cuda.bindings import runtime as cudart

    err, dev_id = cudart.cudaGetDevice()
    if err != cudart.cudaError_t.cudaSuccess:
        raise RuntimeError(f"cudaGetDevice failed with error code {err}")

    err, props = cudart.cudaGetDeviceProperties(dev_id)
    if err != cudart.cudaError_t.cudaSuccess:
        raise RuntimeError(f"cudaGetDeviceProperties failed with error code {err}")

    err, clock_rate = cudart.cudaDeviceGetAttribute(
        cudart.cudaDeviceAttr.cudaDevAttrClockRate, dev_id
    )
    if err != cudart.cudaError_t.cudaSuccess:
        raise RuntimeError(f"cudaDeviceGetAttribute failed with error code {err}")
    err, mem_clock_rate = cudart.cudaDeviceGetAttribute(
        cudart.cudaDeviceAttr.cudaDevAttrMemoryClockRate, dev_id
    )
    if err != cudart.cudaError_t.cudaSuccess:
        raise RuntimeError(f"cudaDeviceGetAttribute failed with error code {err}")

    print("===== device info ======")
    print("GPU-local-id:", dev_id)
    print("GPU-name:", props.name.decode())
    print("GPU-clockRate (MHz):", clock_rate / 1000)
    print("GPU-memoryClockRate (MHz):", mem_clock_rate / 1000)
    print("GPU-nSM:", props.multiProcessorCount)
    print("GPU-major:", props.major)
    print("GPU-minor:", props.minor)
    print("========================")


@jax.jit
def main():
    """
    Main computation.
    """
    batch_size = 2
    state_in = jnp.asarray(jax.random.uniform(key, (batch_size, *space_mode_extents, *space_mode_extents)), dtype=dtype)
    jax.debug.print("Defined input state data buffer.", ordered=True)

    # Original data arrays, broadcast to a leading batch dimension so every operator leaf
    # (data and coefficients) shares the same batch size and maps at axis 0 under vmap.
    n_data = jnp.broadcast_to(
        jnp.asarray(jnp.diag(jnp.arange(space_mode_extents[0])), dtype=dtype),
        (batch_size, space_mode_extents[0], space_mode_extents[0]))
    a_data = jnp.broadcast_to(
        jnp.asarray(jnp.diag(jnp.sqrt(jnp.arange(1, space_mode_extents[1])), k=1), dtype=dtype),
        (batch_size, space_mode_extents[1], space_mode_extents[1]))
    ad_data = jnp.broadcast_to(
        jnp.asarray(jnp.diag(jnp.sqrt(jnp.arange(1, space_mode_extents[1])), k=-1), dtype=dtype),
        (batch_size, space_mode_extents[1], space_mode_extents[1]))
    jax.debug.print("Defined elementary operator data buffers.", ordered=True)

    n_elem_op = ElementaryOperator(n_data)
    ad_elem_op = ElementaryOperator(ad_data)
    a_elem_op = ElementaryOperator(a_data)
    jax.debug.print("Created elementary operator objects.", ordered=True)

    # Create the Hamiltonian and dissipators.
    H = OperatorTerm(space_mode_extents)
    Ls = OperatorTerm(space_mode_extents)
    jax.debug.print("Constructed operator terms from elementary operators.", ordered=True)

    H.append([n_elem_op], modes=[0], duals=[False], coeff=jnp.full(batch_size, 1.0, dtype=dtype))
    Ls.append([ad_elem_op, a_elem_op], modes=[1, 1], duals=[False, True], coeff=jnp.full(batch_size, 1.0, dtype=dtype))
    Ls.append([a_elem_op, ad_elem_op], modes=[1, 1], duals=[False, False], coeff=jnp.full(batch_size, -0.5, dtype=dtype))
    Ls.append([ad_elem_op, a_elem_op], modes=[1, 1], duals=[True, True], coeff=jnp.full(batch_size, -0.5, dtype=dtype))
    jax.debug.print("Constructed operator terms from elementary operators.", ordered=True)

    # Batched coefficients for the Hamiltonian and dissipators.
    coeffs = jnp.array([-1.0j, -1.0j], dtype=dtype)
    coeffs1 = jnp.array([1.0j, 1.0j], dtype=dtype)
    coeffs2 = jnp.array([1.0, 1.0], dtype=dtype)

    liouvillian = Operator(space_mode_extents)
    liouvillian.append(H, dual=False, coeff=coeffs)
    liouvillian.append(H, dual=True, coeff=coeffs1)
    liouvillian.append(Ls, dual=False, coeff=coeffs2)
    jax.debug.print("Constructed operator from operator terms.", ordered=True)

    state_out = jax.vmap(operator_action, in_axes=(liouvillian.in_axes, 0))(liouvillian, state_in)
    jax.debug.print("Performed operator action on the input state.", ordered=True)

    return state_out


if __name__ == "__main__":

    print_device_info()

    key = jax.random.key(42)
    space_mode_extents = (3, 5)
    dtype = jnp.complex128

    main()

    print("Finished computation and exit.")
