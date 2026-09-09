# Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES
#
# SPDX-License-Identifier: BSD-3-Clause

"""
Operator action batched gradient example.
"""

import jax
import jax.numpy as jnp

jax.config.update("jax_enable_x64", True)

from cuquantum.densitymat.jax import (
    ElementaryOperator,
    OperatorTerm,
    Operator,
    operator_action,
)

# Toggle logging from the cuQuantum Python JAX API.
ENABLE_LOGGING = False

if ENABLE_LOGGING:
    import logging
    logging.basicConfig(
        level=logging.INFO,  # logging level can be modified as well
        format='%(name)s [%(levelname)s] %(message)s'
    )

global_key = jax.random.key(42)


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


def coherent_state(n_levels, alpha):
    """
    Create a coherent state |alpha⟩ via the displacement operator.
    """
    # Create annihilation operator a
    a = jnp.diag(jnp.sqrt(jnp.arange(1, n_levels, dtype=jnp.complex128)), k=1)

    # Create creation operator a† (a_dag)
    a_dag = a.conj().T

    # Compute displacement operator D(α) = exp(α a† - α* a)
    displacement_arg = alpha * a_dag - jnp.conj(alpha) * a
    D = jax.scipy.linalg.expm(displacement_arg)

    # Create ground state |0⟩
    ground_state = jnp.zeros(n_levels, dtype=jnp.complex128)
    ground_state = ground_state.at[0].set(1.0)

    # Apply displacement operator: |α⟩ = D(α)|0⟩
    coherent = D @ ground_state

    return coherent


def main(omega, kappa, alpha0):
    """
    Per-element oscillator-population expectation for a single set of parameters.

    This is batched by wrapping the whole function in jax.vmap (see __main__), so every
    operator here is built from per-element scalar parameters and is therefore unbatched
    (no batch_dims needed). The batch dimension lives outside, in the vmap/grad.
    """
    key = global_key

    # Hamiltonian elementary operator, scaled by the (per-element) frequency omega.
    key, subkey = jax.random.split(key)
    h_data = jnp.exp(omega) * jax.random.normal(subkey, (dims[0], dims[0]), dtype=jnp.complex128)
    jax.debug.print("Defined Hamiltonian elementary operator data buffer.", ordered=True)

    h = ElementaryOperator(h_data)
    jax.debug.print("Created Hamiltonian elementary operator.", ordered=True)

    # Dissipation elementary operators, scaled by the (per-element) decay rate kappa.
    key, subkey = jax.random.split(key)
    l_data = jax.random.normal(subkey, (dims[0], dims[0]), dtype=jnp.complex128)
    jax.debug.print("Defined dissipation elementary operator data buffers.", ordered=True)

    l = ElementaryOperator(kappa * l_data)
    ld = ElementaryOperator(jnp.conj(kappa) * l_data.conj().T)  # l†, scaled so it batches like l
    jax.debug.print("Created dissipation elementary operators.", ordered=True)

    # Initial state from the (per-element) coherent amplitude alpha0.
    psi0 = coherent_state(dims[0], alpha0)
    rho0 = jnp.outer(psi0, psi0.conj())
    jax.debug.print("Created initial state data buffer.", ordered=True)

    # Construct operator term for the Hamiltonian
    H = OperatorTerm(dims)
    H.append([h], modes=modes, coeff=jnp.full(batch_size, 1.0, dtype=jnp.complex128))
    jax.debug.print("Constructed Hamiltonian operator term.", ordered=True)

    # Construct operator term for dissipators
    Ls = OperatorTerm(dims)
    Ls.append([l, ld], modes=(0, 0), duals=(False, True), coeff=jnp.full(batch_size, 1.0, dtype=jnp.complex128))
    Ls.append([l, ld], modes=(0, 0), duals=(False, False), coeff=jnp.full(batch_size, -0.5, dtype=jnp.complex128))
    Ls.append([ld, l], modes=(0, 0), duals=(True, True), coeff=jnp.full(batch_size, -0.5, dtype=jnp.complex128))
    jax.debug.print("Constructed dissipator operator term.", ordered=True)

    liouvillian = Operator(dims)
    liouvillian.append(H, dual=False, coeff=jnp.full(batch_size, -1.0j, dtype=jnp.complex128))
    liouvillian.append(H, dual=True, coeff=jnp.full(batch_size, 1.0j, dtype=jnp.complex128))
    liouvillian.append(Ls, dual=False, coeff=jnp.full(batch_size, 1.0, dtype=jnp.complex128))
    jax.debug.print("Constructed Liouvillian operator from operator terms.", ordered=True)

    # No vmap here: operator_action is applied to a single (per-element) operator/state.
    rho1 = operator_action(liouvillian, rho0)
    jax.debug.print("Performed operator action on the input state.", ordered=True)

    key, subkey = jax.random.split(key)
    exp_op = jax.random.normal(subkey, (dims[0], dims[0]), dtype=jnp.complex128)
    return jnp.einsum('ij,ji->', exp_op, rho1).real


if __name__ == "__main__":

    print_device_info()

    dims = (5,)     # Hilbert space dimension
    modes = (0,)
    batch_size = 2

    # Per-element (batched) physical parameters; each entry is one batch element.
    omega = jnp.array([1.0, 1.2], dtype=jnp.complex128)     # frequency
    kappa = jnp.array([0.1, 0.15], dtype=jnp.complex128)    # decay rate
    alpha0 = jnp.array([1.0, 0.9], dtype=jnp.complex128)    # initial coherent amplitude

    # Batch by vmapping the per-element computation (vmap outside main), then take the
    # gradient of the summed batched loss -> a per-element gradient for each parameter.
    result = jax.grad(
        lambda omega, kappa, alpha0: jax.vmap(main, in_axes=0)(omega, kappa, alpha0).sum(),
        argnums=(0, 1, 2),
    )(omega, kappa, alpha0)

    print("Finished computation and exit.")
