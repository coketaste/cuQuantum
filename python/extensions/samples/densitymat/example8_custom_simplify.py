# Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES
#
# SPDX-License-Identifier: BSD-3-Clause

"""
Custom operator-product simplification example.
"""

import jax
import jax.numpy as jnp

jax.config.update("jax_enable_x64", True)

from cuquantum.densitymat.jax import (
    ElementaryOperator,
    OperatorTerm,
    Operator,
    SimplifierConfig,
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


def kron_cond_no_cap(modes_duals):
    """
    Remove the default kron pass's cap on combined modes, so any run of base
    operators with no overlapping mode-dual pairs in between fully collapses into one
    dense base operator, regardless of how many modes it spans.
    """
    combined = set(modes_duals[0]) | set(modes_duals[-1])
    return not any(set(md) & combined for md in modes_duals[1:-1])


def sum_cond_exact_match(modes_duals_src, modes_duals_dst):
    """
    Narrow the default sum pass to only fold a source product into a destination
    whose single base operator's mode-dual footprint exactly matches the source's
    combined footprint, rather than merely containing it.
    """
    if len(modes_duals_dst) > 1:  # required: see SimplifierConfig.sum_cond's docstring
        return False
    combined_src = set().union(*modes_duals_src)
    return combined_src == set(modes_duals_dst[0])


def build_two_three_mode_terms(dims):
    """
    Build two separate operator products, each appended from 3 base operators in one
    call and both spanning the same 3 modes (0, 1, 2), so a fully custom kron pass can
    collapse each to one base operator and a custom sum pass can then fold them
    together.
    """
    d = dims[0]
    a_data = jnp.diag(jnp.sqrt(jnp.arange(1, d, dtype=jnp.complex128)), k=1)  # annihilation
    ad_data = a_data.conj().T
    n_data = ad_data @ a_data  # number operator
    jax.debug.print("Defined elementary operator data buffers.", ordered=True)

    a_elem_op = ElementaryOperator(a_data)
    ad_elem_op = ElementaryOperator(ad_data)
    n_elem_op = ElementaryOperator(n_data)
    m_elem_op = ElementaryOperator(ad_data + a_data)
    jax.debug.print("Created elementary operator objects.", ordered=True)

    H = OperatorTerm(dims)
    H.append([a_elem_op, n_elem_op, m_elem_op], modes=(0, 1, 2), coeff=1.0)
    H.append([ad_elem_op, a_elem_op, n_elem_op], modes=(0, 1, 2), coeff=0.5)
    jax.debug.print("Constructed operator term from elementary operators.", ordered=True)

    return H


if __name__ == "__main__":

    print_device_info()

    dims = (3, 3, 3)  # 3-site chain, extent 3 per site

    H = build_two_three_mode_terms(dims)
    print(f"Hamiltonian built with {len(H)} operator products before simplification.")

    rho0 = jnp.zeros((*dims, *dims), dtype=jnp.complex128)
    rho0 = rho0.at[(0, 0, 0, 0, 0, 0)].set(1.0)
    jax.debug.print("Defined input state data buffer.", ordered=True)

    def total_base_ops(op):
        term = op.op_terms[0]
        return sum(len(term[i]) for i in range(len(term)))

    # Default simplification: kron pass capped at 2 combined modes, so neither
    # product collapses to a single base operator -- and the sum pass then never
    # finds a single-base-operator destination to fold the other product into.
    op_default = Operator(dims, simplify=True)
    op_default.append(H, dual=False, coeff=1.0)
    jax.debug.print("Constructed operator from operator term (default SimplifierConfig).", ordered=True)
    rho1_default = operator_action(op_default, rho0)
    jax.debug.print("Performed operator action on the input state (default SimplifierConfig).", ordered=True)
    print(
        f"Default SimplifierConfig: {len(op_default.op_terms[0])} products, "
        f"{total_base_ops(op_default)} total base operators."
    )

    # Custom simplification: kron pass without the cap fully collapses each product
    # first, which lets the custom exact-match sum pass then fold them together.
    custom_config = SimplifierConfig(
        kron_cond=(kron_cond_no_cap,),
        sum_cond=(sum_cond_exact_match,),
    )
    op_custom = Operator(dims, simplify=custom_config)
    op_custom.append(H, dual=False, coeff=1.0)
    jax.debug.print("Constructed operator from operator term (custom SimplifierConfig).", ordered=True)
    rho1_custom = operator_action(op_custom, rho0)
    jax.debug.print("Performed operator action on the input state (custom SimplifierConfig).", ordered=True)
    print(
        f"Custom SimplifierConfig: {len(op_custom.op_terms[0])} products, "
        f"{total_base_ops(op_custom)} total base operators."
    )

    # Both configurations act identically on the state -- simplification only
    # restructures how the operator is represented, not what it computes.
    print("Results agree:", bool(jnp.allclose(rho1_default, rho1_custom)))

    print("Finished computation and exit.")
