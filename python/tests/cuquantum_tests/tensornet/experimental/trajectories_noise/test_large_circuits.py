# Copyright (c) 2024-2026, NVIDIA CORPORATION & AFFILIATES
#
# SPDX-License-Identifier: BSD-3-Clause

"""
Trajectories based simulation of noisy quantum channels.

This test uses TrajectorySim API to simulate changes in expectation of maxcut cost observable under noise
"""

import numpy as np
import networkx as nx
import pytest
from .quantum_channels import (
    apply_channel_to_mixed_state,
    bitflip_channel,
    QuantumGates,
)
from .network_state_wrap import network_state_config


SEED = 10
np.random.seed(SEED)


@pytest.mark.parametrize("bitflip_p", [0.01, 0.08])
@pytest.mark.parametrize("n_qubits", [10])
def test_bitflip_maxcut_cost(trajectory_sim, bitflip_p, n_qubits, channel_method):
    """
    Take a unitary operator U=exp(C) and its eigenstate.

    Apply the operator to the state with bitflip noise after each gate.

    Evaluate fidelity and expectation value
    """
    n_trajectories = 30
    channel = bitflip_channel(bitflip_p)
    if channel_method == "general":
        channel.set_general()
    G = nx.random_regular_graph(3, n_qubits)
    init_cut_value, (init_flips, _) = nx.approximation.one_exchange(G, seed=SEED)
    cost_dict = {}
    for u, v in G.edges:
        pstring = ["I"] * n_qubits
        pstring[u] = "Z"
        pstring[v] = "Z"
        cost_dict["".join(pstring)] = 0.5

    ensemble_dms = []
    ensemble_exps = []
    for sim in trajectory_sim.iterate_trajectories(n_trajectories):
        # -- Prepare init state
        for q in init_flips:
            sim.apply_gate((q,), QuantumGates.X)
        # -- Apply operator
        for u, v in G.edges:
            gate = QuantumGates.eZZ.reshape((2, 2, 2, 2))
            sim.apply_gate((u, v), gate)
            # -- Apply noise on the gate
            for q in (u, v):
                sim.apply_channel((q,), channel)
        # -- Calculate DM
        dm = sim.rdm()
        ensemble_dms.append(dm)

    ensemble_dm = np.stack(ensemble_dms).mean(axis=0).reshape(2**n_qubits, 2**n_qubits)

    # -- Reference values
    rdm_true = np.ones(2)
    for sim in trajectory_sim.iterate_trajectories(1):
        for q in init_flips:
            sim.apply_gate((q,), QuantumGates.X)
        rdm_true = sim.rdm()
    # --
    # F = <\psi|\rho|\psi>
    fidelity = np.trace(rdm_true.dot(ensemble_dm))
    print(f"{fidelity=}")
    n_noise_gates = 2 * G.number_of_edges()
    expected_fidelity = (1 - bitflip_p) ** (n_noise_gates)
    print(f"{expected_fidelity=}")
    sigma_ = 1 / np.sqrt(n_trajectories)
    print(f"{sigma_=}")
    # - TODO: verify the scaling of fidelity under trajectories
    assert np.abs(fidelity - expected_fidelity) < 2.5 * sigma_


@pytest.mark.parametrize("bitflip_p", [0.01, 0.08])
@pytest.mark.parametrize("as_general", [False, True], ids=["unitary", "general"])
def test_bitflip_maxcut_fidelity_mixed(bitflip_p, as_general):
    """Exact mixed-state fidelity under bitflip noise: F = (1-p)^n_noise_gates.

    Uses a smaller graph (6 qubits) than the trajectory test so the full
    density matrix fits comfortably in memory.
    """
    n_qubits = 6
    G = nx.random_regular_graph(3, n_qubits, seed=SEED)
    _, (init_flips, _) = nx.approximation.one_exchange(G, seed=SEED)

    channel = bitflip_channel(bitflip_p)
    eZZ = QuantumGates.eZZ.reshape((2, 2, 2, 2))
    all_modes = tuple(range(n_qubits))
    dim = 2 ** n_qubits

    with network_state_config(n_qubits, 'tn') as pure:
        for q in init_flips:
            pure.apply_tensor_operator((q,), QuantumGates.X, unitary=True)
        for u, v in G.edges:
            pure.apply_tensor_operator((u, v), eZZ, unitary=True)
        sv_pure = np.asarray(pure.compute_state_vector()).flatten()
    rho_pure = np.outer(sv_pure, sv_pure.conj())

    with network_state_config(n_qubits, 'tn', pure_state=False) as mixed:
        for q in init_flips:
            mixed.apply_tensor_operator((q,), QuantumGates.X, unitary=True)
        for u, v in G.edges:
            mixed.apply_tensor_operator((u, v), eZZ, unitary=True)
            for q in (u, v):
                apply_channel_to_mixed_state(mixed, (q,), channel, as_general)
        rho_noisy = np.asarray(mixed.compute_reduced_density_matrix(all_modes)).reshape(dim, dim)

    fidelity = np.trace(rho_pure @ rho_noisy).real
    # Exact fidelity: each qubit k independently undergoes d_k bitflip(p) channels
    # (one per adjacent edge). Since eZZ is diagonal, the computational basis is
    # preserved between noise events. The single-qubit survival probability after
    # m bitflips is f = 1/2 + 1/2*(1-2p)^m, and the total fidelity is the product.
    expected_fidelity = 1.0
    for k in range(n_qubits):
        d_k = G.degree(k)
        expected_fidelity *= 0.5 + 0.5 * (1 - 2 * bitflip_p) ** d_k
    np.testing.assert_allclose(fidelity, expected_fidelity, atol=1e-10)
