# Copyright (c) 2024-2026, NVIDIA CORPORATION & AFFILIATES
#
# SPDX-License-Identifier: BSD-3-Clause

"""
Trajectories based simulation of noisy quantum channels.

This test uses TrajectorySim API to simulate one-qubit noise channels.
Mixed-state tests at the bottom use pure_state=False for exact (deterministic) verification
against the same analytical references.
"""

import numpy as np
import pytest
from .quantum_channels import (
    apply_channel_to_mixed_state,
    bitflip_channel,
    depolarizing_channel,
    damping_channel,
    QuantumGates,
)
from .network_state_wrap import network_state_config

# in pytest cases, python objects are not displayed nicely, so let's use string tags
np.random.seed(10)

# -- Tests

# This test file uses `trajectory_sim` fixture defined in conftest.py

# Note: The bitflip_p values are selected to catch errors in conversion of
# unitary to general channel (sqrt(p) vs p).
@pytest.mark.parametrize("bitflip_p", [0.1, 0.75])
@pytest.mark.parametrize("n_qubits", [1])
def test_bitflip_channel(trajectory_sim, bitflip_p, channel_method):
    n_trajectories = 300
    channel = bitflip_channel(bitflip_p)
    if channel_method == "general":
        channel.set_general()

    ensemble_probs = []
    for sim in trajectory_sim.iterate_trajectories(n_trajectories):
        sim.apply_channel((0,), channel)
        prob = sim.probs((0,))
        ensemble_probs.append(prob)

    ensemble_probs = np.stack(ensemble_probs).mean(axis=0)
    print("Bitflip ensemble Probs ", ensemble_probs)
    true_probs = np.array([1 - bitflip_p, bitflip_p])
    print("Bitflip ensemble Reference probs ", true_probs)
    # Use 2.5 sigma, which is about 99% CI
    sigma_ = 1 / np.sqrt(n_trajectories)
    assert (np.abs(ensemble_probs - true_probs) < 2.5 * sigma_).all()


@pytest.mark.parametrize("n_qubits", [1, 2])
def test_depolarizing_channel(trajectory_sim, n_qubits, channel_method):
    """
    For n_qubits>1, pads the channel with identity on the left.
    Checks that we get a I/2 state on the corresponding qubit
    """
    n_trajectories = 300
    error = 1
    channel = depolarizing_channel(error)
    if channel_method == "general":
        channel.set_general()
    if n_qubits > 1:
        for _ in range(n_qubits - 1):
            channel.mul_left(np.eye(2))

    ensemble_probs = []
    qubit_id = n_qubits - 1
    for sim in trajectory_sim.iterate_trajectories(n_trajectories):
        sim.apply_channel(tuple(range(n_qubits)), channel)
        probs = sim.probs((qubit_id,))
        ensemble_probs.append(probs)

    # TODO: check that we don't have a I/2 DM on qubit 0
    ensemble_probs = np.stack(ensemble_probs).mean(axis=0)
    print("Depolarizing ensemble probs ", ensemble_probs)
    true_dm = np.ones(2) / 2
    # Use 2.5 sigma, which is about 99% CI
    sigma_ = 1 / np.sqrt(n_trajectories)
    assert (np.abs(ensemble_probs - true_dm) < 2.5 * sigma_).all()


@pytest.mark.parametrize("damping", [0.08])
@pytest.mark.parametrize("n_qubits", [1, 2, 3])
def test_damping_channel(trajectory_sim, damping):
    n_trajectories = 50
    channel = damping_channel(damping)
    n_damping_rounds = 5
    ensemble_probs = []
    for sim in trajectory_sim.iterate_trajectories(n_trajectories):
        sim.apply_gate((0,), QuantumGates.X)
        for _ in range(n_damping_rounds):
            sim.apply_channel((0,), channel)
        prob = sim.probs((0,))
        # print("Damping prob ", prob)
        # probabilities are not normalized for non-unitary channels
        ensemble_probs.append(prob/sum(prob))

    ensemble_probs_raw = np.stack(ensemble_probs)
    print("Bitflip ensemble Probs ", ensemble_probs_raw)
    ensemble_probs = ensemble_probs_raw.mean(axis=0)
    print("Bitflip ensemble Probs mean unnormalized", ensemble_probs)
    damped_prob = (1 - damping)**n_damping_rounds
    true_probs = np.array([1-damped_prob, damped_prob])
    print("Bitflip ensemble Reference probs", true_probs)
    # Use 2.5 sigma, which is about 99% CI
    sigma_ = 1 / np.sqrt(n_trajectories)
    assert (np.abs(ensemble_probs - true_probs) < 2.5 * sigma_).all()


# -- Mixed-state exact tests --
# Same analytical references as above, verified deterministically via pure_state=False.

def _mixed_probs(state, qubits):
    """Extract measurement probabilities from a mixed-state RDM."""
    rdm = state.compute_reduced_density_matrix(qubits)
    dim = 2 ** len(qubits)
    return np.diag(np.asarray(rdm).reshape(dim, dim)).real


@pytest.mark.parametrize("bitflip_p", [0.1, 0.75])
@pytest.mark.parametrize("as_general", [False, True], ids=["unitary", "general"])
def test_bitflip_channel_mixed(bitflip_p, as_general):
    """Exact mixed-state bitflip must give probs [1-p, p]."""
    channel = bitflip_channel(bitflip_p)
    with network_state_config(1, 'tn', pure_state=False) as state:
        apply_channel_to_mixed_state(state, (0,), channel, as_general)
        probs = _mixed_probs(state, (0,))
    true_probs = np.array([1 - bitflip_p, bitflip_p])
    np.testing.assert_allclose(probs, true_probs, atol=1e-12)


@pytest.mark.parametrize("n_qubits", [1, 2])
@pytest.mark.parametrize("as_general", [False, True], ids=["unitary", "general"])
def test_depolarizing_channel_mixed(n_qubits, as_general):
    """Full depolarization on the last qubit must give I/2."""
    channel = depolarizing_channel(1)
    if n_qubits > 1:
        for _ in range(n_qubits - 1):
            channel.mul_left(np.eye(2))

    qubit_id = n_qubits - 1
    with network_state_config(n_qubits, 'tn', pure_state=False) as state:
        apply_channel_to_mixed_state(state, tuple(range(n_qubits)), channel, as_general)
        probs = _mixed_probs(state, (qubit_id,))
    np.testing.assert_allclose(probs, [0.5, 0.5], atol=1e-12)


@pytest.mark.parametrize("damping", [0.08, 0.5])
@pytest.mark.parametrize("n_damping_rounds", [1, 5])
def test_damping_channel_mixed(damping, n_damping_rounds):
    """Amplitude damping on |1> must give p(1)=(1-gamma)^n."""
    channel = damping_channel(damping)
    with network_state_config(1, 'tn', pure_state=False) as state:
        state.apply_tensor_operator((0,), QuantumGates.X, unitary=True)
        for _ in range(n_damping_rounds):
            apply_channel_to_mixed_state(state, (0,), channel)
        probs = _mixed_probs(state, (0,))
    damped_prob = (1 - damping) ** n_damping_rounds
    true_probs = np.array([1 - damped_prob, damped_prob])
    np.testing.assert_allclose(probs, true_probs, atol=1e-12)
