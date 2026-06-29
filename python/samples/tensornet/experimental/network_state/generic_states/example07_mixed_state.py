# Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES
#
# SPDX-License-Identifier: BSD-3-Clause

"""
Tensor network simulation of a mixed quantum state (density matrix) using the doubled tensor network formalism.

This example demonstrates:
  1. Creating a mixed-state NetworkState with pure_state=False
  2. Applying unitary gates and computing the RDM / expectation
  3. Applying a noisy (general) channel and observing its effect on the density matrix
  4. Computing density-matrix properties of the noisy state: the full density matrix,
     diagonal/off-diagonal amplitudes, and the marginal probability distribution

Topology (non-noisy):

Vacuum:         A   B   C   D
                |   |   |   |
one body op     O   O   O   O
                |   |   |   |
two body op     GGGGG   GGGGG
                |   |   |   |
two body op     |   GGGGG   |
                |   |   |   |

For the mixed state, the density matrix rho = |psi><psi| is represented internally
as a doubled tensor network (ket + bra copies).
"""
import numpy as np

from cuquantum.tensornet.experimental import NetworkState, TNConfig

n_qubits = 4
state_mode_extents = (2,) * n_qubits
dtype = 'complex128'

def random_unitary(n, rng):
    mat = rng.standard_normal((2**n, 2**n)) + 1j * rng.standard_normal((2**n, 2**n))
    q, _ = np.linalg.qr(mat)
    return q.reshape((2, 2) * n).astype(np.complex128)

rng = np.random.default_rng(42)
op_one_body = random_unitary(1, rng)
op_two_body = random_unitary(2, rng)

# --- Part 1: Pure state reference ---
print("=== Pure state reference ===")
with NetworkState(state_mode_extents, dtype=dtype, pure_state=True) as pure_state:
    for i in range(n_qubits):
        pure_state.apply_tensor_operator((i,), op_one_body, unitary=True)
    pure_state.apply_tensor_operator((0, 1), op_two_body, unitary=True)
    pure_state.apply_tensor_operator((2, 3), op_two_body, unitary=True)
    pure_state.apply_tensor_operator((1, 2), op_two_body, unitary=True)

    sv = pure_state.compute_state_vector()
    rdm_pure = pure_state.compute_reduced_density_matrix((0, 1))
    expec_pure = pure_state.compute_expectation({'XXXX': 0.3, 'ZIZI': 0.7})

print(f"State vector shape: {sv.shape}")
print(f"RDM(0,1) shape: {rdm_pure.shape}")
print(f"Expectation (pure): {expec_pure}")

# --- Part 2: Same circuit as mixed state (should give identical results for unitary evolution) ---
print("\n=== Mixed state (unitary only, should match pure) ===")
with NetworkState(state_mode_extents, dtype=dtype, pure_state=False) as mixed_state:
    for i in range(n_qubits):
        mixed_state.apply_tensor_operator((i,), op_one_body, unitary=True)
    mixed_state.apply_tensor_operator((0, 1), op_two_body, unitary=True)
    mixed_state.apply_tensor_operator((2, 3), op_two_body, unitary=True)
    mixed_state.apply_tensor_operator((1, 2), op_two_body, unitary=True)

    rdm_mixed = mixed_state.compute_reduced_density_matrix((0, 1))
    expec_mixed = mixed_state.compute_expectation({'XXXX': 0.3, 'ZIZI': 0.7})

print(f"RDM(0,1) shape: {rdm_mixed.shape}")
print(f"Expectation (mixed): {expec_mixed}")
print(f"RDM match: {np.allclose(rdm_pure, rdm_mixed, atol=1e-10)}")
print(f"Expectation match: {np.allclose(expec_pure, expec_mixed, atol=1e-10)}")

# --- Part 3: Mixed state with noise channel ---
print("\n=== Mixed state with depolarizing noise ===")
depolarizing_rate = 0.1
pauli_I = np.eye(2, dtype=np.complex128)
pauli_X = np.array([[0, 1], [1, 0]], dtype=np.complex128)
pauli_Y = np.array([[0, -1j], [1j, 0]], dtype=np.complex128)
pauli_Z = np.array([[1, 0], [0, -1]], dtype=np.complex128)

depolarizing_channel = [
    np.sqrt(1 - depolarizing_rate) * pauli_I,
    np.sqrt(depolarizing_rate / 3) * pauli_X,
    np.sqrt(depolarizing_rate / 3) * pauli_Y,
    np.sqrt(depolarizing_rate / 3) * pauli_Z,
]

with NetworkState(state_mode_extents, dtype=dtype, pure_state=False, config=TNConfig()) as noisy_state:
    for i in range(n_qubits):
        noisy_state.apply_tensor_operator((i,), op_one_body, unitary=True)
    noisy_state.apply_tensor_operator((0, 1), op_two_body, unitary=True)
    noisy_state.apply_tensor_operator((2, 3), op_two_body, unitary=True)

    # Apply depolarizing noise to each qubit
    for i in range(n_qubits):
        noisy_state.apply_general_tensor_channel((i,), depolarizing_channel)

    noisy_state.apply_tensor_operator((1, 2), op_two_body, unitary=True)

    rdm_noisy = noisy_state.compute_reduced_density_matrix((0, 1))
    expec_noisy = noisy_state.compute_expectation({'XXXX': 0.3, 'ZIZI': 0.7})

    # Additional density-matrix accessors (mixed-state mode):
    #   - the full density matrix rho (rank-2N tensor),
    #   - compute_amplitude with a single bitstring -> diagonal element (a probability),
    #   - compute_amplitude with a (ket, bra) pair  -> off-diagonal element,
    #   - compute_reduced_density_matrix(..., diagonal=True) -> diagonal of the reduced density matrix.
    dm = noisy_state.compute_density_matrix()
    zeros = '0' * n_qubits
    ones = '1' * n_qubits
    prob_zeros = noisy_state.compute_amplitude(zeros)
    offdiag = noisy_state.compute_amplitude((zeros, ones))
    marginal = noisy_state.compute_reduced_density_matrix((0, 1), diagonal=True)

print(f"RDM(0,1) shape (noisy): {rdm_noisy.shape}")
print(f"Expectation (noisy):  {expec_noisy}")
print(f"Expectation (clean):  {expec_mixed}")

# Noisy expectation should differ from clean one
print(f"Noise changed result: {not np.allclose(expec_noisy, expec_mixed, atol=1e-6)}")

# RDM trace should be 1 (density matrix is trace-preserving)
rdm_matrix = rdm_noisy.reshape(4, 4)
trace = np.trace(rdm_matrix).real
print(f"RDM trace: {trace:.10f} (should be ~1.0)")

# Full density matrix: rank-2N tensor whose trace is 1
dm_matrix = np.asarray(dm).reshape(2**n_qubits, 2**n_qubits)
print(f"\nDensity matrix shape: {dm.shape}, Tr(rho) = {np.trace(dm_matrix).real:.10f}")
print(f"Diagonal element rho[{zeros},{zeros}] = P({zeros}) = {complex(prob_zeros).real:.6f}")
print(f"Off-diagonal element rho[{zeros},{ones}] = {complex(offdiag):.6f}")
marginal = np.asarray(marginal).real
print(f"Marginal probabilities over qubits (0,1): {marginal.ravel()} (sum = {marginal.sum():.6f})")

print("\nDone.")
