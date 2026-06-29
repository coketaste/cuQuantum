# Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES
#
# SPDX-License-Identifier: BSD-3-Clause

"""
Converting a *noisy* Qiskit circuit to a tensor network with :class:`CircuitToEinsum`.

When the input circuit contains quantum channels (general/Kraus or unitary channels),
:class:`CircuitToEinsum` automatically builds a *density-matrix* tensor network -- there is
no purity argument to set. Properties of the noisy circuit (the full density matrix,
amplitudes, reduced density matrices, and marginal probabilities) can then each be evaluated
with a single tensor network contraction.

This example builds a 3-qubit GHZ-like circuit and applies an amplitude-damping channel
(a non-unitary Kraus channel) to one qubit.
"""
import numpy as np
from qiskit import QuantumCircuit
from qiskit.quantum_info import Kraus

from cuquantum.tensornet import CircuitToEinsum, contract

# Build a small circuit and add a non-unitary (amplitude-damping) channel on qubit 0.
gamma = 0.2
K0 = np.array([[1.0, 0.0], [0.0, np.sqrt(1.0 - gamma)]], dtype=np.complex128)
K1 = np.array([[0.0, np.sqrt(gamma)], [0.0, 0.0]], dtype=np.complex128)

circuit = QuantumCircuit(3)
circuit.h(0)
circuit.cx(0, 1)
circuit.cx(1, 2)
circuit.append(Kraus([K0, K1]), [0])

# CircuitToEinsum detects the channel and switches to a density-matrix network automatically.
converter = CircuitToEinsum(circuit, dtype="complex128", backend="numpy")
n = converter.n_qubits
print(f"Number of qubits: {n}")
print(f"Mixed state (channel detected): {converter.is_mixed}")

# 1) Full density matrix rho (rank-2N tensor) in a single contraction.
expr, operands = converter.density_matrix()
rho = contract(expr, *operands).reshape(2**n, 2**n)
print(f"\nDensity matrix shape: {rho.shape}")
print(f"Tr(rho) = {np.trace(rho).real:.10f}  (should be ~1.0)")

# 2) A diagonal amplitude is a probability: <bs| rho |bs>.
expr, operands = converter.amplitude("000")
p000 = complex(contract(expr, *operands))
print(f"\nP(000) = <000|rho|000> = {p000.real:.6f}")

# 3) A 2-tuple selects an off-diagonal density-matrix element <ket| rho |bra>.
expr, operands = converter.amplitude(("000", "111"))
rho_000_111 = complex(contract(expr, *operands))
print(f"rho[000, 111] = <000|rho|111> = {rho_000_111:.6f}")

# 4) Reduced density matrix over a subset of qubits (tracing out the rest).
where = converter.qubits[:1]
expr, operands = converter.reduced_density_matrix(where)
rdm = np.asarray(contract(expr, *operands)).reshape(2, 2)
print(f"\nReduced density matrix on qubit 0:\n{rdm}")

# 5) Marginal probability distribution (the diagonal of the reduced density matrix).
expr, operands = converter.marginal_probability(where)
probs = np.asarray(contract(expr, *operands)).real
print(f"\nMarginal probabilities on qubit 0: {probs}  (sum = {probs.sum():.6f})")

print("\nDone.")
