# Copyright (c) 2024-2026, NVIDIA CORPORATION & AFFILIATES
#
# SPDX-License-Identifier: BSD-3-Clause

__all__ = [
    "CircuitStateMatrix",
    "EOverNDenominatorTooSmall",
    "GenericStateMatrix",
    "MixedGenericStateMatrix",
    "SimulationConfigMatrix",
    "ExpectationGradientConfig",
    "NetworkOperatorFactory",
    "expectation_gradient_loss_factory",
    "check_e_over_n_well_conditioned",
]

import importlib
import numpy as np
try:
    import torch
except ImportError:
    torch = None
from nvmath.internal.utils import infer_object_package
try:
    import cirq
    from cuquantum.tensornet._internal import circuit_parser_utils_cirq
except ImportError:
    cirq = circuit_parser_utils_cirq = None
try:
    import qiskit
except ImportError:
    qiskit = None

from cuquantum.tensornet import NetworkOptions
from cuquantum.tensornet.experimental import TNConfig, MPSConfig, NetworkOperator
from cuquantum.tensornet.experimental._internal.network_state_utils import get_pauli_map

from ...utils.circuit_matrix import CirqCircuitMatrix, QiskitCircuitMatrix, CircuitMatrixABC, get_qiskit_unitary_gate
from ...utils.helpers import (
    get_rng_iterator,
    get_array_framework_iterator,
    TensorBackend,
)
from .state_factory import StateFactory, NetworkOperatorFactory, _random_hermitian


def get_cirq_random_2q_gate(rng):
    random_state = int(rng.integers(0, high=2023))
    random_unitary = cirq.testing.random_unitary(4, random_state=random_state) # random 2-qubit gate
    random_gate = cirq.MatrixGate(random_unitary)
    return random_gate

def gen_random_layered_cirq_circuit(qubits, num_random_layers, rng):
    n_qubits = len(qubits)
    operations = []
    for n in range(num_random_layers):
        for i in range(n%2, n_qubits-1, 2):
            operations.append(get_cirq_random_2q_gate(rng).on(qubits[i], qubits[i+1]))
    return cirq.Circuit(operations)

def cirq_insert_random_layers(circuit, num_random_layers, rng):
    if num_random_layers == 0:
        return circuit
    qubits = sorted(circuit.all_qubits())
    circuit = circuit_parser_utils_cirq.remove_measurements(circuit)
    pre_circuit = gen_random_layered_cirq_circuit(qubits, num_random_layers, rng)
    return pre_circuit.concat_ragged(circuit)

def gen_random_layered_qiskit_circuit(qubits, num_random_layers, rng):
    n_qubits = len(qubits)
    circuit = qiskit.QuantumCircuit(qubits)
    for n in range(num_random_layers):
        for i in range(n%2, n_qubits-1, 2):
            circuit.append(get_qiskit_unitary_gate(rng), qubits[i:i+2])
    return circuit

def qiskit_insert_random_layers(circuit, num_random_layers, rng):
    if num_random_layers == 0:
        return circuit
    qubits = circuit.qubits
    circuit.remove_final_measurements()
    pre_circuit = gen_random_layered_qiskit_circuit(qubits, num_random_layers, rng)
    circuit.data = pre_circuit.data + circuit.data
    return circuit

class StateMatrixABC(CircuitMatrixABC):
    pass

qiskit_L0_circuits = [
    qiskit_insert_random_layers(circuit, 1, np.random.default_rng(i))
    for i, circuit in enumerate(QiskitCircuitMatrix.L0())
]
qiskit_L1_circuits = [
    qiskit_insert_random_layers(circuit, 2, np.random.default_rng(i))
    for i, circuit in enumerate(QiskitCircuitMatrix.L1())
]
qiskit_L2_circuits = [
    qiskit_insert_random_layers(circuit, 2, np.random.default_rng(i))
    for i, circuit in enumerate(QiskitCircuitMatrix.L2())
]

class QiskitCircuitStateMatrix(StateMatrixABC):
    
    @staticmethod
    def L0():
        return qiskit_L0_circuits

    @staticmethod
    def L1():
        return qiskit_L1_circuits

    @staticmethod
    def L2():
        return qiskit_L2_circuits

cirq_L0_circuits = [
    cirq_insert_random_layers(circuit, 1, np.random.default_rng(i))
    for i, circuit in enumerate(CirqCircuitMatrix.L0())
]
cirq_L1_circuits = [
    cirq_insert_random_layers(circuit, 2, np.random.default_rng(i))
    for i, circuit in enumerate(CirqCircuitMatrix.L1())
]
cirq_L2_circuits = [
    cirq_insert_random_layers(circuit, 2, np.random.default_rng(i))
    for i, circuit in enumerate(CirqCircuitMatrix.L2())
]

class CirqCircuitStateMatrix(StateMatrixABC):

    @staticmethod
    def L0():
        return cirq_L0_circuits

    @staticmethod
    def L1():
        return cirq_L1_circuits

    @staticmethod
    def L2():
        return cirq_L2_circuits

class CircuitStateMatrix(StateMatrixABC):

    @staticmethod
    def L0():
        return CirqCircuitStateMatrix.L0() + QiskitCircuitStateMatrix.L0()
    
    @staticmethod
    def L1():
        return CirqCircuitStateMatrix.L1() + QiskitCircuitStateMatrix.L1()
    
    @staticmethod
    def L2():
        return CirqCircuitStateMatrix.L2() + QiskitCircuitStateMatrix.L2()
    

rng_iterator = get_rng_iterator()
array_framework_iterator = get_array_framework_iterator()

def create_state_factory(*args, **kwargs):
    backend = kwargs.pop("backend", None)
    if backend is None:
        backend = next(array_framework_iterator)
    return StateFactory(*args, **kwargs, backend=backend)


generic_states_L0 = [
    create_state_factory(4, "float32", "SDDS", next(rng_iterator)),
    create_state_factory(5, "float64", "SDMDS", next(rng_iterator), mpo_bond_dim=2),
    create_state_factory((2, 3, 4, 3, 2), "complex64", "SDM", next(rng_iterator), mpo_bond_dim=2),
    create_state_factory(4, "complex128", "SDCS", next(rng_iterator), ct_target_place="first"),
    create_state_factory((2, 5, 3, 2), "complex64", "SDDS", next(rng_iterator), initial_mps_dim=2),
    create_state_factory((2, 5, 2, 2), "float64", "SDDD", next(rng_iterator), adjacent_double_layer=False),
    # Diagonal gate test cases
    create_state_factory(4, "complex128", "SADCAS", next(rng_iterator), ct_target_place="first"),
]

generic_states_L1 = [
    create_state_factory(5, "float32", "SDDS", next(rng_iterator)),
    create_state_factory((2, 3, 4, 3, 2), "complex64", "SDDS", next(rng_iterator)),
    create_state_factory(5, "complex64", "SDMS", next(rng_iterator), adjacent_double_layer=False, mpo_bond_dim=3),
    create_state_factory(4, "float64", "SDMDS", next(rng_iterator), mpo_bond_dim=2, mpo_num_sites=3, mpo_geometry="random-ordered"),
    create_state_factory((2, 5, 3, 2), "complex128", "SDMDS", next(rng_iterator), mpo_bond_dim=2, mpo_geometry="random"),
    create_state_factory(6, "float64", "SDCS", next(rng_iterator), ct_target_place="first"),
    create_state_factory(4, "float64", "SDCS", next(rng_iterator), ct_target_place="middle"),
    create_state_factory(5, "float64", "SDCS", next(rng_iterator), ct_target_place="last", initial_mps_dim=2),
    create_state_factory((2, 3, 3, 2, 2, 2), "float64", "SMDD", next(rng_iterator), mpo_bond_dim=2, adjacent_double_layer=False),
    # Diagonal gate test cases
    # {S, D, A}: all dtypes, exact simulation
    create_state_factory(4, "float32", "SADAAA", next(rng_iterator), adjacent_double_layer=False),
    create_state_factory((2, 5, 3, 2), "complex64", "SADA", next(rng_iterator), adjacent_double_layer=True),
    create_state_factory((2, 3, 2, 3, 4, 2), "complex128", "ASDA", next(rng_iterator), adjacent_double_layer=True),
]

generic_states_L2 = [
    create_state_factory(4, "float32", "SDDS", next(rng_iterator), adjacent_double_layer=False),
    create_state_factory(6, "float64", "SDMDS", next(rng_iterator), mpo_bond_dim=2, mpo_num_sites=4, mpo_geometry="adjacent-ordered"),
    create_state_factory(5, "float64", "SDCDS", next(rng_iterator), ct_target_place="first", initial_mps_dim=2),
    create_state_factory(6, "float64", "SDCSD", next(rng_iterator), ct_target_place="middle", adjacent_double_layer=False),
    create_state_factory(5, "float64", "SDCDS", next(rng_iterator), ct_target_place="last", initial_mps_dim=2),
    create_state_factory(8, "complex128", "SDMDS", next(rng_iterator), mpo_bond_dim=2, mpo_num_sites=5, mpo_geometry="random-ordered", initial_mps_dim=2),
    create_state_factory((3, 3, 3, 3, 3), "complex128", "SDMDS", next(rng_iterator), mpo_bond_dim=3, mpo_num_sites=4, mpo_geometry="random"),
    create_state_factory((2, 3, 2, 3, 4, 2), "complex128", "SDMDS", next(rng_iterator), mpo_bond_dim=2, mpo_geometry="adjacent-ordered"),
    create_state_factory((2, 5, 3, 2), "complex128", "SDMDS", next(rng_iterator), mpo_bond_dim=2, mpo_geometry="random", initial_mps_dim=2),
    # Diagonal gate test cases
    # {S, D, M, A}: double precision, approximate simulation
    create_state_factory((2, 5, 3, 2), "float64", "SADMA", next(rng_iterator), mpo_bond_dim=2),
    create_state_factory((2, 3, 2, 3, 4, 2), "complex128", "SADA", next(rng_iterator), initial_mps_dim=2),
]

mixed_generic_states_L1 = [
    create_state_factory(4, "complex128", "SDDS", next(rng_iterator)),
    create_state_factory(5, "float64", "SDMDS", next(rng_iterator), mpo_bond_dim=2),
    create_state_factory((2, 3, 4, 3, 2), "complex64", "SDM", next(rng_iterator), mpo_bond_dim=2),
    create_state_factory(4, "complex128", "SADCAS", next(rng_iterator), ct_target_place="first"),
]


class GenericStateMatrix(StateMatrixABC):

    @staticmethod
    def L0():
        return generic_states_L0
    
    @staticmethod
    def L1():
        return generic_states_L1

    @staticmethod
    def L2():
        return generic_states_L2


class MixedGenericStateMatrix:

    @staticmethod
    def L1():
        return mixed_generic_states_L1


exact_mps_configs = [
    MPSConfig(mpo_application='exact', gauge_option='free'),
    {'mpo_application': 'exact', 'gauge_option': 'simple'},
]

approx_mps_configs_L0 = [
    {'gauge_option': 'free', 'max_extent': 2, 'algorithm': 'gesvdj', 'abs_cutoff': 0.1},
    {'gauge_option': 'simple', 'rel_cutoff': 0.1},
]

approx_mps_configs_L1 = [
    {'gauge_option': 'free', 'max_extent': 2, 'algorithm': 'gesvdj', 'abs_cutoff': 0.1, 'gesvdj_max_sweeps': 100},
    {'gauge_option': 'free', 'abs_cutoff': 0.1, 'normalization': 'L1', 'canonical_center': 1},
    {'gauge_option': 'simple', 'max_extent': 4, 'rel_cutoff': 0.1, 'discarded_weight_cutoff': 0.1, 'normalization': 'LInf'},
    {'gauge_option': 'simple', 'max_extent': 3, 'canonical_center': 2, 'rel_cutoff': 0.1, 'normalization': 'L2'},
    {'gauge_option': 'simple', 'max_extent': 3, 'rel_cutoff': 0.1, 'normalization': 'L2', 'abs_cutoff': 0.1}, 
]

approx_mps_configs_L2 = [
    {'gauge_option': 'free', 'max_extent': 2, 'canonical_center': 0, 'algorithm': 'gesvdj', 'gesvdj_max_sweeps': 100, 'normalization': 'LInf'}, # fixed extent truncation
    {'gauge_option': 'free', 'abs_cutoff': 0.1, 'rel_cutoff': 0.2, 'discarded_weight_cutoff': 0.1}, # value based truncation
    {'gauge_option': 'free', 'max_extent': 4, 'normalization': 'L1', 'abs_cutoff': 0.1},
    {'gauge_option': 'free', 'max_extent': 3, 'canonical_center': 1, 'rel_cutoff': 0.1, 'normalization': 'L2'},
    {'gauge_option': 'simple', 'max_extent': 2, 'canonical_center': 2, 'normalization': 'LInf'}, # SU with fixed extent truncation
    {'gauge_option': 'simple', 'abs_cutoff': 0.1, 'canonical_center': 1, 'rel_cutoff': 0.2, 'discarded_weight_cutoff': 0.1}, # SU with value based truncation
    {'gauge_option': 'simple', 'algorithm': 'gesvdj', 'gesvdj_max_sweeps': 100, 'max_extent': 3, 'rel_cutoff': 0.1, 'normalization': 'L2'}
]

class MPSConfigMatrix:

    @staticmethod
    def exactConfigs():
        return exact_mps_configs

    @staticmethod
    def approxConfigsL0():
        return approx_mps_configs_L0

    @staticmethod
    def approxConfigsL1():
        return approx_mps_configs_L1
    
    @staticmethod
    def approxConfigsL2():
        return approx_mps_configs_L2
    
class SimulationConfigMatrix:

    @staticmethod
    def exactConfigs():
        return [TNConfig()] + MPSConfigMatrix.exactConfigs()


noisy_state_tests_L0 = [
    create_state_factory(4, "float32", "SDUDS", next(rng_iterator)),
    create_state_factory(4, "float64", "SDGDS", next(rng_iterator)),
    create_state_factory(5, "complex64", "SDuMS", next(rng_iterator), mpo_bond_dim=2),
    create_state_factory(5, "complex128", "SDgDS", next(rng_iterator), initial_mps_dim=2),
    # Diagonal gate test cases
    create_state_factory(5, "complex128", "SDgADS", next(rng_iterator), initial_mps_dim=2)
]

noisy_state_tests_L1 = [
    create_state_factory(6, "complex128", "SDuDS", next(rng_iterator), initial_mps_dim=3),
    create_state_factory(6, "complex128", "SDgDS", next(rng_iterator), adjacent_double_layer=False),
    create_state_factory(5, "complex128", "SDUDS", next(rng_iterator), adjacent_double_layer=False, initial_mps_dim=2),
    create_state_factory(7, "complex128", "SDGDS", next(rng_iterator), adjacent_double_layer=False),
    create_state_factory(6, "complex128", "SDuMS", next(rng_iterator), mpo_bond_dim=2, initial_mps_dim=2),
    create_state_factory(5, "complex128", "SDgMDS", next(rng_iterator), mpo_bond_dim=2, mpo_num_sites=3, mpo_geometry="random"),
    create_state_factory(5, "complex128", "SDuDgDS", next(rng_iterator)),
    # Diagonal gate test cases
    create_state_factory(6, "complex128", "SDAuDS", next(rng_iterator), initial_mps_dim=3),
    create_state_factory(6, "complex128", "SADgDS", next(rng_iterator), adjacent_double_layer=False),
]

class NoisyStateMatrix(StateMatrixABC):

    @staticmethod
    def L0():
        return noisy_state_tests_L0
    
    @staticmethod
    def L1():
        return noisy_state_tests_L1


# --- Expectation gradient test configs ---
# All use StateFactory with S/D layers only (plain gates for TorchRef).
#
# Each level (``expectation_gradient_L*`` and ``expectation_gradient_*_torch``) lists the same
# four coverage cases (distinct circuits / seeds / dtypes may differ by level):
#   1. Hermitian operator (Pauli dict) + unitary gates
#   2. Hermitian + ``mark_non_unitary=True``
#   3. Non-Hermitian (unitary/non-unitary) ``NetworkOperator`` + unitary gates
#   4. Non-Hermitian (unitary/non-unitary) + ``mark_non_unitary=True``
#
# When the factory uses ``mark_non_unitary=True``, ``_exp_grad_config`` sets ``return_norm`` and
# ``state_norm_adjoint`` so the norm-network adjoint path is exercised for explicitly non-unitary gates.

def _exp_grad_config(dtype, backend, factory, hamiltonian=None, **kwargs):
    """Build one expectation-gradient test config dict (shared keys, default adjoints).
    hamiltonian is either a Pauli string dict or a NetworkOperatorFactory."""
    # if dtype == "complex64" or dtype == "complex128":
    #     expectation_value_adjoint = 2.0+1.0j
    #     state_norm_adjoint = -3.0+2.0j
    # else:
    #     expectation_value_adjoint = 2.0
    #     state_norm_adjoint = -3.0
    out = {
        "dtype": dtype,
        "backend": backend,
        "factory": factory,
        # "expectation_value_adjoint": expectation_value_adjoint, # only used with TorchRef0 not TorchRef
        
    }
    if hamiltonian is not None:
        out["hamiltonian"] = hamiltonian
    if getattr(factory, "mark_non_unitary", False):
        out["return_norm"] = True
        # out["state_norm_adjoint"] = state_norm_adjoint # only used with TorchRef0 not TorchRef
    out.update(kwargs)
    return out

expectation_gradient_L0 = [
    # (1) Hermitian operator + unitary gates
    _exp_grad_config(
        "complex64", "cupy",
        create_state_factory(4, "complex64", "SDSDS", np.random.default_rng(41), backend="cupy", mark_gradients=True),
        hamiltonian={"ZZXX": 2.0, "XZXZ": 3.0j},
    ),
    # (2) Hermitian operator + mark_non_unitary + remove_identity
    _exp_grad_config(
        "float32", "numpy",
        create_state_factory(6, "float32", "SDSDD", np.random.default_rng(42), backend="numpy", mark_gradients=True, mark_non_unitary=True),
        hamiltonian={"ZZXIXZ": 2.0, "XIZIIX": 3.0},
        remove_identity=True,
    ),
    # (3) Non-unitary operator + unitary gates
    _exp_grad_config(
        "complex128", "cupy",
        create_state_factory((2, 3, 2, 4, 2, 5, 2, 3), "complex128", "SDDDSSD", np.random.default_rng(43), backend="cupy", mark_gradients=True),
        hamiltonian=NetworkOperatorFactory(
            (2, 3, 2, 4, 2, 5, 2, 3), np.random.default_rng(38), "cupy", dtype="complex128",
            num_repeats=2, real_coefficients=False, use_random_non_unitary=True, add_mpo=False,
        ),
    ),
    # (4) Unitary operator + mark_non_unitary
    _exp_grad_config(
        "float64", "cupy",
        create_state_factory((3, 3, 3, 3), "float64", "SDSD", np.random.default_rng(44), backend="cupy", mark_gradients=True, mark_non_unitary=True),
        hamiltonian=NetworkOperatorFactory(
            (3, 3, 3, 3), np.random.default_rng(39), "cupy", dtype="float64",
            num_repeats=2, real_coefficients=True, use_random_unitary=True, add_mpo=True,
        ),
    ),
]

expectation_gradient_L0_torch = [
    # (1) Hermitian operator + unitary gates (torch, qubits) + remove_identity
    _exp_grad_config(
        "complex128", "torch",
        create_state_factory(4, "complex128", "SDSDS", np.random.default_rng(45), backend="torch", mark_gradients=True),
        hamiltonian={"ZIXX": 1.0, "XIXZ": 3+2.5j},
        remove_identity=True
    ),
    # (2) Hermitian operator + mark_non_unitary + remove_identity
    _exp_grad_config(
        "float32", "torch",
        create_state_factory(6, "float32", "SDSDD", np.random.default_rng(46), backend="torch", mark_gradients=True, mark_non_unitary=True),
        hamiltonian={"ZZIXXI": 2.0, "IZXZIZ": 0.5},
        remove_identity=True
    ),
    # (3) Unitary operator + unitary gates (MPO, qudits)
    _exp_grad_config(
        "float64", "torch",
        create_state_factory((2, 3, 2, 4, 2, 5, 2, 3), "float64", "SDSDSD", np.random.default_rng(47), backend="torch", mark_gradients=True),
        hamiltonian=NetworkOperatorFactory(
            (2, 3, 2, 4, 2, 5, 2, 3), np.random.default_rng(31), "torch", dtype="float64",
            num_repeats=3, real_coefficients=True, use_random_unitary=True, add_mpo=True,
        ),
    ),
    # (4) Non-unitary operator + mark_non_unitary
    _exp_grad_config(
        "complex64", "torch",
        create_state_factory(6, "complex64", "SDSDSD", np.random.default_rng(48), backend="torch", mark_gradients=True, mark_non_unitary=True),
        hamiltonian=NetworkOperatorFactory(
            (2, 2, 2, 2, 2, 2), np.random.default_rng(32), "torch", dtype="complex64",
            num_repeats=3, real_coefficients=False, use_random_non_unitary=True, add_mpo=True,
        ),
    ),
] if torch is not None else []

expectation_gradient_L1 = [
    # (1) Hermitian operator + unitary gates + remove_identity
    _exp_grad_config(
        "complex64", "cupy",
        create_state_factory(8, "complex64", "SDDSD", np.random.default_rng(49), backend="cupy", mark_gradients=True),
        hamiltonian={"ZYIZXZIZ": 5.0, "XZZYIZXZ": 2.0, "ZZYIXZYY": 3.0},
        remove_identity=True,
    ),
    # (2) Hermitian operator + mark_non_unitary + remove_identity
    _exp_grad_config(
        "complex64", "cupy",
        create_state_factory(6, "complex64", "SDSDDSD", np.random.default_rng(50), backend="cupy", mark_gradients=True, mark_non_unitary=True),
        hamiltonian={"ZYIZXZ": 5.0, "XYIZXZ": 2.0+3.5j, "ZIXZYY": 3.0},
        remove_identity=True,
    ),
    # (3) Non-unitary operator + unitary gates (numpy, qudits, MPO)
    _exp_grad_config(
        "float32", "numpy",
        create_state_factory((3, 2, 4, 4, 2, 5), "float32", "SSDSD", np.random.default_rng(51), backend="numpy", mark_gradients=True),
        hamiltonian=NetworkOperatorFactory(
            (3, 2, 4, 4, 2, 5), np.random.default_rng(32), "numpy", dtype="float32",
            num_repeats=3, real_coefficients=True, use_random_non_unitary=True, add_mpo=True,
        ),
    ),
    # (4) Unitary operator + mark_non_unitary
    _exp_grad_config(
        "complex128", "numpy",
        create_state_factory((3, 3, 4, 4, 2, 5), "complex128", "SDSSDSD", np.random.default_rng(52), backend="numpy", mark_gradients=True, mark_non_unitary=True),
        hamiltonian=NetworkOperatorFactory(
            (3, 3, 4, 4, 2, 5), np.random.default_rng(32), "numpy", dtype="complex128",
            num_repeats=3, real_coefficients=False, use_random_unitary=True, add_mpo=True,
        ),
    ),
]

expectation_gradient_L1_torch = [
    # (1) Hermitian operator + unitary gates + remove_identity
    _exp_grad_config(
        "float64", "torch",
        create_state_factory(6, "float64", "SDSDSDS", np.random.default_rng(53), backend="torch", mark_gradients=True),
        hamiltonian={"ZXIXZI": 4.0, "IXZIZX": 3.0},
        remove_identity=True,
    ),
    # (2) Hermitian operator + mark_non_unitary + remove_identity
    _exp_grad_config(
        "complex128", "torch",
        create_state_factory(6, "complex128", "SSDSDS", np.random.default_rng(54), backend="torch", mark_gradients=True, mark_non_unitary=True),
        hamiltonian={"ZXIXZI": 4.0j, "IXZIZX": 3.0+2.5j},
        remove_identity=True,
    ),
    # (3) Unitary operator + unitary gates
    _exp_grad_config(
        "complex64", "torch",
        create_state_factory((2, 3, 2, 3), "complex64", "SDDS", np.random.default_rng(55), backend="torch", mark_gradients=True),
        hamiltonian=NetworkOperatorFactory(
            (2, 3, 2, 3), np.random.default_rng(39), "torch", dtype="complex64",
            num_repeats=3, real_coefficients=False, use_random_unitary=True, add_mpo=True,
        ),
    ),
    # (4) Non-unitary operator + mark_non_unitary
    _exp_grad_config(
        "complex64", "torch",
        create_state_factory((2, 3, 2, 3, 5, 5), "complex64", "SDSDS", np.random.default_rng(56), backend="torch", mark_gradients=True, mark_non_unitary=True),
        hamiltonian=NetworkOperatorFactory(
            (2, 3, 2, 3, 5, 5), np.random.default_rng(39), "torch", dtype="complex64",
            num_repeats=3, real_coefficients=True, use_random_non_unitary=True, add_mpo=True,
        ),
    ),
] if torch is not None else []

expectation_gradient_L2 = [
    # (1) Hermitian operator + unitary gates (8-qubit Pauli, cupy) + remove_identity
    _exp_grad_config(
        "complex128", "cupy",
        create_state_factory(8, "complex128", "SDSDDSD", np.random.default_rng(57), backend="cupy", mark_gradients=True),
        hamiltonian={"ZYIZXZIZ": 1.0j, "XZZYIZXZ": 0.5+0.25j, "ZZYIXZYY": 0.25+0.125j},
        remove_identity=True,
    ),
    # (2) Hermitian operator + mark_non_unitary
    _exp_grad_config(
        "complex64", "cupy",
        create_state_factory(8, "complex64", "SDSDDSD", np.random.default_rng(58), backend="cupy", mark_gradients=True, mark_non_unitary=True),
        hamiltonian={"ZYIZXZIZ": 1.0, "XZZYIZXZ": 0.5, "ZZYIXZYY": 0.25},
        remove_identity=True,
    ),
    # (3) Non-unitary operator + unitary gates (larger qudit system, cupy, MPO)
    _exp_grad_config(
        "float64", "cupy",
        create_state_factory((3, 3, 3, 3, 3), "float64", "SSDDS", np.random.default_rng(59), backend="cupy", mark_gradients=True),
        hamiltonian=NetworkOperatorFactory(
            (3, 3, 3, 3, 3), np.random.default_rng(33), "cupy", dtype="float64",
            num_repeats=4, real_coefficients=True, use_random_non_unitary=True, add_mpo=True,
        ),
    ),
    # (4) Unitary operator + mark_non_unitary
    _exp_grad_config(
        "complex128", "cupy",
        create_state_factory((3, 3, 5, 3, 3), "complex128", "SSDDS", np.random.default_rng(60), backend="cupy", mark_gradients=True, mark_non_unitary=True),
        hamiltonian=NetworkOperatorFactory(
            (3, 3, 5, 3, 3), np.random.default_rng(33), "cupy", dtype="complex128",
            num_repeats=4, real_coefficients=False, use_random_unitary=True, add_mpo=True,
        ),
    ),
]

expectation_gradient_L2_torch = [
    # (1) Hermitian operator + unitary gates + remove_identity
    _exp_grad_config(
        "complex128", "torch",
        create_state_factory(6, "complex128", "SDSDD", np.random.default_rng(61), backend="torch", mark_gradients=True),
        hamiltonian={"ZZXXZZ": 2.0j, "XZXYII": 3.0},
        remove_identity=True,
    ),
    # (2) Hermitian operator + mark_non_unitary + remove_identity
    _exp_grad_config(
        "float32", "torch",
        create_state_factory(6, "float32", "SDSDSD", np.random.default_rng(62), backend="torch", mark_gradients=True, mark_non_unitary=True),
        hamiltonian={"ZZXXZZ": 2.0, "XZXZII": 3.0},
        remove_identity=True,
    ),
    # (3) Unitary operator + unitary gates (torch, qudits, MPO)
    _exp_grad_config(
        "float64", "torch",
        create_state_factory((2, 3, 2, 4, 2, 3), "float64", "SDSDSS", np.random.default_rng(63), backend="torch", mark_gradients=True),
        hamiltonian=NetworkOperatorFactory(
            (2, 3, 2, 4, 2, 3), np.random.default_rng(36), "torch", dtype="float64",
            num_repeats=4, real_coefficients=True, use_random_unitary=True, add_mpo=True,
        ),
    ),
    # (4) Non-unitary operator + mark_non_unitary
    _exp_grad_config(
        "complex64", "torch",
        create_state_factory((3, 2, 4, 5, 2, 3), "complex64", "SDSDSS", np.random.default_rng(64), backend="torch", mark_gradients=True, mark_non_unitary=True),
        hamiltonian=NetworkOperatorFactory(
            (3, 2, 4, 5, 2, 3), np.random.default_rng(36), "torch", dtype="complex64",
            num_repeats=4, real_coefficients=False, use_random_non_unitary=True, add_mpo=True,
        ),
    ),
] if torch is not None else []


class EOverNDenominatorTooSmall(RuntimeError):
    """``Re(N)`` is degenerate (below ``finfo.tiny``); ``E/N`` is not meaningful for the reference test."""



def check_e_over_n_well_conditioned(E, N):
    """Raise :exc:`EOverNDenominatorTooSmall` only when ``Re(N)`` is not a positive normalized float.

    Squared norms can be arbitrarily small (< 1) for legitimate non‑unitary states; rejecting ``|Re(N)| < sqrt(eps)``
    was too aggressive on float32. We only guard the true failure mode: zero / denormal‑scale denominator (use
    ``finfo.tiny``, smallest normalized positive float).
    """
    if torch is None or N is None:
        return
    rd = torch.real(E).dtype
    den = torch.real(N).to(dtype=rd, device=E.device)
    thr = float(torch.finfo(rd).tiny)
    abs_den = float(torch.abs(den).detach().cpu())
    if abs_den < thr:
        raise EOverNDenominatorTooSmall(
            f"e_over_n: |Re(N)|={abs_den:g} below dtype smallest-normal threshold {thr:g}; skip degenerate denominator."
        )


def expectation_gradient_loss_factory(loss_variant):
    """Return ``loss_fn(E, N)`` (real scalar) for parametrized expectation-gradient tests vs CUTN.

    Variants ``e_over_n`` and ``e_times_n`` need ``N`` (``return_norm=True``).

    Variants ``e2_plus_3j_n`` and ``linear_affine_e_n`` also support ``N is None``: then they use
    ``Re(E^2)`` only, or ``Re((2+3j) E)`` / ``3 E`` respectively (affine terms drop the ``N`` part).

    ``e_over_n``: ``Re(E / N_real)`` with ``N`` cast to ``E``'s real dtype. If ``Re(N)`` is not safely above
    ``finfo.tiny`` (smallest normalized value), raises :exc:`EOverNDenominatorTooSmall` so tests can skip the
    degenerate zero-/denominator case only (small but normal ``‖ψ‖²`` remains valid).

    ``linear_affine_e_n``: real ``E`` → ``3E + 4N`` (or ``3E`` if ``N`` is omitted); complex ``E`` →
    ``Re((2+3j) E + 6j N)`` or ``Re((2+3j) E)`` when ``N`` is omitted.
    """
    if torch is None:
        raise ImportError("expectation_gradient_loss_factory requires PyTorch")

    if loss_variant == "e_over_n":

        def loss_fn(E, N):
            if N is None:
                raise ValueError("e_over_n requires N (use return_norm=True)")
            check_e_over_n_well_conditioned(E, N)
            rd = torch.real(E).dtype
            den = torch.real(N).to(dtype=rd, device=E.device)
            return torch.real(E / den)

        return loss_fn

    if loss_variant == "e2_plus_3j_n":

        def loss_fn(E, N):
            if N is None:
                Ec = (
                    E
                    if torch.is_complex(E)
                    else E.to(torch.complex64 if E.dtype == torch.float32 else torch.complex128)
                )
                return torch.real(Ec * Ec)
            if not torch.is_complex(E):
                E = E.to(torch.complex64 if E.dtype == torch.float32 else torch.complex128)
            three_j = 3.0 * torch.tensor(1j, dtype=E.dtype, device=E.device)
            Nc = torch.as_tensor(torch.real(N), dtype=E.dtype, device=E.device)
            z = E * E + three_j * Nc
            return torch.real(z)

        return loss_fn

    if loss_variant == "e_times_n":

        def loss_fn(E, N):
            if N is None:
                raise ValueError("e_times_n requires N (use return_norm=True)")
            return torch.real(E * torch.as_tensor(torch.real(N), dtype=E.dtype, device=E.device))

        return loss_fn

    if loss_variant == "linear_affine_e_n":

        def loss_fn(E, N):
            if N is None:
                if torch.is_complex(E):
                    c_e = torch.tensor(2 + 3j, dtype=E.dtype, device=E.device)
                    return torch.real(c_e * E)
                return 3.0 * E
            Nr = torch.as_tensor(torch.real(N), dtype=(torch.real(E).dtype), device=E.device)
            if torch.is_complex(E):
                c_e = torch.tensor(2 + 3j, dtype=E.dtype, device=E.device)
                six_j = torch.tensor(6j, dtype=E.dtype, device=E.device)
                z = c_e * E + six_j * Nr.to(dtype=E.dtype)
                return torch.real(z)
            return 3.0 * E + 4.0 * Nr

        return loss_fn

    raise ValueError(f"unknown loss_variant={loss_variant!r}")


class ExpectationGradientConfig:
    """Config lists for expectation gradient tests (compute_expectation_with_gradients vs TorchRef)."""

    @staticmethod
    def L0():
        return expectation_gradient_L0 + expectation_gradient_L0_torch

    @staticmethod
    def L1():
        return expectation_gradient_L1 + expectation_gradient_L1_torch

    @staticmethod
    def L2():
        return expectation_gradient_L2 + expectation_gradient_L2_torch