# Copyright (c) 2023-2026, NVIDIA CORPORATION & AFFILIATES
#
# SPDX-License-Identifier: BSD-3-Clause

import pytest
import numpy as np

from ..utils.circuit_ifc import QuantumStateTestHelper, PropertyComputeHelper
from ._internal.state_factory import get_random_network_operator, StateFactory
from ._internal.state_matrix import MPSConfigMatrix, NoisyStateMatrix, SimulationConfigMatrix
from ._internal.state_tester import BaseNoisyStateTester
from ._internal.state_utils import compute_noisy_sv, verify_state_sampling, analyze_trajectory_deviation
from ..utils.helpers import _BaseTester, get_contraction_tolerance

from cuquantum.tensornet import CircuitToEinsum, contract
from cuquantum.tensornet.experimental import NetworkState, MPSConfig, TNConfig

NUM_TESTS_PER_CONFIG = 3
NUM_TRAJECTORIES_PER_CONFIG = 10


def skip_unsupported_noisy_state_tests(noisy_factory, config, pure_state=True):
    has_general_channel = 'g' in noisy_factory.layers or 'G' in noisy_factory.layers
    if has_general_channel:
        if isinstance(config, TNConfig) or config == {}:
            if pure_state:
                pytest.skip("TNConfig is not supported for general channel simulation with pure_state=True")
        elif isinstance(config, MPSConfig):
            if config.gauge_option == 'simple':
                pytest.skip("Simple update is not supported for general channel simulation")
        elif isinstance(config, dict):
            if config.get('gauge_option', 'free') == 'simple':
                pytest.skip("Simple update is not supported for general channel simulation")
        else:
            raise ValueError(f"Unknown config type: {type(config)}")

@pytest.fixture(params=NoisyStateMatrix.L0(), scope="class")
def noisy_factory_L0(request):
    return request.param

@pytest.fixture(params=MPSConfigMatrix.approxConfigsL0(), scope="class")
def approx_mps_config_L0(request):
    return request.param

@pytest.fixture(scope="class")
def noisy_state_results_L0(noisy_factory_L0, approx_mps_config_L0):
    skip_unsupported_noisy_state_tests(noisy_factory_L0, approx_mps_config_L0)
    # NOTE: torch seems to be holding on to the result tensors, so here we force numpy as reference
    force_numpy = noisy_factory_L0.backend.name == 'torch'
    return compute_noisy_sv(noisy_factory_L0, approx_mps_config_L0, return_probabilities=True, force_numpy=force_numpy)

class TestTrajectoryNoisyStateFunctionality(_BaseTester):

    def test_release_operators(self, noisy_factory_L0, approx_mps_config_L0):
        skip_unsupported_noisy_state_tests(noisy_factory_L0, approx_mps_config_L0)
        with noisy_factory_L0.to_network_state(config=approx_mps_config_L0) as state:
            state.compute_output_state(release_operators=True)
            sv = state.compute_state_vector()
            # when release_operator is set to True, 
            # subsequent property calculations should be all corresponding to fixed final state
            n_qubits = sv.ndim
            bitstring = '0' * n_qubits
            amp = state.compute_amplitude(bitstring)
            QuantumStateTestHelper.verify_amplitude(sv, bitstring, amp)

            fixed = {0: '1', 1: '0'}
            batched_amps = state.compute_batched_amplitudes(fixed)
            QuantumStateTestHelper.verify_batched_amplitudes(sv, fixed, batched_amps)

            where = (0, 1)
            rdm = state.compute_reduced_density_matrix(where)
            QuantumStateTestHelper.verify_reduced_density_matrix(sv, where, rdm)
            if set(state.state_mode_extents) == {2} and "complex" in state.dtype:
                operators = {
                    'X' * n_qubits: 0.2,
                    'Y' * n_qubits: 0.3,
                    'Z' * n_qubits: 0.4,
                    'I' * n_qubits: 0.1,
                } 
                exp = state.compute_expectation(operators)
                QuantumStateTestHelper.verify_expectation(sv, operators, exp)
            modes = list(range(n_qubits))
            verify_state_sampling(state, modes, 1000, [sv, ], 3)
    

    NUM_MAX_TRAJECTORIES = 50000
    NUM_TRAJECTORIES_PER_CHECK = 500
    @pytest.mark.parametrize(
        "factory", (StateFactory(4, "float64", "SDuDS", np.random.default_rng(0)),
                    StateFactory(4, "float64", "SDgDS", np.random.default_rng(1)),
                    StateFactory(5, "float64", "SDugS", np.random.default_rng(2)))
    )
    @pytest.mark.parametrize(
        "config", (TNConfig(), 
                   MPSConfig(max_extent=2, gauge_option='simple'), 
                   MPSConfig(max_extent=2, gauge_option='free'))
    )
    def test_distribution(self, factory, config):
        skip_unsupported_noisy_state_tests(factory, config)
        rng = self._get_rng(factory, config, "distribution")
        operator = get_random_network_operator(
            factory.state_dims, 
            rng, 
            factory.backend.name,
            num_repeats=1,
            dtype=factory.dtype, 
        )
        results = compute_noisy_sv(factory, config, return_probabilities=True)
        
        expecs = []
        probs = []
        for p, sv in results:
            expecs.append(PropertyComputeHelper.expectation_from_sv(sv, operator))
            probs.append(p)
        
        expecs = np.asarray(expecs).reshape(1, -1)

        with factory.to_network_state(config=config) as state:
            expec_traj_results = []
            prob_deviation = 1
            for i in range(1, self.NUM_MAX_TRAJECTORIES + 1):
                expec_traj_results.append(state.compute_expectation(operator))
                if i % self.NUM_TRAJECTORIES_PER_CHECK == 0:
                    min_deviation, prob_deviation = analyze_trajectory_deviation(expec_traj_results, expecs, probs)
                    assert np.isclose(min_deviation, 0), f"trajectory results not found in the configuration space, min_deviation: {min_deviation}"
                    if prob_deviation < 0.05:
                        break
                    print(f"num_trajectories: {i}, prob_deviation: {prob_deviation}, adding another {self.NUM_TRAJECTORIES_PER_CHECK} trajectories")
            if prob_deviation < 0.05:
                print(f"Passed after {i} trajectories, {prob_deviation=}")
            else:
                assert False, f"Failed after {i} trajectories, {prob_deviation=}"


@pytest.fixture(params=NoisyStateMatrix.L1(), scope="class")
def noisy_factory_L1(request):
    return request.param

@pytest.fixture(params=MPSConfigMatrix.approxConfigsL1()+SimulationConfigMatrix.exactConfigs(), scope="class")
def noisy_config(request):
    return request.param

@pytest.fixture(scope="class")
def noisy_factory_svs_L1(noisy_factory_L1, noisy_config):
    skip_unsupported_noisy_state_tests(noisy_factory_L1, noisy_config)
    # NOTE: torch seems to be holding on to the result tensors, so here we force numpy as reference
    force_numpy = noisy_factory_L1.backend.name == 'torch'
    return compute_noisy_sv(noisy_factory_L1, noisy_config, return_probabilities=False, force_numpy=force_numpy)

class TestTrajectoryNoisyStateCorrectness(BaseNoisyStateTester):

    def test_state_vector(self, noisy_factory_L1, noisy_config, noisy_factory_svs_L1):
        with noisy_factory_L1.to_network_state(config=noisy_config) as state:
            super()._test_state_vector(noisy_factory_svs_L1, state, NUM_TRAJECTORIES_PER_CONFIG)
    
    def test_amplitude(self, noisy_factory_L1, noisy_config, noisy_factory_svs_L1):
        rng = self._get_rng(noisy_factory_L1, "amplitude")
        with noisy_factory_L1.to_network_state(config=noisy_config) as state:
            super()._test_amplitude(noisy_factory_svs_L1, state, NUM_TESTS_PER_CONFIG, rng, NUM_TRAJECTORIES_PER_CONFIG)
    
    def test_batched_amplitudes(self, noisy_factory_L1, noisy_config, noisy_factory_svs_L1):
        rng = self._get_rng(noisy_factory_L1, "batched_amplitudes")
        with noisy_factory_L1.to_network_state(config=noisy_config) as state:
            super()._test_batched_amplitudes(noisy_factory_svs_L1, state, NUM_TESTS_PER_CONFIG, rng, NUM_TRAJECTORIES_PER_CONFIG)

    def test_expectation(self, noisy_factory_L1, noisy_config, noisy_factory_svs_L1):
        rng = self._get_rng(noisy_factory_L1, "expectation")
        with noisy_factory_L1.to_network_state(config=noisy_config) as state:
            super()._test_expectation(noisy_factory_svs_L1, state, rng, NUM_TRAJECTORIES_PER_CONFIG)

    def test_reduced_density_matrix(self, noisy_factory_L1, noisy_config, noisy_factory_svs_L1):
        rng = self._get_rng(noisy_factory_L1, "reduced_density_matrix")
        with noisy_factory_L1.to_network_state(config=noisy_config) as state:
            super()._test_reduced_density_matrix(noisy_factory_svs_L1, state, NUM_TESTS_PER_CONFIG, rng, NUM_TRAJECTORIES_PER_CONFIG)

    def test_sampling(self, noisy_factory_L1, noisy_config, noisy_factory_svs_L1):
        with noisy_factory_L1.to_network_state(config=noisy_config) as state:
            super()._test_sampling(noisy_factory_svs_L1, state, NUM_TRAJECTORIES_PER_CONFIG)


# --- Mixed-state noisy tests ---
# Reuse the factory infrastructure: compute_noisy_sv enumerates all trajectory
# branches as (probability, state_vector) pairs.  The mixed-state density matrix
# must equal sum_i p_i * RDM(psi_i).

_mixed_noisy_factories = [
    StateFactory(4, "complex128", "SDUDS", np.random.default_rng(7000)),
    StateFactory(4, "complex128", "SDGDS", np.random.default_rng(7001)),
    StateFactory(4, "float64", "SDUDS", np.random.default_rng(7002)),
    StateFactory(5, "complex128", "SDuDgDS", np.random.default_rng(7003)),
]

def _mixed_reference_rdm_from_dm(dm_mat, where, state_dims):
    """Compute reference RDM by tracing out non-where modes from a full density matrix."""
    n = len(state_dims)
    dm_tensor = dm_mat.reshape(tuple(state_dims) * 2)
    trace_out = [i for i in range(n) if i not in where]
    for offset, mode in enumerate(sorted(trace_out)):
        bra_mode = mode + n - 2 * offset
        dm_tensor = np.trace(dm_tensor, axis1=mode - offset, axis2=bra_mode)
    return dm_tensor


def _mixed_reference_rdm(prob_sv_pairs, where):
    """Compute reference RDM from enumerated (probability, state_vector) pairs."""
    rdm = None
    for p, sv in prob_sv_pairs:
        sv_np = np.asarray(sv)
        rdm_i = np.asarray(PropertyComputeHelper.reduced_density_matrix_from_sv(sv_np, where))
        if rdm is None:
            rdm = p * rdm_i
        else:
            rdm = rdm + p * rdm_i
    return rdm


@pytest.fixture(params=_mixed_noisy_factories, scope="class")
def mixed_noisy_factory(request):
    return request.param

@pytest.fixture(scope="class")
def mixed_noisy_reference(mixed_noisy_factory):
    return compute_noisy_sv(mixed_noisy_factory, TNConfig(), return_probabilities=True, force_numpy=True)


class TestMixedNoisyState(_BaseTester):
    """Mixed-state noisy simulation must match the trajectory-enumerated reference."""

    def test_property_consistency(self, mixed_noisy_factory):
        """All property APIs must be mutually consistent for a mixed noisy state."""
        with mixed_noisy_factory.to_network_state(config=TNConfig(), pure_state=False) as state:
            n = state.n
            dim = int(np.prod(state.state_mode_extents))
            all_modes = tuple(range(n))

            dm = state.compute_density_matrix()
            dm_mat = np.asarray(dm).reshape(dim, dim)
            tol = get_contraction_tolerance(state.dtype)

            bs = tuple(0 for _ in range(n))
            amp = state.compute_amplitude((bs, bs))
            idx = bs
            ref_diag = dm[idx + idx]
            np.testing.assert_allclose(complex(amp).real, ref_diag.real, **tol)

            where = tuple(range(min(2, n)))
            rdm = state.compute_reduced_density_matrix(where)
            rdm_ref = _mixed_reference_rdm_from_dm(dm_mat, where, state.state_mode_extents)
            np.testing.assert_allclose(np.asarray(rdm), rdm_ref, **tol)

            if set(state.state_mode_extents) == {2} and "complex" in state.dtype:
                operators = {
                    'Z' * n: 0.5,
                    'I' * n: 0.5,
                }
                exp = state.compute_expectation(operators)
                pauli_z_full = np.diag([(-1) ** bin(i).count('1') for i in range(dim)])
                exp_ref = 0.5 * np.trace(dm_mat @ pauli_z_full) + 0.5 * np.trace(dm_mat)
                np.testing.assert_allclose(exp, exp_ref, **tol)

    def test_rdm_matches_reference(self, mixed_noisy_factory, mixed_noisy_reference):
        where = (0, 1)
        rdm_ref = _mixed_reference_rdm(mixed_noisy_reference, where)
        with mixed_noisy_factory.to_network_state(config=TNConfig(), pure_state=False) as state:
            rdm = state.compute_reduced_density_matrix(where)
        tol = get_contraction_tolerance(mixed_noisy_factory.dtype)
        np.testing.assert_allclose(np.asarray(rdm), rdm_ref, **tol)

    def test_expectation_matches_reference(self, mixed_noisy_factory, mixed_noisy_reference):
        if set(mixed_noisy_factory.state_dims) != {2} or "complex" not in mixed_noisy_factory.dtype:
            pytest.skip("Pauli expectation requires qubits with complex dtype")
        rng = np.random.default_rng(8000)
        n = mixed_noisy_factory.num_qudits
        pauli_chars = 'IXYZ'
        pauli_str = ''.join(rng.choice(list(pauli_chars)) for _ in range(n))
        operator = {pauli_str: 1.0}
        expec_ref = sum(
            p * PropertyComputeHelper.expectation_from_sv(np.asarray(sv), operator)
            for p, sv in mixed_noisy_reference
        )
        with mixed_noisy_factory.to_network_state(config=TNConfig(), pure_state=False) as state:
            expec = state.compute_expectation(operator)
        tol = get_contraction_tolerance(mixed_noisy_factory.dtype)
        np.testing.assert_allclose(expec, expec_ref, **tol)

    def test_full_rdm_trace_matches_reference(self, mixed_noisy_factory, mixed_noisy_reference):
        """Full density matrix trace must match the trajectory-weighted norm.

        The factory's standard gate layers ('S') use Hermitian matrices normalized
        to Frobenius norm 1, which are not unitary.  The overall circuit is therefore
        not trace-preserving, so Tr(rho) != 1 in general.  The correct reference
        trace is sum_i p_i * ||psi_i||^2.
        """
        n = mixed_noisy_factory.num_qudits
        dim = int(np.prod(mixed_noisy_factory.state_dims))
        all_modes = tuple(range(n))

        ref_trace = sum(
            p * float((np.asarray(sv).conj().ravel() @ np.asarray(sv).ravel()).real)
            for p, sv in mixed_noisy_reference
        )

        with mixed_noisy_factory.to_network_state(config=TNConfig(), pure_state=False) as state:
            rdm = state.compute_reduced_density_matrix(all_modes)
        trace = np.trace(np.asarray(rdm).reshape(dim, dim)).real
        tol = get_contraction_tolerance(mixed_noisy_factory.dtype)
        np.testing.assert_allclose(trace, ref_trace, **tol)


# --- Cross-validation: from_circuit with embedded channels ---

def _make_qiskit_noisy_circuit():
    from qiskit import QuantumCircuit
    from qiskit.quantum_info import Kraus
    from qiskit_aer.noise import depolarizing_error
    qc = QuantumCircuit(2)
    qc.h(0)
    qc.cx(0, 1)
    qc.append(Kraus(depolarizing_error(0.1, 1)), [0])
    return qc

def _make_cirq_noisy_circuit():
    import cirq
    q0, q1 = cirq.LineQubit.range(2)
    return cirq.Circuit([cirq.H(q0), cirq.CNOT(q0, q1), cirq.DepolarizingChannel(p=0.1).on(q0)])

_NOISY_CIRCUIT_FACTORIES = {
    "qiskit_depolarizing": _make_qiskit_noisy_circuit,
    "cirq_depolarizing": _make_cirq_noisy_circuit,
}

def _c2e_ref_density_matrix(circuit):
    """Reference density matrix via CircuitToEinsum contraction (already validated)."""
    converter = CircuitToEinsum(circuit, dtype='complex128', backend='numpy')
    expr, operands = converter.density_matrix()
    return np.asarray(contract(expr, *operands))


class TestMixedNoisyFromCircuit(_BaseTester):
    """Mixed-state NetworkState.from_circuit / from_converter with circuits containing quantum channels.

    Uses CircuitToEinsum contraction as the reference (validated in test_circuit_converter.py).
    """

    @pytest.fixture(params=list(_NOISY_CIRCUIT_FACTORIES.keys()))
    def noisy_circuit(self, request):
        try:
            return _NOISY_CIRCUIT_FACTORIES[request.param]()
        except ImportError:
            pytest.skip(f"dependencies for {request.param} not available")

    def test_from_circuit_noisy_density_matrix(self, noisy_circuit):
        rho_ref = _c2e_ref_density_matrix(noisy_circuit)
        with NetworkState.from_circuit(noisy_circuit, pure_state=False, backend='numpy') as state:
            rho = state.compute_density_matrix()
            tol = get_contraction_tolerance(state.dtype)
            np.testing.assert_allclose(rho, rho_ref, **tol)

    def test_from_circuit_noisy_rdm(self, noisy_circuit):
        rho_ref = _c2e_ref_density_matrix(noisy_circuit)
        with NetworkState.from_circuit(noisy_circuit, pure_state=False, backend='numpy') as state:
            rdm = state.compute_reduced_density_matrix(state.state_labels[:1])
            n = state.n
            dim = 2 ** n
            rho_mat = rho_ref.reshape(dim, dim)
            rdm_ref = np.zeros((2, 2), dtype=complex)
            for i in range(2):
                for j in range(2):
                    for k in range(dim // 2):
                        rdm_ref[i, j] += rho_mat[i * (dim // 2) + k, j * (dim // 2) + k]
            tol = get_contraction_tolerance(state.dtype)
            np.testing.assert_allclose(np.asarray(rdm).reshape(2, 2), rdm_ref, **tol)

    def test_from_circuit_noisy_amplitude(self, noisy_circuit):
        rho_ref = _c2e_ref_density_matrix(noisy_circuit)
        with NetworkState.from_circuit(noisy_circuit, pure_state=False, backend='numpy') as state:
            n = state.n
            bs = tuple(0 for _ in range(n))
            amp = state.compute_amplitude((bs, bs))
            ref_diag = rho_ref[bs + bs]
            tol = get_contraction_tolerance(state.dtype)
            np.testing.assert_allclose(complex(amp).real, ref_diag.real, **tol)

    def test_from_circuit_noisy_expectation(self, noisy_circuit):
        rho_ref = _c2e_ref_density_matrix(noisy_circuit)
        with NetworkState.from_circuit(noisy_circuit, pure_state=False, backend='numpy') as state:
            n = state.n
            dim = 2 ** n
            rho_mat = rho_ref.reshape(dim, dim)
            pauli_z_full = np.diag([(-1) ** bin(i).count('1') for i in range(dim)])
            exp_ref = np.trace(rho_mat @ pauli_z_full).real
            exp = state.compute_expectation('Z' * n)
            tol = get_contraction_tolerance(state.dtype)
            np.testing.assert_allclose(np.real(exp), exp_ref, **tol)

    def test_from_converter_noisy(self, noisy_circuit):
        rho_ref = _c2e_ref_density_matrix(noisy_circuit)
        converter = CircuitToEinsum(noisy_circuit, dtype='complex128', backend='numpy')
        with NetworkState.from_converter(converter, pure_state=False) as state:
            rho = state.compute_density_matrix()
            tol = get_contraction_tolerance(state.dtype)
            np.testing.assert_allclose(rho, rho_ref, **tol)

    def test_from_circuit_real_dtype_channel(self):
        """NetworkState.from_circuit with real dtype and a real-valued channel (bit-flip)."""
        try:
            from qiskit import QuantumCircuit
            from qiskit.quantum_info import Kraus
        except ImportError:
            pytest.skip("qiskit not available")
        p = 0.15
        K0 = np.sqrt(1 - p) * np.eye(2)
        K1 = np.sqrt(p) * np.array([[0, 1], [1, 0]])
        qc = QuantumCircuit(2)
        qc.x(0)
        qc.append(Kraus([K0, K1]), [0])
        rho_ref = _c2e_ref_density_matrix(qc)
        with NetworkState.from_circuit(qc, dtype='float64', pure_state=False, backend='numpy') as state:
            rho = state.compute_density_matrix()
            tol = get_contraction_tolerance(state.dtype)
            np.testing.assert_allclose(rho, rho_ref.real, **tol)

    def test_from_circuit_noisy_pure_raises(self):
        """General channels with TNConfig + pure must raise at the NetworkState level."""
        try:
            circuit = _make_qiskit_noisy_circuit()
        except ImportError:
            pytest.skip("qiskit or qiskit_aer not available")
        with pytest.raises(ValueError, match="general channel"):
            NetworkState.from_circuit(circuit, pure_state=True, backend='numpy')

    def test_from_converter_noisy_pure_raises(self):
        try:
            circuit = _make_qiskit_noisy_circuit()
        except ImportError:
            pytest.skip("qiskit or qiskit_aer not available")
        converter = CircuitToEinsum(circuit, dtype='complex128', backend='numpy')
        with pytest.raises(ValueError, match="general channel"):
            NetworkState.from_converter(converter, pure_state=True)
