# Copyright (c) 2025-2026, NVIDIA CORPORATION & AFFILIATES
#
# SPDX-License-Identifier: BSD-3-Clause

"""Tests for the pythonic stabilizer API."""

import pytest
import numpy as np
import math
import stim

try:
    import deltakit_stim as _deltakit_stim
except ImportError:
    _deltakit_stim = None
import time
from typing import Union
import logging
from enum import Enum

logger = logging.getLogger("pythonic")
log = logger.debug

Array = Union[np.ndarray, "cp.ndarray"]

try:
    import cupy as cp
except ImportError:
    cp = np
from cuquantum.bindings import custabilizer as custab
from cuquantum.stabilizer import Circuit, FrameSimulator, LeakageFrameSimulator, Options

pytestmark = pytest.mark.custabilizer


def test_circuit_smoke():
    """Test creating a circuit."""
    circ = Circuit("H 0\nCNOT 0 1\nM 0 1")
    assert circ.circuit is not None


def test_circuit_attributes():
    # Chosen so every attribute is non-trivial:
    # qubits=2, measurements=4 (M 0 1 + REPEAT 2 { M 0 }), detectors=1, repeat_blocks=1,
    # resets=2 (R 0 1), 1q=1 (H), 2q=1 (CX), has_noise=1 (DEPOLARIZE1).
    circ = Circuit(
        "R 0 1\nH 0\nCX 0 1\nDEPOLARIZE1(0.1) 0\nM 0 1\nDETECTOR rec[-1]\nREPEAT 2 {\nM 0\n}"
    )
    assert circ.num_qubits == 2
    assert circ.num_measurements == 4
    assert circ.num_measurement_gates == 4
    assert circ.num_detectors == 1
    assert circ.num_repeat_blocks == 1
    assert circ.num_resets == 2
    assert circ.num_1q_gates == 1
    assert circ.num_2q_gates == 1
    assert circ.has_noise is True
    assert circ.has_leakage is False
    assert circ.num_leakage_readouts == 0


def test_circuit_leakage_attributes():
    circ = Circuit("R 0 1\nLEAKAGE_MARK1(0.5) 0\nHERALD_LEAKAGE_EVENT 0\nM 0 1")
    assert circ.has_leakage is True
    assert circ.num_leakage_readouts == 1
    assert circ.num_measurement_gates == 2
    assert circ.num_measurements == 3  # gates + herald readouts


def test_from_circuit_matches_explicit():
    circ = Circuit("R 0 1\nH 0\nCNOT 0 1\nM 0 1")
    a = FrameSimulator.from_circuit(circ, 1024)
    b = FrameSimulator(
        circ.num_qubits,
        1024,
        num_measurements=circ.num_measurements,
        num_detectors=circ.num_detectors,
    )
    assert a.num_qubits == b.num_qubits
    assert a.num_measurements == b.num_measurements
    with pytest.raises(TypeError):
        FrameSimulator.from_circuit(circ, 1024, num_qubits=99)


def test_frame_simulator_rejects_leakage_circuit():
    circ = Circuit("LEAKAGE_MARK1(1) 0\nM 0")
    sim = FrameSimulator(1, 1024, num_measurements=1)
    with pytest.raises(custab.cuStabilizerError):
        sim.apply(circ)


def test_leakage_simulator_smoke():
    sim = LeakageFrameSimulator(2, 1024, num_measurements=1)
    assert sim.num_qubits == 2
    leakage = sim.get_leakage_bits(bit_packed=False)
    assert leakage.shape == (2, 1024)
    assert not leakage.any()  # initial state is unleaked


def test_leakage_from_circuit_sizing():
    circ = Circuit("R 0 1\nLEAKAGE_MARK1(0.5) 0\nHERALD_LEAKAGE_EVENT 0\nM 0 1")
    sim = LeakageFrameSimulator.from_circuit(circ, 1024)
    # Measurement rows count measurement gates plus herald leakage readouts.
    assert sim.num_measurements == circ.num_measurements == 3
    with pytest.raises(TypeError):
        LeakageFrameSimulator.from_circuit(circ, 1024, num_measurements=99)


def test_leakage_simulation_no_leakage_instructions():
    circ = Circuit("X_ERROR(1) 0\nH 0 1\nCNOT 1 2\nM 2\n")
    sim = LeakageFrameSimulator(3, 1024, num_measurements=1, randomize_measurements=False)
    sim.apply(circ)
    mbits = sim.get_measurement_bits(bit_packed=False)
    assert mbits.shape == (1, 1024)
    leakage = sim.get_leakage_bits(bit_packed=False)
    assert not leakage.any()  # circuit has no leakage instructions


def test_leakage_simulation_deterministic():
    # LEAKAGE_MARK1(1) leaks qubit 0 in every shot; the herald copies the
    # leakage flag into the measurement table.
    circ = Circuit("LEAKAGE_MARK1(1) 0\nHERALD_LEAKAGE_EVENT 0")
    sim = LeakageFrameSimulator.from_circuit(circ, 1024, randomize_measurements=False)
    sim.apply(circ)
    leakage = sim.get_leakage_bits(bit_packed=False)
    assert leakage[0].all()
    herald = sim.get_measurement_bits(bit_packed=False)
    assert herald.shape == (1, 1024)
    assert herald[0].all()


def test_frame_simulator_smoke():
    """Test creating a circuit."""
    sim = FrameSimulator(2, 1024, num_measurements=1)
    assert sim.num_qubits == 2
    sim = FrameSimulator(2, 1027, num_measurements=1)
    assert sim.num_paulis == 1027
    sim = FrameSimulator(0, 27, num_measurements=15)
    assert sim.num_measurements == 15


def test_simulation_basic():
    """Test creating a circuit."""
    circ = Circuit("X_ERROR(1) 0\nZ_ERROR(1) 1\nY_ERROR(1) 4\nH 0 1\nCNOT 1 2\n M 2 3\n")
    possible = ("ZXY.Y", "ZXX.Y", "ZXYZY", "ZXXZY")
    sim = FrameSimulator(len(possible[0]), 1024, num_measurements=2, randomize_measurements=False)
    sim.apply(circ)
    table = sim.get_pauli_table()
    assert table[0].to_string() in possible
    assert table[487].to_string() in possible
    assert table[1023].to_string() in possible
    assert sim.num_qubits == len(possible[0])
    mbits: Array = sim.get_measurement_bits()
    assert (mbits[0] == 255).all()
    assert (mbits[1] == 0).all()
    mbits: Array = sim.get_measurement_bits(bit_packed=False)
    assert len(mbits[0]) == 1024
    assert (mbits[0] == 1).all()
    assert (mbits[1] == 0).all()


def calculate_table_population(*args):
    t = tuple(a.sum(axis=-1) for a in args)
    return np.concatenate(t)


def stim_circuit_filter_gates(circuit: stim.Circuit, gates: list[str]) -> stim.Circuit:
    new_circuit = stim.Circuit()
    for gate in circuit:
        if isinstance(gate, stim.CircuitRepeatBlock):
            newbloc_body = stim_circuit_filter_gates(gate.body_copy(), gates)
            newblock = stim.CircuitRepeatBlock(gate.repeat_count, newbloc_body, tag=gate.tag)
            new_circuit.append(newblock)
            continue
        if gate.name not in gates:
            new_circuit.append(gate)
            new_circuit.append('tick')

    return new_circuit


class Circuits(Enum):
    rare_events1 = """
       REPEAT 30 {
           X_ERROR(0.001) 0 1 2 5
           Z_ERROR(0.001) 0 4 1 3
           DEPOLARIZE1(0.005) 0 4
           DEPOLARIZE2(0.006) 3 7 6 0
       }
       M(0.009) 0 2 4
       MRY(0.009) 7 5
       """


def get_circuit(circuit_name: str, d, r, p) -> str:
    if "memory" in circuit_name:
        circuit = str(stim.Circuit.generated(
            "surface_code:" + circuit_name,
            distance=d,
            rounds=r,
            after_clifford_depolarization=p,
            before_round_data_depolarization=p,
            before_measure_flip_probability=p,
        ))
    else:
        circuit = str(Circuits[circuit_name].value)
    return circuit

@pytest.mark.parametrize(
    ("d", "r", "p", "nshots", "circuit_name", "randomize_measurements"),
    # fmt: off
    [
        ( 4, 4, 0.001, 1024 * 200, "rotated_memory_z",          False,),     #
        ( 7, 3, 0.01,  1024 * 80,  "rotated_memory_x",          False,),     #
        ( 8, 9, 0.002, 1024 * 100, "unrotated_memory_z",        False,),     #
        ( 5, 4, 0.1,   32 * 357,   "unrotated_memory_x",        True, ),     #
        ( 0, 0, 0,     32 * 357,   Circuits.rare_events1.name,  False,),     #
        ( 0, 0, 0,     128     ,   Circuits.rare_events1.name,  False,),     #
     ],
    # fmt: on
)
def test_statistical_wrt_stim(d, r, p, nshots, circuit_name, randomize_measurements):
    log(
        f"Surface code test {d=} {r=} {p=} {nshots=} {circuit_name=} {randomize_measurements=}"
    )
    circuit = stim.Circuit(get_circuit(circuit_name, d, r, p))
    cuda_circuit = Circuit(circuit)
    sim = FrameSimulator(
        circuit.num_qubits,
        nshots,
        circuit.num_measurements,
        num_detectors=circuit.num_detectors,
        randomize_measurements=randomize_measurements,
        seed=0,
        package="cupy",
    )
    sim.apply(cuda_circuit)
    xzbits = sim.get_pauli_xz_bits(bit_packed=False)
    mbits = sim.get_measurement_bits(bit_packed=False)

    def get_stim_probs(seed):
        sim_ref = stim.FlipSimulator(
            num_qubits=circuit.num_qubits,
            batch_size=nshots,
            seed=seed,
            disable_stabilizer_randomization=not randomize_measurements,
        )
        stim_start = time.time()
        sim_ref.do(circuit)
        stim_end = time.time()
        log(f"Stim time: {(stim_end - stim_start) * 1000} ms")
        xref, zref, mref, dref, oref = sim_ref.to_numpy(
            bit_packed=False, output_xs=True, output_zs=True, output_measure_flips=True
        )
        probs_ref = calculate_table_population(xref, zref, mref) / nshots
        return probs_ref

    prob = calculate_table_population(xzbits[0], xzbits[1], mbits) / nshots
    prob = prob.get()
    prob_ref = get_stim_probs(0)
    log(f"Shapes: {prob.shape=}, {prob_ref.shape=}")
    print_num = 30

    num_tries = 4
    retry_violations_below = 5
    K = len(prob)
    # TODO: this is still giving false negatives for small p
    z = np.sqrt(2 * np.log(K) + 15)
    log(f"Using statistical z={z} for {K} probabilities")
    individual_FP = math.erfc(z / np.sqrt(2))
    log(f"Individual false positive rate: {individual_FP}")
    log(f"Joint false positive rate: {1 - (1 - individual_FP) ** K}")

    original_printoptions = np.get_printoptions()
    try:
        np.set_printoptions(
            edgeitems=30, linewidth=100000, formatter=dict(float=lambda x: "%.5f" % x)
        )

        violations_by_try: dict[int, tuple] = {}
        for tryix in range(num_tries):
            log("IX=%s", ' '.join(f"{ix:7}" for ix in range(min(print_num, K))))
            log(f"prob={prob[:print_num]}")
            log(f"pref={prob_ref[:print_num]}")
            assert not np.all(prob == 0), "Probs should not be all zeros"
            assert np.all((prob_ref == 0) == (prob == 0))

            diff = np.abs(prob_ref - prob)
            log(f"diff={diff[:print_num]}")
            sigma = np.std(prob_ref[prob_ref != 0])
            atol = sigma * 2 / np.sqrt(nshots)

            pref = prob_ref
            atol = z * np.sqrt(pref * (1 - pref) / nshots)  # vector atol
            rtol = 1 / nshots
            log(f"atol={atol[:print_num]}")
            log(f"rtol={rtol}")
            max_diff = np.max(diff)
            max_diff_ix = np.argmax(diff)
            log(
                f"Max difference: {max_diff} at {max_diff_ix} ({prob_ref[max_diff_ix]}(ref) vs {prob[max_diff_ix]})"
            )

            close = np.isclose(prob, pref, atol=atol, rtol=rtol)
            violations = np.where(~close)[0]
            if len(violations) != 0:
                logger.warning(f"{len(violations)} violations at try {tryix}!")
                log(f"Violations: {violations[:print_num]}")
                log(f"Values res: {prob[violations][:print_num]}")
                log(f"Values ref: {pref[violations][:print_num]}")
                log(f"diff      : {diff[violations][:print_num]}")
                log(f"atol      : {atol[violations][:print_num]}")
                assert len(violations) < retry_violations_below
                violations_by_try[tryix] = tuple(violations.tolist())
                prob_ref = get_stim_probs(tryix + 1)
            else:
                break

        assert len(violations_by_try) < num_tries, "All retries had violations"
    finally:
        np.set_printoptions(**original_printoptions)


@pytest.mark.parametrize("disable_frame_randomization", [True, False])
@pytest.mark.parametrize("h_layer", [True, False])
def test_h_layer_measure_all(disable_frame_randomization, h_layer):
    """Optional H on all 5 qubits then measure all. Compare measurement statistics against stim."""
    num_qubits = 5
    nshots = 1024 * 50
    qubits_str = " ".join(str(i) for i in range(num_qubits))
    circuit_str = ("H " + qubits_str + "\n" if h_layer else "") + "M " + qubits_str

    stim_circuit = stim.Circuit(circuit_str)
    cuda_circuit = Circuit(circuit_str)
    randomize_measurements = not disable_frame_randomization

    sim = FrameSimulator(
        num_qubits,
        nshots,
        num_measurements=num_qubits,
        randomize_measurements=randomize_measurements,
        seed=0,
    )
    sim.apply(cuda_circuit)
    mbits = sim.get_measurement_bits(bit_packed=False)
    xbits, zbits = sim.get_pauli_xz_bits(bit_packed=False)

    sim_ref = stim.FlipSimulator(
        num_qubits=num_qubits,
        batch_size=nshots,
        seed=0,
        disable_stabilizer_randomization=disable_frame_randomization,
    )
    sim_ref.do(stim_circuit)
    xref, zref, mref, _, _ = sim_ref.to_numpy(
        bit_packed=False, output_xs=True, output_zs=True, output_measure_flips=True
    )

    if disable_frame_randomization:
        assert np.array_equal(xbits, xref), "X table mismatch"
        assert np.array_equal(zbits, zref), "Z table mismatch"
        assert np.array_equal(mbits, mref), "M table mismatch"
    else:
        probs = calculate_table_population(xbits, zbits, mbits) / nshots
        probs_ref = calculate_table_population(xref, zref, mref) / nshots
        K = len(probs)
        z = np.sqrt(2 * np.log(K) + 15)
        atol = z * np.sqrt(2 * probs_ref * (1 - probs_ref) / nshots)
        rtol = 1 / nshots
        assert np.allclose(probs, probs_ref, atol=atol, rtol=rtol), \
            f"Statistical mismatch: probs={probs}, probs_ref={probs_ref}, atol={atol}"


def test_multiple_circuits_same_simulator():
    """Test reusing same simulator for multiple circuits."""
    circ = Circuit(
        """
       X_ERROR(0.1) 0 2 5
       Z_ERROR(0.3) 1 2 4
       H 0 1 3
       CNOT 0 1 5 2
       X_ERROR(0.002) 0 1 2 5
       Z_ERROR(0.002) 0 1 2 5
       DEPOLARIZE2(0.005) 1 4
       M 0 2 4
       """
    )
    nshots = 1024 * 5
    nqubits = 6
    nmeas = 3
    sim = FrameSimulator(nqubits, nshots, num_measurements=nmeas, randomize_measurements=False)

    seed = 15
    # Apply first circuit
    sim.apply(circ, seed=seed)
    m1 = sim.get_measurement_bits(bit_packed=False)
    x1, z1 = sim.get_pauli_xz_bits()
    x1, z1 = x1.copy(), z1.copy()

    sim.apply(circ, seed=seed)
    m2 = sim.get_measurement_bits(bit_packed=False)
    x2, z2 = sim.get_pauli_xz_bits()
    x2, z2 = x2.copy(), z2.copy()

    # Reset tables and apply second circuit
    x_table = np.zeros((nqubits, nshots), dtype=np.uint8)
    z_table = np.zeros((nqubits, nshots), dtype=np.uint8)
    m_table = np.zeros((nmeas, nshots // 8), dtype=np.uint8)  # bit-packed format

    sim.set_input_tables(x_table, z_table, bit_packed=False)
    sim.set_input_tables(x=None, z=None, m=m_table, bit_packed=True)
    sim.apply(circ, seed=seed)
    m3 = sim.get_measurement_bits(bit_packed=False)
    x3, z3 = sim.get_pauli_xz_bits()
    assert x1.shape == x2.shape

    assert not np.array_equal(x1, x2)
    assert not np.array_equal(z1, z2)
    assert not np.array_equal(m1, m2)
    assert not np.all(m1 == 0)
    assert not np.all(m2 == 0)
    assert not np.all(m3 == 0)
    assert np.array_equal(x1, x3)
    assert np.array_equal(z1, z3)
    assert np.array_equal(m1, m3)

def test_multiple_runs_same_simulator():
    """
    Concatenate results of multiple runs and compare against one run with nshots=sum(nshots_small).

    Notes:
    - The circuit should contain small probabilities to trigger rare event sampling
    - Number of small runs should be large enough to avoid false failures of the test
    - Since the gate errors are small, use REPEAT instruction to accumulate errors
    """
    circ = Circuit(Circuits.rare_events1.value)
    stim_circ = stim.Circuit(circ.circuit_string)

    small_runs = [1] * 128
    big_run = sum(small_runs)
    nqubits = stim_circ.num_qubits
    nmeas = stim_circ.num_measurements
    logger = logging.getLogger("Ignore")
    logger.setLevel(level=logging.WARNING)
    options = Options(logger=logger)
    seed = 10
    sim_small = FrameSimulator(
        nqubits,
        max(small_runs),
        num_measurements=nmeas,
        randomize_measurements=False,
        seed=seed,
        options=options,
    )
    sim_big = FrameSimulator(
        nqubits,
        big_run,
        num_measurements=nmeas,
        randomize_measurements=False,
        seed=seed,
    )

    def reset(sim, nqubits, nmeas, nshots):
        x_table = np.zeros((nqubits, nshots), dtype=np.uint8)
        z_table = np.zeros((nqubits, nshots), dtype=np.uint8)
        m_table = np.zeros((nmeas, nshots), dtype=np.uint8)
        sim.set_input_tables(x=x_table, z=z_table, m=m_table, bit_packed=False)

    sm_results = []
    for i, run in enumerate(small_runs):
        reset(sim_small, nqubits, nmeas, max(small_runs))
        sim_small.apply(circ)
        x, z = sim_small.get_pauli_xz_bits(bit_packed=False)
        mbits = sim_small.get_measurement_bits(bit_packed=False)
        # log(f"run={i} xbits={x.flatten()} zbits={z.flatten()} mbits={mbits.flatten()}")
        counts_i = calculate_table_population(x[:, :run], z[:, :run], mbits[:, :run])
        # log(f"run={i} counts={counts_i}")
        sm_results.append(counts_i)

    sim_big.apply(circ)
    xzbits = sim_big.get_pauli_xz_bits(bit_packed=False)
    mbits = sim_big.get_measurement_bits(bit_packed=False)
    probs_big = calculate_table_population(xzbits[0], xzbits[1], mbits) / big_run
    probs_sm = np.sum(sm_results, axis=0) / np.sum(small_runs)
    print_num = 20
    original_printoptions = np.get_printoptions()
    try:
        np.set_printoptions(
            edgeitems=30, linewidth=100000, formatter=dict(float=lambda x: "%.6f" % x)
        )
        log(f"probs_big={probs_big[:print_num]}")
        log(f"probs_sml={probs_sm[:print_num]}")

        K = len(probs_big)
        z = np.sqrt(2 * np.log(K) + 15)
        atol = z * np.sqrt(probs_sm * (1 - probs_sm) / big_run)  # vector atol
        rtol = 1 / big_run
        diff = np.abs(probs_big - probs_sm)
        log(f"diff = {diff[:print_num]}")
        log(f"atol = {atol[:print_num]} rtol={rtol}")
        assert np.allclose(probs_big, probs_sm, atol=atol, rtol=rtol)
    finally:
        np.set_printoptions(**original_printoptions)



def test_leakage_instructions_analytical():
    """One circuit exercising each leakage instruction family; batch-check
    l_table, x_table, z_table, and m_table marginals against analytical
    targets.
    """
    nshots = 1024 * 200

    # LEAKAGE_PROPAGATE rates (pairs 0-17)
    s01, s10, m01, m10 = 0.30, 0.25, 0.20, 0.15
    s01_c, m01_c = 0.10, 0.05
    s10_c, m10_c = 0.08, 0.06
    s01_lo, m10_lo = 0.005, 0.008
    all_s01, all_s10, all_m01, all_m10 = 0.20, 0.15, 0.10, 0.05
    # LEAKAGE_MARK1, LEAKAGE_MARK2, LEAKAGE_PROPAGATE_MARK, LEAKAGE_RELAX
    mk_p_a, mk_p_lo = 0.35, 0.007
    mk2_ll, mk2_li, mk2_il = 0.20, 0.15, 0.10
    pm_s01, pm_m01 = 0.25, 0.10
    relax_p = 0.30
    # LEAKAGE1, LEAKAGE2, LEAKAGE_PAULI1
    lk1_p = 0.20
    lk2_ll, lk2_li, lk2_il = 0.15, 0.10, 0.05
    pa_x0, pa_y0, pa_z0 = 0.10, 0.05, 0.15
    pa_x1, pa_y1, pa_z1 = 0.08, 0.12, 0.20

    circuit_str = f"""
    LEAKAGE_MARK1(1) 0 3 4 7 8 11 12 15
    LEAKAGE_MARK1(0.5) 16 17
    CX 0 1
    LEAKAGE_PROPAGATE({s01},0,0,0) 0 1
    CX 2 3
    LEAKAGE_PROPAGATE(0,{s10},0,0) 2 3
    CX 4 5
    LEAKAGE_PROPAGATE(0,0,{m01},0) 4 5
    CX 6 7
    LEAKAGE_PROPAGATE(0,0,0,{m10}) 6 7
    CX 8 9
    LEAKAGE_PROPAGATE({s01_c},0,{m01_c},0) 8 9
    CX 10 11
    LEAKAGE_PROPAGATE(0,{s10_c},0,{m10_c}) 10 11
    CX 12 13
    LEAKAGE_PROPAGATE({s01_lo},0,0,0) 12 13
    CX 14 15
    LEAKAGE_PROPAGATE(0,0,0,{m10_lo}) 14 15
    CX 16 17
    LEAKAGE_PROPAGATE({all_s01},{all_s10},{all_m01},{all_m10}) 16 17
    LEAKAGE_MARK1({mk_p_a}) 18
    LEAKAGE_MARK1({mk_p_lo}) 19
    LEAKAGE_MARK2({mk2_ll},{mk2_li},{mk2_il}) 20 21
    LEAKAGE_MARK1(1) 22
    LEAKAGE_PROPAGATE_MARK({pm_s01},0,{pm_m01},0) 22 23
    LEAKAGE_MARK1(1) 24
    LEAKAGE_RESET 24
    LEAKAGE_MARK1(1) 25
    LEAKAGE_RELAX({relax_p}) 25
    LEAKAGE_MARK1(1) 26
    LEAKAGE_SCRAMBLE 26
    LEAKAGE_MARK1(1) 27
    LEAKAGE_SCRAMBLE_PARTNER 27 28
    LEAKAGE1({lk1_p}) 29
    LEAKAGE2({lk2_ll},{lk2_li},{lk2_il}) 30 31
    LEAKAGE_PAULI1({pa_x0},{pa_y0},{pa_z0},0,0,0) 32
    LEAKAGE_MARK1(1) 33
    LEAKAGE_PAULI1(0,0,0,{pa_x1},{pa_y1},{pa_z1}) 33
    LEAKAGE_MARK1(1) 34
    HERALD_LEAKAGE_EVENT 34 35
    """
    sim = LeakageFrameSimulator(
        36, nshots, num_measurements=2, seed=0, randomize_measurements=False,
    )
    sim.apply(Circuit(circuit_str))

    l_obs = sim.get_leakage_bits(bit_packed=False).mean(axis=1)
    x_obs, z_obs = (t.mean(axis=1) for t in sim.get_pauli_xz_bits(bit_packed=False))
    m_obs = sim.get_measurement_bits(bit_packed=False).mean(axis=1)

    l_exp = np.array([
        # LEAKAGE_PROPAGATE (0..17)
        1.0,           s01,
        s10,           1.0,
        1 - m01,       m01,
        m10,           1 - m10,
        1 - m01_c,     s01_c + m01_c,
        s10_c + m10_c, 1 - m10_c,
        1.0,           s01_lo,
        m10_lo,        1 - m10_lo,
        0.5 + 0.25 * (all_s10 + all_m10 - all_m01),
        0.5 + 0.25 * (all_s01 + all_m01 - all_m10),
        # LEAKAGE_MARK1 (18, 19)
        mk_p_a, mk_p_lo,
        # LEAKAGE_MARK2 (20, 21)
        mk2_li + mk2_ll, mk2_il + mk2_ll,
        # LEAKAGE_PROPAGATE_MARK (22, 23)
        1 - pm_m01, pm_s01 + pm_m01,
        # LEAKAGE_RESET (24), LEAKAGE_RELAX (25)
        0.0, 1 - relax_p,
        # LEAKAGE_SCRAMBLE (26): l unchanged
        1.0,
        # LEAKAGE_SCRAMBLE_PARTNER (27, 28): l unchanged
        1.0, 0.0,
        # LEAKAGE1 (29)
        lk1_p,
        # LEAKAGE2 (30, 31)
        lk2_li + lk2_ll, lk2_il + lk2_ll,
        # LEAKAGE_PAULI1 unleaked (32), leaked (33)
        0.0, 1.0,
        # HERALD_LEAKAGE_EVENT source (34), partner (35)
        1.0, 0.0,
    ])
    x_exp = np.array([
        # LEAKAGE_PROPAGATE (partner-scramble on qubits 1,2,5,6,9,10,13,14)
        0, 0.5, 0.5, 0, 0, 0.5, 0.5, 0,
        0, 0.5, 0.5, 0, 0, 0.5, 0.5, 0,
        0.25, 0.25,
        # Mark family (18-23) leaves x untouched
        0, 0, 0, 0, 0, 0,
        # LEAKAGE_RESET (24), LEAKAGE_RELAX (25)
        0, 0,
        # LEAKAGE_SCRAMBLE (26): X or Y flip = 0.5 XOR with 0 = 0.5
        0.5,
        # LEAKAGE_SCRAMBLE_PARTNER (27, 28)
        0, 0.5,
        # LEAKAGE1 (29): P(mark) * P(X or Y | scramble) = p * 0.5
        lk1_p * 0.5,
        # LEAKAGE2 (30, 31)
        (lk2_li + lk2_ll) * 0.5, (lk2_il + lk2_ll) * 0.5,
        # LEAKAGE_PAULI1 (32): P(X)+P(Y) = pa_x0 + pa_y0
        pa_x0 + pa_y0,
        # LEAKAGE_PAULI1 (33): leaked path
        pa_x1 + pa_y1,
        # HERALD (34, 35)
        0, 0,
    ])
    z_exp = np.array([
        0, 0.5, 0.5, 0, 0, 0.5, 0.5, 0,
        0, 0.5, 0.5, 0, 0, 0.5, 0.5, 0,
        0.25, 0.25,
        0, 0, 0, 0, 0, 0,
        0, 0,
        0.5,
        0, 0.5,
        lk1_p * 0.5,
        (lk2_li + lk2_ll) * 0.5, (lk2_il + lk2_ll) * 0.5,
        # LEAKAGE_PAULI1: P(Y)+P(Z)
        pa_y0 + pa_z0,
        pa_y1 + pa_z1,
        0, 0,
    ])
    m_exp = np.array([1.0, 0.0])

    obs = np.concatenate([l_obs, x_obs, z_obs, m_obs])
    exp = np.concatenate([l_exp, x_exp, z_exp, m_exp])
    K = len(exp)
    z_score = np.sqrt(2 * np.log(K) + 15)
    atol = z_score * np.sqrt(exp * (1 - exp) / nshots)
    assert np.allclose(obs, exp, atol=atol, rtol=1 / nshots), \
        f"obs={obs}\nexp={exp}\ndiff={obs - exp}\natol={atol}"


def test_leakage_statistical_wrt_deltakit_stim():
    """Statistical measurement comparison against deltakit-stim on a
    leakage-augmented rotated_memory_z surface code.
    """
    if _deltakit_stim is None:
        pytest.skip("deltakit_stim not installed")
    deltakit_stim = _deltakit_stim
    from cuquantum.stabilizer._internal.deltakit_parser import rewrite_deltakit_stim

    # nshots x p_lk sized for a few expected events per HERALD_LEAKAGE_EVENT
    # row; nshots additionally sized so per-bit tolerance is seed-stable.
    # p_m parametric for future coverage.
    d, r, nshots = 5, 3, 1024 * 300
    p_depol, p_lk, p_s, p_m = 0.005, 0.015, 0.005, 0
    base_text = str(stim.Circuit.generated(
        "surface_code:rotated_memory_z",
        distance=d, rounds=r,
        after_clifford_depolarization=p_depol,
        before_round_data_depolarization=p_depol,
        after_reset_flip_probability=p_depol,
    ))

    dk_lines: list[str] = []
    for line in base_text.splitlines():
        stripped = line.strip()
        if not stripped or stripped.startswith("#") or stripped == "}":
            dk_lines.append(line)
            continue
        indent = line[: len(line) - len(line.lstrip())]
        name = stripped.split("(", 1)[0].split()[0]
        targets = stripped.split(None, 1)[1] if " " in stripped else ""
        if name in ("H", "H_XZ"):
            dk_lines.append(line)
            dk_lines.append(f"{indent}LEAKAGE({p_lk}) {targets}")
        elif name in ("CX", "CNOT"):
            dk_lines.append(f"{indent}CX({p_s}, {p_s}, {p_m}, {p_m}) {targets}")
        else:
            dk_lines.append(line)
    dk_text = "\n".join(dk_lines) + "\n"
    _nq = deltakit_stim.Circuit(dk_text).num_qubits
    dk_text += "HERALD_LEAKAGE_EVENT " + " ".join(str(q) for q in range(_nq)) + "\n"

    dk_circuit = deltakit_stim.Circuit(dk_text)
    dk_m = np.asarray(dk_circuit.compile_sampler(seed=0).sample(shots=nshots))

    cust_text = rewrite_deltakit_stim(dk_text, reject_unsupported=True)
    cust_circuit = Circuit(cust_text)
    sim = LeakageFrameSimulator.from_circuit(
        cust_circuit, nshots, seed=0, randomize_measurements=True,
    )
    sim.apply(cust_circuit)
    cust_m = np.asarray(sim.get_measurement_bits(bit_packed=False)).T

    dk_p = dk_m.mean(axis=0)
    cust_p = cust_m.mean(axis=0)
    K = len(dk_p)
    z_score = np.sqrt(2 * np.log(K) + 15)
    atol = z_score * np.sqrt(dk_p * (1 - dk_p) / nshots)
    assert np.allclose(cust_p, dk_p, atol=atol, rtol=1 / nshots), \
        f"cust={cust_p} dk={dk_p}"


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
