# Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES
#
# SPDX-License-Identifier: BSD-3-Clause

"""LeakageFrameSimulator tour: every leakage instruction, one section each.

Instructions fall into three families:

- mark family    -- change leakage flags only, X/Z frame untouched:
                    LEAKAGE_MARK1, LEAKAGE_MARK2, LEAKAGE_PROPAGATE_MARK,
                    LEAKAGE_RESET, LEAKAGE_RELAX
- scramble family -- leakage-conditioned Pauli-frame noise:
                    LEAKAGE_SCRAMBLE, LEAKAGE_SCRAMBLE_PARTNER,
                    LEAKAGE_PAULI1, LEAKAGE_PAULI2
- combined family -- mark + scramble of newly leaked shots in one step:
                    LEAKAGE1, LEAKAGE2, LEAKAGE_PROPAGATE
- readout        -- HERALD_LEAKAGE_EVENT copies leakage flags into the
                    measurement table; herald rows are included in
                    Circuit.num_measurements (and counted separately via
                    Circuit.num_leakage_readouts).
"""

import argparse

import numpy as np

from cuquantum.stabilizer import Circuit, LeakageFrameSimulator


def run(name, circuit_string, shots, seed):
    """Parse, simulate, and report leakage / frame / herald populations."""
    circ = Circuit(circuit_string)
    sim = LeakageFrameSimulator.from_circuit(
        circ, shots, randomize_measurements=False, seed=seed
    )
    sim.apply(circ)

    leak = sim.get_leakage_bits(bit_packed=False)       # (num_qubits, shots)
    x, z = sim.get_pauli_xz_bits(bit_packed=False)      # (num_qubits, shots)

    print(f"\n== {name} ==")
    print("  circuit:", " | ".join(circuit_string.strip().splitlines()))
    print(f"  leaked fraction per qubit:    {leak.mean(axis=1)}")
    print(f"  X-frame fraction per qubit:   {x.mean(axis=1)}")
    print(f"  Z-frame fraction per qubit:   {z.mean(axis=1)}")
    if circ.num_measurements:
        m = sim.get_measurement_bits(bit_packed=False)  # (num_measurements, shots)
        print(f"  measurement-row fractions:    {m.mean(axis=1)}")
        print(f"  (gates={circ.num_measurement_gates}, heralds={circ.num_leakage_readouts})")
    return sim


def main():
    parser = argparse.ArgumentParser(description="LeakageFrameSimulator instruction tour")
    parser.add_argument("--shots", type=int, default=1024 * 64, help="Number of shots")
    parser.add_argument("--seed", type=int, default=42, help="RNG seed")
    args = parser.parse_args()
    shots, seed = args.shots, args.seed

    # -- mark family: leakage flags only, X/Z untouched ----------------------

    # LEAKAGE_MARK1(p) q...  -- leak each target independently with prob p.
    run("LEAKAGE_MARK1", "LEAKAGE_MARK1(0.3) 0 1", shots, seed)

    # LEAKAGE_MARK2(p_LL, p_LI, p_IL) q0 q1 ...  -- per pair pick both/first/
    # second-leaks (or nothing); expect q0: p_LL+p_LI, q1: p_LL+p_IL.
    run("LEAKAGE_MARK2", "LEAKAGE_MARK2(0.1, 0.2, 0.3) 0 1", shots, seed)

    # LEAKAGE_PROPAGATE_MARK(p_move_01, p_move_10, p_spread_01, p_spread_10)
    # q0 q1 ...  -- move/spread existing leakage along each pair; here qubit 0
    # is always leaked first, then spreads to qubit 1 half the time.
    run(
        "LEAKAGE_PROPAGATE_MARK",
        "LEAKAGE_MARK1(1) 0\nLEAKAGE_PROPAGATE_MARK(0, 0, 0.5, 0) 0 1",
        shots,
        seed,
    )

    # LEAKAGE_RESET q...  -- deterministically clear leakage flags.
    run("LEAKAGE_RESET", "LEAKAGE_MARK1(1) 0 1\nLEAKAGE_RESET 0", shots, seed)

    # LEAKAGE_RELAX(p) q...  -- clear each currently leaked shot with prob p.
    run("LEAKAGE_RELAX", "LEAKAGE_MARK1(1) 0\nLEAKAGE_RELAX(0.75) 0", shots, seed)

    # -- scramble family: leakage-conditioned Pauli-frame noise --------------

    # LEAKAGE_SCRAMBLE q...  -- for leaked shots, XOR independent 50% X and Z
    # masks into the frame (fully depolarize the leaked subspace).
    run("LEAKAGE_SCRAMBLE", "LEAKAGE_MARK1(1) 0\nLEAKAGE_SCRAMBLE 0", shots, seed)

    # LEAKAGE_SCRAMBLE_PARTNER q0 q1 ...  -- a leaked member of each pair
    # scrambles its partner's frame for the same shots.
    run(
        "LEAKAGE_SCRAMBLE_PARTNER",
        "LEAKAGE_MARK1(1) 0\nLEAKAGE_SCRAMBLE_PARTNER 0 1",
        shots,
        seed,
    )

    # LEAKAGE_PAULI1(px, py, pz, px_l, py_l, pz_l) q...  -- one-qubit Pauli
    # channel; first triple applies to unleaked shots, second to leaked ones.
    # Qubit 0 is leaked (second triple: always X), qubit 1 is not (first: Z).
    run(
        "LEAKAGE_PAULI1",
        "LEAKAGE_MARK1(1) 0\nLEAKAGE_PAULI1(0, 0, 0.5, 1, 0, 0) 0 1",
        shots,
        seed,
    )

    # LEAKAGE_PAULI2(60 probs) q0 q1 ...  -- two-qubit Pauli channel with one
    # 15-outcome table per pair leakage state (II, IL, LI, LL; 4 x 15 = 60).
    # Here the pair is unleaked, and the II table always applies XX (IX=1 on
    # both qubits is outcome index 5 -> XX below).
    pauli2_args = ["0"] * 60
    pauli2_args[4] = "1"  # II-state table, outcome 5 of 15 (XX)
    run(
        "LEAKAGE_PAULI2",
        f"LEAKAGE_PAULI2({', '.join(pauli2_args)}) 0 1",
        shots,
        seed,
    )

    # -- combined family: mark + scramble newly leaked shots -----------------

    # LEAKAGE1(p) q...  -- leak with prob p and scramble only the shots that
    # transition unleaked -> leaked here (unlike MARK1 + SCRAMBLE, which
    # scrambles everything currently leaked).
    run("LEAKAGE1", "LEAKAGE1(0.5) 0", shots, seed)

    # LEAKAGE2(p_LL, p_LI, p_IL) q0 q1 ...  -- pair leakage outcomes with
    # scrambling of the newly leaked members.
    run("LEAKAGE2", "LEAKAGE2(0.1, 0.2, 0.3) 0 1", shots, seed)

    # LEAKAGE_PROPAGATE(p_move_01, p_move_10, p_spread_01, p_spread_10)
    # q0 q1 ...  -- propagate leakage and scramble the partner of each
    # originally leaked source.
    run(
        "LEAKAGE_PROPAGATE",
        "LEAKAGE_MARK1(1) 0\nLEAKAGE_PROPAGATE(0, 0, 0.5, 0) 0 1",
        shots,
        seed,
    )

    # -- readout: heralds and detectors ---------------------------------------

    # HERALD_LEAKAGE_EVENT q...  -- copy each target's leakage row into the
    # next measurement row; DETECTOR can reference it via rec[-k]. The herald
    # row for qubit 0 reads 1 in every shot, the M 0 1 rows stay 0.
    sim = run(
        "HERALD_LEAKAGE_EVENT",
        "R 0 1\nLEAKAGE_MARK1(1) 0\nHERALD_LEAKAGE_EVENT 0\nM 0 1\nDETECTOR rec[-3]",
        shots,
        seed,
    )
    herald = sim.get_measurement_bits(bit_packed=False)[0]
    assert herald.all(), "herald row should read 1 in every shot"

    print("\nAll leakage instructions demonstrated.")


if __name__ == "__main__":
    main()
