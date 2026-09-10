# Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES
#
# SPDX-License-Identifier: BSD-3-Clause

"""Demonstrate cuStateVec Ex simulator operations with hand-computable examples."""

import math

import numpy as np

from cuquantum.bindings import custatevec as cusv
from cuquantum.bindings import custatevecEx as cusvex

from . import assertions
from .conftest import over_dtypes, sv_across_distributions, sv_on_single
from .sv_tools import (
    GATE_H,
    GATE_X,
    MatrixData,
    WiresData,
    abs2sum_over_wires,
    apply_matrix,
    get_accessible_amplitudes,
    set_accessible_amplitudes_from_full_state,
)


@sv_across_distributions
@over_dtypes
def test_apply_matrix_x_on_every_wire(sv_factory, dtype):
    """Rotate |0...0> to |1...1> by applying X gates."""
    num_wires = 3
    sv_descriptor, sv_meta = sv_factory(dtype=dtype, num_wires=num_wires)

    for wire in range(num_wires):
        apply_matrix(
            sv_descriptor,
            MatrixData(GATE_X, dtype=dtype),
            WiresData([wire]),  # one target, no controls
        )

    assertions.assert_normalized(sv_descriptor, sv_meta)
    assertions.assert_basis_state(
        sv_descriptor,
        sv_meta,
        sv_meta.created_num_elements - 1,  # |1...1>
    )


@sv_on_single
@over_dtypes
def test_apply_pauli_rotation_x_half_pi(sv_factory, dtype):
    """Map |000> to +i|001> with a pi/2 X rotation on wire 0."""
    sv_descriptor, sv_meta = sv_factory(dtype=dtype, num_wires=3)

    cusvex.apply_pauli_rotation(
        state_vector=sv_descriptor,
        theta=math.pi / 2.0,
        paulis=[cusv.Pauli.X],
        targets=[0],
        num_targets=1,
        controls=[],
        control_bit_values=0,
        num_controls=0,
    )
    cusvex.state_vector_synchronize(sv_descriptor)

    assertions.assert_normalized(sv_descriptor, sv_meta)
    state = get_accessible_amplitudes(sv_descriptor, sv_meta)
    assertions.assert_allclose(
        state[1],
        1j,
        err_msg="X(pi/2) did not map |0> to +i|1>",
    )
    assertions.assert_near_zero(
        state[0],
        err_msg="X(pi/2) left amplitude on |0>",
    )


@sv_across_distributions
@over_dtypes
def test_abs2sum_array_zero_state(sv_factory, dtype):
    """Verify that the zero state has unit probability at outcome zero."""
    sv_descriptor, sv_meta = sv_factory(dtype=dtype, num_wires=3)

    probabilities = abs2sum_over_wires(sv_descriptor)
    assertions.assert_collective_result_consistent(
        sv_meta.mpi_context,
        probabilities,
        err_msg="abs2sum_array returned different probabilities across ranks",
    )
    assertions.assert_allclose(
        probabilities[0],
        1.0,
        err_msg="zero-state probability at outcome zero is not one",
    )
    assertions.assert_near_zero(
        probabilities[1:],
        err_msg="zero-state probability is nonzero away from outcome zero",
    )


@sv_across_distributions
@over_dtypes
def test_expectation_all_x_on_hadamard_state(sv_factory, dtype):
    """Compute the all-X expectation of an all-Hadamard state."""
    num_wires = 3
    sv_descriptor, sv_meta = sv_factory(dtype=dtype, num_wires=num_wires)

    for wire in range(num_wires):
        apply_matrix(
            sv_descriptor,
            MatrixData(GATE_H, dtype=dtype),
            WiresData([wire]),
        )

    # One Pauli product, X on every wire.
    expectation = np.zeros(1, dtype=np.float64)
    cusvex.compute_expectation_on_pauli_basis(
        state_vector=sv_descriptor,
        expectation_values=expectation.ctypes.data,
        pauli_operator_arrays=[[cusv.Pauli.X] * num_wires],
        num_pauli_operator_arrays=1,
        basis_wires_array=[list(range(num_wires))],
        num_basis_wires_array=[num_wires],
    )
    cusvex.state_vector_synchronize(sv_descriptor)

    assertions.assert_collective_result_consistent(
        sv_meta.mpi_context,
        expectation[0],
        err_msg="expectation returned different values across ranks",
    )
    assertions.assert_normalized(sv_descriptor, sv_meta)
    assertions.assert_allclose(
        expectation[0],
        1.0,
        err_msg="<X...X> on the all-Hadamard state is not one",
    )


@sv_across_distributions
@over_dtypes
def test_expectation_noneigenstate_fractional(sv_factory, dtype):
    """Compute a fractional Z expectation value for a non-eigenstate.

    The state cos(theta)|000> + sin(theta)|001> is no eigenstate of Z on wire 0,
    so the expectation lands between the eigenvalues at cos(2 theta).
    """
    sv_descriptor, sv_meta = sv_factory(dtype=dtype, num_wires=3)
    theta = math.pi / 6.0
    psi = np.zeros(sv_meta.created_num_elements, dtype=dtype)
    psi[0] = math.cos(theta)
    psi[1] = math.sin(theta)
    set_accessible_amplitudes_from_full_state(sv_descriptor, sv_meta, psi)

    expectation = np.zeros(1, dtype=np.float64)
    cusvex.compute_expectation_on_pauli_basis(
        state_vector=sv_descriptor,
        expectation_values=expectation.ctypes.data,
        pauli_operator_arrays=[[cusv.Pauli.Z]],
        num_pauli_operator_arrays=1,
        basis_wires_array=[[0]],
        num_basis_wires_array=[1],
    )
    cusvex.state_vector_synchronize(sv_descriptor)

    assertions.assert_collective_result_consistent(
        sv_meta.mpi_context,
        expectation[0],
        err_msg="fractional expectation returned different values across ranks",
    )
    assertions.assert_normalized(sv_descriptor, sv_meta)
    assertions.assert_allclose(
        expectation[0],
        math.cos(2.0 * theta),
        err_msg="<Z_0> on the prepared state does not equal cos(2 theta)",
    )


@sv_across_distributions
@over_dtypes
def test_compute_matrix_expectation(sv_factory, dtype):
    """Compute the complex expectation of a dense matrix observable."""
    sv_descriptor, sv_meta = sv_factory(dtype=dtype, num_wires=3)

    # (|0> + i|1>)/sqrt(2) on wire 0.
    psi = np.zeros(sv_meta.created_num_elements, dtype=dtype)
    psi[0] = 1.0 / math.sqrt(2.0)
    psi[1] = 1j / math.sqrt(2.0)
    set_accessible_amplitudes_from_full_state(sv_descriptor, sv_meta, psi)

    # The expectation value is not symmetric with transposition of the buffer,
    # so both layouts describe the same matrix only if the flag matches the buffer.
    # Declaring the wrong layout gives -i/2 instead of +i/2.
    cases = (
        (cusv.MatrixLayout.ROW, cusv.MatrixLayout.ROW, 0.5j),
        (cusv.MatrixLayout.COL, cusv.MatrixLayout.COL, 0.5j),
        (cusv.MatrixLayout.ROW, cusv.MatrixLayout.COL, -0.5j),  # incorrect
    )
    # Run cases before assert to avoid collective hang in tests.
    measured = []
    for buffer_layout, declared_layout, _ in cases:
        observable = MatrixData([[0, 1], [0, 0]], dtype=dtype, layout=buffer_layout)
        expectation = np.zeros(1, dtype=np.complex128)
        cusvex.compute_expectation(
            state_vector=sv_descriptor,
            expectation_values=expectation.ctypes.data,
            matrices=observable.data.ctypes.data,
            matrix_data_type=observable.data_type,
            layout=declared_layout,
            num_matrices=1,
            basis_wires=[0],
            num_basis_wires=1,
        )
        cusvex.state_vector_synchronize(sv_descriptor)
        assertions.assert_collective_result_consistent(
            sv_meta.mpi_context,
            expectation[0],
            err_msg="matrix expectation returned different values across ranks",
        )
        measured.append(expectation[0])

    # Computing an expectation must leave the state untouched.
    assertions.assert_normalized(sv_descriptor, sv_meta)
    for (buffer_layout, declared_layout, expected_value), value in zip(cases, measured):
        assertions.assert_allclose(
            value,
            expected_value,
            err_msg=f"{buffer_layout.name} buffer declared as {declared_layout.name} gave the wrong expectation",
        )


@sv_across_distributions
@over_dtypes
def test_measure_two_branch_collapses_to_selected(sv_factory, dtype):
    """Measure two-outcome state and collapse."""
    # Probability of measuring |0>, random number, and the outcome it selects.
    cases = (
        (0.5, 0.0, 0),
        (0.5, np.nextafter(1.0, 0.0), 1),
    )
    for probability_zero, randnum, expected_outcome in cases:
        sv_descriptor, sv_meta = sv_factory(dtype=dtype, num_wires=3)

        # sqrt(p)|000> + sqrt(1 - p)|001>: two branches differing on wire 0 only.
        psi = np.zeros(sv_meta.created_num_elements, dtype=dtype)
        psi[0] = np.sqrt(probability_zero)
        psi[1] = np.sqrt(1.0 - probability_zero)
        set_accessible_amplitudes_from_full_state(sv_descriptor, sv_meta, psi)

        bit_string = cusvex.measure(
            state_vector=sv_descriptor,
            bit_string_ordering=[0],  # wire 0, reported as bit 0
            bit_string_ordering_len=1,
            randnum=randnum,
            collapse=cusv.CollapseOp.NORMALIZE_AND_ZERO,
            reserved=0,
        )
        cusvex.state_vector_synchronize(sv_descriptor)
        outcome = int(bit_string)

        assertions.assert_collective_result_consistent(
            sv_meta.mpi_context,
            outcome,
            err_msg="measure returned different outcomes across ranks",
        )
        # Collapse discards the unselected branch and renormalizes what remains.
        assertions.assert_normalized(sv_descriptor, sv_meta)
        assert outcome == expected_outcome, f"randnum {randnum} selected outcome {outcome}"
        assertions.assert_basis_state(sv_descriptor, sv_meta, expected_outcome)


@sv_across_distributions
@over_dtypes
def test_sample_uniform_covers_all_outcomes(sv_factory, dtype):
    """Sample every outcome exactly once from a uniform state."""
    num_wires = 3
    sv_descriptor, sv_meta = sv_factory(dtype=dtype, num_wires=num_wires)

    # Prepare uniform state.
    for wire in range(num_wires):
        apply_matrix(
            sv_descriptor,
            MatrixData(GATE_H, dtype=dtype),
            WiresData([wire]),
        )

    num_shots = sv_meta.created_num_elements  # one shot per outcome
    bit_strings = np.empty(num_shots, dtype=np.int64)
    cusvex.sample(
        state_vector=sv_descriptor,
        bit_strings=bit_strings.ctypes.data,
        bit_string_ordering=list(range(num_wires)),
        bit_string_ordering_len=num_wires,
        randnums=[(shot + 0.5) / num_shots for shot in range(num_shots)],
        num_shots=num_shots,
        output=cusv.SamplerOutput.ASCENDING_ORDER,
        abs2sums=0,  # null: the API computes them
    )
    cusvex.state_vector_synchronize(sv_descriptor)

    assertions.assert_collective_result_consistent(
        sv_meta.mpi_context,
        bit_strings,
        err_msg="sample returned different bit strings across ranks",
    )
    # Sampling does not collapse the state.
    assertions.assert_normalized(sv_descriptor, sv_meta)
    # Every outcome appears once, and ASCENDING_ORDER returns them sorted.
    np.testing.assert_array_equal(bit_strings, np.arange(num_shots))
