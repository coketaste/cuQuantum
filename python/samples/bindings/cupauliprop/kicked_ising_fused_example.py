#!/usr/bin/env python3
# Copyright (c) 2025-2026, NVIDIA CORPORATION & AFFILIATES
#
# SPDX-License-Identifier: BSD-3-Clause

"""
Example: IBM 127-qubit Kicked Ising simulation using fused Clifford application (bindings)

This is the low-level cuPauliProp bindings analogue of the pythonic-API sample
``samples/pauliprop/kicked_ising_fused_example.py``. It demonstrates the fused
multi-operator API, namely:

    cupauliprop.pauli_expansion_view_prepare_operator_fused_application
    cupauliprop.pauli_expansion_view_compute_operator_fused_application

The kicked Ising circuit alternates a non-Clifford Rx(pi/4) layer with three
Rzz(-pi/2) entangling layers. An ``Rzz(-pi/2) = exp(i*pi/4*ZZ)`` rotation is a
Clifford, so each entangling layer is rewritten in terms of Clifford gates and the
entire (3-colour) entangling block of every Trotter step is applied in a *single*
fused call. Because a fused application may fail on the fly (e.g. when the operator
count exceeds a device shared-memory limit), the bindings raise ``cuPauliPropError``
with an ``INSUFFICIENT_*`` status; on such a failure the batch is halved and retried --
mirroring the realistic usage of applying the operators in smaller fused batches.

As in the simple bindings example, the Z_62 observable is back-propagated through the
adjoint circuit (20 Trotter steps, X angle pi/4) and the expectation value with respect
to the initial state |0...0> is evaluated.
"""

import time

import numpy as np
import cupy as cp
from cuquantum.bindings import cupauliprop as cupp
from nvmath.internal.typemaps import NAME_TO_DATA_TYPE


# Memory (fixed sizes known to be sufficient for these parameters)
FIXED_EXPANSION_PAULI_MEM = 16 * (1 << 20)  # 16 MiB
FIXED_EXPANSION_COEF_MEM = 4 * (1 << 20)    # 4 MiB
FIXED_WORKSPACE_MEM = 32 * (1 << 20)        # 32 MiB

# Circuit constants (IBM heavy-hex kicked Ising)
NUM_CIRCUIT_QUBITS = 127
NUM_ROTATIONS_PER_LAYER = 48
PI = np.pi
ZZ_ROTATION_ANGLE = -PI / 2.0

ZZ_QUBITS_RED = np.array([
    [  2,   1],  [ 33,  39], [ 59,  60], [ 66,  67], [ 72,  81], [118, 119],
    [ 21,  20],  [ 26,  25], [ 13,  12], [ 31,  32], [ 70,  74], [122, 123],
    [ 96,  97],  [ 57,  56], [ 63,  64], [107, 108], [103, 104], [ 46,  45],
    [ 28,  35],  [  7,   6], [ 79,  78], [  5,   4], [109, 114], [ 62,  61],
    [ 58,  71],  [ 37,  52], [ 76,  77], [  0,  14], [ 36,  51], [106, 105],
    [ 73,  85],  [ 88,  87], [ 68,  55], [116, 115], [ 94,  95], [100, 110],
    [ 17,  30],  [ 92, 102], [ 50,  49], [ 83,  84], [ 48,  47], [ 98,  99],
    [  8,   9],  [121, 120], [ 23,  24], [ 44,  43], [ 22,  15], [ 53,  41]
], dtype=np.int32)

ZZ_QUBITS_BLUE = np.array([
    [ 53,  60], [123, 124], [ 21,  22], [ 11,  12], [ 67,  68], [  2,   3],
    [ 66,  65], [122, 121], [110, 118], [  6,   5], [ 94,  90], [ 28,  29],
    [ 14,  18], [ 63,  62], [111, 104], [100,  99], [ 45,  44], [  4,  15],
    [ 20,  19], [ 57,  58], [ 77,  71], [ 76,  75], [ 26,  27], [ 16,   8],
    [ 35,  47], [ 31,  30], [ 48,  49], [ 69,  70], [125, 126], [ 89,  74],
    [ 80,  79], [116, 117], [114, 113], [ 10,   9], [106,  93], [101, 102],
    [ 92,  83], [ 98,  91], [ 82,  81], [ 54,  64], [ 96, 109], [ 85,  84],
    [ 87,  86], [108, 112], [ 34,  24], [ 42,  43], [ 40,  41], [ 39,  38]
], dtype=np.int32)

ZZ_QUBITS_GREEN = np.array([
    [ 10,  11], [ 54,  45], [111, 122], [ 64,  65], [ 60,  61], [103, 102],
    [ 72,  62], [  4,   3], [ 33,  20], [ 58,  59], [ 26,  16], [ 28,  27],
    [  8,   7], [104, 105], [ 73,  66], [ 87,  93], [ 85,  86], [ 55,  49],
    [ 68,  69], [ 89,  88], [ 80,  81], [117, 118], [101, 100], [114, 115],
    [ 96,  95], [ 29,  30], [106, 107], [ 83,  82], [ 91,  79], [  0,   1],
    [ 56,  52], [ 90,  75], [126, 112], [ 36,  32], [ 46,  47], [ 77,  78],
    [ 97,  98], [ 17,  12], [119, 120], [ 22,  23], [ 24,  25], [ 43,  34],
    [ 42,  41], [ 40,  39], [ 37,  38], [125, 124], [ 50,  51], [ 18,  19]
], dtype=np.int32)


_DEVICE = cupp.Memspace.DEVICE
_SCRATCH = cupp.WorkspaceKind.WORKSPACE_SCRATCH

# Fused on-the-fly failures that are recoverable by shrinking the fused batch. Any other
# cuPauliPropError indicates a genuine error and is re-raised rather than retried.
_FUSED_FALLBACK_STATUSES = frozenset({
    int(cupp.Status.INSUFFICIENT_WORKSPACE),
    int(cupp.Status.INSUFFICIENT_OUT_EXPANSION),
    int(cupp.Status.INSUFFICIENT_DEVICE_PROPERTY),
})


def get_pauli_string_as_packed_integers(paulis, qubits):
    """Convert a Pauli string to packed integer (X|Z mask) representation."""
    num_packed_ints = cupp.get_num_packed_integers(NUM_CIRCUIT_QUBITS)
    out = np.zeros(num_packed_ints * 2, dtype=np.uint64)
    x_ptr = out[:num_packed_ints]
    z_ptr = out[num_packed_ints:]
    for pauli, qubit in zip(paulis, qubits):
        int_ind = qubit // 64
        bit_ind = qubit % 64
        if pauli in (cupp.PauliKind.PAULI_X, cupp.PauliKind.PAULI_Y):
            x_ptr[int_ind] |= np.uint64(1 << bit_ind)
        if pauli in (cupp.PauliKind.PAULI_Z, cupp.PauliKind.PAULI_Y):
            z_ptr[int_ind] |= np.uint64(1 << bit_ind)
    return out


def get_x_rotation_layer(handle, angle):
    """A layer of single-qubit Rx(angle) rotations on every qubit."""
    paulis = np.array([cupp.PauliKind.PAULI_X], dtype=np.int32)
    return [
        cupp.create_pauli_rotation_gate_operator(
            handle, angle, 1, np.array([i], dtype=np.int32), paulis)
        for i in range(NUM_CIRCUIT_QUBITS)
    ]


def get_zz_layer_adjoint_clifford_operators(handle, *topologies):
    """Clifford operators implementing the *adjoint* of the given Rzz(-pi/2) layers.

    Each ``Rzz(-pi/2) = exp(i*pi/4*ZZ)`` rotation on edge ``(a, b)`` is a Clifford.
    Back-propagation applies the adjoint of every gate, and
    ``Rzz(-pi/2)^dagger = Rzz(+pi/2) = CX(a, b) . S(a) . CX(a, b)``, so we emit that
    Clifford triple per edge (with no per-operator adjoint, since the adjoint is already
    baked in). The phase gate sits on the *first* qubit of CX(a, b) -- the qubit whose Z
    the CX spreads (Z_a -> Z_a Z_b) in this library's convention.

    Within one heavy-hex colour class the edges are disjoint and within a Trotter step the
    Rzz rotations are mutually commuting (all diagonal), so the whole entangling block can
    be emitted as a single same-kind Clifford sequence.
    """
    CX = cupp.CliffordGateKind.CLIFFORD_GATE_CX
    S = cupp.CliffordGateKind.CLIFFORD_GATE_S
    ops = []
    for topology in topologies:
        for pair in topology:
            a, b = int(pair[0]), int(pair[1])
            ops.append(cupp.create_clifford_gate_operator(handle, CX, [a, b]))
            ops.append(cupp.create_clifford_gate_operator(handle, S, [a]))
            ops.append(cupp.create_clifford_gate_operator(handle, CX, [a, b]))
    return ops


def apply_single_gate(handle, in_exp, out_exp, gate, adjoint, trunc_strats,
                      d_workspace_buffer, workspace_mem, workspace):
    """Apply one operator in-order (single-operator API), returning (result, spare)."""
    n_terms = cupp.pauli_expansion_get_num_terms(handle, in_exp)
    view_in = cupp.pauli_expansion_get_contiguous_range(handle, in_exp, 0, n_terms)
    num_trunc = len(trunc_strats) if trunc_strats else 0
    try:
        cupp.pauli_expansion_view_prepare_operator_application(
            handle, view_in, gate, 0, 0,
            num_trunc, trunc_strats if num_trunc > 0 else None,
            workspace_mem, workspace)
        # Prepare detaches the buffer from the workspace; re-attach it before compute.
        cupp.workspace_set_memory(handle, workspace, _DEVICE, _SCRATCH,
                                  d_workspace_buffer.ptr, workspace_mem)
        cupp.pauli_expansion_view_compute_operator_application(
            handle, view_in, out_exp, gate, adjoint, 0, 0,
            num_trunc, trunc_strats if num_trunc > 0 else None, workspace, 0)
    finally:
        cupp.destroy_pauli_expansion_view(view_in)
    return out_exp, in_exp


def apply_clifford_block(handle, in_exp, out_exp, operators, trunc_strats,
                         d_workspace_buffer, workspace_mem, out_term_capacity,
                         ws_min, ws_avg, ws_max, workspace, label):
    """Apply a Clifford operator block via one fused call, halving the batch on failure.

    The whole block is attempted in a single fused application using the failsafe (maximum)
    workspace/capacity tier. A fused application may fail on the fly (e.g. the operator count
    exceeds a device shared-memory limit, or the chosen workspace/output capacity proves
    insufficient), which the bindings surface by raising ``cuPauliPropError`` with one of the
    ``INSUFFICIENT_*`` status codes; in that case the batch is halved and each half retried in
    order. Truncation is applied only after the very last operator of the whole block. Returns
    ``(result_exp, spare_exp)``.
    """
    n_ops = len(operators)
    adjoints = [0] * n_ops  # adjoint already baked into the decomposition
    num_trunc = len(trunc_strats) if trunc_strats else 0
    n_terms = cupp.pauli_expansion_get_num_terms(handle, in_exp)
    view_in = cupp.pauli_expansion_get_contiguous_range(handle, in_exp, 0, n_terms)

    try:
        # Size the failsafe scenario via the (distinct) min/average/max workspace descriptors.
        min_cap, avg_cap, max_cap = (
            cupp.pauli_expansion_view_prepare_operator_fused_application(
                handle, view_in, n_ops, operators, adjoints,
                num_trunc, trunc_strats if num_trunc > 0 else None,
                workspace_mem, ws_min, ws_avg, ws_max))

        req_ws = cupp.workspace_get_memory_size(handle, ws_max, _DEVICE, _SCRATCH)
        assert req_ws <= workspace_mem
        assert max_cap <= out_term_capacity
        # The prepare detaches buffers; attach our buffer to the compute workspace.
        cupp.workspace_set_memory(handle, workspace, _DEVICE, _SCRATCH,
                                  d_workspace_buffer.ptr, workspace_mem)
        cupp.pauli_expansion_view_compute_operator_fused_application(
            handle, view_in, out_exp, n_ops, operators, adjoints,
            num_trunc, trunc_strats if num_trunc > 0 else None, workspace, 0)
        return out_exp, in_exp
    except cupp.cuPauliPropError as exc:
        # Only on-the-fly fused failures are recoverable by shrinking the batch. A single
        # operator that still fails, or any other status, is a genuine error -- re-raise it.
        if exc.status not in _FUSED_FALLBACK_STATUSES or n_ops == 1:
            raise
        failure_status = cupp.Status(exc.status)
    finally:
        cupp.destroy_pauli_expansion_view(view_in)

    mid = n_ops // 2
    print(f"  [{label}] fused application of {n_ops} operators reported "
          f"{failure_status!r}; retrying in halves of {mid} and {n_ops - mid}")
    in_exp, out_exp = apply_clifford_block(
        handle, in_exp, out_exp, operators[:mid], None,
        d_workspace_buffer, workspace_mem, out_term_capacity,
        ws_min, ws_avg, ws_max, workspace, label)
    in_exp, out_exp = apply_clifford_block(
        handle, in_exp, out_exp, operators[mid:], trunc_strats,
        d_workspace_buffer, workspace_mem, out_term_capacity,
        ws_min, ws_avg, ws_max, workspace, label)
    return in_exp, out_exp


def main():
    print("cuPauliProp IBM Heavy-hex Ising Example (bindings, fused Cliffords)")
    print("=" * 67)
    print()

    # Library setup
    device_id = 0
    cp.cuda.Device(device_id).use()
    handle = cupp.create()

    # Expansion / workspace buffers (ping-pong pair of expansions + one workspace)
    expansion_pauli_mem = FIXED_EXPANSION_PAULI_MEM
    expansion_coef_mem = FIXED_EXPANSION_COEF_MEM
    workspace_mem = FIXED_WORKSPACE_MEM

    num_packed_ints = cupp.get_num_packed_integers(NUM_CIRCUIT_QUBITS)
    term_xz_bytes = 2 * num_packed_ints * np.dtype(np.uint64).itemsize
    out_term_capacity = min(expansion_pauli_mem // term_xz_bytes,
                            expansion_coef_mem // np.dtype(np.float64).itemsize)

    d_in_pauli = cp.cuda.alloc(expansion_pauli_mem)
    d_in_coef = cp.cuda.alloc(expansion_coef_mem)
    d_out_pauli = cp.cuda.alloc(expansion_pauli_mem)
    d_out_coef = cp.cuda.alloc(expansion_coef_mem)

    # Observable Z_62 (coefficient 1.0), created on host then copied to device.
    print("Observable: Z_62")
    h_obs_pauli = np.zeros((1, num_packed_ints * 2), dtype=np.uint64, order="C")
    h_obs_coef = np.array([[1.0]], dtype=np.float64, order="C")
    h_obs_pauli[0, :] = get_pauli_string_as_packed_integers([cupp.PauliKind.PAULI_Z], [62])

    h_expansion = cupp.create_pauli_expansion(
        handle, NUM_CIRCUIT_QUBITS,
        h_obs_pauli.ctypes.data, h_obs_pauli.nbytes,
        h_obs_coef.ctypes.data, h_obs_coef.nbytes,
        NAME_TO_DATA_TYPE['float64'], 1, 1, 1)
    h_obs_view = cupp.pauli_expansion_get_contiguous_range(handle, h_expansion, 0, 1)

    in_expansion = cupp.create_pauli_expansion(
        handle, NUM_CIRCUIT_QUBITS,
        d_in_pauli.ptr, expansion_pauli_mem, d_in_coef.ptr, expansion_coef_mem,
        NAME_TO_DATA_TYPE['float64'], 0, 0, 0)
    out_expansion = cupp.create_pauli_expansion(
        handle, NUM_CIRCUIT_QUBITS,
        d_out_pauli.ptr, expansion_pauli_mem, d_out_coef.ptr, expansion_coef_mem,
        NAME_TO_DATA_TYPE['float64'], 0, 0, 0)

    cupp.pauli_expansion_populate_from_view(handle, h_obs_view, in_expansion, 0)
    cupp.destroy_pauli_expansion_view(h_obs_view)
    cupp.destroy_pauli_expansion(h_expansion)

    # Workspaces: one (with buffer) for single-op + compute, and three *distinct* bufferless
    # descriptors required by the fused prepare (min / max / average sizing tiers).
    workspace = cupp.create_workspace_descriptor(handle)
    ws_min = cupp.create_workspace_descriptor(handle)
    ws_max = cupp.create_workspace_descriptor(handle)
    ws_avg = cupp.create_workspace_descriptor(handle)
    d_workspace_buffer = cp.cuda.alloc(workspace_mem)

    # Truncation: |coef| < 1e-4 or Pauli weight > 8.
    coef_params = cupp.CoefficientTruncationParams()
    coef_params.cutoff = 1e-4
    weight_params = cupp.PauliWeightTruncationParams()
    weight_params.cutoff = 8
    coef_strategy = cupp.TruncationStrategy()
    coef_strategy.strategy = cupp.TruncationStrategyKind.TRUNCATION_STRATEGY_COEFFICIENT_BASED
    coef_strategy.param_struct = coef_params.ptr
    weight_strategy = cupp.TruncationStrategy()
    weight_strategy.strategy = cupp.TruncationStrategyKind.TRUNCATION_STRATEGY_PAULI_WEIGHT_BASED
    weight_strategy.param_struct = weight_params.ptr
    truncation_strategies = [coef_strategy, weight_strategy]

    # Circuit parameters and the (reused) per-Trotter-step layers.
    x_rotation_angle = PI / 4.0
    num_trotter_steps = 20
    x_gate_truncation_cadence = 10

    # The forward step is [Rx, Rzz_RED, Rzz_BLUE, Rzz_GREEN]; back-propagation processes it in
    # reverse (GREEN, BLUE, RED, then Rx), applying the adjoint of every gate.
    clifford_block = get_zz_layer_adjoint_clifford_operators(
        handle, ZZ_QUBITS_GREEN, ZZ_QUBITS_BLUE, ZZ_QUBITS_RED)
    x_layer = get_x_rotation_layer(handle, x_rotation_angle)

    print(f"Circuit: 127-qubit IBM heavy-hex Ising")
    print(f"  Trotter steps:           {num_trotter_steps}")
    print(f"  Rx angle:                {x_rotation_angle} (pi/4)")
    print(f"  Cliffords per ZZ block:  {len(clifford_block)} (fused per Trotter step)")
    print()

    start_time = time.time()
    max_num_terms = 1

    for step in range(num_trotter_steps):
        # Entangling block of this step, applied in one fused Clifford call (with fallback).
        in_expansion, out_expansion = apply_clifford_block(
            handle, in_expansion, out_expansion, clifford_block, truncation_strategies,
            d_workspace_buffer, workspace_mem, out_term_capacity,
            ws_min, ws_avg, ws_max, workspace, f"step {step}")

        # Non-Clifford Rx layer, applied one gate at a time with adjoint=True.
        for i, gate in enumerate(x_layer):
            active = truncation_strategies if (i % x_gate_truncation_cadence == 0) else None
            in_expansion, out_expansion = apply_single_gate(
                handle, in_expansion, out_expansion, gate, True, active,
                d_workspace_buffer, workspace_mem, workspace)

        max_num_terms = max(max_num_terms, cupp.pauli_expansion_get_num_terms(handle, in_expansion))

    # Expectation value <Z_62> = Tr(in_expansion * |0><0|).
    num_out_terms = cupp.pauli_expansion_get_num_terms(handle, in_expansion)
    out_view = cupp.pauli_expansion_get_contiguous_range(handle, in_expansion, 0, num_out_terms)
    cupp.pauli_expansion_view_prepare_trace_with_zero_state(
        handle, out_view, workspace_mem, workspace)
    cupp.workspace_set_memory(handle, workspace, _DEVICE, _SCRATCH,
                              d_workspace_buffer.ptr, workspace_mem)
    trace_significand = np.zeros(1, dtype=np.float64)
    trace_exponent = np.zeros(1, dtype=np.float64)
    cupp.pauli_expansion_view_compute_trace_with_zero_state(
        handle, out_view,
        trace_significand.ctypes.data, trace_exponent.ctypes.data, workspace, 0)
    expec = trace_significand[0] * np.exp2(trace_exponent[0])
    cupp.destroy_pauli_expansion_view(out_view)

    duration = time.time() - start_time

    print()
    print(f"Expectation value:       {expec}")
    print(f"Final number of terms:   {num_out_terms}")
    print(f"Maximum number of terms: {max_num_terms}")
    print(f"Runtime:                 {duration} seconds")
    print()

    # Clean up
    for gate in clifford_block:
        cupp.destroy_operator(gate)
    for gate in x_layer:
        cupp.destroy_operator(gate)
    cupp.destroy_workspace_descriptor(workspace)
    cupp.destroy_workspace_descriptor(ws_min)
    cupp.destroy_workspace_descriptor(ws_max)
    cupp.destroy_workspace_descriptor(ws_avg)
    cupp.destroy_pauli_expansion(in_expansion)
    cupp.destroy_pauli_expansion(out_expansion)
    cupp.destroy(handle)


main()
