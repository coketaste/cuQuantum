# Copyright (c) 2021-2026, NVIDIA CORPORATION & AFFILIATES
#
# SPDX-License-Identifier: BSD-3-Clause

import importlib

import numpy as np
from qiskit import QuantumCircuit
from qiskit.circuit import Barrier, ControlledGate, Delay, Gate, Measure
try:
    from qiskit.circuit import AnnotatedOperation
except ImportError:
    AnnotatedOperation = None  # older Qiskit versions don't have this
from qiskit.circuit.library import (
    UnitaryGate, ZGate, SGate, SdgGate, TGate, 
    TdgGate, PhaseGate, RZGate, CZGate, CCZGate, RZZGate,
    DiagonalGate, IGate, U1Gate
)

from qiskit.quantum_info import (
    Operator, 
    Choi,
    SuperOp,
    Kraus,
    Stinespring,
    Chi,
    PTM,
)

DIAGONAL_GATE_CLASSES = (
    ZGate, SGate, SdgGate, TGate, TdgGate,
    PhaseGate, RZGate, CZGate, CCZGate, RZZGate,
    DiagonalGate, IGate, U1Gate,
)

from .circuit_converter_utils import GateEntry
from .helpers import _get_backend_asarray_func, get_dtype_name

# https://docs.quantum.ibm.com/api/qiskit/quantum_info#channels
NOISY_CHANNEL_TYPES = (
    Operator, 
    Choi,
    SuperOp,
    Kraus,
    Stinespring,
    Chi,
    PTM,
)

def remove_measurements(circuit):
    """
    Return a circuit with final measurement operations removed.
    """
    circuit = circuit.copy()
    circuit.remove_final_measurements()
    for instruction in circuit.data:
        if isinstance(instruction.operation, Measure):
            raise ValueError('mid-circuit measurement not supported in tensor network simulation')
    return circuit

def get_inverse_circuit(circuit):
    """
    Return a circuit with all gate operations inversed.
    """
    return circuit.inverse()

def should_parse_operation_operand(operation, qubits, decompose_gates):
    """
    Return whether the input operation should be parsed as full operand or further decomposed.
    If the operation is a Gate instance and also satisifies at least one of the following conditions, directly parse the operand:
        1. is a customized unitary gate
        2. number of qubits no larger than two
        3. number of qubits larger than two but decompose_gate is set to False
    """
    return isinstance(operation, Gate) and (isinstance(operation, UnitaryGate) or len(qubits) <= 2 or not decompose_gates)

def is_diagonal_gate(inst):
    if isinstance(inst, ControlledGate):
        return is_diagonal_gate(inst.base_gate)
    return isinstance(inst, DIAGONAL_GATE_CLASSES)

def _parse_channel_to_kraus(operation, gate_qubits, asarray, dtype):
    """Convert a NOISY_CHANNEL_TYPES operation to a list of Kraus operator tensors."""
    kraus_data = Kraus(operation).data
    n_qubits = len(gate_qubits)
    kraus_ops = []
    for k in kraus_data:
        tensor = k.reshape((2, 2) * n_qubits)
        if get_dtype_name(dtype).startswith("float"):
            if not np.isreal(tensor).all():
                imag_max = abs(tensor.imag).max()
                raise RuntimeError(f"channel Kraus operand found to have imaginary part {imag_max=} while real dtype {dtype} is specified")
            tensor = tensor.real
        kraus_ops.append(asarray(tensor, dtype=dtype))
    return kraus_ops

def parse_gate_sequence(
    circuit, 
    *, 
    asarray=None, 
    dtype='complex128', 
    qubit_map=None, 
    gates=None, 
    global_phase=0, 
    decompose_gates=True, 
    check_diagonal=True, 
    gates_are_diagonal=None,
):
    """
    Return the gate sequence for the given circuit.
    Args:
        circuit: A :class:`qiskit.QuantumCircuit` object. 
        asarray: An asarray function to convert the operand to the ndarray for a specific package.
            If is None, the original operation will be returned
        dtype: The dtype to convert the operation operand to
        qubit_map: A dictionary mapping the local operation qubits to the global qubits.
        gates: The current gate sequences
        global_phase: An additional global phase to add to.
        decompose_gate: Whether the operation should be decomposed when operand is being parsed into ndarrays.

    Channels (when present) are always converted to Kraus operator tensors; the pure-vs-mixed
    distinction is handled downstream by the consumer.
    """
    if gates is None:
        gates = []
    if gates_are_diagonal is None:
        gates_are_diagonal = []
    global_phase += circuit.global_phase
    for instruction in circuit.data:
        operation = instruction.operation
        gate_qubits = instruction.qubits
        if qubit_map:
            gate_qubits = [qubit_map[q] for q in gate_qubits]
        if isinstance(operation, (Barrier, Delay)):
            continue
        if not isinstance(operation, NOISY_CHANNEL_TYPES) and getattr(operation, 'name', None) == 'kraus':
            operation = Kraus(operation)
        if isinstance(operation, NOISY_CHANNEL_TYPES):
            if asarray is None:
                gates.append((operation, gate_qubits))
                continue
            kraus_ops = _parse_channel_to_kraus(operation, gate_qubits, asarray, dtype)
            gates.append((kraus_ops, gate_qubits[::-1]))
            gates_are_diagonal.append(None)
            continue
        if AnnotatedOperation is not None and isinstance(operation, AnnotatedOperation):
            raise ValueError(
                f"AnnotatedOperation '{operation.name}' is not supported. "
                "Please switch to gates without annotation."
            )
        if asarray is None:
            gates.append((operation, gate_qubits))
            continue
        if should_parse_operation_operand(operation, gate_qubits, decompose_gates):
            tensor = Operator(operation).data.reshape((2,2)*len(gate_qubits))
            if get_dtype_name(dtype).startswith("float"):
                if not np.isreal(tensor).all():
                    imag_max = abs(tensor.imag).max()
                    raise RuntimeError(f"gate operand found to have imaginary part {imag_max=} while real dtype {dtype} is specified")
                tensor = tensor.real
            tensor = asarray(tensor, dtype=dtype)
            # in qiskit notation, qubits are labelled in the inverse order
            gates.append((tensor, gate_qubits[::-1]))
            gates_are_diagonal.append(check_diagonal and is_diagonal_gate(operation))
            continue
        # Instruction as composite gate
        if not isinstance(operation.definition, QuantumCircuit):            
            raise ValueError(f'operation type {type(operation)} not supported')
        # for composite gate, must provide a map from the sub circuit to the original circuit
        next_qubit_map = dict(zip(operation.definition.qubits, gate_qubits))
        gates, global_phase, gates_are_diagonal = parse_gate_sequence(
            operation.definition, 
            asarray=asarray, 
            dtype=dtype, 
            qubit_map=next_qubit_map, 
            gates=gates, 
            global_phase=global_phase,
            decompose_gates=decompose_gates,
            check_diagonal=check_diagonal,
            gates_are_diagonal=gates_are_diagonal,
        )
    return gates, global_phase, gates_are_diagonal

def _to_gate_entries(gates, gates_are_diagonal):
    """Convert parallel (gates, gates_are_diagonal) lists to a list of GateEntry."""
    entries = []
    for (operand, qubits), is_diag in zip(gates, gates_are_diagonal):
        if is_diag is None and isinstance(operand, list):
            entries.append(GateEntry(kind='general_channel', operand=operand, qubits=qubits))
        else:
            entries.append(GateEntry(kind='gate', operand=operand, qubits=qubits, is_diagonal=is_diag))
    return entries


def unfold_circuit(circuit, backend, dtype, *, decompose_gates=True, check_diagonal=True):
    """
    Unfold the circuit to obtain the qubits and all gate entries.

    Channels (when present) are always converted to Kraus operator tensors; the pure-vs-mixed
    distinction is handled downstream by the consumer.

    Args:
        circuit: A :class:`qiskit.QuantumCircuit` object. All parameters in the circuit must be binded.
        dtype: Data type for the tensor operands.
        backend: The package the tensor operands belong to.
        decompose_gates: Whether to decompose composite gates down to at most two qubits.

    Returns:
        A tuple ``(qubits, gate_entries)`` where *gate_entries* is a list of
        :class:`~cuquantum.tensornet._internal.circuit_converter_utils.GateEntry` objects.
    """
    if circuit.parameters:
        raise ValueError(f"Input circuit contains following parameters: {circuit.parameters}. Must be fully parameterized")
    package = importlib.import_module(backend)
    asarray = _get_backend_asarray_func(package)
    qubits = circuit.qubits

    gates, global_phase, gates_are_diagonal = parse_gate_sequence(
        circuit, 
        asarray=asarray,
        dtype=dtype,
        global_phase=0, 
        decompose_gates=decompose_gates,
        check_diagonal=check_diagonal,
    )
    if global_phase != 0:
        phase = np.exp(1j*global_phase)
        if get_dtype_name(dtype).startswith("float"):
            if not np.isreal(phase).all():
                raise RuntimeError(f"global phase {global_phase} has imaginary part {abs(phase.imag).max()}")
            phase = phase.real
        phase_gate = asarray([[phase, 0], [0, phase]], dtype=dtype)
        gates = [(phase_gate, qubits[:1]), ] + gates
        gates_are_diagonal = [True] + gates_are_diagonal

    return qubits, _to_gate_entries(gates, gates_are_diagonal)

def get_lightcone_circuit(circuit, coned_qubits):
    """
    Use unitary reversed lightcone cancellation technique to reduce the effective circuit size based on the qubits to be coned. 

    Args:
        circuit: A :class:`qiskit.QuantumCircuit` object. 
        coned_qubits: An iterable of qubits to be coned.

    Returns:
        A :class:`qiskit.QuantumCircuit` object that potentially contains less number of gates
    """
    coned_qubits = set(coned_qubits)
    gates, global_phase, _ = parse_gate_sequence(circuit, asarray=None, decompose_gates=False, check_diagonal=False)
    newqc = QuantumCircuit(circuit.qubits)
    ix = len(gates)
    tail_operations = []
    while len(coned_qubits) != circuit.num_qubits and ix>0:
        ix -= 1
        operation, gate_qubits = gates[ix]
        qubit_set = set(gate_qubits)
        is_channel = isinstance(operation, NOISY_CHANNEL_TYPES)
        if is_channel or (qubit_set & coned_qubits):
            tail_operations.append([operation, gate_qubits])
            coned_qubits |= qubit_set
    for operation, gate_qubits in gates[:ix] + tail_operations[::-1]:
        newqc.append(operation, gate_qubits)
    newqc.global_phase = global_phase
    return newqc
