# Copyright (c) 2021-2025, NVIDIA CORPORATION & AFFILIATES
#
# SPDX-License-Identifier: BSD-3-Clause


import importlib
import numpy as np

import cirq
from cirq import protocols, unitary, Circuit, MeasurementGate

from .circuit_converter_utils import GateEntry
from .helpers import _get_backend_asarray_func, get_dtype_name

def remove_measurements(circuit):
    """
    Return a circuit with final measurement operations removed
    """
    circuit = circuit.copy()
    if circuit.has_measurements():
        if not circuit.are_all_measurements_terminal():
            raise ValueError('mid-circuit measurement not supported in tensor network simulation')
        else:
            predicate = lambda operation: isinstance(operation.gate, MeasurementGate)
            measurement_gates = list(circuit.findall_operations(predicate))
            circuit.batch_remove(measurement_gates)
    return circuit

def get_inverse_circuit(circuit):
    """
    Return a circuit with all gate operations inversed
    """
    return protocols.inverse(circuit)

def _is_channel(operation):
    """Check if a Cirq operation is a channel (not a single unitary gate).

    Returns True for both unitary-mixture channels (e.g. depolarizing) and
    general channels (e.g. amplitude damping). The further distinction is
    made downstream via ``cirq.has_mixture()``.
    """
    return cirq.has_kraus(operation) and not cirq.has_unitary(operation)


def _parse_channel_operands(kraus_ops, n_qubits_gate, dtype, asarray):
    """Convert raw Kraus matrices to reshaped/typed tensors with real-dtype validation."""
    tensors = []
    for k in kraus_ops:
        tensor = k.reshape((2,) * 2 * n_qubits_gate)
        if get_dtype_name(dtype).startswith("float"):
            if not np.isreal(tensor).all():
                imag_max = abs(tensor.imag).max()
                raise RuntimeError(
                    f"channel Kraus operand found to have imaginary part {imag_max=} "
                    f"while real dtype {dtype} is specified")
            tensor = tensor.real
        tensors.append(asarray(tensor, dtype=dtype))
    return tensors


def unfold_circuit(circuit, backend, dtype, check_diagonal=True, **kwargs):
    """
    Unfold the circuit to obtain the qubits and all gate entries.

    Channels (when present) are always converted to Kraus/unitary-channel entries; the
    pure-vs-mixed distinction is handled downstream by the consumer.

    Args:
        circuit: A :class:`cirq.Circuit` object. All parameters in the circuit must be resolved.
        dtype: Data type for the tensor operands.
        backend: The package the tensor operands belong to.

    Returns:
        A tuple ``(qubits, gate_entries)`` where *gate_entries* is a list of
        :class:`~cuquantum.tensornet._internal.circuit_converter_utils.GateEntry` objects.
    """
    qubits = sorted(circuit.all_qubits())
    package = importlib.import_module(backend)
    asarray = _get_backend_asarray_func(package)
    gate_entries = []
    for moment in circuit.moments:
        for operation in moment:
            gate_qubits = operation.qubits
            if _is_channel(operation):
                n_qubits_gate = len(gate_qubits)
                if cirq.has_mixture(operation):
                    mixture_data = cirq.mixture(operation)
                    probs, unitaries = zip(*mixture_data)
                    operands = _parse_channel_operands(unitaries, n_qubits_gate, dtype, asarray)
                    gate_entries.append(GateEntry(
                        kind='unitary_channel', operand=operands,
                        qubits=gate_qubits,
                        probabilities=tuple(float(p) for p in probs)))
                    continue
                kraus_ops = cirq.kraus(operation)
                kraus_tensors = _parse_channel_operands(kraus_ops, n_qubits_gate, dtype, asarray)
                gate_entries.append(GateEntry(
                    kind='general_channel', operand=kraus_tensors, qubits=gate_qubits))
                continue
            operand = unitary(operation.gate)
            is_diag = check_diagonal and cirq.is_diagonal(operand, atol=1e-14)
            tensor = operand.reshape((2,) * 2 * len(gate_qubits))
            if get_dtype_name(dtype).startswith("float"):
                if not np.isreal(tensor).all():
                    imag_max = abs(tensor.imag).max()
                    raise RuntimeError(f"gate operand found to have imaginary part {imag_max=} while real dtype {dtype} is specified")
                tensor = tensor.real
            tensor = asarray(tensor, dtype=dtype)
            gate_entries.append(GateEntry(
                kind='gate', operand=tensor, qubits=gate_qubits, is_diagonal=is_diag))
    return qubits, gate_entries

def get_lightcone_circuit(circuit, coned_qubits):
    """
    Use unitary reversed lightcone cancellation technique to reduce the effective circuit size based on the qubits to be coned. 

    Args:
        circuit: A :class:`cirq.Circuit` object. 
        coned_qubits: An iterable of qubits to be coned.

    Returns:
        A :class:`cirq.Circuit` object that potentially contains less number of gates
    """
    coned_qubits = set(coned_qubits)
    all_operations = list(circuit.all_operations())
    n_qubits = len(circuit.all_qubits())
    ix = len(all_operations)
    tail_operations = []
    while len(coned_qubits) != n_qubits and ix>0:
        ix -= 1
        operation = all_operations[ix]
        qubit_set = set(operation.qubits)
        is_channel = _is_channel(operation)
        if is_channel or (qubit_set & coned_qubits):
            tail_operations.append(operation)
            coned_qubits |= qubit_set
    newqc = Circuit(all_operations[:ix]+tail_operations[::-1])
    return newqc
