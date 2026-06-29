# Copyright (c) 2021-2026, NVIDIA CORPORATION & AFFILIATES
#
# SPDX-License-Identifier: BSD-3-Clause

"""
A converter that translates a quantum circuit to tensor network Einsum equations.
"""

__all__ = ['CircuitToEinsum', 'CirqParserOptions', 'QiskitParserOptions']

import collections.abc
import importlib
from dataclasses import dataclass
import numpy as np

from nvmath.internal import utils

from ._internal import circuit_converter_utils as circ_utils
from ._internal.helpers import get_auto_backend, get_dtype_name, _get_backend_asarray_func

EMPTY_DICT = circ_utils.EMPTY_DICT

@dataclass
class CirqParserOptions:
    """
    A data class for providing Cirq parser options to the :class:`CircuitToEinsum` class.

    Attributes:
        check_diagonal: If True (default), the parser will check if the gate operand can be represented in a diagonal form.
    
    .. note::

        Setting ``check_diagonal=True`` will perform value-based comparison for gate operands in :class:`cirq.Circuit` object to check if the gate operand can be represented in a diagonal form. 
        This may lead to performance degradation during the parsing stage, but can potentially reduce the size of the output tensor network and thus improve the contraction performance.
    """
    check_diagonal: bool = True


@dataclass
class QiskitParserOptions:
    """
    A data class for providing Qiskit parser options to the :class:`CircuitToEinsum` class.

    Attributes:
        decompose_gates: If True (default), the parser will decompose all standard gates into a sequence of gates that act on at most
            two qubits. Custom (user-defined or opaque) gates will be preserved as-is. If False, each gate in the circuit is treated as a logical unit, 
            regardless of its qubit width. 
        check_diagonal: If True (default), the parser will check if the gate operand can be represented in a diagonal form.
    """
    decompose_gates: bool = True
    check_diagonal: bool = True


class CircuitToEinsum:
    """
    Create a converter object that can generate Einstein summation expressions and tensor operands for a given circuit.

    The supported circuit types include :class:`cirq.Circuit` and :class:`qiskit.QuantumCircuit`. The input circuit must 
    be fully parameterized and can not contain operations that are not well-defined in tensor network simulation, for instance, 
    resetting the quantum state or performing any intermediate measurement. 

    The converter automatically targets a density-matrix (mixed-state) network when the input circuit
    contains quantum channels (Kraus/general and unitary channels), and a state-vector (pure-state)
    network otherwise. In the mixed case, the accessors return density-matrix quantities; see
    :meth:`density_matrix`, :meth:`amplitude`, and :meth:`batched_amplitudes` for details. For
    trajectory-based noisy simulation of a pure state, use
    :meth:`~cuquantum.tensornet.experimental.NetworkState.from_circuit` instead.

    Args:
        circuit : A fully parameterized :class:`cirq.Circuit` or :class:`qiskit.QuantumCircuit` object.
        dtype : The datatype for the output tensor operands. Currently supports ``'float32'``, ``'float64'``, ``'complex64'``, and ``'complex128'``.
            If not specified, double complex is used. Note that real dtype should only be used when all gate operands are expected to be real.
        backend: A string specifying the ndarray backend for the output tensor operands. 
            Currently supports ``'auto'`` (default), ``'numpy'``, ``'cupy'``, and ``'torch'``.
            If ``'auto'``, ``'cupy'`` is used when it is available, otherwise ``'numpy'`` is used.
        options: Specify the parser options as a :class:`CirqParserOptions` for :class:`cirq.Circuit` 
            or a :class:`QiskitParserOptions` for :class:`qiskit.QuantumCircuit`. 
            Alternatively, a `dict` containing the parameters for the :class:`CirqParserOptions` or :class:`QiskitParserOptions` constructor can also be provided. 
            If not specified, the value will be set to the default-constructed :class:`CirqParserOptions` or :class:`QiskitParserOptions` based on the circuit type. 

    Examples:

        Examples using Qiskit:

        >>> import qiskit.circuit.random
        >>> from cuquantum.tensornet import contract, CircuitToEinsum

        Generate a random quantum circuit:
        
        >>> qc = qiskit.circuit.random.random_circuit(num_qubits=8, depth=7)

        Create a :class:`CircuitToEinsum` object:

        >>> converter = CircuitToEinsum(qc, backend='cupy')

        Find the Einstein summation expression and tensor operands for the state vector:

        >>> expression, operands = converter.state_vector()

        Contract the equation above to compute the state vector:
        
        >>> sv = contract(expression, *operands)
        >>> print(sv.shape)
        (2, 2, 2, 2, 2, 2, 2, 2)

        Find the Einstein summation expression and tensor operands for computing the probability amplitude of bitstring 00000000:

        >>> expression, operands = converter.amplitude('00000000')

        Contract the equation above to compute the amplitude:

        >>> amplitude = contract(expression, *operands)

        Find the Einstein summation expression and tensor operands for computing reduced density matrix on the
        first two qubits with the condition that the last qubit is fixed at state ``1``:

        >>> where = qc.qubits[:2]
        >>> fixed = {qc.qubits[-1]: '1'}
        >>> expression, operands = converter.reduced_density_matrix(where, fixed=fixed)

        Contract the equation above to compute the reduced density matrix:

        >>> rdm = contract(expression, *operands)
        >>> print(rdm.shape)
        (2, 2, 2, 2)

    """
    def __init__(self, circuit, *, dtype='complex128', backend="auto", options=None):
        # infer library-specific parser
        self.parser = circ_utils.infer_parser(circuit)

        circuit = self.parser.remove_measurements(circuit)
        self.circuit = circuit
        if backend == "auto":
            backend = get_auto_backend()
        elif isinstance(backend, str):
            backend = importlib.import_module(backend)
        self.backend = backend
        self.backend_name = backend.__name__

        circuit_package = utils.infer_object_package(circuit)
        if circuit_package == 'qiskit':
            self.options = utils.check_or_create_options(QiskitParserOptions, options, 'QiskitParserOptions')
        elif circuit_package == 'cirq':
            self.options = utils.check_or_create_options(CirqParserOptions, options, 'CirqParserOptions')
        else:
            raise ValueError(f"Unsupported circuit type: {circuit_package}")

        self.check_diagonal = self.options.check_diagonal
        self.decompose_gates = getattr(self.options, 'decompose_gates', True)
        
        if isinstance(dtype, str):
            try:
                dtype = getattr(backend, dtype)
            except AttributeError:
                dtype = getattr(backend, np.dtype(dtype).name)

        self.dtype = dtype
        self.dtype_name = get_dtype_name(dtype)

        # unfold circuit metadata; channels are always parsed into channel entries
        self._qubits, self._gate_entries = self.parser.unfold_circuit(
            circuit, self.backend_name, self.dtype, check_diagonal=self.check_diagonal, decompose_gates=self.decompose_gates)
        # density-matrix (mixed) mode is enabled automatically when the circuit contains
        # quantum channels; a channel-free circuit is simulated as a pure state.
        self.is_mixed = any(e.kind != 'gate' for e in self._gate_entries)
        self.n_qubits = len(self.qubits)
        self._metadata = None
    
    @property
    def qubits(self):
        """A sequence of all qubits in the circuit."""
        return self._qubits

    @property
    def gates(self):
        """
        A sequence of 2-tuple (``gate_operand``, ``qubits``) representing all gates and quantum
        channels in the circuit:

        Returns:
            - tuple ``gates``:
                - ``gate_operand``: Always a single ndarray-like tensor object (never a list).
                  For a unitary gate acting on ``k`` qubits it is a rank-``2k`` tensor whose modes
                  are ordered as ``AB...ab...``, where ``AB...`` denotes all output modes and
                  ``ab...`` denotes all input modes. For a quantum channel (present only when the
                  circuit is noisy) it is a rank-``(2k+1)`` tensor stacking the ``m`` Kraus operators
                  along a leading mode, i.e. shape ``(m, AB...ab...)``. Unitary channels are returned
                  in Kraus form :math:`\\sqrt{p_k}\\, K_k`, so the stacked tensor fully describes the
                  channel action :math:`\\sum_k K_k \\rho K_k^\\dagger` (the individual probabilities
                  :math:`p_k` are not returned separately). Gate operands therefore have even rank
                  ``2k`` and channel operands odd rank ``2k+1``; the parity of ``gate_operand.ndim``
                  distinguishes the two.
                - ``qubits``: A list of arrays corresponding to all the qubits and gate tensor operands.
        """
        result = []
        asarray = _get_backend_asarray_func(self.backend)
        for e in self._gate_entries:
            if e.kind == 'gate':
                result.append((e.operand, e.qubits))
            else:
                result.append((circ_utils.stack_kraus_operators(e, self.backend, asarray, self.dtype), e.qubits))
        return result

    @property
    def _gates_are_diagonal(self):
        """Backward-compatible accessor for diagonal flags."""
        return [e.is_diagonal for e in self._gate_entries]
        
    def state_vector(self):
        """
        Generate the Einstein summation expression and tensor operands to compute the statevector for the input circuit.

        This is only supported for channel-free (pure-state) circuits. For circuits containing
        quantum channels (which are simulated as mixed states) use :meth:`density_matrix` to obtain
        the full density matrix.

        Returns:
            The Einstein summation expression and a list of tensor operands. The output shape is
            ``(d_0, d_1, ..., d_{N-1})`` and the order of the output mode labels is consistent with
            :attr:`CircuitToEinsum.qubits`.
            For :class:`cirq.Circuit`, this order corresponds to all qubits in the circuit sorted in ascending order. 
            For :class:`qiskit.QuantumCircuit`, this order is the same as :attr:`qiskit.QuantumCircuit.qubits`.
        """
        if self.is_mixed:
            raise TypeError("state_vector() is not supported for mixed-state (channel-containing) circuits. "
                            "Use density_matrix() to obtain the full density matrix.")
        return self.batched_amplitudes(dict())

    def density_matrix(self):
        """
        Generate the Einstein summation expression and tensor operands to compute the full density matrix for the input circuit.

        This is supported for both channel-free circuits (where it yields :math:`\\rho = |\\psi\\rangle\\langle\\psi|`)
        and circuits with channels (mixed states). For a channel-free circuit, :meth:`state_vector` is the cheaper rank-N alternative.

        Returns:
            The Einstein summation expression and a list of tensor operands. The output shape is
            ``(d_0, ..., d_{N-1}, d_0, ..., d_{N-1})`` where the first N modes are ket (row) indices
            and the last N modes are bra (column) indices, each in :attr:`CircuitToEinsum.qubits` order.
            For :class:`cirq.Circuit`, this order corresponds to all qubits in the circuit sorted in ascending order. 
            For :class:`qiskit.QuantumCircuit`, this order is the same as :attr:`qiskit.QuantumCircuit.qubits`.
        """
        return self.reduced_density_matrix(self.qubits, lightcone=False)

    def batched_amplitudes(self, fixed):
        """
        Generate the Einstein summation expression and tensor operands to compute a slice of the
        underlying state tensor for the input circuit.

        For channel-free (pure-state) circuits, returns a slice of the state vector
        :math:`\\langle \\text{bs} | \\psi \\rangle` over the open ket modes.

        For circuits with channels (mixed states), returns a slice of the density matrix with the
        requested ket and bra modes projected independently. Output modes are ordered as
        ``(open_ket_modes, open_bra_modes)`` (each in :attr:`CircuitToEinsum.qubits` order).

        Args:    
            fixed: A dictionary mapping qubits to fixed states; qubits absent from the dictionary
                are left open.

                * For pure-state (channel-free) circuits, each value is a single state ``0`` or ``1`` (as
                  :class:`int` or ``'0'``/``'1'`` :class:`str`).
                * For mixed-state (channel-containing) circuits, each value selects the ket and bra index for that qubit
                  and may be either

                  - a single state ``0``/``1`` -- shorthand for fixing the ket and bra to the same
                    value (a diagonal/symmetric projection), or
                  - a 2-tuple ``(ket, bra)`` where each entry is ``0``/``1`` or ``None``; ``None``
                    leaves that side open, enabling independent ket/bra projection
                    (e.g. ``(0, None)`` fixes only the ket).

        Returns:
            The Einstein summation expression and a list of tensor operands. The order of the
            output mode labels is consistent with :attr:`CircuitToEinsum.qubits`.
            For :class:`cirq.Circuit`, this order corresponds to all qubits in the circuit sorted in ascending order. 
            For :class:`qiskit.QuantumCircuit`, this order is the same as :attr:`qiskit.QuantumCircuit.qubits`.
        """
        if self.is_mixed:
            if not isinstance(fixed, collections.abc.Mapping):
                raise TypeError("for mixed-state (channel-containing) circuits, `fixed` must be a dict "
                                "mapping qubits to 0/1 or a (ket, bra) 2-tuple")
            fixed_ket, fixed_bra = circ_utils.split_mixed_fixed(fixed)
            return circ_utils.build_mixed_dm_slice_tn(
                self.qubits, self._gate_entries, self.dtype, self.backend_name,
                fixed_ket, fixed_bra)
        if not isinstance(fixed, collections.abc.Mapping):
            raise TypeError("for pure-state (channel-free) circuits, `fixed` must be a dictionary")
        input_mode_labels, input_operands, qubits_frontier = self._get_inputs()
        
        fixed_qubits, fixed_bitstring = circ_utils.parse_fixed_qubits(fixed)
        fixed_mode_labels = [[qubits_frontier[q]] for q in fixed_qubits]    
        mode_labels = input_mode_labels + fixed_mode_labels
        
        operands = input_operands + circ_utils.get_bitstring_tensors(fixed_bitstring, self.backend_name, self.dtype)
        output_mode_labels = [qubits_frontier[q] for q in self.qubits if q not in fixed]

        expression = circ_utils.convert_mode_labels_to_expression(mode_labels, output_mode_labels)
        return expression, operands 
    
    def amplitude(self, bitstring):
        """Generate the Einstein summation expression and tensor operands to compute a single
        element of the underlying state tensor for the input circuit.

        For channel-free (pure-state) circuits, returns the complex amplitude :math:`\\langle \\text{bitstring} | \\psi \\rangle`.

        For circuits with channels (mixed states), returns the density matrix element
        :math:`\\langle \\text{ket\\_bitstring} | \\rho | \\text{bra\\_bitstring} \\rangle`. Use the
        same bitstring on both sides to extract a diagonal element (probability).

        Args:    
            bitstring:
                * For pure-state (channel-free) circuits, a sequence of 0/1 specifying the desired measured state.
                * For mixed-state (channel-containing) circuits, either a single length-N sequence of 0/1 -- interpreted
                  symmetrically as ``ket == bra`` to extract a diagonal element (probability) -- or
                  a 2-tuple ``(ket_bitstring, bra_bitstring)`` of length-N sequences specifying the
                  row (ket) and column (bra) indices of the density matrix element to compute.

                The order of the bitstring(s) is consistent with :attr:`CircuitToEinsum.qubits`.
                For :class:`cirq.Circuit`, this order corresponds to all qubits in the circuit sorted in ascending order. 
                For :class:`qiskit.QuantumCircuit`, this order is the same as :attr:`qiskit.QuantumCircuit.qubits`.

        Returns:
            The Einstein summation expression and a list of tensor operands
        """
        if self.is_mixed:
            ket_spec, bra_spec = circ_utils.split_mixed_bitstring(bitstring, self.n_qubits)
            ket_bs = circ_utils.parse_bitstring(ket_spec, n_qubits=self.n_qubits)
            bra_bs = circ_utils.parse_bitstring(bra_spec, n_qubits=self.n_qubits)
            fixed_ket = dict(zip(self.qubits, ket_bs))
            fixed_bra = dict(zip(self.qubits, bra_bs))
            return circ_utils.build_mixed_dm_slice_tn(
                self.qubits, self._gate_entries, self.dtype, self.backend_name,
                fixed_ket, fixed_bra)
        bitstring = circ_utils.parse_bitstring(bitstring, n_qubits=self.n_qubits)
        input_mode_labels, input_operands, qubits_frontier = self._get_inputs()
        mode_labels = input_mode_labels + [[qubits_frontier[q]] for q in self.qubits]
        output_mode_labels = []

        expression = circ_utils.convert_mode_labels_to_expression(mode_labels, output_mode_labels)
        operands = input_operands + circ_utils.get_bitstring_tensors(bitstring, self.backend_name, self.dtype)
        return expression, operands 
    
    def reduced_density_matrix(self, where, *, fixed=EMPTY_DICT, lightcone=True, diagonal=False):
        r"""
        reduced_density_matrix(where, fixed=None, lightcone=True, diagonal=False)

        Generate the Einstein summation expression and tensor operands to compute the reduced density matrix for
        the input circuit. This is supported for both channel-free (pure-state) circuits and circuits with
        channels, in which case the reduced density matrix is obtained from the mixed state :math:`\rho`.

        Unitary reverse lightcone cancellation refers to removing the identity formed by a unitary gate (from
        the ket state) and its inverse (from the bra state) when there exists no additional operators
        in-between. One can take advantage of this technique to reduce the effective network size by
        only including the *causal* gates (gates residing in the lightcone).

        Args:    
            where: A sequence of qubits specifying where the density matrix are reduced onto. 
            fixed: Optional, a dictionary that maps certain qubits to the corresponding fixed states 0 or 1.
            lightcone: Whether to apply the unitary reverse lightcone cancellation technique to reduce the number of tensors in density matrix computation.
            diagonal: If ``False`` (default), the full reduced density matrix is computed. If ``True``, the bra modes are
                contracted onto the ket modes so that only the diagonal of the reduced density matrix is computed, i.e. the
                marginal probability distribution over ``where``. With ``diagonal=True`` this is functionally equivalent to
                :meth:`marginal_probability`.
            
        Returns:
            The Einstein summation expression and a list of tensor operands.
            With ``diagonal=False`` the mode labels for the output of the expression has the same order as the where argument.
            For example, if where = (:math:`a, b`), the mode labels for the reduced density matrix would be (:math:`a, b, a^{\prime}, b^{\prime}`).
            With ``diagonal=True`` the output carries only the ket modes (e.g. (:math:`a, b`)), holding the diagonal entries.
        
        .. seealso:: `unitary reverse lightcone cancellation <https://quimb.readthedocs.io/en/latest/tensor-circuit.html#Unitary-Reverse-Lightcone-Cancellation>`_
        """
        if self.is_mixed:
            expression, operands = self._build_mixed_tn(where, fixed=fixed, lightcone=lightcone)
            if diagonal:
                expression = self._collapse_rdm_expression_to_diagonal(expression, len(where))
            return expression, operands

        n_qubits = self.n_qubits
        coned_qubits = list(where) + list(fixed.keys())
        input_mode_labels, input_operands, qubits_frontier, next_frontier, inverse_gate_entries = self._get_forward_inverse_metadata(lightcone, coned_qubits)

        # handle tensors/mode labels for qubits with fixed state
        fixed_qubits, fixed_bitstring = circ_utils.parse_fixed_qubits(fixed)
        fixed_operands = circ_utils.get_bitstring_tensors(fixed_bitstring, self.backend_name, self.dtype)

        mode_labels = input_mode_labels + [[qubits_frontier[ix]] for ix in fixed_qubits]
        for iqubit in fixed_qubits:
            qubits_frontier[iqubit] = next_frontier
            mode_labels.append([next_frontier])
            next_frontier += 1
        operands = input_operands + fixed_operands * 2

        output_mode_labels_info = dict()
        for iqubit in where:
            output_mode_labels_info[iqubit] = [qubits_frontier[iqubit], next_frontier]
            qubits_frontier[iqubit] = next_frontier
            next_frontier += 1

        igate_mode_labels, igate_operands = circ_utils.parse_gates_to_mode_labels_operands(
            inverse_gate_entries, qubits_frontier, next_frontier)
        mode_labels += igate_mode_labels
        operands += igate_operands
        
        mode_labels += [[qubits_frontier[ix]] for ix in self.qubits]
        operands += input_operands[:n_qubits]
        
        output_left_mode_labels = []
        output_right_mode_labels = []
        for _, (left_mode_labels, right_mode_labels) in output_mode_labels_info.items():
            output_left_mode_labels.append(left_mode_labels)
            output_right_mode_labels.append(right_mode_labels)
        output_mode_labels = output_left_mode_labels + output_right_mode_labels
        expression = circ_utils.convert_mode_labels_to_expression(mode_labels, output_mode_labels)
        if diagonal:
            expression = self._collapse_rdm_expression_to_diagonal(expression, len(where))
        return expression, operands

    @staticmethod
    def _collapse_rdm_expression_to_diagonal(expression, num_target_qubits):
        """
        Rewrite a reduced-density-matrix einsum expression (output modes ``ket... bra...``) so that the bra
        modes are identified with the corresponding ket modes and only the ket modes are emitted, yielding the
        RDM diagonal (the marginal probability distribution).
        """
        input_modes, output_modes = expression.split('->')
        ket_modes = output_modes[:num_target_qubits]
        bra_modes = output_modes[num_target_qubits:]
        for ket_mode, bra_mode in zip(ket_modes, bra_modes):
            input_modes = input_modes.replace(bra_mode, ket_mode)
        return f"{input_modes}->{ket_modes}"

    def marginal_probability(self, where, *, fixed=EMPTY_DICT, lightcone=True):
        r"""
        marginal_probability(where, fixed=None, lightcone=True)

        Generate the Einstein summation expression and tensor operands to compute the marginal probability for
        the input circuit.

        Unitary reverse lightcone cancellation refers to removing the identity formed by a unitary gate (from
        the ket state) and its inverse (from the bra state) when there exists no additional operators
        in-between. One can take advantage of this technique to reduce the effective network size by
        only including the *causal* gates (gates residing in the lightcone).

        Args:    
            where: A sequence of qubits specifying where the marginal probability are computed. 
            fixed: Optional, a dictionary that maps certain qubits to the corresponding fixed states 0 or 1.
            lightcone: Whether to apply the unitary reverse lightcone cancellation technique to reduce the number of tensors in marginal probability computation.
            
        Returns:
            The Einstein summation expression and a list of tensor operands.
            The mode labels for output of the expression has the same order as the where argument.
        
        .. note::

            The marginal probability resulting from the contraction may be a complex tensor with zero imaginary part depending on the underlying data type.

        .. note::

            This is functionally equivalent to :meth:`reduced_density_matrix` with ``diagonal=True``.
         
        .. seealso:: `unitary reverse lightcone cancellation <https://quimb.readthedocs.io/en/latest/tensor-circuit.html#Unitary-Reverse-Lightcone-Cancellation>`_
        """
        return self.reduced_density_matrix(where, fixed=fixed, lightcone=lightcone, diagonal=True)

    
    def expectation(self, pauli_string, lightcone=True):
        """
        Generate the Einstein summation expression and tensor operands to compute the expectation value of a Pauli
        string for the input circuit. This is supported for both channel-free (pure-state) circuits and circuits
        with channels, in which case the expectation value :math:`\\mathrm{Tr}(\\rho P)` is computed from the
        mixed state :math:`\\rho`.

        Unitary reverse lightcone cancellation refers to removing the identity formed by a unitary gate (from
        the ket state) and its inverse (from the bra state) when there exists no additional operators
        in-between. One can take advantage of this technique to reduce the effective network size by
        only including the *causal* gates (gates residing in the lightcone).

        Args:    
            pauli_string: The Pauli string for expectation value computation. It can be:

                - a sequence of characters ``'I'``/``'X'``/``'Y'``/``'Z'``. The length must be equal to the number of qubits.
                - a dictionary mapping the selected qubits to Pauli characters. Qubits not specified are
                  assumed to be applied with the identity operator ``'I'``.
            
            lightcone: Whether to apply the unitary reverse lightcone cancellation technique to reduce the number of tensors in expectation value computation.
            
        Returns:
            The Einstein summation expression and a list of tensor operands.
        
        .. note::

            When ``lightcone=True``, the identity Pauli operators will be omitted in the output operands. The unitary reverse lightcone cancellation technique is then 
            applied based on the remaining causal qubits to further reduce the size of the network. The reduction effect depends on the circuit topology and the input Pauli string 
            (so the contraction path cannot be reused for the contraction of different Pauli strings). When ``lightcone=False``, the identity Pauli operators are preserved in the output operands such that the output tensor network has the identical topology for different Pauli strings, and the contraction path only needs to be computed once and can be reused for all Pauli strings.
        
        .. note::

            When the underlying dtype is real, Pauli Y operator is not supported.

        .. seealso:: `unitary reverse lightcone cancellation <https://quimb.readthedocs.io/en/latest/tensor-circuit.html#Unitary-Reverse-Lightcone-Cancellation>`_
        """
        if isinstance(pauli_string, collections.abc.Sequence):
            if len(pauli_string) != self.n_qubits:
                raise ValueError('pauli_string must be of equal size as the number of qubits in the circuit')
            pauli_string = dict(zip(self.qubits, pauli_string))
        else:
            if not isinstance(pauli_string, collections.abc.Mapping):
                raise TypeError('pauli_string must be either a sequence of pauli characters or a dictionary')
        
        n_qubits = self.n_qubits
        if lightcone:
            pauli_map = {qubit: pauli_char for qubit, pauli_char in pauli_string.items() if pauli_char!='I'}
        else:
            pauli_map = pauli_string

        if self.dtype_name.startswith("float"):
            pauli_chars = set(pauli_map.values())
            if 'Y' in pauli_chars:
                raise ValueError(f"Pauli Y operator is not supported when the underlying dtype is {self.dtype_name}")

        if self.is_mixed:
            return self._build_mixed_expectation(pauli_map, lightcone=lightcone)

        coned_qubits = pauli_map.keys()
        input_mode_labels, input_operands, qubits_frontier, next_frontier, inverse_gate_entries = self._get_forward_inverse_metadata(lightcone, coned_qubits)

        pauli_entries = circ_utils.get_pauli_gates(pauli_map, self.backend_name, self.dtype)
        combined_entries = pauli_entries + inverse_gate_entries

        gate_mode_labels, gate_operands = circ_utils.parse_gates_to_mode_labels_operands(
            combined_entries, qubits_frontier, next_frontier)
        
        mode_labels = input_mode_labels + gate_mode_labels + [[qubits_frontier[ix]] for ix in self.qubits]
        operands = input_operands + gate_operands + input_operands[:n_qubits]

        output_mode_labels = []
        expression = circ_utils.convert_mode_labels_to_expression(mode_labels, output_mode_labels)
        return expression, operands

    def _get_mixed_gate_entries(self, lightcone, coned_qubits):
        """Get gate entries for the mixed path, applying lightcone if requested."""
        parser = self.parser
        if lightcone:
            circuit = parser.get_lightcone_circuit(self.circuit, coned_qubits)
            _, gate_entries = parser.unfold_circuit(
                circuit, self.backend_name, self.dtype,
                decompose_gates=self.decompose_gates, check_diagonal=self.check_diagonal)
        else:
            gate_entries = self._gate_entries
        return gate_entries

    def _build_mixed_tn(self, where, *, fixed=EMPTY_DICT, lightcone=True):
        """Build the doubled TN for mixed-state reduced density matrix."""
        coned_qubits = list(where) + list(fixed.keys())
        gate_entries = self._get_mixed_gate_entries(lightcone, coned_qubits)
        return circ_utils.build_mixed_tn(
            self.qubits, gate_entries, self.dtype, self.backend_name, where, fixed)

    def _build_mixed_expectation(self, pauli_map, *, lightcone=True):
        """Build the doubled TN for mixed-state expectation value of a Pauli string."""
        coned_qubits = list(pauli_map.keys())
        gate_entries = self._get_mixed_gate_entries(lightcone, coned_qubits)
        return circ_utils.build_mixed_expectation_tn(
            self.qubits, gate_entries, self.dtype, self.backend_name, pauli_map)

    def _get_inputs(self):
        """transform the qubits and gates in the circuit to a prelimary Einsum form.

        Returns:
            metadata: A 3-tuple (``mode_labels``, ``operands``, ``qubits_frontier``):

                - ``mode_labels`` :  A list of list of int, each corresponding to the mode labels for the tensor operands.
                - ``operands`` : A list of arrays corresponding to all the qubits and gate tensor operands.
                - ``qubits_frontier`` : A dictionary that maps all qubits to their current mode labels.
        """
        if self._metadata is None:
            self._metadata = circ_utils.parse_inputs(self.qubits, self._gate_entries, self.dtype, self.backend_name)
        return self._metadata
    
    def _get_forward_inverse_metadata(self, lightcone, coned_qubits):
        """parse the metadata for forward and inverse circuit.

        Args:
            lightcone: Whether to apply the unitary reverse lightcone cancellation technique to reduce the number of tensors in expectation value computation.
            coned_qubits: An iterable of qubits to be coned.

        Returns:
            tuple: A 6-tuple (``input_mode_labels``, ``input_operands``, ``qubits_frontier``, ``next_frontier``, ``inverse_gate_entries``, ...):

                - ``input_mode_labels`` :  A sequence of mode labels for initial states and gate tensors.
                - ``input_operands`` :  A sequence of operands for initial states and gate tensors.
                - ``qubits_frontier``: A dictionary mapping all qubits to their current mode labels.
                - ``next_frontier``: The next mode label to use.
                - ``inverse_gate_entries``: A list of :class:`GateEntry` for the inverse circuit.
        """
        parser = self.parser
        if lightcone:
            circuit = parser.get_lightcone_circuit(self.circuit, coned_qubits)
            _, gate_entries = parser.unfold_circuit(circuit, self.backend_name, self.dtype, decompose_gates=self.decompose_gates, check_diagonal=self.check_diagonal)
            input_mode_labels, input_operands, qubits_frontier = circ_utils.parse_inputs(self.qubits, gate_entries, self.dtype, self.backend_name)
        else:
            circuit = self.circuit
            input_mode_labels, input_operands, qubits_frontier = self._get_inputs()
            # avoid inplace modification on metadata
            qubits_frontier = qubits_frontier.copy()
        
        next_frontier = max(qubits_frontier.values()) + 1
        # inverse circuit
        inverse_circuit = parser.get_inverse_circuit(circuit)
        _, inverse_gate_entries = parser.unfold_circuit(inverse_circuit, self.backend_name, self.dtype, decompose_gates=self.decompose_gates, check_diagonal=self.check_diagonal)
        return input_mode_labels, input_operands, qubits_frontier, next_frontier, inverse_gate_entries
