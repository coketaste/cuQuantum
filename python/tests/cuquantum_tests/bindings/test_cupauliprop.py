# Copyright (c) 2025-2026, NVIDIA CORPORATION & AFFILIATES
#
# SPDX-License-Identifier: BSD-3-Clause

import contextlib
import functools

import pytest
import numpy as np

cp = pytest.importorskip("cupy")
from cupy import testing

from nvmath.internal.typemaps import NAME_TO_DATA_TYPE
from cuquantum.bindings import cupauliprop as cupp


###################################################################
#
# Following the test_cutensornet.py philosophy:
# The sole goal is to ensure the Python arguments are properly
# passed to the C level. We do not ensure coverage nor correctness.
#
###################################################################

def manage_resource(name):
    """Decorator to manage resource lifecycle in tests."""
    def decorator(impl):
        @functools.wraps(impl)
        def test_func(self, *args, **kwargs):
            try:
                if name == 'handle':
                    h = cupp.create()
                elif name == 'workspace':
                    h = cupp.create_workspace_descriptor(self.handle)
                elif name == 'pauli_expansion':
                    # Create a simple Pauli expansion on device
                    num_qubits = getattr(self, 'num_qubits', 4)
                    num_terms = getattr(self, 'num_terms', 2)
                    dtype = getattr(self, 'dtype', np.complex128)
                    
                    num_packed_ints = cupp.get_num_packed_integers(num_qubits)
                    xz_bits_size = num_terms * 2 * num_packed_ints * np.dtype(np.uint64).itemsize
                    coef_size = num_terms * np.dtype(dtype).itemsize
                    
                    d_xz_bits = cp.cuda.alloc(xz_bits_size)
                    d_coefs = cp.cuda.alloc(coef_size)
                    
                    h = cupp.create_pauli_expansion(
                        self.handle, num_qubits,
                        d_xz_bits.ptr, xz_bits_size,
                        d_coefs.ptr, coef_size,
                        NAME_TO_DATA_TYPE[np.dtype(dtype).name],
                        num_terms, 0, 1)  # not sorted, has duplicates
                    
                    # Keep memory alive
                    self._xz_bits_mem = d_xz_bits
                    self._coefs_mem = d_coefs
                    
                elif name == 'pauli_expansion_view':
                    # Create view of entire expansion
                    num_terms = cupp.pauli_expansion_get_num_terms(
                        self.handle, self.pauli_expansion)
                    h = cupp.pauli_expansion_get_contiguous_range(
                        self.handle, self.pauli_expansion, 0, num_terms)
                    
                elif name == 'clifford_operator':
                    # Create a simple Hadamard gate on qubit 0
                    h = cupp.create_clifford_gate_operator(
                        self.handle, cupp.CliffordGateKind.CLIFFORD_GATE_H, [0])
                    
                elif name == 'pauli_rotation_operator':
                    # Create a Pauli rotation gate
                    angle = getattr(self, 'angle', np.pi / 4)
                    h = cupp.create_pauli_rotation_gate_operator(
                        self.handle, angle, 2, [0, 1],
                        [cupp.PauliKind.PAULI_X, cupp.PauliKind.PAULI_X])
                    
                elif name == 'noise_channel_operator':
                    # Create a single-qubit depolarizing channel
                    probs = [0.9, 0.033, 0.033, 0.034]  # I, X, Y, Z
                    h = cupp.create_pauli_noise_channel_operator(
                        self.handle, 1, [0], probs)
                else:
                    assert False, f'name "{name}" not recognized'
                    
                setattr(self, name, h)
                impl(self, *args, **kwargs)
            except:
                print(f'managing resource {name} failed')
                raise
            finally:
                if name == 'handle' and hasattr(self, name):
                    cupp.destroy(self.handle)
                    del self.handle
                elif name == 'workspace' and hasattr(self, name):
                    cupp.destroy_workspace_descriptor(self.workspace)
                    del self.workspace
                elif name == 'pauli_expansion' and hasattr(self, name):
                    cupp.destroy_pauli_expansion(self.pauli_expansion)
                    del self.pauli_expansion
                    if hasattr(self, '_xz_bits_mem'):
                        del self._xz_bits_mem
                    if hasattr(self, '_coefs_mem'):
                        del self._coefs_mem
                elif name == 'pauli_expansion_view' and hasattr(self, name):
                    cupp.destroy_pauli_expansion_view(self.pauli_expansion_view)
                    del self.pauli_expansion_view
                elif name in ('clifford_operator', 'pauli_rotation_operator', 'noise_channel_operator'):
                    if hasattr(self, name):
                        cupp.destroy_operator(getattr(self, name))
                        delattr(self, name)
        return test_func
    return decorator


def _alloc_and_attach_workspace(handle, workspace):
    sz = int(cupp.workspace_get_memory_size(
        handle, workspace,
        cupp.Memspace.DEVICE,
        cupp.WorkspaceKind.WORKSPACE_SCRATCH))
    d_mem = cp.cuda.alloc(sz)
    cupp.workspace_set_memory(
        handle, workspace,
        cupp.Memspace.DEVICE,
        cupp.WorkspaceKind.WORKSPACE_SCRATCH,
        d_mem.ptr, sz)
    
    # calle rmust retain d_mem reference to defer de-alloc
    return d_mem


def _alloc_pauli_expansion_pair(num_qubits, num_terms, coef_dtype):
    npack = cupp.get_num_packed_integers(num_qubits)
    xz_b = num_terms * 2 * npack * np.dtype(np.uint64).itemsize
    coef_b = num_terms * np.dtype(coef_dtype).itemsize
    return cp.cuda.alloc(xz_b), cp.cuda.alloc(coef_b), xz_b, coef_b


@contextlib.contextmanager
def _sorted_deduplicated_view(
    handle,
    workspace,
    view_in,
    *,
    num_qubits,
    num_terms,
    data_type_int,
    coef_dtype,
    max_workspace_bytes,
    stream_ptr,
):
    """Sort ``view_in`` then deduplicate; yield a contiguous view on the deduped expansion."""

    sx, sc, xz_b, cf_b = _alloc_pauli_expansion_pair(num_qubits, num_terms, coef_dtype)
    dx, dc, _, _ = _alloc_pauli_expansion_pair(num_qubits, num_terms, coef_dtype)
    sorted_exp = sorted_view = dedup_exp = dedup_view = None
    try:
        cupp.pauli_expansion_view_prepare_sort(
            handle, view_in,
            cupp.SortOrder.LITTLE_ENDIAN_BITWISE,
            max_workspace_bytes, workspace)

        # must retain buffer1 to avoid premature workspace dealloc
        buffer1 = _alloc_and_attach_workspace(handle, workspace)

        sorted_exp = cupp.create_pauli_expansion(
            handle, num_qubits, sx.ptr, xz_b, sc.ptr, cf_b,
            data_type_int, num_terms, 0, 1)
        cupp.pauli_expansion_view_execute_sort(
            handle, view_in, sorted_exp,
            cupp.SortOrder.LITTLE_ENDIAN_BITWISE,
            workspace, stream_ptr)
        sorted_view = cupp.pauli_expansion_get_contiguous_range(
            handle, sorted_exp, 0, num_terms)

        cupp.pauli_expansion_view_prepare_deduplication(
            handle, sorted_view, int(cupp.SortOrder.NONE),
            max_workspace_bytes, workspace)

        # must retain buffer2 to avoid premature workspace dealloc
        buffer2 = _alloc_and_attach_workspace(handle, workspace)

        dedup_exp = cupp.create_pauli_expansion(
            handle, num_qubits, dx.ptr, xz_b, dc.ptr, cf_b,
            data_type_int, num_terms, 0, 0)
        cupp.pauli_expansion_view_execute_deduplication(
            handle, sorted_view, dedup_exp,
            int(cupp.SortOrder.NONE), workspace, stream_ptr)

        nt = cupp.pauli_expansion_get_num_terms(handle, dedup_exp)
        dedup_view = cupp.pauli_expansion_get_contiguous_range(
            handle, dedup_exp, 0, nt)
        cp.cuda.get_current_stream().synchronize()

        yield dedup_view
    finally:
        if dedup_view is not None:
            cupp.destroy_pauli_expansion_view(dedup_view)
        if dedup_exp is not None:
            cupp.destroy_pauli_expansion(dedup_exp)
        if sorted_view is not None:
            cupp.destroy_pauli_expansion_view(sorted_view)
        if sorted_exp is not None:
            cupp.destroy_pauli_expansion(sorted_exp)


@contextlib.contextmanager
def _fused_workspaces(handle):
    """Yield three distinct (bufferless) workspace descriptors for the fused prepare."""
    ws = [cupp.create_workspace_descriptor(handle) for _ in range(3)]
    try:
        yield ws
    finally:
        for w in ws:
            cupp.destroy_workspace_descriptor(w)


@contextlib.contextmanager
def _fused_io_expansions(handle, num_qubits, num_terms, coef_dtype, out_capacity=None):
    """Yield ``(view_in, expansion_in, expansion_out)`` for fused-application tests."""
    if out_capacity is None:
        out_capacity = num_terms
    data_type = NAME_TO_DATA_TYPE[np.dtype(coef_dtype).name]
    dxi, dci, xz_in, cf_in = _alloc_pauli_expansion_pair(num_qubits, num_terms, coef_dtype)
    dxo, dco, xz_out, cf_out = _alloc_pauli_expansion_pair(num_qubits, max(out_capacity, 1), coef_dtype)
    exp_in = exp_out = view_in = None
    try:
        exp_in = cupp.create_pauli_expansion(
            handle, num_qubits, dxi.ptr, xz_in, dci.ptr, cf_in, data_type, num_terms, 0, 1)
        exp_out = cupp.create_pauli_expansion(
            handle, num_qubits, dxo.ptr, xz_out, dco.ptr, cf_out, data_type, 0, 0, 1)
        view_in = cupp.pauli_expansion_get_contiguous_range(handle, exp_in, 0, num_terms)
        yield view_in, exp_in, exp_out
    finally:
        if view_in is not None:
            cupp.destroy_pauli_expansion_view(view_in)
        if exp_out is not None:
            cupp.destroy_pauli_expansion(exp_out)
        if exp_in is not None:
            cupp.destroy_pauli_expansion(exp_in)
        # Retain device buffers until the expansions are destroyed.
        del dxi, dci, dxo, dco


@contextlib.contextmanager
def _operators(handle, specs):
    """Create operators from compact specs, destroying them on exit.

    Each spec is either ``("clifford", gate_kind, qubits)`` or
    ``("rotation", angle, qubits, paulis)``.
    """
    ops = []
    try:
        for spec in specs:
            if spec[0] == "clifford":
                _, kind, qubits = spec
                ops.append(cupp.create_clifford_gate_operator(handle, kind, qubits))
            else:
                _, angle, qubits, paulis = spec
                ops.append(cupp.create_pauli_rotation_gate_operator(
                    handle, angle, len(paulis), qubits, paulis))
        yield ops
    finally:
        for op in ops:
            cupp.destroy_operator(op)


class TestLibHelper:
    
    def test_get_version(self):
        ver = cupp.get_version()
        assert isinstance(ver, int)
    
    def test_get_error_string(self):
        msg = cupp.get_error_string(cupp.Status.SUCCESS)
        assert isinstance(msg, str)
    
    def test_get_num_packed_integers(self):
        for num_qubits in [1, 10, 64, 100]:
            num_packed = cupp.get_num_packed_integers(num_qubits)
            assert isinstance(num_packed, int)
            assert num_packed > 0


class TestHandle:
    
    @manage_resource('handle')
    def test_handle_create_destroy(self):
        # Simple round-trip test
        pass


class TestWorkspaceDescriptor:
    
    @manage_resource('handle')
    @manage_resource('workspace')
    def test_workspace_create_destroy(self):
        # Simple round-trip test
        pass
    
    @manage_resource('handle')
    @manage_resource('workspace')
    def test_workspace_set_get_memory(self):
        # Allocate some memory
        mem_size = 1024 * 1024  # 1 MB
        d_mem = cp.cuda.alloc(mem_size)
        
        # Set memory
        cupp.workspace_set_memory(
            self.handle, self.workspace,
            cupp.Memspace.DEVICE,
            cupp.WorkspaceKind.WORKSPACE_SCRATCH,
            d_mem.ptr, mem_size)
        
        # Get memory back
        ptr, size = cupp.workspace_get_memory(
            self.handle, self.workspace,
            cupp.Memspace.DEVICE,
            cupp.WorkspaceKind.WORKSPACE_SCRATCH)
        
        assert ptr == d_mem.ptr
        assert size == mem_size

    @manage_resource('handle')
    @manage_resource('workspace')
    def test_workspace_set_memory_misaligned(self):
        mem_size = 1024 * 1024
        d_mem = cp.cuda.alloc(mem_size)

        misaligned_ptr = d_mem.ptr + 1
        with pytest.raises(cupp.cuPauliPropError):
            cupp.workspace_set_memory(
                self.handle, self.workspace,
                cupp.Memspace.DEVICE,
                cupp.WorkspaceKind.WORKSPACE_SCRATCH,
                misaligned_ptr, mem_size - 1)


@testing.parameterize(*testing.product({
    'num_qubits': (4, 8, 16),
    'num_terms': (1, 10, 100),
    'dtype': (np.complex64, np.complex128),
}))
class TestPauliExpansion:
    
    @manage_resource('handle')
    @manage_resource('pauli_expansion')
    def test_pauli_expansion_create_destroy(self):
        # Simple round-trip test
        pass
    
    @manage_resource('handle')
    @manage_resource('pauli_expansion')
    def test_pauli_expansion_get_properties(self):
        # Test all getter functions
        num_qubits = cupp.pauli_expansion_get_num_qubits(
            self.handle, self.pauli_expansion)
        assert num_qubits == self.num_qubits
        
        num_terms = cupp.pauli_expansion_get_num_terms(
            self.handle, self.pauli_expansion)
        assert num_terms == self.num_terms
        
        data_type = cupp.pauli_expansion_get_data_type(
            self.handle, self.pauli_expansion)
        assert data_type == NAME_TO_DATA_TYPE[np.dtype(self.dtype).name]
        
        sort_order = cupp.pauli_expansion_get_sort_order(
            self.handle, self.pauli_expansion)
        assert isinstance(sort_order, (int, np.integer))
        
        is_dedup = cupp.pauli_expansion_is_deduplicated(
            self.handle, self.pauli_expansion)
        assert isinstance(is_dedup, (int, np.integer))
    
    @manage_resource('handle')
    @manage_resource('pauli_expansion')
    def test_pauli_expansion_get_storage_buffer(self):
        xz_ptr, xz_size, coef_ptr, coef_size, num_terms, location = \
            cupp.pauli_expansion_get_storage_buffer(
                self.handle, self.pauli_expansion)
        
        assert xz_ptr == self._xz_bits_mem.ptr
        assert coef_ptr == self._coefs_mem.ptr
        assert num_terms == self.num_terms
        assert location == cupp.Memspace.DEVICE
    
    @manage_resource('handle')
    @manage_resource('pauli_expansion')
    @manage_resource('pauli_expansion_view')
    def test_pauli_expansion_create_view(self):
        # View is created by decorator
        num_terms = cupp.pauli_expansion_view_get_num_terms(
            self.handle, self.pauli_expansion_view)
        assert num_terms == self.num_terms


@testing.parameterize(*testing.product({
    'dtype': (np.complex64, np.complex128),
}))
class TestPauliExpansionViewOperations:
    
    num_qubits = 4
    num_terms = 10
    
    @manage_resource('handle')
    @manage_resource('pauli_expansion')
    @manage_resource('pauli_expansion_view')
    @manage_resource('workspace')
    def test_view_prepare_deduplication(self):
        max_workspace = 1 << 30  # 1 GB
        cupp.pauli_expansion_view_prepare_deduplication(
            self.handle, self.pauli_expansion_view,
            1,  # make_sorted
            max_workspace, self.workspace)
        
        # Query workspace size
        ws_size = cupp.workspace_get_memory_size(
            self.handle, self.workspace,
            cupp.Memspace.DEVICE,
            cupp.WorkspaceKind.WORKSPACE_SCRATCH)
        assert isinstance(ws_size, (int, np.integer))
    
    @manage_resource('handle')
    @manage_resource('pauli_expansion')
    @manage_resource('pauli_expansion_view')
    @manage_resource('workspace')
    def test_view_prepare_sort(self):
        max_workspace = 1 << 30
        cupp.pauli_expansion_view_prepare_sort(
            self.handle, self.pauli_expansion_view,
            cupp.SortOrder.LITTLE_ENDIAN_BITWISE,
            max_workspace, self.workspace)
        
        ws_size = cupp.workspace_get_memory_size(
            self.handle, self.workspace,
            cupp.Memspace.DEVICE,
            cupp.WorkspaceKind.WORKSPACE_SCRATCH)
        assert isinstance(ws_size, (int, np.integer))
    
    @manage_resource('handle')
    @manage_resource('pauli_expansion')
    @manage_resource('pauli_expansion_view')
    @manage_resource('workspace')
    def test_view_prepare_trace_with_zero_state(self):
        max_workspace = 1 << 30
        cupp.pauli_expansion_view_prepare_trace_with_zero_state(
            self.handle, self.pauli_expansion_view,
            max_workspace, self.workspace)
        
        ws_size = cupp.workspace_get_memory_size(
            self.handle, self.workspace,
            cupp.Memspace.DEVICE,
            cupp.WorkspaceKind.WORKSPACE_SCRATCH)
        assert isinstance(ws_size, (int, np.integer))

    @manage_resource('handle')
    @manage_resource('pauli_expansion')
    @manage_resource('pauli_expansion_view')
    @manage_resource('workspace')
    def test_view_prepare_trace_with_zero_state_backward_diff(self):
        max_workspace = 1 << 30
        xz_size, coef_size = cupp.pauli_expansion_view_prepare_trace_with_zero_state_backward_diff(
            self.handle, self.pauli_expansion_view,
            max_workspace, self.workspace)

        assert isinstance(xz_size, (int, np.integer))
        assert isinstance(coef_size, (int, np.integer))
        assert xz_size >= 0
        assert coef_size >= 0

        ws_size = cupp.workspace_get_memory_size(
            self.handle, self.workspace,
            cupp.Memspace.DEVICE,
            cupp.WorkspaceKind.WORKSPACE_SCRATCH)
        assert isinstance(ws_size, (int, np.integer))

    @manage_resource('handle')
    @manage_resource('pauli_expansion')
    @manage_resource('pauli_expansion_view')
    @manage_resource('workspace')
    def test_view_prepare_trace_with_expansion_view_backward_diff(self):
        # C prepare requires expansions that report deduplicated Pauli strings; fixture does not.
        max_workspace = 1 << 30
        dtype = self.dtype
        with _sorted_deduplicated_view(
            self.handle,
            self.workspace,
            self.pauli_expansion_view,
            num_qubits=self.num_qubits,
            num_terms=self.num_terms,
            data_type_int=NAME_TO_DATA_TYPE[np.dtype(dtype).name],
            coef_dtype=dtype,
            max_workspace_bytes=max_workspace,
            stream_ptr=cp.cuda.get_current_stream().ptr,
        ) as dv:
            xz_size_1, coef_size_1, xz_size_2, coef_size_2 = (
                cupp.pauli_expansion_view_prepare_trace_with_expansion_view_backward_diff(
                    self.handle, dv, dv, max_workspace, self.workspace))

        assert isinstance(xz_size_1, (int, np.integer))
        assert isinstance(coef_size_1, (int, np.integer))
        assert isinstance(xz_size_2, (int, np.integer))
        assert isinstance(coef_size_2, (int, np.integer))
        assert xz_size_1 >= 0
        assert coef_size_1 >= 0
        assert xz_size_2 >= 0
        assert coef_size_2 >= 0

        ws_size = cupp.workspace_get_memory_size(
            self.handle, self.workspace,
            cupp.Memspace.DEVICE,
            cupp.WorkspaceKind.WORKSPACE_SCRATCH)
        assert isinstance(ws_size, (int, np.integer))


@pytest.mark.parametrize('gate_kind', [
    cupp.CliffordGateKind.CLIFFORD_GATE_H,
    cupp.CliffordGateKind.CLIFFORD_GATE_X,
    cupp.CliffordGateKind.CLIFFORD_GATE_CX,
    cupp.CliffordGateKind.CLIFFORD_GATE_SWAP,
])
class TestCliffordOperators:
    
    @manage_resource('handle')
    def test_clifford_operator_create_destroy(self, gate_kind):
        # Determine number of qubits based on gate
        if gate_kind in (cupp.CliffordGateKind.CLIFFORD_GATE_CX,
                         cupp.CliffordGateKind.CLIFFORD_GATE_CZ,
                         cupp.CliffordGateKind.CLIFFORD_GATE_SWAP):
            qubits = [0, 1]
        else:
            qubits = [0]
        
        oper = cupp.create_clifford_gate_operator(
            self.handle, gate_kind, qubits)
        
        try:
            pass
        finally:
            cupp.destroy_operator(oper)


@pytest.mark.parametrize('angle', [0.0, np.pi/4, np.pi/2, np.pi])
@pytest.mark.parametrize('paulis', [
    [cupp.PauliKind.PAULI_X],
    [cupp.PauliKind.PAULI_Z, cupp.PauliKind.PAULI_Z],
    [cupp.PauliKind.PAULI_X, cupp.PauliKind.PAULI_Y],
])
class TestPauliRotationOperators:
    
    @manage_resource('handle')
    def test_pauli_rotation_create_destroy(self, angle, paulis):
        num_qubits = len(paulis)
        qubits = list(range(num_qubits))
        
        oper = cupp.create_pauli_rotation_gate_operator(
            self.handle, angle, num_qubits, qubits, paulis)
        
        try:
            pass
        finally:
            cupp.destroy_operator(oper)


@testing.parameterize(*testing.product({
    'dtype': (np.float64, np.complex128),
}))
class TestCotangentBuffer:
    """Test attach/get cotangent buffer round-trip and edge cases."""

    @manage_resource('handle')
    def test_get_cotangent_buffer_no_attach(self):
        angle = np.pi / 4
        oper = cupp.create_pauli_rotation_gate_operator(
            self.handle, angle, 1, [0], [cupp.PauliKind.PAULI_X])
        try:
            buf_ptr, num_elements, data_type, location = \
                cupp.quantum_operator_get_cotangent_buffer(self.handle, oper)
            assert buf_ptr == 0
            assert num_elements == 1
            assert data_type == 0
        finally:
            cupp.destroy_operator(oper)

    @manage_resource('handle')
    def test_attach_and_get_cotangent_buffer(self):
        angle = np.pi / 4
        oper = cupp.create_pauli_rotation_gate_operator(
            self.handle, angle, 1, [0], [cupp.PauliKind.PAULI_X])
        try:
            _, num_elements, _, _ = \
                cupp.quantum_operator_get_cotangent_buffer(self.handle, oper)

            grad_buf = np.zeros(num_elements, dtype=self.dtype)
            buf_size_bytes = grad_buf.nbytes
            cuda_dtype = NAME_TO_DATA_TYPE[np.dtype(self.dtype).name]

            cupp.quantum_operator_attach_cotangent_buffer(
                self.handle, oper, grad_buf.ctypes.data,
                buf_size_bytes, cuda_dtype, int(cupp.Memspace.HOST))

            buf_ptr, num_elems_out, dt_out, loc_out = \
                cupp.quantum_operator_get_cotangent_buffer(self.handle, oper)
            assert buf_ptr == grad_buf.ctypes.data
            assert num_elems_out == num_elements
            assert dt_out == cuda_dtype
            assert loc_out == int(cupp.Memspace.HOST)
        finally:
            cupp.destroy_operator(oper)

    @manage_resource('handle')
    def test_attach_cotangent_buffer_oversized(self):
        angle = np.pi / 4
        oper = cupp.create_pauli_rotation_gate_operator(
            self.handle, angle, 1, [0], [cupp.PauliKind.PAULI_X])
        try:
            _, num_elements, _, _ = \
                cupp.quantum_operator_get_cotangent_buffer(self.handle, oper)

            grad_buf = np.zeros(num_elements * 4, dtype=self.dtype)
            buf_size_bytes = grad_buf.nbytes
            cuda_dtype = NAME_TO_DATA_TYPE[np.dtype(self.dtype).name]

            cupp.quantum_operator_attach_cotangent_buffer(
                self.handle, oper, grad_buf.ctypes.data,
                buf_size_bytes, cuda_dtype, int(cupp.Memspace.HOST))

            buf_ptr, _, _, _ = \
                cupp.quantum_operator_get_cotangent_buffer(self.handle, oper)
            assert buf_ptr == grad_buf.ctypes.data
        finally:
            cupp.destroy_operator(oper)

    @manage_resource('handle')
    def test_attach_cotangent_buffer_too_small(self):
        angle = np.pi / 4
        oper = cupp.create_pauli_rotation_gate_operator(
            self.handle, angle, 1, [0], [cupp.PauliKind.PAULI_X])
        try:
            grad_buf = np.zeros(1, dtype=self.dtype)
            cuda_dtype = NAME_TO_DATA_TYPE[np.dtype(self.dtype).name]
            with pytest.raises(cupp.cuPauliPropError):
                cupp.quantum_operator_attach_cotangent_buffer(
                    self.handle, oper, grad_buf.ctypes.data,
                    0, cuda_dtype, int(cupp.Memspace.HOST))
        finally:
            cupp.destroy_operator(oper)

    @manage_resource('handle')
    def test_clifford_no_differentiable_params(self):
        oper = cupp.create_clifford_gate_operator(
            self.handle, cupp.CliffordGateKind.CLIFFORD_GATE_H, [0])
        try:
            buf_ptr, num_elements, data_type, location = \
                cupp.quantum_operator_get_cotangent_buffer(self.handle, oper)
            assert buf_ptr == 0
            assert num_elements == 0
            assert data_type == 0
        finally:
            cupp.destroy_operator(oper)


class TestNoiseChannelOperators:
    
    @manage_resource('handle')
    def test_single_qubit_depolarizing_channel(self):
        probs = [0.9, 0.033, 0.033, 0.034]
        oper = cupp.create_pauli_noise_channel_operator(
            self.handle, 1, [0], probs)
        
        try:
            pass
        finally:
            cupp.destroy_operator(oper)
    
    @manage_resource('handle')
    def test_two_qubit_noise_channel(self):
        # 16 probabilities for 2-qubit channel
        probs = [1.0/16] * 16
        oper = cupp.create_pauli_noise_channel_operator(
            self.handle, 2, [0, 1], probs)
        
        try:
            pass
        finally:
            cupp.destroy_operator(oper)


@testing.parameterize(*testing.product({
    'dtype': (np.complex64, np.complex128),
}))
class TestOperatorApplicationWorkflow:
    """Test end-to-end operator application workflow."""
    
    num_qubits = 4
    num_terms = 5
    
    @manage_resource('handle')
    @manage_resource('workspace')
    def test_operator_application_workflow(self):
        
        # Establish buffer sizes
        data_type = NAME_TO_DATA_TYPE[np.dtype(self.dtype).name]
        d_xz_in, d_coef_in, xz_bits_size, coef_size = _alloc_pauli_expansion_pair(
            self.num_qubits, self.num_terms, self.dtype)
        d_xz_out, d_coef_out, _, _ = _alloc_pauli_expansion_pair(
            self.num_qubits, self.num_terms, self.dtype)

        # Create input and output expansions
        expansion_in = cupp.create_pauli_expansion(
            self.handle, self.num_qubits,
            d_xz_in.ptr, xz_bits_size,
            d_coef_in.ptr, coef_size,
            data_type, self.num_terms, 0, 1)
        expansion_out = cupp.create_pauli_expansion(
            self.handle, self.num_qubits,
            d_xz_out.ptr, xz_bits_size,
            d_coef_out.ptr, coef_size,
            data_type, 0, 0, 1)
        
        try:
            # Create view
            view_in = cupp.pauli_expansion_get_contiguous_range(
                self.handle, expansion_in, 0, self.num_terms)
            
            try:
                # Create operator (Hadamard on qubit 0)
                oper = cupp.create_clifford_gate_operator(
                    self.handle, cupp.CliffordGateKind.CLIFFORD_GATE_H, [0])
                
                try:
                    # Prepare operator application
                    max_workspace = 1 << 30
                    xz_out_size, coef_out_size = cupp.pauli_expansion_view_prepare_operator_application(
                        self.handle, view_in, oper,
                        0,  # make_sorted
                        1,  # keep_duplicates
                        0,  # num_truncation_strategies
                        None,  # truncation_strategies (no truncation)
                        max_workspace, self.workspace)
                    
                    assert isinstance(xz_out_size, (int, np.integer))
                    assert isinstance(coef_out_size, (int, np.integer))
                    assert xz_out_size > 0
                    assert coef_out_size > 0
                    assert xz_out_size <= xz_bits_size
                    assert coef_out_size <= coef_size
                    
                    # must retain buffer to avoid premature workspace dealloc
                    buffer = _alloc_and_attach_workspace(self.handle, self.workspace)

                    cupp.pauli_expansion_view_compute_operator_application(
                        self.handle, view_in, expansion_out, oper,
                        0, 0, 1, 0, None, self.workspace, 0)
                    
                finally:
                    cupp.destroy_operator(oper)
            finally:
                cupp.destroy_pauli_expansion_view(view_in)
        finally:
            cupp.destroy_pauli_expansion(expansion_out)
            cupp.destroy_pauli_expansion(expansion_in)


class TestTruncationParams:
    """Test truncation parameter structures."""
    
    def test_coefficient_truncation_params(self):
        params = cupp.CoefficientTruncationParams()
        params.cutoff = 1e-6
        assert params.cutoff == 1e-6
        assert params.ptr != 0
    
    def test_pauli_weight_truncation_params(self):
        params = cupp.PauliWeightTruncationParams()
        params.cutoff = 10
        assert params.cutoff == 10
        assert params.ptr != 0
    
    def test_truncation_strategy(self):
        # Test creating a truncation strategy struct
        params = cupp.CoefficientTruncationParams()
        params.cutoff = 1e-4
        
        strategy = cupp.TruncationStrategy()
        strategy.strategy = cupp.TruncationStrategyKind.TRUNCATION_STRATEGY_COEFFICIENT_BASED
        strategy.param_struct = params.ptr
        
        assert strategy.strategy == cupp.TruncationStrategyKind.TRUNCATION_STRATEGY_COEFFICIENT_BASED
        assert strategy.param_struct == params.ptr


@testing.parameterize(*testing.product({
    'dtype': (np.complex64, np.complex128),
}))
class TestOperatorApplicationWithTruncation:
    """Test operator application with truncation strategies."""
    
    num_qubits = 4
    num_terms = 5
    
    @manage_resource('handle')
    @manage_resource('workspace')
    def test_operator_application_with_truncation(self):

        # Establish buffer sizes
        data_type = NAME_TO_DATA_TYPE[np.dtype(self.dtype).name]
        d_xz_in, d_coef_in, xz_bits_size, coef_size = _alloc_pauli_expansion_pair(
            self.num_qubits, self.num_terms, self.dtype)
        d_xz_out, d_coef_out, _, _ = _alloc_pauli_expansion_pair(
            self.num_qubits, self.num_terms, self.dtype)
        
        # Create input and output expansions
        expansion_in = cupp.create_pauli_expansion(
            self.handle, self.num_qubits,
            d_xz_in.ptr, xz_bits_size,
            d_coef_in.ptr, coef_size,
            data_type, self.num_terms, 0, 1)
        expansion_out = cupp.create_pauli_expansion(
            self.handle, self.num_qubits,
            d_xz_out.ptr, xz_bits_size,
            d_coef_out.ptr, coef_size,
            data_type, 0, 0, 1)
        
        try:
            # Create view
            view_in = cupp.pauli_expansion_get_contiguous_range(
                self.handle, expansion_in, 0, self.num_terms)
            
            try:
                # Create operator (Hadamard on qubit 0)
                oper = cupp.create_clifford_gate_operator(
                    self.handle, cupp.CliffordGateKind.CLIFFORD_GATE_H, [0])
                
                try:
                    # Create truncation strategies
                    coef_params = cupp.CoefficientTruncationParams()
                    coef_params.cutoff = 1e-4
                    
                    weight_params = cupp.PauliWeightTruncationParams()
                    weight_params.cutoff = 8
                    
                    # Create truncation strategy structs
                    coef_strategy = cupp.TruncationStrategy()
                    coef_strategy.strategy = cupp.TruncationStrategyKind.TRUNCATION_STRATEGY_COEFFICIENT_BASED
                    coef_strategy.param_struct = coef_params.ptr
                    
                    weight_strategy = cupp.TruncationStrategy()
                    weight_strategy.strategy = cupp.TruncationStrategyKind.TRUNCATION_STRATEGY_PAULI_WEIGHT_BASED
                    weight_strategy.param_struct = weight_params.ptr
                    
                    # Create list of strategy objects
                    strategies = [coef_strategy, weight_strategy]
                    
                    # Prepare operator application with truncation
                    max_workspace = 1 << 30
                    xz_out_size, coef_out_size = cupp.pauli_expansion_view_prepare_operator_application(
                        self.handle, view_in, oper,
                        0,  # make_sorted
                        1,  # keep_duplicates
                        2,  # num_truncation_strategies
                        strategies,  # truncation_strategies
                        max_workspace, self.workspace)
                    
                    assert isinstance(xz_out_size, (int, np.integer))
                    assert isinstance(coef_out_size, (int, np.integer))
                    assert xz_out_size > 0
                    assert coef_out_size > 0
                    assert xz_out_size <= xz_bits_size
                    assert coef_out_size <= coef_size
                    
                    # must retain buffer1 to avoid premature workspace dealloc
                    buffer = _alloc_and_attach_workspace(self.handle, self.workspace)
                    
                    cupp.pauli_expansion_view_compute_operator_application(
                        self.handle, view_in, expansion_out, oper,
                        0, 0, 1, 2, strategies, self.workspace, 0)
                    
                finally:
                    cupp.destroy_operator(oper)
            finally:
                cupp.destroy_pauli_expansion_view(view_in)
        finally:
            cupp.destroy_pauli_expansion(expansion_out)
            cupp.destroy_pauli_expansion(expansion_in)


@testing.parameterize(*testing.product({
    'dtype': (np.complex64, np.complex128),
}))
class TestFusedOperatorApplication:
    """Argument-passing and negative tests for the fused multi-operator API.

    Correctness coverage is established by the pythonic-API tests; here we only verify that
    the Python arguments reach the C layer and that documented invalid inputs raise.
    """

    num_qubits = 4
    num_terms = 5

    # Two same-kind (Clifford) operators acting within the 4-qubit register.
    _CLIFFORD_SPECS = [
        ("clifford", cupp.CliffordGateKind.CLIFFORD_GATE_H, [0]),
        ("clifford", cupp.CliffordGateKind.CLIFFORD_GATE_CX, [0, 1]),
    ]

    @manage_resource('handle')
    def test_prepare_fused_application(self):
        with _fused_io_expansions(self.handle, self.num_qubits, self.num_terms, self.dtype) as (view_in, _, _), \
             _fused_workspaces(self.handle) as (ws_min, ws_max, ws_avg), \
             _operators(self.handle, self._CLIFFORD_SPECS) as ops:
            min_cap, avg_cap, max_cap = (
                cupp.pauli_expansion_view_prepare_operator_fused_application(
                    self.handle, view_in, len(ops), ops, [0] * len(ops),
                    0, None, 1 << 30, ws_min, ws_avg, ws_max))

            # Each tier is a valid capacity; overflowed quantities would come back as -1.
            for cap in (min_cap, avg_cap, max_cap):
                assert isinstance(cap, (int, np.integer))
                assert cap >= 0

            # Each sizing tier's workspace descriptor should be independently queryable.
            for ws in (ws_min, ws_max, ws_avg):
                size = cupp.workspace_get_memory_size(self.handle, ws, cupp.Memspace.DEVICE,
                                                      cupp.WorkspaceKind.WORKSPACE_SCRATCH)
                assert isinstance(size, (int, np.integer))

    @manage_resource('handle')
    def test_compute_fused_application(self):
        with _fused_io_expansions(self.handle, self.num_qubits, self.num_terms, self.dtype) as (view_in, _, exp_out), \
             _fused_workspaces(self.handle) as (ws_min, ws_max, ws_avg), \
             _operators(self.handle, self._CLIFFORD_SPECS) as ops:
            cupp.pauli_expansion_view_prepare_operator_fused_application(
                self.handle, view_in, len(ops), ops, [0] * len(ops),
                0, None, 1 << 30, ws_min, ws_avg, ws_max)

            # Attach a buffer sized to the failsafe (max) tier, then compute. A successful
            # fused application returns ``None``; any on-the-fly failure raises.
            buffer = _alloc_and_attach_workspace(self.handle, ws_max)  # noqa: F841 (retain alloc)
            assert cupp.pauli_expansion_view_compute_operator_fused_application(
                self.handle, view_in, exp_out, len(ops), ops, [0] * len(ops),
                0, None, ws_max, 0) is None

    @manage_resource('handle')
    def test_prepare_fused_zero_operators_raises(self):
        with _fused_io_expansions(self.handle, self.num_qubits, self.num_terms, self.dtype) as (view_in, _, _), \
             _fused_workspaces(self.handle) as (ws_min, ws_max, ws_avg):
            with pytest.raises(cupp.cuPauliPropError):
                cupp.pauli_expansion_view_prepare_operator_fused_application(
                    self.handle, view_in, 0, [], [], 0, None, 1 << 30, ws_min, ws_avg, ws_max)

    @manage_resource('handle')
    def test_prepare_fused_inconsistent_kinds_raises(self):
        specs = [
            ("clifford", cupp.CliffordGateKind.CLIFFORD_GATE_H, [0]),
            ("rotation", np.pi / 4, [0], [cupp.PauliKind.PAULI_X]),
        ]
        with _fused_io_expansions(self.handle, self.num_qubits, self.num_terms, self.dtype) as (view_in, _, _), \
             _fused_workspaces(self.handle) as (ws_min, ws_max, ws_avg), \
             _operators(self.handle, specs) as ops:
            with pytest.raises(cupp.cuPauliPropError):
                cupp.pauli_expansion_view_prepare_operator_fused_application(
                    self.handle, view_in, len(ops), ops, [0] * len(ops),
                    0, None, 1 << 30, ws_min, ws_avg, ws_max)

    @manage_resource('handle')
    def test_prepare_fused_unsupported_kind_raises(self):
        # A consistent sequence of (unsupported) Pauli rotation gates: rejected as NotSupported.
        specs = [
            ("rotation", np.pi / 4, [0], [cupp.PauliKind.PAULI_X]),
            ("rotation", np.pi / 4, [1], [cupp.PauliKind.PAULI_X]),
        ]
        with _fused_io_expansions(self.handle, self.num_qubits, self.num_terms, self.dtype) as (view_in, _, _), \
             _fused_workspaces(self.handle) as (ws_min, ws_max, ws_avg), \
             _operators(self.handle, specs) as ops:
            with pytest.raises(cupp.cuPauliPropError):
                cupp.pauli_expansion_view_prepare_operator_fused_application(
                    self.handle, view_in, len(ops), ops, [0] * len(ops),
                    0, None, 1 << 30, ws_min, ws_avg, ws_max)

    @manage_resource('handle')
    def test_prepare_fused_non_distinct_workspaces_raises(self):
        with _fused_io_expansions(self.handle, self.num_qubits, self.num_terms, self.dtype) as (view_in, _, _), \
             _fused_workspaces(self.handle) as (ws_min, _ws_max, _ws_avg), \
             _operators(self.handle, self._CLIFFORD_SPECS) as ops:
            # Passing the same descriptor for all three sizing tiers is rejected.
            with pytest.raises(cupp.cuPauliPropError):
                cupp.pauli_expansion_view_prepare_operator_fused_application(
                    self.handle, view_in, len(ops), ops, [0] * len(ops),
                    0, None, 1 << 30, ws_min, ws_min, ws_min)

    @manage_resource('handle')
    def test_compute_fused_inconsistent_kinds_raises(self):
        specs = [
            ("clifford", cupp.CliffordGateKind.CLIFFORD_GATE_H, [0]),
            ("rotation", np.pi / 4, [0], [cupp.PauliKind.PAULI_X]),
        ]
        with _fused_io_expansions(self.handle, self.num_qubits, self.num_terms, self.dtype) as (view_in, _, exp_out), \
             _operators(self.handle, specs) as ops:
            d_ws = cp.cuda.alloc(1 << 20)
            ws = cupp.create_workspace_descriptor(self.handle)
            cupp.workspace_set_memory(self.handle, ws, cupp.Memspace.DEVICE,
                                      cupp.WorkspaceKind.WORKSPACE_SCRATCH, d_ws.ptr, 1 << 20)
            try:
                with pytest.raises(cupp.cuPauliPropError):
                    cupp.pauli_expansion_view_compute_operator_fused_application(
                        self.handle, view_in, exp_out, len(ops), ops, [0] * len(ops),
                        0, None, ws, 0)
            finally:
                cupp.destroy_workspace_descriptor(ws)

    @manage_resource('handle')
    def test_compute_fused_insufficient_out_capacity_raises(self):
        # Output expansion capacity (1) is smaller than the input view's term count.
        with _fused_io_expansions(self.handle, self.num_qubits, self.num_terms, self.dtype,
                                  out_capacity=1) as (view_in, _, exp_out), \
             _operators(self.handle, self._CLIFFORD_SPECS) as ops:
            d_ws = cp.cuda.alloc(1 << 20)
            ws = cupp.create_workspace_descriptor(self.handle)
            cupp.workspace_set_memory(self.handle, ws, cupp.Memspace.DEVICE,
                                      cupp.WorkspaceKind.WORKSPACE_SCRATCH, d_ws.ptr, 1 << 20)
            try:
                with pytest.raises(cupp.cuPauliPropError):
                    cupp.pauli_expansion_view_compute_operator_fused_application(
                        self.handle, view_in, exp_out, len(ops), ops, [0] * len(ops),
                        0, None, ws, 0)
            finally:
                cupp.destroy_workspace_descriptor(ws)


class TestEnums:
    """Test that all enums are accessible."""
    
    def test_status_enum(self):
        assert hasattr(cupp.Status, 'SUCCESS')
        assert hasattr(cupp.Status, 'INVALID_VALUE')
    
    def test_compute_type_enum(self):
        assert hasattr(cupp.ComputeType, 'COMPUTE_32F')
        assert hasattr(cupp.ComputeType, 'COMPUTE_64F')
    
    def test_memspace_enum(self):
        assert hasattr(cupp.Memspace, 'DEVICE')
        assert hasattr(cupp.Memspace, 'HOST')
    
    def test_pauli_kind_enum(self):
        assert hasattr(cupp.PauliKind, 'PAULI_I')
        assert hasattr(cupp.PauliKind, 'PAULI_X')
        assert hasattr(cupp.PauliKind, 'PAULI_Y')
        assert hasattr(cupp.PauliKind, 'PAULI_Z')
    
    def test_clifford_gate_enum(self):
        assert hasattr(cupp.CliffordGateKind, 'CLIFFORD_GATE_H')
        assert hasattr(cupp.CliffordGateKind, 'CLIFFORD_GATE_CX')
    
    def test_truncation_strategy_kind_enum(self):
        assert hasattr(cupp.TruncationStrategyKind, 'TRUNCATION_STRATEGY_COEFFICIENT_BASED')
        assert hasattr(cupp.TruncationStrategyKind, 'TRUNCATION_STRATEGY_PAULI_WEIGHT_BASED')
    
    def test_fused_failure_status_codes(self):
        assert hasattr(cupp.Status, 'MANDATORY_MEMORY_OVERFLOWED')
        assert hasattr(cupp.Status, 'INSUFFICIENT_WORKSPACE')
        assert hasattr(cupp.Status, 'INSUFFICIENT_OUT_EXPANSION')
        assert hasattr(cupp.Status, 'INSUFFICIENT_DEVICE_PROPERTY')
    
    def test_sort_order_enum(self):
        assert hasattr(cupp.SortOrder, 'NONE')
        assert hasattr(cupp.SortOrder, 'INTERNAL')
        assert hasattr(cupp.SortOrder, 'LITTLE_ENDIAN_BITWISE')
    
    def test_sort_order_enum_values_match_cpp(self):
        """
        Verify that SortOrder enum values in Cython bindings match C++ header.
        
        The C++ header (cupauliprop.h) defines:
            CUPAULIPROP_SORT_ORDER_NONE = 0
            CUPAULIPROP_SORT_ORDER_INTERNAL = 1
            CUPAULIPROP_SORT_ORDER_LITTLE_ENDIAN_BITWISE = 2
        
        Note: NULL is renamed to NONE to avoid Cython reserved word conflict.
        """
        assert int(cupp.SortOrder.NONE) == 0, f"SORT_ORDER_NONE should be 0, got {int(cupp.SortOrder.NONE)}"
        assert int(cupp.SortOrder.INTERNAL) == 1, f"SORT_ORDER_INTERNAL should be 1, got {int(cupp.SortOrder.INTERNAL)}"
        assert int(cupp.SortOrder.LITTLE_ENDIAN_BITWISE) == 2, f"SORT_ORDER_LITTLE_ENDIAN_BITWISE should be 2, got {int(cupp.SortOrder.LITTLE_ENDIAN_BITWISE)}"
    
    def test_pauli_kind_enum_values_match_cpp(self):
        """
        Verify that PauliKind enum values in Cython bindings match C++ header.
        
        The C++ header (cupauliprop.h) defines:
            CUPAULIPROP_PAULI_I = 0
            CUPAULIPROP_PAULI_X = 1
            CUPAULIPROP_PAULI_Y = 2
            CUPAULIPROP_PAULI_Z = 3
        
        This test catches any mismatch between C++ enum ordering and Python bindings.
        """
        assert int(cupp.PauliKind.PAULI_I) == 0, f"PAULI_I should be 0, got {int(cupp.PauliKind.PAULI_I)}"
        assert int(cupp.PauliKind.PAULI_X) == 1, f"PAULI_X should be 1, got {int(cupp.PauliKind.PAULI_X)}"
        assert int(cupp.PauliKind.PAULI_Y) == 2, f"PAULI_Y should be 2, got {int(cupp.PauliKind.PAULI_Y)}"
        assert int(cupp.PauliKind.PAULI_Z) == 3, f"PAULI_Z should be 3, got {int(cupp.PauliKind.PAULI_Z)}"
    
    def test_clifford_gate_enum_values_match_cpp(self):
        """
        Verify that CliffordGateKind enum values in Cython bindings match C++ header.
        
        The C++ header (cupauliprop.h) defines:
            CUPAULIPROP_CLIFFORD_GATE_I = 0
            CUPAULIPROP_CLIFFORD_GATE_X = 1
            CUPAULIPROP_CLIFFORD_GATE_Y = 2
            CUPAULIPROP_CLIFFORD_GATE_Z = 3
            CUPAULIPROP_CLIFFORD_GATE_H = 4
            CUPAULIPROP_CLIFFORD_GATE_S = 5
        """
        assert int(cupp.CliffordGateKind.CLIFFORD_GATE_I) == 0
        assert int(cupp.CliffordGateKind.CLIFFORD_GATE_X) == 1
        assert int(cupp.CliffordGateKind.CLIFFORD_GATE_Y) == 2
        assert int(cupp.CliffordGateKind.CLIFFORD_GATE_Z) == 3
        assert int(cupp.CliffordGateKind.CLIFFORD_GATE_H) == 4
        assert int(cupp.CliffordGateKind.CLIFFORD_GATE_S) == 5