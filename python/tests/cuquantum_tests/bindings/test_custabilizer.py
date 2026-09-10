# Copyright (c) 2021-2025, NVIDIA CORPORATION & AFFILIATES
#
# SPDX-License-Identifier: BSD-3-Clause

import numpy as np
import pytest
import cuda.bindings.runtime as cudart
from cuquantum.bindings import custabilizer as custab


def test_version():
    v = custab.get_version()
    assert v is not None
    assert v > 0
    print(f"cuStabilizer version: {v}")

def test_circuit():
  handle = custab.create();
  assert handle != 0, "Failed to create handle"

  buffer_size = 0;
  circuit_string = "H 0\nCNOT 0 1\nREPEAT 2 {\nZ_ERROR(0.1) 2\nREPEAT 3 {\nCNOT 0 2\nM 0 2\n}\n}\nM 0 1 2"

  buffer_size = custab.circuit_size_from_string(handle, circuit_string);
  assert buffer_size > 0, "Failed to get circuit size"
  #  create gpu buffer to pass to function
  err, device_ptr = cudart.cudaMalloc(buffer_size)
  assert err == 0, "Failed to create buffer"
  circuit = None
  try:
    circuit = custab.create_circuit_from_string(handle, circuit_string, device_ptr, buffer_size)
    assert circuit != 0, "Failed to create circuit on device"
  finally:
    if circuit is not None:
      custab.destroy_circuit(circuit)
    custab.destroy(handle)
    err = cudart.cudaFree(device_ptr)[0]
    assert err == 0

def test_circuit_get_attribute():
  handle = custab.create()
  # Every attribute non-trivial: qubits=4, measurements=5 (MX 0 + MY 1 + REPEAT 3 {M 2}),
  # detectors=1, repeat_blocks=1, resets=2 (RX 0 1), 1q=2 (H,S), 2q=2 (CX,CZ), noise=1.
  circuit_string = (
      "RX 0 1\nH 2\nS 3\nCX 0 2\nCZ 1 3\nDEPOLARIZE2(0.01) 0 1\n"
      "MX 0\nMY 1\nDETECTOR rec[-1]\nREPEAT 3 {\nM 2\n}"
  )
  buffer_size = custab.circuit_size_from_string(handle, circuit_string)
  err, device_ptr = cudart.cudaMalloc(buffer_size)
  assert err == 0
  circuit = None
  try:
    circuit = custab.create_circuit_from_string(handle, circuit_string, device_ptr, buffer_size)

    A = custab.CircuitAttribute
    expected = {
        A.NUM_QUBITS: 4,
        A.NUM_MEASUREMENT_BITS: 5,
        A.NUM_DETECTORS: 1,
        A.NUM_REPEAT_BLOCKS: 1,
        A.NUM_RESETS: 2,
        A.NUM_1Q_GATES: 2,
        A.NUM_2Q_GATES: 2,
        A.HAS_NOISE: 1,
        A.NUM_LEAKAGE_INSTRUCTIONS: 0,
        A.NUM_LEAKAGE_READOUTS: 0,
        A.NUM_MEASUREMENT_BITS: 5,
    }
    for attr, want in expected.items():
      out = np.empty(1, dtype=custab.get_circuit_attribute_dtype(attr))
      custab.circuit_get_attribute(handle, circuit, attr, out.ctypes.data, out.itemsize)
      assert int(out[0]) == want, f"{attr}: got {int(out[0])}, want {want}"

    # Error paths all map to CUSTABILIZER_STATUS_INVALID_VALUE -> cuStabilizerError.
    out = np.empty(1, dtype=custab.get_circuit_attribute_dtype(A.NUM_QUBITS))
    with pytest.raises(custab.cuStabilizerError):
      custab.circuit_get_attribute(0, circuit, A.NUM_QUBITS, out.ctypes.data, out.itemsize)  # null handle
    with pytest.raises(custab.cuStabilizerError):
      custab.circuit_get_attribute(handle, 0, A.NUM_QUBITS, out.ctypes.data, out.itemsize)  # null circuit
    with pytest.raises(custab.cuStabilizerError):
      custab.circuit_get_attribute(handle, circuit, A.NUM_QUBITS, out.ctypes.data, 4)  # buffer too small
    with pytest.raises(custab.cuStabilizerError):
      custab.circuit_get_attribute(handle, circuit, 999, out.ctypes.data, out.itemsize)  # bad attribute
  finally:
    if circuit is not None:
      custab.destroy_circuit(circuit)
    custab.destroy(handle)
    # The device buffer is caller-owned, so the caller frees it.
    err = cudart.cudaFree(device_ptr)[0]
    assert err == 0

def test_circuit_leakage_attributes():
  handle = custab.create()
  circuit_string = "R 0 1\nLEAKAGE_MARK1(0.5) 0\nHERALD_LEAKAGE_EVENT 0\nM 0 1"
  buffer_size = custab.circuit_size_from_string(handle, circuit_string)
  err, device_ptr = cudart.cudaMalloc(buffer_size)
  assert err == 0
  circuit = None
  try:
    circuit = custab.create_circuit_from_string(handle, circuit_string, device_ptr, buffer_size)

    A = custab.CircuitAttribute
    for attr, want in (
        (A.NUM_LEAKAGE_INSTRUCTIONS, 2),
        (A.NUM_LEAKAGE_READOUTS, 1),
        (A.NUM_MEASUREMENT_GATES, 2),
        (A.NUM_MEASUREMENT_BITS, 3),
    ):
      out = np.empty(1, dtype=custab.get_circuit_attribute_dtype(attr))
      custab.circuit_get_attribute(handle, circuit, attr, out.ctypes.data, out.itemsize)
      assert int(out[0]) == want, f"{attr}: got {int(out[0])}, want {want}"
  finally:
    if circuit is not None:
      custab.destroy_circuit(circuit)
    custab.destroy(handle)
    err = cudart.cudaFree(device_ptr)[0]
    assert err == 0

def test_leakage_frame_simulator():
  handle = custab.create()
  num_qubits, num_shots, num_measurements = 2, 32, 2
  stride = ((num_shots + 31) // 32) * 4
  circuit_string = "R 0 1\nH 0\nCX 0 1\nM 0 1"

  bit_bytes = num_qubits * stride
  m_bytes = num_measurements * stride
  tables = []
  for nbytes in (bit_bytes, bit_bytes, bit_bytes, m_bytes):  # x, z, l, m
    err, ptr = cudart.cudaMalloc(nbytes)
    assert err == 0
    tables.append(ptr)

  buffer_size = custab.circuit_size_from_string(handle, circuit_string)
  err, device_ptr = cudart.cudaMalloc(buffer_size)
  assert err == 0

  sim = None
  circuit = None
  try:
    sim = custab.create_leakage_frame_simulator(
        handle, num_qubits, num_shots, num_measurements, stride)
    circuit = custab.create_circuit_from_string(handle, circuit_string, device_ptr, buffer_size)
    custab.leakage_frame_simulator_apply_circuit(
        handle, sim, circuit, 0, 42, *tables, 0)
  finally:
    if circuit is not None:
      custab.destroy_circuit(circuit)
    if sim is not None:
      custab.destroy_leakage_frame_simulator(sim)
    custab.destroy(handle)
    for ptr in (*tables, device_ptr):
      err = cudart.cudaFree(ptr)[0]
      assert err == 0

if __name__ == "__main__":
  test_version()
  test_circuit()
  test_circuit_get_attribute()
  print("All bindings tests passed!")
