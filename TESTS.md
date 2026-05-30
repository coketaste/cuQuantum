# cuQuantum Python Test Catalog

Generated from:
- `python/tests/` — main test suite (cuQuantum Python bindings & high-level APIs)
- `python/extensions/tests/` — pythonic cuDensityMat extensions
- `benchmarks/tests/` — `nv-quantum-benchmarks` CLI

## 0. Pytest markers (from `python/tests/pytest.ini`)

| Marker | Meaning |
|---|---|
| `custatevec` | tests for cuStateVec |
| `cutensornet` | tests for cuTensorNet |
| `cudensitymat` | tests for cuDensityMat |
| `cupauliprop` | tests for cuPauliProp |
| `custabilizer` | tests for cuStabilizer |
| `utility` | internal utilities |
| `mpi` | requires MPI (skipped unless `-m mpi`) |

Run a subset, e.g.: `pytest -m "custatevec and not mpi"`.

---

## 1. Low-level Cython bindings — `python/tests/cuquantum_tests/bindings/`

Direct exercises of the Cython wrappers around the C libraries. Heaviest test set; these are the ones that surfaced the earlier `dlopen libcustatevec` and missing-`setuptools` issues.

### 1.1 cuStateVec — `test_custatevec.py`

| Class | Description |
|---|---|
| `TestSV` / `TestBatchedSV` / `TestMultiGpuSV` | Base fixtures for single, batched, and multi-GPU state vectors. |
| `TestLibHelper` | Library version & utility helpers. |
| `TestMathMode` | `cusvMathMode` setter/getter. |
| `TestHandle` | Handle lifecycle, stream binding. |
| `TestInitSV` | `initializeStateVector`. |
| `TestAbs2Sum` / `TestBatchedAbs2Sum` | Marginal probabilities (`|ψ|²` reductions). |
| `TestCollapse` / `TestBatchedCollapse` | Post-measurement collapse. |
| `TestMeasure` / `TestMeasureBatched` | Single & batched measurement on Z basis. |
| `TestApply` / `TestBatchedApply` | `applyMatrix` (gate application). |
| `TestExpect` / `TestBatchedExpect` | Expectation-value computations. |
| `TestSampler` | Bit-string sampling. |
| `TestAccessor` | State-vector slice get/set. |
| `TestTestMatrixType` | Detect unitary / Hermitian matrices. |
| `TestBatchMeasureWithSubSV` | Distributed measurement across sub-statevectors. |
| `TestSwap` / `TestMultiGPUSwap` | Index-bit swap (single & multi-GPU). |
| `TestCommunicator` | Built-in / external communicator hooks. |
| `TestParameters` | SubSV scheduling parameter object. |
| `TestWorker` / `TestScheduler` | Distributed worker + scheduler. |
| `TestSubSVMigrator` | Page-migration between host / device for very large states. |
| `TestMemHandler` | External device-memory handler plug-in. |
| `TestLogger` | C-level logger callbacks. |

### 1.2 cuTensorNet — `test_cutensornet.py`

| Class | Description |
|---|---|
| `TestLibHelper`, `TestHandle` | Library/handle basics. |
| `TestTensorNetworkBase` | Common fixture. |
| `TestTensorNetworkDescriptor` | Describe a tensor network. |
| `TestOptimizerInfo`, `TestOptimizerConfig`, `TestAutotunePreference` | Path-finder configs. |
| `TestContraction` | End-to-end contraction (forward + gradient workflow). |
| `TestStateBase`, `TestStateAPIs` | Network-state object APIs. |
| `TestMPSOvercompleteExtentsSUGauge`, `TestMPSNonContiguousStrides` | MPS edge cases. |
| `TestSliceGroup` | Slice-group bookkeeping. |
| `TestMemHandler` | Pluggable device memory. |
| `TestTensorQR`, `TestTensorSVD`, `TestTensorSVDConfig` | Decompositions. |
| `TestTensorGate` | Gate split/contract. |
| `TestDistributed` | Multi-process contraction (NCCL/MPI). |
| `TestTensorProductModeConvention` | Mode-label conventions. |
| `TestLogger`, `TestMisc` | Logging + miscellaneous. |

### 1.3 cuPauliProp — `test_cupauliprop.py`

| Class | Description |
|---|---|
| `TestLibHelper`, `TestHandle` | Library/handle basics. |
| `TestWorkspaceDescriptor` | Workspace allocation. |
| `TestPauliExpansion`, `TestPauliExpansionViewOperations` | Build/inspect Pauli expansions. |
| `TestCliffordOperators`, `TestPauliRotationOperators`, `TestNoiseChannelOperators` | Operator constructors. |
| `TestCotangentBuffer` | Backward-mode buffer. |
| `TestOperatorApplicationWorkflow` | Apply operator to expansion. |
| `TestTruncationParams`, `TestOperatorApplicationWithTruncation` | Truncated Heisenberg evolution. |
| `TestEnums` | Enum coverage check. |

### 1.4 cuStabilizer — `test_custabilizer.py`

| Function | Description |
|---|---|
| `test_version` | Library version sanity. |
| `test_circuit` | Smoke-build a stabilizer circuit. |

### 1.5 Internal — `test_internal.py`

| Function | Description |
|---|---|
| `test_compute_type` | Compute-type enum coverage per library. |
| `test_compute_type_alignment_with_nvmath` | Verify compute-type enum alignment with `nvmath-python`. |
| `test_data_type_alignment_with_nvmath` | Verify data-type enum alignment with `nvmath-python`. |

---

## 2. cuTensorNet pythonic API — `python/tests/cuquantum_tests/tensornet/`

High-level `cuquantum.tensornet` API.

### 2.1 Core contraction
| File / Class | Description |
|---|---|
| `test_contract.py::TestContractFunctionality` | `contract()` API surface (returns, dtypes, etc.). |
| `test_contract.py::TestContract` | Numerical correctness of `contract()`. |
| `test_contract.py::TestEinsum` | `cuquantum.einsum` correctness vs NumPy. |
| `test_contract_path.py::TestContractPath` | Path finder + `contract_path()`. |
| `test_network.py::TestNetworkFunctionality` | `Network` class lifecycle, options. |
| `test_network.py::TestNetworkCorrectness` | Numerical correctness via `Network`. |
| `test_network.py::test_none_memory_pool` | Behaviour without an external memory pool. |

### 2.2 Tensor decompositions
| File / Class | Description |
|---|---|
| `test_tensor.py::TestDecomposeFunctionality` | API of `tensor.decompose`. |
| `test_tensor.py::TestDecomposeCorrectness` | Numerical correctness of QR/SVD. |
| `test_tensor.py::test_memory_limit` | Memory-limit enforcement. |
| `test_tensor.py::TestDecompositionOptions` | Options object validation. |
| `test_tensor.py::TestSVDMethod`, `TestSVDInfo` | SVD algorithm flags + info reporting. |

### 2.3 Options / tuning
| File / Class | Description |
|---|---|
| `test_options.py::TestNetworkOptions` | Top-level options validation. |
| `test_options.py::TestOptimizerOptions`, `TestOptimizerInfo` | Path-optimiser options. |
| `test_options.py::TestPathFinderOptions`, `TestReconfigOptions`, `TestSlicerOptions` | Sub-options. |

### 2.4 Internal
| File | Description |
|---|---|
| `test_internal.py::TestGetSymbol` | Internal mode-symbol generator. |

### 2.5 Circuit converters
| File / Class | Description |
|---|---|
| `test_circuit_converter.py::TestParserOptions` | Parser options. |
| `test_circuit_converter.py::TestCircuitToEinsumFunctionality` | Generic circuit→einsum API. |
| `test_circuit_converter.py::TestQiskitCorrectness` | Qiskit circuit correctness. |
| `test_circuit_converter.py::TestCirqCorrectness` | Cirq circuit correctness. |

### 2.6 Experimental — `tensornet/experimental/`
| File / Class | Description |
|---|---|
| `test_network_state.py::TestNetworkStateBasicFunctionality` | NetworkState API. |
| `test_network_state.py::TestExactCircuitSimulation`, `TestExactGenericState` | Exact-state simulation. |
| `test_network_state.py::TestApproxCircuitSimulation`, `TestApproxGenericState` | Approximate (MPS/TT) simulation. |
| `test_network_state.py::TestExpectationGradient` | Reverse-mode expectation gradients. |
| `test_network_state.py::TestAdjointGateCancellation` | Optimisation: redundant-gate cancellation. |
| `test_network_state.py::TestNetworkOperator`, `test_network_operator.py::TestNetworkOperator` | NetworkOperator API. |
| `test_noisy_state.py::TestNoisyStateFunctionality`, `TestNoisyStateCorrectness` | Noisy-state simulation. |
| `test_contract_decompose.py::TestContractDecomposeFunctionality`, `TestContractDecompose` | Fused contract+decompose. |
| `test_contract_decompose.py::test_memory_limit`, `TestContractDecomposeAlgorithm`, `TestContractDecomposeInfo` | Knobs + reporting. |
| `test_internal.py::TestCircuitState`, `TestGenericState` | Internal state objects. |

### 2.7 Trajectory-noise sub-suite — `experimental/trajectories_noise/`
| File / Test | Description |
|---|---|
| `test_onequbit_channel.py::test_bitflip_channel` | Bit-flip channel correctness. |
| `test_onequbit_channel.py::test_depolarizing_channel` | Depolarising channel. |
| `test_onequbit_channel.py::test_damping_channel` | Amplitude-damping channel. |
| `test_mid_circuit_measurement.py::test_3qubit_parity` | Mid-circuit parity correction. |
| `test_quantum_volume_mid_circuit.py::test_quantum_volume` | Quantum-volume circuits with mid-circuit measurement under noise. |
| `test_large_circuits.py::test_bitflip_maxcut_cost` | Large-circuit MaxCut cost under bit-flip noise. |

---

## 3. cuDensityMat — `python/tests/cuquantum_tests/densitymat/`

Density-matrix / Liouvillian engine.

### 3.1 Single-process
| File / Class | Description |
|---|---|
| `test_work_stream.py::TestWorkStream` | WorkStream lifecycle (multi-GPU branches skip when `<2` GPUs). |
| `test_state.py::TestState` | State construction, scaling, accumulation, copy/inner-product. |
| `test_operators.py::TestOperators` | Build dense/multidiagonal operators, action on states. |
| `test_elementary_operator.py::TestDenseOperatorUnaryOperations` | Dense op: scalar mul, conj, dual, herm. |
| `test_elementary_operator.py::TestDenseOperatorBinaryOperations` | Dense op: add/sub/matmul. |
| `test_elementary_operator.py::TestMultidiagonalOperatorUnaryOperations` | Diagonal op unary algebra. |
| `test_elementary_operator.py::TestMultidiagonalOperatorBinaryOperations` | Diagonal op binary algebra. |
| `test_elementary_operator.py::TestMixedOperations` | Cross dense × diagonal arithmetic. |

### 3.2 Distributed (`-m mpi`, requires `mpi4py` + system MPI; NCCL paths require NCCL)
| File / Function | Description |
|---|---|
| `test_state_mpi.py::test_creation` | Distributed state creation across MPI/NCCL providers. |
| `test_state_compute_mpi.py::TestStateAPI` | Distributed compute APIs (norm, scale, accumulate, expectation). |
| `test_work_stream_mpi.py::test_work_stream_communicator_from_mpi_comm` | Build WorkStream from `MPI.Comm`. |
| `test_work_stream_mpi.py::test_work_stream_mpi_communicator_from_pointer` | From raw MPI pointer (tuple/list). |
| `test_work_stream_mpi.py::test_work_stream_mpi_communicator_from_int_pointer` | From integer MPI pointer. |
| `test_work_stream_mpi.py::test_work_stream_nccl_communicator_from_pointer` | From raw NCCL handle (tuple/list). |
| `test_work_stream_mpi.py::test_work_stream_nccl_communicator_from_int_pointer` | From integer NCCL handle. |

### 3.3 Pythonic extensions — `python/extensions/tests/`
| File / Class | Description |
|---|---|
| `test_context.py::TestCudensitymatContext`, `TestOperatorContext`, `TestStateContext` | High-level context managers (lib / operator / state scopes). |
| `test_elementary_operator.py::TestElementaryOperator` | Elementary-operator wrapper. |
| `test_matrix_operator.py::TestMatrixOperator` | Matrix-operator helper. |
| `test_operator_term.py::TestOperatorTerm` | Single Liouvillian term. |
| `test_operator.py::TestOperator` | Composite operator builder. |

---

## 4. cuStabilizer — `python/tests/cuquantum_tests/stabilizer/`

Stabilizer (Clifford) simulator + decoders.

### 4.1 Pythonic API — `test_pythonic_api.py`
| Test | Description |
|---|---|
| `test_circuit_smoke`, `test_frame_simulator_smoke` | Basic API smoke tests. |
| `test_simulation_basic` | End-to-end simulation. |
| `test_statistical_wrt_stim` | Statistical agreement with `stim`. |
| `test_h_layer_measure_all` | H-layer measurement. |
| `test_multiple_circuits_same_simulator`, `test_multiple_runs_same_simulator` | Reuse semantics. |

### 4.2 Simulator inputs — `test_simulator_inputs.py`
| Test | Description |
|---|---|
| `test_odd_shots_packed`, `test_odd_shots_unpacked` | Odd-shot count packed/unpacked I/O. |
| `test_package_semantics_no_inputs_kwarg` | Default-input behaviour. |
| `test_simulator_input_tables_packed`, `test_simulator_input_tables_unpacked` | Input table layouts. |
| `test_simulator_constructor_measurement_table_packed` | Constructor accepting measurement table. |
| `test_simulator_set_input_tables_packed` | Setter for input tables. |
| `test_simulator_get_pauli_xz_bits_uses_exact_memory_size` | Memory-size invariant. |

### 4.3 Frame simulator — `test_frame_simulator.py`
| Test | Description |
|---|---|
| `test_frame_simulator` | Frame-simulator smoke. |

### 4.4 Detector-error-model sampler — `test_dem_sampling.py`
| Test | Description |
|---|---|
| `test_dem_sampler_shapes_packed_and_unpacked` | DEM sampler shape tests. |
| `test_matrix_sampler_numpy_io` | NumPy I/O round-trip. |
| `test_extract_from_dem_known` | Known-DEM extraction. |
| `test_correctness_small`, `test_correctness_wrt_stim_dem_surface_code` | Correctness vs hand-computed and `stim` DEM. |
| `test_csr_get_errors_returns_bit_matrix_csr`, `test_csr_get_errors_numpy_package`, `test_csr_to_scipy_sparse`, `test_csr_from_sparse`, `test_sampler_accepts_sparse` | CSR sparse interop. |
| `test_runtime_error_before_sample` | Error before sampling. |
| `test_seed_reproducibility`, `test_consecutive_calls_differ` | RNG semantics. |
| `test_max_shots_exceeded` | Shot-count limit. |
| `test_bitmatrix_sampler_alias` | Alias coverage. |
| `test_parity_invariant` | Parity invariant. |
| `test_deterministic_all_ones` | Deterministic-input case. |

### 4.5 Error strings — `test_error_string.py`
| Test | Description |
|---|---|
| `test_get_error_string_success` | `cusbStatusSuccess` → string. |
| `test_get_error_string_invalid_value` | Invalid-value status. |
| `test_get_error_string_alloc_failed` | Allocation-failure status. |
| `test_get_error_string_internal_error` | Internal-error status. |
| `test_get_error_string_insufficient_workspace` | Workspace-too-small status. |
| `test_get_error_string_not_supported` | Not-supported status. |
| `test_get_error_string_cuda_error` | CUDA-error status. |
| `test_error_in_exception` | Exception-path round-trip. |

---

## 5. Top-level utilities — `python/tests/cuquantum_tests/test_cuquantum.py`

| Class | Description |
|---|---|
| `TestModuleUtils` | `cuquantum` package-level utilities (lazy submodule loading, version helpers, etc.). |

---

## 6. Sample-program smoke tests — `python/tests/samples_tests/`

Each suite runs every script under the corresponding `python/samples/...` tree as a subprocess and asserts it executes successfully.

| File | Class |
|---|---|
| `custatevec_tests/test_custatevec_samples.py` | `TestcuStateVecSamples` |
| `cutensornet_tests/test_cutensornet_samples.py` | `TestcuTensorNetSamples`, `TestNotebooks` |
| `cudensitymat_tests/test_cudensitymat_samples.py` | `TestcuDensityMatSamples` |
| `custabilizer_tests/test_custabilizer_samples.py` | `TestcuStabilizerSamples` |
| `cupauliprop_tests/test_cupauliprop_samples.py` | `TestcuPauliPropSamples` |
| `test_internal.py::test_non_empty_testing_samples` | Sample discovery is non-empty. |
| `test_internal.py::test_samples_included` | Every sample is registered. |

---

## 7. Benchmark CLI — `benchmarks/tests/nv_quantum_benchmarks_tests/test_run.py`

| Class | Description |
|---|---|
| `TestCmdCircuit` | `cuquantum-benchmarks circuit …` (parameter sweeps over QFT, QAOA, etc., across backends). |
| `TestCmdApi` | `cuquantum-benchmarks api …` (cuStateVec / cuTensorNet primitive benchmarks). |

---

## 8. How to run subsets

```bash
# Everything fast and offline (skip MPI tests)
pytest python/tests -m "not mpi"

# Only the cuStateVec layer
pytest python/tests -m custatevec

# Only the high-level Python tensor-network API (skip Cython binding tests)
pytest python/tests/cuquantum_tests/tensornet \
       python/tests/cuquantum_tests/tensornet/experimental

# Distributed cuDensityMat tests (needs MPI + NCCL)
mpirun -n 2 pytest python/tests/cuquantum_tests/densitymat -m mpi

# Sample programs only
pytest python/tests/samples_tests

# Benchmark CLI
pytest benchmarks/tests
```

Common knobs:

```bash
# Skip the heavyweight CFFI parametrisations (also avoids the setuptools dep)
pytest python/tests -k "not cffi"

# Quick collect-only listing of every parametrized test ID
pytest python/tests --collect-only -q > tests_collected.txt
```

---

## 9. Known environmental requirements

| Requirement | Needed by |
|---|---|
| NVIDIA driver + CUDA-13 capable GPU | every test except a handful of pure-Python helpers |
| `libcustatevec.so.1` etc. on `LD_LIBRARY_PATH` (or installed via `cuquantum-cu13` wheels) | all binding tests |
| `setuptools` in the active venv | any `-cffi` / `-cffi_struct` parametrisation on Python ≥ 3.12 |
| `mpi4py` + system MPI (e.g. OpenMPI) | tests under `densitymat/*_mpi.py` and `bindings/test_cutensornet.py::TestDistributed` |
| NCCL | NCCL-provider variants in `densitymat/test_*_mpi.py` |
| `qiskit`, `cirq` | `tensornet/test_circuit_converter.py` |
| `stim` | `stabilizer/test_pythonic_api.py::test_statistical_wrt_stim`, `stabilizer/test_dem_sampling.py::test_correctness_wrt_stim_dem_surface_code` |
| `jax` | a subset of tensornet experimental tests (auto-skip otherwise) |
| `nbformat`, `jupyter` | `samples_tests/cutensornet_tests/test_cutensornet_samples.py::TestNotebooks` |

If a dependency is missing the corresponding tests are typically *skipped* rather than failed; a hard failure usually points to a missing **runtime shared library** or **`setuptools`** rather than missing optional deps.
