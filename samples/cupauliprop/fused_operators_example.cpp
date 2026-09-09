/*
 * Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
 *
 * SPDX-License-Identifier: BSD-3-Clause
 */

/**
 * @file fused_operators_example.cpp
 * @brief Demonstrates cuPauliProp fused-operator execution by comparing two ways
 *        to apply long operator sequences to a Pauli expansion:
 *          (1) applying each operator individually, in-turn, and
 *          (2) applying the same sequence through the fused API.
 *        This example first grows a random input expansion, then
 *        benchmarks both approaches on random Pauli noise channels and random
 *        Clifford gates, and reports the measured runtime speedup.
 */

#include "example_utils.hpp"

#include <algorithm>
#include <array>
#include <chrono>
#include <cstdint>
#include <cstdlib>
#include <iomanip>
#include <iostream>
#include <random>
#include <vector>



// Sphinx: #1
// ========================================================================
// HYPER-PARAMETERS (can be changed)
// ========================================================================


constexpr int32_t NUM_QUBITS = 700;

constexpr int64_t TARGET_NUM_TERMS = 10'000;

constexpr int32_t NUM_CLIFFORD_GATES = 300;

constexpr int32_t NUM_PAULI_NOISE_CHANNELS = 300;



// Sphinx: #2
// ========================================================================
// CONSTANTS (are strictly fixed)
// ========================================================================


// Our Pauli expansions maintain real, double-precision coefficients
constexpr auto COEF_TYPE = CUDA_R_64F;

// We will only apply operators in the Schrodinger picture
constexpr int32_t ADJOINT = 0;

// We disable any sorting or de-duplication of pauli expansion terms, since they
// do not affect the interaction with, nor performance of, the fused-operator API
constexpr auto SORT_ORDER = CUPAULIPROP_SORT_ORDER_NONE;
constexpr int DUPLICATES = 1;

// We disable truncation for simplicity, since although accepted by the
// fused-operator API, it has only a trivial/additive performance overhead
constexpr int32_t NUM_TRUNC_STRATS = 0;
constexpr cupaulipropTruncationStrategy_t* TRUNC_STRATS = nullptr;



// Sphinx: #3
// ========================================================================
// SMALL HELPER FUNCTIONS
// ========================================================================


auto getPauliStrAndCoefSizes() {

  // Size of Pauli string depends upon the number of qubits
  int32_t numPackedIntsPerXZMask;
  HANDLE_CUPP_ERROR( cupaulipropGetNumPackedIntegers(NUM_QUBITS, &numPackedIntsPerXZMask) );
  size_t numBytesPerPauliStr = 2 * numPackedIntsPerXZMask * sizeof(cupaulipropPackedIntegerType_t);

  // Size of coefficient depends upon type
  size_t numBytesPerCoef = sizeof(double); // COEF_TYPE = CUDA_R_64F

  return std::array{numBytesPerPauliStr, numBytesPerCoef};
}


auto getNewExpansionBufferSizes() {

  auto [numBytesPerPauliStr, numBytesPerCoef] = getPauliStrAndCoefSizes();
  size_t numBytesPerTerm = numBytesPerPauliStr + numBytesPerCoef;

  // Allocate enough for TWICE our desired expansion size, to handle if we over-populate
  auto numTerms = 2 * TARGET_NUM_TERMS;
  size_t pauliBufferSize = numBytesPerPauliStr * numTerms;
  size_t coefBufferSize  = numBytesPerCoef     * numTerms;
  return std::array{ pauliBufferSize, coefBufferSize };
}


auto getExistingExpansionBufferSizes(cupaulipropHandle_t handle, cupaulipropPauliExpansion_t expansion) {

  int64_t xzBitsBufferSize;
  int64_t coefBufferSize;

  void* dummyBuffer;
  int64_t dummyNumTerms;
  cupaulipropMemspace_t dummyLocatin;

  HANDLE_CUPP_ERROR( cupaulipropPauliExpansionGetStorageBuffer(
    handle, expansion, 
    &dummyBuffer, &xzBitsBufferSize, 
    &dummyBuffer, &coefBufferSize, 
    &dummyNumTerms, &dummyLocatin) );

  return std::array{xzBitsBufferSize, coefBufferSize};
}


int64_t getExistingExpansionTermCapacity(cupaulipropHandle_t handle, cupaulipropPauliExpansion_t expansion) {

  auto [xzBitsBufferSize, coefBufferSize] = getExistingExpansionBufferSizes(handle, expansion);
  auto [numBytesPerPauliStr, numBytesPerCoef] = getPauliStrAndCoefSizes();

  return std::min(
    xzBitsBufferSize / static_cast<int64_t>(numBytesPerPauliStr),
    coefBufferSize   / static_cast<int64_t>(numBytesPerCoef));
}


int64_t getPopulatedExpansionMemorySize(int64_t numTerms) {

  int32_t numPackedIntsPerXZMask;
  HANDLE_CUPP_ERROR( cupaulipropGetNumPackedIntegers(NUM_QUBITS, &numPackedIntsPerXZMask) );

  int64_t numBytesPerPauliStr =
    2LL * numPackedIntsPerXZMask * static_cast<int64_t>(sizeof(cupaulipropPackedIntegerType_t));

  int64_t numBytesPerCoef = sizeof(double); // COEF_TYPE = CUDA_R_64F
  return numTerms * (numBytesPerPauliStr + numBytesPerCoef);
}


auto createViewOfAllTerms(cupaulipropHandle_t handle, cupaulipropPauliExpansion_t expansion) {

  int64_t numTerms;
  HANDLE_CUPP_ERROR( cupaulipropPauliExpansionGetNumTerms(handle, expansion, &numTerms) );

  cupaulipropPauliExpansionView_t inView;
  HANDLE_CUPP_ERROR( cupaulipropPauliExpansionGetContiguousRange(
    handle, expansion, 0, numTerms, &inView) );

  return inView;
}



// Sphinx: #4
// ========================================================================
// INDIVIDUAL AND FUSED OPERATOR APPLICATION
// ========================================================================


// Sphinx: #5
void applySingleOperator(
  cupaulipropHandle_t handle,
  cudaStream_t stream,
  cupaulipropPauliExpansion_t inExpansion,
  cupaulipropPauliExpansion_t outExpansion,
  cupaulipropQuantumOperator_t quantumOp,
  cupaulipropWorkspaceDescriptor_t workspace,
  void* d_workspaceBuffer,
  size_t workspaceBufferSize)
{
  // Create a view of all terms in the expansion, in order for all to be processed
  auto inView = createViewOfAllTerms(handle, inExpansion);

  // Call Prepare() to obtain the REQUIRED expansion and workspace sizes; if
  // these are not satisfied, the corresponding Compute() call will error
  int64_t requiredPauliSize;
  int64_t requiredCoefSize;
  HANDLE_CUPP_ERROR( cupaulipropPauliExpansionViewPrepareOperatorApplication(
    handle, inView, quantumOp,
    SORT_ORDER, DUPLICATES, NUM_TRUNC_STRATS, TRUNC_STRATS,
    workspaceBufferSize, &requiredPauliSize, &requiredCoefSize, workspace) );

  // The required workspace size is attached to the descriptor
  int64_t requiredWorkspaceSize;
  HANDLE_CUPP_ERROR( cupaulipropWorkspaceGetMemorySize(
    handle, workspace,
    CUPAULIPROP_MEMSPACE_DEVICE, CUPAULIPROP_WORKSPACE_SCRATCH,
    &requiredWorkspaceSize) );

  // Ensure sufficient memory, guaranteeing the Compute() call will succeed
  auto [pauliBufferSize, coefBufferSize] = getExistingExpansionBufferSizes(handle, outExpansion);
  if (requiredPauliSize     > static_cast<int64_t>(pauliBufferSize) ||
      requiredCoefSize      > static_cast<int64_t>(coefBufferSize)  ||
      requiredWorkspaceSize > static_cast<int64_t>(workspaceBufferSize))
  {
    std::cout
      << "Insufficient outExpansion capacity and/or workspace buffer size "
      << "to perform single-operator application. Exiting..."
      << std::endl;
    std::abort();
  }

  // Re-attach the workspace memory buffer, which is detached by the Prepare() call
  HANDLE_CUPP_ERROR( cupaulipropWorkspaceSetMemory(
    handle, workspace,
    CUPAULIPROP_MEMSPACE_DEVICE, CUPAULIPROP_WORKSPACE_SCRATCH,
    d_workspaceBuffer, workspaceBufferSize) );

  // Apply the individual operator, which always exceeds (except due to a bug!)
  HANDLE_CUPP_ERROR( cupaulipropPauliExpansionViewComputeOperatorApplication(
    handle, inView, outExpansion, quantumOp,
    ADJOINT, SORT_ORDER, DUPLICATES,
    NUM_TRUNC_STRATS, TRUNC_STRATS, workspace, stream) );

  // Destroy the view, which has been invalidated by the above Compute() call
  HANDLE_CUPP_ERROR( cupaulipropDestroyPauliExpansionView(inView) );
}


// Sphinx: #6
void applyFusedOperators(
  cupaulipropHandle_t handle,
  cudaStream_t stream,
  cupaulipropPauliExpansion_t inExpansion,
  cupaulipropPauliExpansion_t outExpansion,
  const std::vector<cupaulipropQuantumOperator_t>& quantumOps,
  void* d_workspaceBuffer,
  size_t workspaceBufferSize,
  cupaulipropWorkspaceDescriptor_t minWorkspace,
  cupaulipropWorkspaceDescriptor_t maxWorkspace,
  cupaulipropWorkspaceDescriptor_t avgWorkspace)
{
  // Create a view of all terms in the expansion, in order for all to be processed
  auto inView = createViewOfAllTerms(handle, inExpansion);
  const std::vector<int32_t> adjoints(quantumOps.size(), ADJOINT);

  // Sphinx: #7

  // Call Prepare() to obtain the expansion capacities and workspaces necessary to
  // handle the best-case, worst-case and average-case Compute() scenarios. These
  // correspond to different magnitudes of expansion growth under the operators.
  int64_t minOutCapacity;
  int64_t maxOutCapacity; // -1 if overflowed
  int64_t avgOutCapacity; // -1 if overflowed
  cupaulipropStatus_t prepareStatus = cupaulipropPauliExpansionViewPrepareOperatorFusedApplication(
    handle, inView, quantumOps.size(), quantumOps.data(), adjoints.data(),
    NUM_TRUNC_STRATS, TRUNC_STRATS,
    static_cast<int64_t>(workspaceBufferSize),
    &minOutCapacity, minWorkspace,
    &avgOutCapacity, avgWorkspace,
    &maxOutCapacity, maxWorkspace);

  // Sphinx: #8

  // Process the additional, "recoverable" failure modes of the fused-operator API,
  // which are not exhibited by the single-operator API.
  switch (prepareStatus)
  {
    case CUPAULIPROP_STATUS_SUCCESS:
      break;
    case CUPAULIPROP_STATUS_INSUFFICIENT_DEVICE_PROPERTY:
      std::cout
        << "Too many fused operators (" << quantumOps.size() << ") were given, exceeding "
        << "a hardware limitation of the user's device. One could dynamically shrink "
        << "the number passed, but in this simple example, we treat this as an error."
        << std::endl;
      std::abort();
    default:
      HANDLE_CUPP_ERROR(prepareStatus);

    // Below are not thrown by the Prepare() call
    // case CUPAULIPROP_STATUS_INSUFFICIENT_WORKSPACE:
    // case CUPAULIPROP_STATUS_INSUFFICIENT_OUT_EXPANSION:
  }

  // Sphinx: #9

  // Learn how many terms our pre-allocated outExpansion can store
  int64_t outTermCapacity = getExistingExpansionTermCapacity(handle, outExpansion);

  // Learn the workspace buffer sizes advised by Prepare() above
  int64_t minWorkspaceSize;
  int64_t avgWorkspaceSize;
  int64_t maxWorkspaceSize;
  HANDLE_CUPP_ERROR( cupaulipropWorkspaceGetMemorySize(
    handle, minWorkspace,
    CUPAULIPROP_MEMSPACE_DEVICE, CUPAULIPROP_WORKSPACE_SCRATCH,
    &minWorkspaceSize) );
  HANDLE_CUPP_ERROR( cupaulipropWorkspaceGetMemorySize(
    handle, avgWorkspace,
    CUPAULIPROP_MEMSPACE_DEVICE, CUPAULIPROP_WORKSPACE_SCRATCH,
    &avgWorkspaceSize) );
  HANDLE_CUPP_ERROR( cupaulipropWorkspaceGetMemorySize(
    handle, maxWorkspace,
    CUPAULIPROP_MEMSPACE_DEVICE, CUPAULIPROP_WORKSPACE_SCRATCH,
    &maxWorkspaceSize) );

  // Compare our pre-allocated expansion and workspace buffers to the min-memory scenario,
  // for which the outputs cannot overflow without a returned error code. Note that we
  // already know the workspace is sufficient, since the above Prepare() call would 
  // otherwise return error CUPAULIPROP_STATUS_INSUFFICIENT_WORKSPACE.
  if (minOutCapacity > outTermCapacity || minWorkspaceSize > workspaceBufferSize)
  {
    std::cout
      << "The best-case memory sizes exceed our outExpansion and/or workspace buffer sizes. "
      << "This is invalid and will trigger an error in the Compute() call. Exiting..."
      << std::endl;
    std::abort();
  }

  // Compare our pre-allocated expansion and workspace buffers to the max-memory scenario
  if (maxOutCapacity == -1 || maxWorkspaceSize == -1)
  {
    std::cout
      << "  (The worst-case memory sizes overflowed and are reported as -1. "
      << "This example will continue, with a non-zero likelihood of failure.)"
      << std::endl;
  } 
  else if (maxOutCapacity > outTermCapacity || maxWorkspaceSize > workspaceBufferSize)
  {
    std::cout
      << "  (The worst-case memory sizes exceed our outExpansion and/or workspace buffer sizes. "
      << "This example will continue, with a non-zero likelihood of failure.)"
      << std::endl;
  }
  else
  {
    std::cout
      << "  (The worst-case memory sizes are smaller than our pre-allocated buffers. "
      << "As such, the fused API is guaranteed to succeed.)"
      << std::endl;
  }

  // Compare our pre-allocated expansion and workspace buffers to the average-memory scenario
  if (avgOutCapacity == -1 || avgWorkspaceSize == -1)
  {
    std::cout
      << "  (The average-case memory estimate overflowed and is reported as -1. "
      << "This example will continue, with a considerable likelihood of failure.)"
      << std::endl;
  } 
  else if (avgOutCapacity > outTermCapacity || avgWorkspaceSize > workspaceBufferSize)
  {
    std::cout
      << "  (The average-case memory sizes exceed our outExpansion and/or workspace buffer sizes. "
      << "This example will continue, with a possibly large likelihood of failure.)"
      << std::endl;
  } 
  else
  {
    // When the buffers exceed the average case, there's nothing significant to report!  
  }

  // Sphinx: #10

  // Attach our pre-allocated workspace buffer to a workspace; we arbitrarily choose avgWorkspace
  HANDLE_CUPP_ERROR( cupaulipropWorkspaceSetMemory(
    handle, avgWorkspace,
    CUPAULIPROP_MEMSPACE_DEVICE, CUPAULIPROP_WORKSPACE_SCRATCH,
    d_workspaceBuffer, workspaceBufferSize) );

  // Sphinx: #11

  // Attempt to apply all operators at once
  cupaulipropStatus_t computeStatus = cupaulipropPauliExpansionViewComputeOperatorFusedApplication(
    handle, inView, outExpansion, quantumOps.size(), quantumOps.data(), adjoints.data(),
    NUM_TRUNC_STRATS, TRUNC_STRATS, avgWorkspace, stream);

  switch (computeStatus) {
    case CUPAULIPROP_STATUS_SUCCESS:
      break;
    case CUPAULIPROP_STATUS_INSUFFICIENT_WORKSPACE:
    case CUPAULIPROP_STATUS_INSUFFICIENT_OUT_EXPANSION:
      std::cout
        << "Fused application failed on-the-fly, due to the provided expansion and/or "
        << "workspace being insufficient to contain the expansion growth under the given "
        << "operators. One could dynamically shrink the number of passed operators, or "
        << "enlarge the memory capacities, but in this simple example, we opt to exit."
        << std::endl;
      std::abort();
    default:
      HANDLE_CUPP_ERROR(computeStatus);

    // Below are made impossible by our prior Prepare() call:
    // case CUPAULIPROP_FAILURE_NUM_OPERATORS_EXCEEDS_DEVICE_LIMITATION:
  }

  HANDLE_CUPP_ERROR( cupaulipropDestroyPauliExpansionView(inView) );

  // Sphinx: #12
}



// Sphinx: #13
// ========================================================================
// BENCHMARKING
// ========================================================================


// Sphinx: #14
auto timeSingleOperatorSequence(
  cupaulipropHandle_t handle,
  cudaStream_t stream,
  cupaulipropPauliExpansion_t inExpansion,
  cupaulipropPauliExpansion_t outExpansion,
  const std::vector<cupaulipropQuantumOperator_t>& quantumOps,
  void* d_workspaceBuffer,
  size_t workspaceBufferSize)
{
  cupaulipropWorkspaceDescriptor_t workspace;
  HANDLE_CUPP_ERROR( cupaulipropCreateWorkspaceDescriptor(handle, &workspace) );

  // Warm up before timing (leaves inExpansion unchanged)
  applySingleOperator(
    handle, stream,
    inExpansion, outExpansion, quantumOps[0],
    workspace, d_workspaceBuffer, workspaceBufferSize);
  HANDLE_CUDA_ERROR( cudaStreamSynchronize(stream) );

  const auto start = std::chrono::steady_clock::now();

  // Apply every operator in-turn
  for (auto quantumOp : quantumOps) {

    applySingleOperator(
      handle, stream,
      inExpansion, outExpansion, quantumOp,
      workspace, d_workspaceBuffer, workspaceBufferSize);

    std::swap(inExpansion, outExpansion);
  }

  HANDLE_CUDA_ERROR( cudaStreamSynchronize(stream) );

  const auto end = std::chrono::steady_clock::now();

  // Destroy workspace outside of timing
  HANDLE_CUPP_ERROR( cupaulipropDestroyWorkspaceDescriptor(workspace) );

  return end - start;
}


// Sphinx: #15
auto timeFusedOperatorSequence(
  cupaulipropHandle_t handle,
  cudaStream_t stream,
  cupaulipropPauliExpansion_t inExpansion,
  cupaulipropPauliExpansion_t outExpansion,
  const std::vector<cupaulipropQuantumOperator_t>& quantumOps,
  void* d_workspaceBuffer,
  size_t workspaceBufferSize)
{
  cupaulipropWorkspaceDescriptor_t minWorkspace;
  cupaulipropWorkspaceDescriptor_t maxWorkspace;
  cupaulipropWorkspaceDescriptor_t avgWorkspace;
  HANDLE_CUPP_ERROR( cupaulipropCreateWorkspaceDescriptor(handle, &minWorkspace) );
  HANDLE_CUPP_ERROR( cupaulipropCreateWorkspaceDescriptor(handle, &maxWorkspace) );
  HANDLE_CUPP_ERROR( cupaulipropCreateWorkspaceDescriptor(handle, &avgWorkspace) );

  // Warm up before timing (leaves inExpansion unchanged)
  applyFusedOperators(
    handle, stream,
    inExpansion, outExpansion, quantumOps,
    d_workspaceBuffer, workspaceBufferSize,
    minWorkspace, maxWorkspace, avgWorkspace);
  HANDLE_CUDA_ERROR( cudaStreamSynchronize(stream) );

  const auto start = std::chrono::steady_clock::now();

  applyFusedOperators(
    handle, stream,
    inExpansion, outExpansion, quantumOps,
    d_workspaceBuffer, workspaceBufferSize,
    minWorkspace, maxWorkspace, avgWorkspace);
  HANDLE_CUDA_ERROR( cudaStreamSynchronize(stream) );

  const auto end = std::chrono::steady_clock::now();

  HANDLE_CUPP_ERROR( cupaulipropDestroyWorkspaceDescriptor(minWorkspace) );
  HANDLE_CUPP_ERROR( cupaulipropDestroyWorkspaceDescriptor(maxWorkspace) );
  HANDLE_CUPP_ERROR( cupaulipropDestroyWorkspaceDescriptor(avgWorkspace) );

  return end - start;
}


// Sphinx: #16
void benchmarkOperatorSequence(
  cupaulipropHandle_t handle,
  cudaStream_t stream,
  cupaulipropPauliExpansion_t inExpansion,
  cupaulipropPauliExpansion_t outExpansion,
  const std::vector<cupaulipropQuantumOperator_t>& quantumOps,
  void* d_workspaceBuffer, size_t workspaceBufferSize)
{
  // Time the fused API first, since it does not modify inExpansion
  auto fusedDuration = timeFusedOperatorSequence(
    handle, stream,
    inExpansion, outExpansion, quantumOps,
    d_workspaceBuffer, workspaceBufferSize);
  
  // Time the single-operator API second, because it involves ping-ponging
  // between inExpansion and outExpansion, mutating both! 
  auto sequentialDuration = timeSingleOperatorSequence(
    handle, stream,
    inExpansion, outExpansion, quantumOps,
    d_workspaceBuffer, workspaceBufferSize);

  // Report the measured speedup of the fused API over the single-operator API.
  const auto fusedMs      = std::chrono::duration<double, std::milli>(fusedDuration     ).count();
  const auto sequentialMs = std::chrono::duration<double, std::milli>(sequentialDuration).count();
  std::cout << std::endl;
  std::cout << "Single-operator API: " << sequentialMs << " ms" << std::endl;
  std::cout << "Fused-operator API:  " << fusedMs      << " ms" << std::endl;
  std::cout << "Speedup:             " << sequentialMs / fusedMs << "x\n" << std::endl;
}



// Sphinx: #17
// ========================================================================
// MAIN
// ========================================================================


int main()
{
  std::cout << "cuPauliProp fused operators example" << std::endl;
  std::cout << "===================================" << std::endl << std::endl;


  // Sphinx: #18
  /*
   * PREPARE LIBRARY AND MEMORY BUFFERS
   */

  HANDLE_CUDA_ERROR( cudaSetDevice(0) );
  cudaStream_t stream = 0;

  cupaulipropHandle_t handle{};
  HANDLE_CUPP_ERROR( cupaulipropCreate(&handle) );

  // Prepare buffers for two Pauli expansions
  const auto [pauliBufferSize, coefBufferSize] = getNewExpansionBufferSizes();
  void* d_inPauliBuffer;
  void* d_inCoefBuffer;
  void* d_outPauliBuffer;
  void* d_outCoefBuffer;
  HANDLE_CUDA_ERROR( cudaMalloc(&d_inPauliBuffer,   pauliBufferSize) );
  HANDLE_CUDA_ERROR( cudaMalloc(&d_inCoefBuffer,    coefBufferSize ) );
  HANDLE_CUDA_ERROR( cudaMalloc(&d_outPauliBuffer,  pauliBufferSize) );
  HANDLE_CUDA_ERROR( cudaMalloc(&d_outCoefBuffer,   coefBufferSize ) );

  // Initialise the "in" Pauli expansion to a single term; X on qubit 0
  const cupaulipropPackedIntegerType_t initX = 1;
  const double initCoef = 1.0;
  HANDLE_CUDA_ERROR( cudaMemset(d_inPauliBuffer, 0, pauliBufferSize) );
  HANDLE_CUDA_ERROR( cudaMemcpy(d_inPauliBuffer, &initX,    sizeof(initX),    cudaMemcpyHostToDevice) );
  HANDLE_CUDA_ERROR( cudaMemcpy(d_inCoefBuffer,  &initCoef, sizeof(initCoef), cudaMemcpyHostToDevice) );

  // Create two Pauli expansions, to ping-pong as input and output to the cuPP API
  cupaulipropPauliExpansion_t inExpansion;
  cupaulipropPauliExpansion_t outExpansion;
  HANDLE_CUPP_ERROR( cupaulipropCreatePauliExpansion(
    handle, NUM_QUBITS,
    d_inPauliBuffer, pauliBufferSize,
    d_inCoefBuffer,  coefBufferSize, COEF_TYPE,
    /*numTerms=*/1,
    SORT_ORDER, DUPLICATES,
    &inExpansion) );
  HANDLE_CUPP_ERROR( cupaulipropCreatePauliExpansion(
    handle, NUM_QUBITS,
    d_outPauliBuffer, pauliBufferSize,
    d_outCoefBuffer,  coefBufferSize, COEF_TYPE,
    /*numTerms=*/0,
    SORT_ORDER, DUPLICATES,
    &outExpansion) );

  // Allocate workspace buffer; descriptors are created where needed.
  void* d_workspaceBuffer;
  size_t workspaceBufferSize = 2 * (pauliBufferSize + coefBufferSize);
  HANDLE_CUDA_ERROR( cudaMalloc(&d_workspaceBuffer, workspaceBufferSize) ); // attached in Prepare()


  // Sphinx: #19
  /*
   * PREPARE A RANDOM EXPANSION
   */

  // Grow inExpansion (not outExpansion) to size >= TARGET_NUM_TERMS via random Pauli rotations
  growExpansionViaRandomRotations(
    handle, stream,
    inExpansion, outExpansion,
    NUM_QUBITS, TARGET_NUM_TERMS,
    pauliBufferSize, coefBufferSize,
    d_workspaceBuffer, workspaceBufferSize);

  int64_t numInputTerms;
  HANDLE_CUPP_ERROR( cupaulipropPauliExpansionGetNumTerms(handle, inExpansion, &numInputTerms) );
  const int64_t inputExpansionBytes = getPopulatedExpansionMemorySize(numInputTerms);
  const double inputExpansionMiB = static_cast<double>(inputExpansionBytes) / (1024.0 * 1024.0);
  std::cout << "Input expansion size: " << numInputTerms
            << " terms (" << NUM_QUBITS << " qubits)" << std::endl;
  std::cout << "Input expansion memory: " << inputExpansionBytes
            << " bytes (" << std::fixed << std::setprecision(2) << inputExpansionMiB
            << " MiB)" << std::endl << std::endl;


  // Sphinx: #20
  /*
   * BENCHMARK RANDOM PAULI NOISE CHANNELS
   */

  std::cout << "Applying " << NUM_PAULI_NOISE_CHANNELS << " Pauli noise channels...\n" << std::endl;

  // Prepare a list of random Pauli channels (1 and 2 qubit)
  std::mt19937_64 rng(5678);
  std::vector<cupaulipropQuantumOperator_t> pauliNoiseChannels;
  for (int32_t i = 0; i < NUM_PAULI_NOISE_CHANNELS; ++i)
    pauliNoiseChannels.push_back(createRandomPauliNoiseChannel(handle, rng, NUM_QUBITS));

  benchmarkOperatorSequence(
    handle, stream,
    inExpansion, outExpansion, pauliNoiseChannels,
    d_workspaceBuffer, workspaceBufferSize);


  // Sphinx: #21
  /*
   * BENCHMARK RANDOM CLIFFORD GATES
   */

   std::cout << "Applying " << NUM_CLIFFORD_GATES << " random Clifford gates...\n" << std::endl;

  // Prepare a list of random Clifford gates (1 and 2 qubit, and random kinds)
  std::vector<cupaulipropQuantumOperator_t> cliffordGates;
  for (int32_t i = 0; i < NUM_CLIFFORD_GATES; ++i)
    cliffordGates.push_back(createRandomCliffordGate(handle, rng, NUM_QUBITS));

  // Comparative benchmarking penalises fused API with print statements - no big deal!
  benchmarkOperatorSequence(
    handle, stream,
    inExpansion, outExpansion, cliffordGates,
    d_workspaceBuffer, workspaceBufferSize);


  // Sphinx: #22
  /*
   * CLEAN UP
   */
  
  for (auto gate : cliffordGates)
    HANDLE_CUPP_ERROR( cupaulipropDestroyOperator(gate) );
  for (auto channel : pauliNoiseChannels)
    HANDLE_CUPP_ERROR( cupaulipropDestroyOperator(channel) );
  HANDLE_CUPP_ERROR( cupaulipropDestroyPauliExpansion(inExpansion) );
  HANDLE_CUPP_ERROR( cupaulipropDestroyPauliExpansion(outExpansion) );
  HANDLE_CUPP_ERROR( cupaulipropDestroy(handle) );
  HANDLE_CUDA_ERROR( cudaFree(d_workspaceBuffer) );
  HANDLE_CUDA_ERROR( cudaFree(d_inPauliBuffer) );
  HANDLE_CUDA_ERROR( cudaFree(d_inCoefBuffer) );
  HANDLE_CUDA_ERROR( cudaFree(d_outPauliBuffer) );
  HANDLE_CUDA_ERROR( cudaFree(d_outCoefBuffer) );

  return EXIT_SUCCESS;
}
