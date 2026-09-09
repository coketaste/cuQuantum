/*
 * Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
 *
 * SPDX-License-Identifier: BSD-3-Clause
 */

#include "example_utils.hpp"

#include <algorithm>
#include <array>
#include <cmath>
#include <cstdlib>
#include <iostream>
#include <numeric>
#include <vector>



/*
 * PRIVATE FUNCTIONS
 */


namespace {

std::vector<double> sampleSingleQubitNoiseProbabilities(std::mt19937_64& rng)
{
  // Lazily choose max probability such that probs are normalised
  const double maxProb = 0.05;
  std::uniform_real_distribution<double> errorRateDist(0.0, maxProb);

  const double pX = errorRateDist(rng);
  const double pY = errorRateDist(rng);
  const double pZ = errorRateDist(rng);
  return {1.0 - pX - pY - pZ, pX, pY, pZ};
}


std::vector<double> sampleTwoQubitNoiseProbabilities(std::mt19937_64& rng)
{
  // Lazily choose max probability such that probs are normalised
  const double maxProb = 0.05;
  const size_t numProbs = 16;
  std::vector<double> probabilities(numProbs, 0.0);
  std::uniform_real_distribution<double> errorRateDist(0.0, maxProb);
  
  double totalErrorRate = 0.0;
  for (size_t i = 1; i < numProbs; ++i) {
    auto prob = errorRateDist(rng);
    probabilities[i] = prob;
    totalErrorRate += prob;
  }

  probabilities[0] = 1.0 - totalErrorRate;
  return probabilities;
}


void prepareSingleOperatorApplication(
  cupaulipropHandle_t handle,
  cupaulipropPauliExpansionView_t inView,
  cupaulipropQuantumOperator_t gate,
  cupaulipropWorkspaceDescriptor_t workspace,
  void* d_workspaceBuffer,
  size_t pauliBufferSize,
  size_t coefBufferSize,
  size_t workspaceBufferSize,
  cupaulipropSortOrder_t sortOrder,
  int32_t keepDuplicates,
  int32_t numTruncationStrategies,
  const cupaulipropTruncationStrategy_t truncationStrategies[])
{
  // Learn required expansion sizes, and attach required workspace size
  int64_t requiredPauliSize;
  int64_t requiredCoefSize;
  HANDLE_CUPP_ERROR( cupaulipropPauliExpansionViewPrepareOperatorApplication(
    handle, inView, gate,
    sortOrder, keepDuplicates, numTruncationStrategies, truncationStrategies,
    workspaceBufferSize, &requiredPauliSize, &requiredCoefSize, workspace) );

  // Learn required attached workspace size
  int64_t requiredWorkspaceSize;
  HANDLE_CUPP_ERROR( cupaulipropWorkspaceGetMemorySize(
    handle, workspace,
    CUPAULIPROP_MEMSPACE_DEVICE, CUPAULIPROP_WORKSPACE_SCRATCH,
    &requiredWorkspaceSize) );

  // Ensure the pre-allocated buffers are all sufficiently sized
  if (requiredPauliSize     > static_cast<int64_t>(pauliBufferSize) ||
      requiredCoefSize      > static_cast<int64_t>(coefBufferSize)  ||
      requiredWorkspaceSize > static_cast<int64_t>(workspaceBufferSize))
  {
    std::cout
      << "Insufficient outExpansion capacity and/or workspace buffer size "
      << "to perform operator application. Exiting..."
      << std::endl;
    std::abort();
  }

  // Re-attach the existing workspace buffer
  HANDLE_CUPP_ERROR( cupaulipropWorkspaceSetMemory(
    handle, workspace,
    CUPAULIPROP_MEMSPACE_DEVICE, CUPAULIPROP_WORKSPACE_SCRATCH,
    d_workspaceBuffer, workspaceBufferSize) );
}

} // namespace



/*
 * PUBLIC FUNCTIONS
 */


cupaulipropQuantumOperator_t createRandomPauliRotationGate(
  cupaulipropHandle_t handle,
  std::mt19937_64& rng,
  int32_t numQubits)
{
  cupaulipropQuantumOperator_t gate;

  // Target all qubits; some will receive PAULI_I
  std::vector<int32_t> qubits(numQubits);
  std::iota(qubits.begin(), qubits.end(), 0);

  // Generate a random Pauli string
  std::vector<cupaulipropPauliKind_t> paulis(static_cast<size_t>(numQubits));
  auto pauliChoices = std::array{
    CUPAULIPROP_PAULI_I,
    CUPAULIPROP_PAULI_X,
    CUPAULIPROP_PAULI_Y,
    CUPAULIPROP_PAULI_Z};
  std::uniform_int_distribution<int> pauliDist(0, 3);
  for (auto& pauli : paulis)
    pauli = pauliChoices[pauliDist(rng)];

  // Generate a random coefficient in (approx) [-pi, pi]
  std::uniform_real_distribution<double> angleDist(-3.14, 3.14);
  const double angle = angleDist(rng);

  // Create a new operator, which the caller must later free
  HANDLE_CUPP_ERROR( cupaulipropCreatePauliRotationGateOperator(
      handle, angle, numQubits, qubits.data(), paulis.data(), &gate) );

  return gate;
}


cupaulipropQuantumOperator_t createRandomPauliNoiseChannel(
  cupaulipropHandle_t handle,
  std::mt19937_64& rng,
  int32_t numQubits)
{
  cupaulipropQuantumOperator_t channel;

  // Prepare distributions for target sampling, and numTargets (1 or 2) sampling
  std::uniform_int_distribution<int32_t> targetQubitDist(0, numQubits - 1);
  std::bernoulli_distribution useTwoQubitsDist(0.5); // 1 and 2 qubit equally likely

  if (numQubits >= 2 && useTwoQubitsDist(rng)) {

    // Generate 2 targets independently, retrying until distinct
    int32_t firstQubit = targetQubitDist(rng);
    int32_t secondQubit = targetQubitDist(rng);
    while (secondQubit == firstQubit)
      secondQubit = targetQubitDist(rng);
    const int32_t qubits[2] = {firstQubit, secondQubit};

    const auto probabilities = sampleTwoQubitNoiseProbabilities(rng);
    HANDLE_CUPP_ERROR( cupaulipropCreatePauliNoiseChannelOperator(
      handle, /*numTargets=*/2, qubits, probabilities.data(), &channel) );

  } else {

    // Trivially generate a random 1-qubit channel
    const int32_t qubits[1] = {targetQubitDist(rng)};
    const auto probabilities = sampleSingleQubitNoiseProbabilities(rng);
    HANDLE_CUPP_ERROR( cupaulipropCreatePauliNoiseChannelOperator(
      handle, /*numTargets=*/1, qubits, probabilities.data(), &channel) );
  }

  return channel;
}


cupaulipropQuantumOperator_t createRandomCliffordGate(
  cupaulipropHandle_t handle,
  std::mt19937_64& rng,
  int32_t numQubits)
{
  cupaulipropQuantumOperator_t gate;

  // Options for the Clifford gate kind
  static constexpr auto singleQubitGates = std::array{
    CUPAULIPROP_CLIFFORD_GATE_I,
    CUPAULIPROP_CLIFFORD_GATE_X,
    CUPAULIPROP_CLIFFORD_GATE_Y,
    CUPAULIPROP_CLIFFORD_GATE_Z,
    CUPAULIPROP_CLIFFORD_GATE_H,
    CUPAULIPROP_CLIFFORD_GATE_S,
    CUPAULIPROP_CLIFFORD_GATE_SQRTX,
    CUPAULIPROP_CLIFFORD_GATE_SQRTY,
    CUPAULIPROP_CLIFFORD_GATE_SQRTZ};
  static constexpr auto twoQubitGates = std::array{
    CUPAULIPROP_CLIFFORD_GATE_CX,
    CUPAULIPROP_CLIFFORD_GATE_CY,
    CUPAULIPROP_CLIFFORD_GATE_CZ,
    CUPAULIPROP_CLIFFORD_GATE_SWAP,
    CUPAULIPROP_CLIFFORD_GATE_ISWAP};

  // Distributions for qubit indices, gate kinds, and whether to use 1 or 2 qubit cliffords
  std::uniform_int_distribution<int32_t> targetQubitDist(0, numQubits - 1);
  std::uniform_int_distribution<size_t> singleQubitGateDist(0, singleQubitGates.size() - 1);
  std::uniform_int_distribution<size_t> twoQubitGateDist(0, twoQubitGates.size() - 1);
  std::bernoulli_distribution useTwoQubitGateDist(0.5); // 1 and 2 qubit equally likely

  if (numQubits >= 2 && useTwoQubitGateDist(rng)) {

    // Generate 2 targets independently, retrying until distinct
    int32_t target1 = targetQubitDist(rng);
    int32_t target2 = targetQubitDist(rng);
    while (target1 == target2)
      target2 = targetQubitDist(rng);

    const int32_t targets[2] = {target1, target2};
    HANDLE_CUPP_ERROR( cupaulipropCreateCliffordGateOperator(
      handle, twoQubitGates[twoQubitGateDist(rng)], targets, &gate) );
  } else {

    const int32_t targets[1] = {targetQubitDist(rng)};
    HANDLE_CUPP_ERROR( cupaulipropCreateCliffordGateOperator(
      handle, singleQubitGates[singleQubitGateDist(rng)], targets, &gate) );
  }
  
  return gate;
}



/*
 * SUBROUTINES
 */


void growExpansionViaRandomRotations(
  cupaulipropHandle_t handle,
  cudaStream_t stream,
  cupaulipropPauliExpansion_t& inOutExpansion,
  cupaulipropPauliExpansion_t& tempExpansion,
  int32_t numQubits,
  int64_t targetNumTerms,
  size_t pauliBufferSize,
  size_t coefBufferSize,
  void* d_workspaceBuffer,
  size_t workspaceBufferSize)
{
  constexpr int32_t takeAdjoint = 0;
  constexpr auto sortOrder = CUPAULIPROP_SORT_ORDER_NONE;
  constexpr int32_t keepDuplicates = 1;
  constexpr int32_t numTruncationStrategies = 0;
  const cupaulipropTruncationStrategy_t* truncationStrategies = nullptr;

  std::mt19937_64 rng(1234);
  cupaulipropWorkspaceDescriptor_t workspace;
  HANDLE_CUPP_ERROR( cupaulipropCreateWorkspaceDescriptor(handle, &workspace) );

  int64_t numTerms;
  HANDLE_CUPP_ERROR( cupaulipropPauliExpansionGetNumTerms(handle, inOutExpansion, &numTerms) );

  // Repeatedly apply random rotations until the targeted expansion size is reached
  while (numTerms < targetNumTerms) {

    cupaulipropPauliExpansionView_t inView;
    HANDLE_CUPP_ERROR( cupaulipropPauliExpansionGetContiguousRange(
      handle, inOutExpansion, 0, numTerms, &inView) );

    cupaulipropQuantumOperator_t gate = createRandomPauliRotationGate(handle, rng, numQubits);

    prepareSingleOperatorApplication(
      handle, inView, gate,
      workspace, d_workspaceBuffer,
      pauliBufferSize, coefBufferSize, workspaceBufferSize,
      sortOrder, keepDuplicates, numTruncationStrategies, truncationStrategies);

    HANDLE_CUPP_ERROR( cupaulipropPauliExpansionViewComputeOperatorApplication(
      handle, inView, tempExpansion, gate,
      takeAdjoint, sortOrder, keepDuplicates, numTruncationStrategies, truncationStrategies,
      workspace, stream));

    HANDLE_CUPP_ERROR( cupaulipropDestroyPauliExpansionView(inView) );
    HANDLE_CUPP_ERROR( cupaulipropDestroyOperator(gate) );

    // Retain latest state in inOutExpansion
    std::swap(inOutExpansion, tempExpansion);

    HANDLE_CUPP_ERROR( cupaulipropPauliExpansionGetNumTerms(handle, inOutExpansion, &numTerms) );
  }

  HANDLE_CUPP_ERROR( cupaulipropDestroyWorkspaceDescriptor(workspace) );
}
