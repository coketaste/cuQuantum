/*
 * Copyright (c) 2025-2026, NVIDIA CORPORATION & AFFILIATES.
 *
 * SPDX-License-Identifier: BSD-3-Clause
 */

/**
 * @file kicked_ising_fused_example.cpp
 * @brief An example demonstrating cuPauliProp simulation of the IBM 127-qubit kicked Ising experiment,
 *        as presented in Nature volume 618, pages 500–505 (2023). Specifically, we simulate the Z_62
 *        20-Trotter-step experiment of Fig 4. b), the only circuit with a full 127-qubit lightcone, 
 *        at an X-rotation angle of PI/4. The fixed Rzz(-PI/2) gates are decomposed into Cliffords,
 *        and a contiguous sequence of Cliffords is simulated with a single call to cuPauliProp's
 *        "fused" API; the remaining operators are simulated one at a time. For simplicity, we do not simulate
 *        the error channels nor twirling process of the experimental circuit, though inhomogeneous one
 *        and two qubit Pauli channels are supported by cuPauliProp - and can actually accelerate
 *        simulation by enabling more permissive truncation strategies!
 */

#include <cupauliprop.h>
#include <cuda_runtime.h>
#include <iostream>
#include <algorithm>
#include <vector>
#include <cassert>
#include <cmath>
#include <chrono>


// Sphinx: #1
// ========================================================================
// CUDA and cuPauliProp error handling
// ========================================================================

#define HANDLE_CUPP_ERROR(x)                               \
{                                                          \
  const auto err = x;                                      \
  if (err != CUPAULIPROP_STATUS_SUCCESS)                   \
  {                                                        \
    printf("cuPauliProp error in line %d\n", __LINE__);    \
    fflush(stdout);                                        \
    std::abort();                                          \
  }                                                        \
};


#define HANDLE_CUDA_ERROR(x)                                \
{                                                           \
  const auto err = x;                                       \
  if (err != cudaSuccess)                                   \
  {                                                         \
    const char * error = cudaGetErrorString(err);           \
    printf("CUDA Error: %s in line %d\n", error, __LINE__); \
    fflush(stdout);                                         \
    std::abort();                                           \
  }                                                         \
};


// Sphinx: #2
// ========================================================================
// Memory usage
// ========================================================================

// Each Pauli expansion has two pre-allocated GPU buffers, storing packed
// integers (which encode Pauli strings) and corresponding coefficients.
// As much memory can be dedicated as your hardware allows, while the min-
// imum required is specific and very sensitive to the simulated circuit,
// studied observable, and the chosen truncation hyperparameters.
// Some operations also require additional workspace memory which is also
// ideally pre-allocated, and can be established using the API 'Prepare'
// functions (e.g. cupaulipropPauliExpansionViewPrepareTraceWithZeroState).
// In this demo, we dedicate either (a percentage of) the entirety of GPU
// memory uniformly between the required memory buffers, or instead use a
// fixed hardcoded amount which has been tested with
// our other simulation parameters (like truncations); these choices are
// toggled via USE_MAX_VRAM below. If changing the below hyperparameters of
// this example, such as Trotter depth, set USE_MAX_VRAM=true.

// =true to use MAX_VRAM_PERCENT of VRAM, and =false to use fixed memories below
bool USE_MAX_VRAM = false;
double MAX_VRAM_PERCENT = 90; // 0-100%

size_t FIXED_EXPANSION_PAULI_MEM = 16 * (1LLU << 20); // bytes = 16 MiB
size_t FIXED_EXPANSION_COEF_MEM  =  4 * (1LLU << 20); // bytes = 4  MiB
size_t FIXED_WORKSPACE_MEM       = 32 * (1LLU << 20); // bytes = 32 MiB


// Sphinx: #3
// ========================================================================
// Circuit preparation (Trotterised Ising on IBM heavy hex topology)
// ========================================================================

// This demo simulates the circuits experimentally executed by IBM in article
// 'Nature volume 618, pages 500–505 (2023)'. This is a circuit Trotterising
// the evolution operator of a 2D transverse-field Ising model, but where the
// prescribed ZZ rotations have a fixed angle of -pi/2, and where the X angles
// are arbitrarily set/swept; later, we will fix the X angle to be pi/4. The
// Hamiltonian ZZ interactions are confined to a heavy-hex topology, matching
// the connectivity of the IBM Eagle processor 'ibm_kyiv', as we fix below.

const int NUM_CIRCUIT_QUBITS = 127;           // fixed by circuit
const int NUM_ZZ_ROTATIONS_PER_LAYER = 48;    // fixed by circuit
const int NUM_QUBITS_PER_X_ROTATION = 1;      // definitional
const int NUM_QUBITS_PER_ZZ_ROTATION = 2;     // definitional
const int NUM_ZZ_LAYERS_PER_TROTTER_STEP = 3; // fixed by circuit
const int NUM_CLIFFORDS_PER_ZZ_ROTATION = 3;  // fixed by Rzz(-pi/2) decomposition

const double PI = 3.14159265358979323846;

// Indices of ZZ-interacting qubits which undergo the first (red) Trotter round
const int32_t ZZ_QUBITS_RED[NUM_ZZ_ROTATIONS_PER_LAYER][NUM_QUBITS_PER_ZZ_ROTATION] = {
  {  2,   1},  { 33,  39}, { 59,  60}, { 66,  67}, { 72,  81}, {118, 119},
  { 21,  20},  { 26,  25}, { 13,  12}, { 31,  32}, { 70,  74}, {122, 123},
  { 96,  97},  { 57,  56}, { 63,  64}, {107, 108}, {103, 104}, { 46,  45},
  { 28,  35},  {  7,   6}, { 79,  78}, {  5,   4}, {109, 114}, { 62,  61},
  { 58,  71},  { 37,  52}, { 76,  77}, {  0,  14}, { 36,  51}, {106, 105},
  { 73,  85},  { 88,  87}, { 68,  55}, {116, 115}, { 94,  95}, {100, 110},
  { 17,  30},  { 92, 102}, { 50,  49}, { 83,  84}, { 48,  47}, { 98,  99},
  {  8,   9},  {121, 120}, { 23,  24}, { 44,  43}, { 22,  15}, { 53,  41}
};

// Indices of ZZ-interacting qubits which undergo the second (blue) Trotter round
const int32_t ZZ_QUBITS_BLUE[NUM_ZZ_ROTATIONS_PER_LAYER][NUM_QUBITS_PER_ZZ_ROTATION] = {
  { 53,  60}, {123, 124}, { 21,  22}, { 11,  12}, { 67,  68}, {  2,   3},
  { 66,  65}, {122, 121}, {110, 118}, {  6,   5}, { 94,  90}, { 28,  29},
  { 14,  18}, { 63,  62}, {111, 104}, {100,  99}, { 45,  44}, {  4,  15},
  { 20,  19}, { 57,  58}, { 77,  71}, { 76,  75}, { 26,  27}, { 16,   8},
  { 35,  47}, { 31,  30}, { 48,  49}, { 69,  70}, {125, 126}, { 89,  74},
  { 80,  79}, {116, 117}, {114, 113}, { 10,   9}, {106,  93}, {101, 102},
  { 92,  83}, { 98,  91}, { 82,  81}, { 54,  64}, { 96, 109}, { 85,  84},
  { 87,  86}, {108, 112}, { 34,  24}, { 42,  43}, { 40,  41}, { 39,  38}
};

// Indices of ZZ-interacting qubits which undergo the third (green) Trotter round
const int32_t ZZ_QUBITS_GREEN[NUM_ZZ_ROTATIONS_PER_LAYER][NUM_QUBITS_PER_ZZ_ROTATION] = {
  { 10,  11}, { 54,  45}, {111, 122}, { 64,  65}, { 60,  61}, {103, 102},
  { 72,  62}, {  4,   3}, { 33,  20}, { 58,  59}, { 26,  16}, { 28,  27},
  {  8,   7}, {104, 105}, { 73,  66}, { 87,  93}, { 85,  86}, { 55,  49},
  { 68,  69}, { 89,  88}, { 80,  81}, {117, 118}, {101, 100}, {114, 115},
  { 96,  95}, { 29,  30}, {106, 107}, { 83,  82}, { 91,  79}, {  0,   1},
  { 56,  52}, { 90,  75}, {126, 112}, { 36,  32}, { 46,  47}, { 77,  78},
  { 97,  98}, { 17,  12}, {119, 120}, { 22,  23}, { 24,  25}, { 43,  34},
  { 42,  41}, { 40,  39}, { 37,  38}, {125, 124}, { 50,  51}, { 18,  19}
};


// Sphinx: #4
// ========================================================================
// Circuit construction
// ========================================================================

// Each Trotter step alternates a layer of single-qubit X rotations on every
// qubit then three layers of two-qubit Rzz(-pi/2) rotations on the heavy-hex
// topology. Since Rzz(-pi/2) is Clifford, we represent its adjoint as a
// CX-S-CX sequence. An entire contiguous sequence of Clifford operators can
// be passed to the cuPauliProp API, so we group them separately from the X
// rotations.


std::vector<cupaulipropQuantumOperator_t> getXRotationLayer(
  cupaulipropHandle_t handle, double xRotationAngle
) {  
  std::vector<cupaulipropQuantumOperator_t> layer(NUM_CIRCUIT_QUBITS);

  const cupaulipropPauliKind_t paulis[NUM_QUBITS_PER_X_ROTATION] = {CUPAULIPROP_PAULI_X};

  for (int32_t i=0; i<NUM_CIRCUIT_QUBITS; i++) {
    HANDLE_CUPP_ERROR(cupaulipropCreatePauliRotationGateOperator(
      handle, xRotationAngle, NUM_QUBITS_PER_X_ROTATION, &i, paulis, &layer[i]));
  }

  return layer;
}


std::vector<cupaulipropQuantumOperator_t> getZZRotationLayersAdjointAsCliffords(
  cupaulipropHandle_t handle
) {
  std::vector<cupaulipropQuantumOperator_t> sequence;
  sequence.reserve(
    NUM_ZZ_LAYERS_PER_TROTTER_STEP
    * NUM_CLIFFORDS_PER_ZZ_ROTATION
    * NUM_ZZ_ROTATIONS_PER_LAYER);

  // The forward layer order is red-blue-green, so the adjoint order is
  // green-blue-red.
  const int32_t (*topologies[])[NUM_QUBITS_PER_ZZ_ROTATION] = {
    ZZ_QUBITS_GREEN,
    ZZ_QUBITS_BLUE,
    ZZ_QUBITS_RED
  };

  // Traverse the layers in reverse to construct the adjoint sequence.
  for (const auto topology : topologies) {

    // Rotations within a layer have disjoint support and therefore commute,
    // so their original order can be retained.
    for (uint32_t i=0; i<NUM_ZZ_ROTATIONS_PER_LAYER; i++) {

      // adjoint(Rzz(-pi/2)) = CX S CX (up to global phase), where
      // the S acts upon the target of CX (Rzz is agnostic to qubit order)
      cupaulipropQuantumOperator_t cxBefore;
      cupaulipropQuantumOperator_t sGate;
      cupaulipropQuantumOperator_t cxAfter;
      HANDLE_CUPP_ERROR(cupaulipropCreateCliffordGateOperator(
        handle, CUPAULIPROP_CLIFFORD_GATE_CX, topology[i], &cxBefore));
      HANDLE_CUPP_ERROR(cupaulipropCreateCliffordGateOperator(
        handle, CUPAULIPROP_CLIFFORD_GATE_S, &topology[i][0], &sGate));
      HANDLE_CUPP_ERROR(cupaulipropCreateCliffordGateOperator(
        handle, CUPAULIPROP_CLIFFORD_GATE_CX, topology[i], &cxAfter));

      sequence.push_back(cxBefore);
      sequence.push_back(sGate);
      sequence.push_back(cxAfter);
    }
  }

  return sequence;
}


// Sphinx: #5
// ========================================================================
// Observable preparation
// ========================================================================

// This demo simulates the IBM circuit via back-propagating the measurement
// observable through the adjoint circuit. As such, we encode the measured
// observable into our initial Pauli expansion, in the format recognised by
// cuPauliProp. Pauli strings are represented with "packed integers" wherein
// every bit encodes a Pauli operator upon a corresponding qubit. We maintain
// two masks which separately encode the position of X and Z Pauli operators,
// indicated by a set bit at the qubit index, with a common set bit encoding
// a Y Pauli operator. Simulating more qubits than exist bits in the packed
// integer type (64) requires using multiple packed integers for each X and Z
// mask. We store a Pauli string's constituent X and Z masks contiguously in
// a single array, where the final mask of each string is padded with zero
// bits to be an integer multiple of the packed integer size (64 bits).

// The below function accepts a single Pauli string (i.e. a tensor product of
// the given Pauli operators at the specified qubit indices) and returns the
// sequence of packed integers which encode it as per the cuPauliProp API;
// this sequence can be copied directly to the GPU buffer of a Pauli expansion.

std::vector<cupaulipropPackedIntegerType_t> getPauliStringAsPackedIntegers(
  std::vector<cupaulipropPauliKind_t> paulis, 
  std::vector<uint32_t> qubits
) {
  assert(paulis.size() == qubits.size());
  assert(*std::max_element(qubits.begin(), qubits.end()) < NUM_CIRCUIT_QUBITS);

  int32_t numPackedInts;
  HANDLE_CUPP_ERROR(cupaulipropGetNumPackedIntegers(NUM_CIRCUIT_QUBITS, &numPackedInts));

  // A single Pauli string is composed of separate X and Z masks, one after the other
  std::vector<cupaulipropPackedIntegerType_t> out(numPackedInts * 2, 0);
  auto xPtr = &out[0];
  auto zPtr = &out[numPackedInts];

  // Process one input (pauli, qubit) pair at a time
  for (auto i=0; i<qubits.size(); i++) {

    // The qubit corresponds to a specific bit of a specific packed integer
    auto numBitsPerPackedInt = 8 * sizeof(cupaulipropPackedIntegerType_t);
    auto intInd = qubits[i] / numBitsPerPackedInt;
    auto bitInd = qubits[i] % numBitsPerPackedInt;

    // Overwrite a bit of either the X or Z masks (or both when pauli==Y)
    if (paulis[i] == CUPAULIPROP_PAULI_X || paulis[i] == CUPAULIPROP_PAULI_Y)
      xPtr[intInd] = xPtr[intInd] | (1ULL << bitInd);
    if (paulis[i] == CUPAULIPROP_PAULI_Z || paulis[i] == CUPAULIPROP_PAULI_Y)
      zPtr[intInd] = zPtr[intInd] | (1ULL << bitInd);
  }

  return out;
}


// Sphinx: #6
// ========================================================================
// Main
// ========================================================================

// Simulation of the IBM utility experiment proceeds as follows. We set up the
// cuPauliProp library using the default stream, then create two
// Pauli expansions (since the API is out-of-place, as elaborated upon below).
// One expansion is initialised to the measured observable of the IBM circuit.
// We prepare workspace memory and truncation hyperparameters, then create the
// non-Clifford X layer and the Clifford decomposition of the adjoint entangling
// sequence. In each Trotter step, the complete Clifford sequence is applied in one
// "fused" call before the X rotations are applied individually. Thereafter we
// compute the overlap between the final back-propagated observable and the
// experimental initial state (the all-zero state), producing an estimate of
// the experimental expectation value. Finally, we free all allocated memory.


int main(int argc, char** argv) {
  std::cout << "cuPauliProp IBM Heavy-hex Ising Fused Example" << std::endl;
  std::cout << "=============================================" << std::endl << std::endl;


  // Sphinx: #7
  // ========================================================================
  // Library setup
  // ========================================================================

  int deviceId = 0;
  HANDLE_CUDA_ERROR(cudaSetDevice(deviceId));

  // cuPauliProp operations accept a cudaStream_t argument for asynchronous usage
  // We use the default stream (0) in this example
  cudaStream_t stream = 0;
  cupaulipropHandle_t handle;
  HANDLE_CUPP_ERROR(cupaulipropCreate(&handle));


  // Sphinx: #8
  // ========================================================================
  // Decide memory usage
  // ========================================================================

  // As outlined in the 'Memory usage' section above, we either uniformly
  // allocate all (or a high percentage of) available memory between the needed
  // memory buffers, or use the pre-decided fixed values. This demo will create
  // a total of two Pauli expansions (each of which accepts two separate buffers
  // to store Pauli strings and their corresponding coefficients; these have
  // different sizes) and one workspace, hence we arrange for an allocation of
  // five buffers in total.

  size_t expansionPauliMem;
  size_t expansionCoefMem;
  size_t workspaceMem;
  size_t totalUsedMem;

  if (USE_MAX_VRAM) {

    // Find usable device memory
    size_t totalFreeMem, totalGlobalMem;
    HANDLE_CUDA_ERROR(cudaMemGetInfo(&totalFreeMem, &totalGlobalMem));
    size_t totalUsableMem = static_cast<size_t>(totalFreeMem * MAX_VRAM_PERCENT/100);

    // Divide it between the three instances (two expansions, one workspace)
    size_t instanceMem = totalUsableMem / 3;

    // Determine the ideal ratio between an expansion's Pauli and coef buffers
    int32_t numPackedInts;
    HANDLE_CUPP_ERROR(cupaulipropGetNumPackedIntegers(NUM_CIRCUIT_QUBITS, &numPackedInts));
    size_t pauliMemPerTerm = 2 * numPackedInts * sizeof(cupaulipropPackedIntegerType_t);
    size_t coefMemPerTerm = sizeof(double);
    size_t totalMemPerTerm = pauliMemPerTerm + coefMemPerTerm;

    expansionPauliMem = (instanceMem * pauliMemPerTerm) / totalMemPerTerm;
    expansionCoefMem  = (instanceMem * coefMemPerTerm ) / totalMemPerTerm;
    workspaceMem      = instanceMem;

    totalUsedMem = 2*expansionPauliMem + 2*expansionCoefMem + workspaceMem;
    std::cout << "Dedicated memory: " << MAX_VRAM_PERCENT << "% of " << totalFreeMem;
    std::cout << " B free = " << totalUsedMem << " B, divided into..." << std::endl;

  } else {

    // Use pre-decided buffer sizes
    expansionPauliMem = FIXED_EXPANSION_PAULI_MEM;
    expansionCoefMem  = FIXED_EXPANSION_COEF_MEM;
    workspaceMem      = FIXED_WORKSPACE_MEM;

    totalUsedMem = 2*expansionPauliMem + 2*expansionCoefMem + workspaceMem;
    std::cout << "Dedicated memory: " << totalUsedMem << " B = 72 MiB, divided into..." << std::endl;
  }

  std::cout << "  expansion Pauli buffer: " << expansionPauliMem << " B" << std::endl;
  std::cout << "  expansion coef buffer:  " << expansionCoefMem << " B" << std::endl;
  std::cout << "  workspace buffer:       " << workspaceMem << " B\n" << std::endl;


  // Sphinx: #9
  // ========================================================================
  // Pauli expansion preparation
  // ========================================================================

  // Create buffers for two Pauli expansions, which will serve as 'input' and
  // 'output' to the out-of-place cuPauliProp API. Note that the capacities of
  // these buffers constrain the maximum number of Pauli strings maintained
  // during simulation and therefore affect the accuracy of the result. The
  // sufficient buffer sizes are specific to the simulated system, and we
  // choose a surprisingly small capacity as admitted by the studied circuit.

  void * d_inExpansionPauliBuffer;
  void * d_outExpansionPauliBuffer;
  void * d_inExpansionCoefBuffer;
  void * d_outExpansionCoefBuffer;
  HANDLE_CUDA_ERROR(cudaMalloc(&d_inExpansionPauliBuffer,  expansionPauliMem));
  HANDLE_CUDA_ERROR(cudaMalloc(&d_inExpansionCoefBuffer,   expansionCoefMem));
  HANDLE_CUDA_ERROR(cudaMalloc(&d_outExpansionPauliBuffer, expansionPauliMem));
  HANDLE_CUDA_ERROR(cudaMalloc(&d_outExpansionCoefBuffer,  expansionCoefMem));

  // Prepare the X and Z masks which encode the experimental observable Z_62,
  // which has a coefficient of unity, as seen in Figure 4. b) of the IBM work.
  std::cout << "Observable: Z_62\n" << std::endl;
  int64_t numObservableTerms = 1;
  double observableCoef = 1.0;
  std::vector<cupaulipropPauliKind_t> observablePaulis = {CUPAULIPROP_PAULI_Z};
  std::vector<uint32_t> observableQubits = {62};

  // Overwrite the 'input' Pauli expansion buffers with the observable data
  auto observablePackedInts = getPauliStringAsPackedIntegers(observablePaulis, observableQubits);
  size_t numObservableBytes = observablePackedInts.size() * sizeof(observablePackedInts[0]);
  HANDLE_CUDA_ERROR(cudaMemcpy(
    d_inExpansionPauliBuffer, observablePackedInts.data(), numObservableBytes, cudaMemcpyHostToDevice));
  HANDLE_CUDA_ERROR(cudaMemcpy(
    d_inExpansionCoefBuffer, &observableCoef, sizeof(observableCoef), cudaMemcpyHostToDevice));

  // Create two Pauli expansions, which will serve as 'input and 'output' to the API.
  // The observable is Hermitian, and conjugation by the circuit preserves real
  // coefficients in its Pauli expansion, so a real data type is sufficient.
  // The input contains one unique term; no sort order is requested.

  cupaulipropPauliExpansion_t inExpansion;
  cupaulipropPauliExpansion_t outExpansion;

  cupaulipropSortOrder_t sortOrder = CUPAULIPROP_SORT_ORDER_NONE;
  int32_t hasDuplicates = 0;  // isUnique = !hasDuplicates
  cudaDataType_t dataType = CUDA_R_64F;

  HANDLE_CUPP_ERROR(cupaulipropCreatePauliExpansion( // init to above
    handle, NUM_CIRCUIT_QUBITS,
    d_inExpansionPauliBuffer, expansionPauliMem,
    d_inExpansionCoefBuffer,  expansionCoefMem,
    dataType, numObservableTerms, sortOrder, hasDuplicates, 
    &inExpansion));
  HANDLE_CUPP_ERROR(cupaulipropCreatePauliExpansion( // init to empty
    handle, NUM_CIRCUIT_QUBITS,
    d_outExpansionPauliBuffer, expansionPauliMem,
    d_outExpansionCoefBuffer,  expansionCoefMem,
    dataType, 0, CUPAULIPROP_SORT_ORDER_NONE, 0,
    &outExpansion));

  
  // Sphinx: #10
  // ========================================================================
  // Workspace preparation
  // ========================================================================

  // Some API functions require additional workspace memory which we bind to a
  // workspace descriptor. Ordinarily we use the 'Prepare' functions, such as
  // cupaulipropPauliExpansionViewPrepareOperatorApplication(), to precisely
  // bound upfront the needed workspace memory. For simplicity, this demo
  // instead uses a workspace buffer which is known to be sufficient. We call
  // the Prepare functions only to check our fixed allocations are sufficient.

  // The Rzz(-pi/2) layers are simulated as one sequence of Cliffords using
  // the "fused" API, for which the 'Prepare' function accepts three workspaces.
  cupaulipropWorkspaceDescriptor_t workspace;
  cupaulipropWorkspaceDescriptor_t minWorkspace;
  cupaulipropWorkspaceDescriptor_t avgWorkspace;
  HANDLE_CUPP_ERROR(cupaulipropCreateWorkspaceDescriptor(handle, &workspace));
  HANDLE_CUPP_ERROR(cupaulipropCreateWorkspaceDescriptor(handle, &minWorkspace));
  HANDLE_CUPP_ERROR(cupaulipropCreateWorkspaceDescriptor(handle, &avgWorkspace));

  void* d_workspaceBuffer;
  HANDLE_CUDA_ERROR(cudaMalloc(&d_workspaceBuffer, workspaceMem));
  
  // We do not attach this buffer yet because each Prepare call detaches any
  // existing buffer. It is attached immediately before the corresponding Compute.


  // Sphinx: #11
  // ========================================================================
  // Truncation parameter preparation
  // ========================================================================

  // The Pauli propagation simulation technique has memory and runtime costs which
  // (for generic circuits) grow exponentially with the circuit length, which we
  // curtail through "truncation"; dynamic discarding of Pauli strings in our
  // expansion which are predicted to contribute negligibly to the final output
  // expectation value. This file demonstrates simultaneous usage of two truncation
  // techniques; discarding of Pauli strings with an absolute coefficient less than
  // 0.0001, or a "Pauli weight" (the number of non-identity operators in the string)
  // exceeding eight. For example, given a Pauli expansion containing:
  //    0.1 XYZXYZII + 1E-5 ZIIIIIII + 0.2 XXXXYYYY,
  // our truncation parameters below would see the latter two strings discarded due
  // to coefficient and weight truncation respectively.

  cupaulipropCoefficientTruncationParams_t coefTruncParams;
  coefTruncParams.cutoff = 1E-4;

  cupaulipropPauliWeightTruncationParams_t weightTruncParams;
  weightTruncParams.cutoff = 8;

  const uint32_t numTruncStrats = 2;
  cupaulipropTruncationStrategy_t truncStrats[] = {
    {
      CUPAULIPROP_TRUNCATION_STRATEGY_COEFFICIENT_BASED,
      &coefTruncParams
    },
    {
      CUPAULIPROP_TRUNCATION_STRATEGY_PAULI_WEIGHT_BASED,
      &weightTruncParams
    }
  };

  // It is not necessary to perform truncation after every operator, however,
  // since the Pauli expansion size may not have grown substantially, and
  // truncation incurs memory enumeration costs. To control term-count growth,
  // we truncate periodically only during each branching X rotation layer.
  const int numXRotationsBetweenTruncations = 10;

  std::cout << "Coefficient truncation threshold:  " << coefTruncParams.cutoff << std::endl;
  std::cout << "Pauli weight truncation threshold: " << weightTruncParams.cutoff << std::endl;
  std::cout << "Truncation performed after every:  "
            << numXRotationsBetweenTruncations << " X rotation gates\n" << std::endl;


  // Sphinx: #12
  // ========================================================================
  // Back-propagation of the observable through the circuit
  // ========================================================================

  // We now back-propagate Z_62, applying the adjoint circuit, through 20
  // Trotter steps. In each step, the three adjoint Rzz layers are already
  // encoded in one sequence of Clifford gates, and are passed in their
  // entirety to cupaulipropPauliExpansionViewComputeOperatorFusedApplication().
  // In contrast, every operator in an X rotation layer is applied one at a time,
  // via cupaulipropPauliExpansionViewComputeOperatorApplication().

  double xRotationAngle = PI / 4.;
  int numTrotterSteps = 20;
  auto xLayer = getXRotationLayer(handle, xRotationAngle);
  auto zzAdjointCliffordSequence = getZZRotationLayersAdjointAsCliffords(handle);

  // Note Clifford adjoints are false, since CX-S-CX decomposition already adjoints.
  int32_t numCliffords = static_cast<int32_t>(zzAdjointCliffordSequence.size());
  std::vector<int32_t> zzCliffordAdjoints(numCliffords, false);

  int32_t numPackedInts;
  HANDLE_CUPP_ERROR(cupaulipropGetNumPackedIntegers(NUM_CIRCUIT_QUBITS, &numPackedInts));
  int64_t expansionTermCapacity = std::min(
    expansionPauliMem / (2 * numPackedInts * sizeof(cupaulipropPackedIntegerType_t)),
    expansionCoefMem / sizeof(double));

  std::cout << "Circuit: 127 qubit IBM heavy-hex Ising circuit, with..." << std::endl;
  std::cout << "  Trotter steps:          " << numTrotterSteps << std::endl;
  std::cout << "  Rx angle:               " << xRotationAngle << " (i.e. PI/4)" << std::endl;
  std::cout << std::endl;

  // During simulation, we demand the PauliExpansion is always free of
  // duplicate Pauli strings, but permit any sort ordering.
  sortOrder = CUPAULIPROP_SORT_ORDER_NONE;
  uint32_t keepDuplicates = false;

  // Ensure setup work on the stream is complete before starting the timer.
  HANDLE_CUDA_ERROR(cudaStreamSynchronize(stream));
  auto startTime = std::chrono::steady_clock::now();
  int64_t maxNumTerms = 1;

  // Repeatedly apply an adjoint Trotter step
  for (int step=0; step<numTrotterSteps; ++step) {

    // Every operation is applied upon a view of the full expansion (we
    // might alternatively operate upon a smaller view to trade-off memory
    // and runtime costs, hybridising breadth and depth-first evaluation)
    int64_t numExpansionTerms;
    cupaulipropPauliExpansionView_t inView;
    HANDLE_CUPP_ERROR(cupaulipropPauliExpansionGetNumTerms(
      handle, inExpansion, &numExpansionTerms));
    HANDLE_CUPP_ERROR(cupaulipropPauliExpansionGetContiguousRange(
      handle, inExpansion, 0, numExpansionTerms, &inView));
    maxNumTerms = std::max(maxNumTerms, numExpansionTerms);

    // Check that our preallocated workspace buffer is sufficient to handle
    // the best-case minimum-memory scenario of the fused API - which, since our
    // fused operators are Clifford, happens to be equivalent to the average and
    // maximum-memory scenarios.
    int64_t outCapacity;
    HANDLE_CUPP_ERROR(cupaulipropPauliExpansionViewPrepareOperatorFusedApplication(
      handle, inView, numCliffords,
      zzAdjointCliffordSequence.data(), zzCliffordAdjoints.data(),
      0, nullptr, workspaceMem,
      &outCapacity, minWorkspace,
      &outCapacity, avgWorkspace,
      &outCapacity, workspace));

    int64_t reqWorkspaceMem;
    HANDLE_CUPP_ERROR(cupaulipropWorkspaceGetMemorySize(
      handle, workspace,
      CUPAULIPROP_MEMSPACE_DEVICE, CUPAULIPROP_WORKSPACE_SCRATCH,
      &reqWorkspaceMem));
    if (outCapacity > expansionTermCapacity ||
        reqWorkspaceMem > workspaceMem)
    {
      std::cout
        << "Insufficient outExpansion capacity and/or workspace buffer size "
        << "to perform fused operator application. Exiting..."
        << std::endl;
      std::abort();
    }

    // We must re-attach the memory buffer to the workspace after 'Prepare' detachment
    HANDLE_CUPP_ERROR(cupaulipropWorkspaceSetMemory(
      handle, workspace,
      CUPAULIPROP_MEMSPACE_DEVICE, CUPAULIPROP_WORKSPACE_SCRATCH,
      d_workspaceBuffer, workspaceMem));

    // Apply the adjoint Rzz layers as one fused sequence of Cliffords.
    HANDLE_CUPP_ERROR(cupaulipropPauliExpansionViewComputeOperatorFusedApplication(
      handle, inView, outExpansion,
      numCliffords, zzAdjointCliffordSequence.data(), zzCliffordAdjoints.data(),
      0, nullptr, workspace, stream));

    // Destroy the temporary input view, then use the result as the next input.
    HANDLE_CUPP_ERROR(cupaulipropDestroyPauliExpansionView(inView));
    std::swap(inExpansion, outExpansion);

    // Apply the X layer in reverse order, adjointing every gate.
    for (size_t ind=0; ind<xLayer.size(); ++ind) {
      const size_t gateInd = xLayer.size() - 1 - ind;
      const int32_t gateAdj = true;

      // Prepare a new view of all terms in the expansion
      HANDLE_CUPP_ERROR(cupaulipropPauliExpansionGetNumTerms(
        handle, inExpansion, &numExpansionTerms));
      maxNumTerms = std::max(maxNumTerms, numExpansionTerms);
      HANDLE_CUPP_ERROR(cupaulipropPauliExpansionGetContiguousRange(
        handle, inExpansion, 0, numExpansionTerms, &inView));

      // Decide whether to truncate after this gate
      const bool truncateAfterThisRotation =
        (ind + 1) % numXRotationsBetweenTruncations == 0;
      int numPassedTruncStrats = truncateAfterThisRotation ? numTruncStrats : 0;

      // Check the expansion and workspace memories needed to apply the current gate
      // and perform the truncation, ensuring our prior allocations are sufficient
      int64_t reqExpansionPauliMem;
      int64_t reqExpansionCoefMem;
      int64_t reqWorkspaceMem;
      HANDLE_CUPP_ERROR(cupaulipropPauliExpansionViewPrepareOperatorApplication(
        handle, inView, xLayer[gateInd], sortOrder, keepDuplicates,
        numPassedTruncStrats, numPassedTruncStrats > 0 ? truncStrats : nullptr,
        workspaceMem,
        &reqExpansionPauliMem, &reqExpansionCoefMem, workspace));
      HANDLE_CUPP_ERROR(cupaulipropWorkspaceGetMemorySize(
        handle, workspace,
        CUPAULIPROP_MEMSPACE_DEVICE, CUPAULIPROP_WORKSPACE_SCRATCH,
        &reqWorkspaceMem));
      if (reqExpansionPauliMem > static_cast<int64_t>(expansionPauliMem) ||
          reqExpansionCoefMem  > static_cast<int64_t>(expansionCoefMem)  ||
          reqWorkspaceMem      > static_cast<int64_t>(workspaceMem))
      {
        std::cout
          << "Insufficient outExpansion capacity and/or workspace buffer size "
          << "to perform operator application. Exiting..."
          << std::endl;
        std::abort();
      }

      // Reattach the workspace buffer after 'Prepare' detachment
      HANDLE_CUPP_ERROR(cupaulipropWorkspaceSetMemory(
        handle, workspace,
        CUPAULIPROP_MEMSPACE_DEVICE, CUPAULIPROP_WORKSPACE_SCRATCH,
        d_workspaceBuffer, workspaceMem));

      // Apply the gate upon the prepared view of the input expansion, evolving the
      // Pauli strings pointed to within, truncating the result. The input expansion
      // is unchanged while the output expansion is entirely overwritten.
      HANDLE_CUPP_ERROR(cupaulipropPauliExpansionViewComputeOperatorApplication(
        handle, inView, outExpansion, xLayer[gateInd],
        gateAdj, sortOrder, keepDuplicates,
        numPassedTruncStrats, numPassedTruncStrats > 0 ? truncStrats : nullptr,
        workspace, stream));
      HANDLE_CUPP_ERROR(cupaulipropDestroyPauliExpansionView(inView));

      std::swap(inExpansion, outExpansion);
    }
  }

  // Restore outExpansion to being the final output for clarity.
  std::swap(inExpansion, outExpansion);


  // Sphinx: #13
  // ========================================================================
  // Evaluation of the expectation value of the observable
  // ========================================================================

  // The output expansion is now a proxy for the observable back-propagated
  // through to the front of the circuit (though having discarded components
  // which negligibly influence the subsequent overlap). The expectation value
  // of the IBM experiment is the overlap of the output expansion with the
  // zero state, i.e. Tr(outExpansion * |0><0|), as we now compute.

  // Obtain a view of the full output expansion (we'll free it in 'Clean up')
  cupaulipropPauliExpansionView_t outView;
  int64_t numOutTerms;
  HANDLE_CUPP_ERROR(cupaulipropPauliExpansionGetNumTerms(handle, outExpansion, &numOutTerms));
  maxNumTerms = std::max(maxNumTerms, numOutTerms);
  HANDLE_CUPP_ERROR(cupaulipropPauliExpansionGetContiguousRange(
    handle, outExpansion, 0, numOutTerms, &outView));

  // Check that the existing workspace memory is sufficient to compute the trace 
  int64_t reqWorkspaceMem;
  HANDLE_CUPP_ERROR(cupaulipropPauliExpansionViewPrepareTraceWithZeroState(
    handle, outView, workspaceMem, workspace));
  HANDLE_CUPP_ERROR(cupaulipropWorkspaceGetMemorySize(
    handle, workspace,
    CUPAULIPROP_MEMSPACE_DEVICE, CUPAULIPROP_WORKSPACE_SCRATCH,
    &reqWorkspaceMem));
  if (reqWorkspaceMem > static_cast<int64_t>(workspaceMem)) {
    std::cout
      << "Insufficient workspace buffer size to compute the trace. Exiting..."
      << std::endl;
    std::abort();
  }

  // Beware that we must now reattach the buffer to the workspace
  HANDLE_CUPP_ERROR(cupaulipropWorkspaceSetMemory(
    handle, workspace,
    CUPAULIPROP_MEMSPACE_DEVICE, CUPAULIPROP_WORKSPACE_SCRATCH,
    d_workspaceBuffer, workspaceMem));

  // Compute the trace; the main and final output of this simulation!
  double expecSignificand;
  double expecExponent;
  HANDLE_CUPP_ERROR(cupaulipropPauliExpansionViewComputeTraceWithZeroState(
    handle, outView, &expecSignificand, &expecExponent, workspace, stream));

  // The CUDA stream must be synchronized before accessing the trace output.
  HANDLE_CUDA_ERROR(cudaStreamSynchronize(stream));
  double expec = expecSignificand * std::pow(2.0, expecExponent);

  // End timing after trace is evaluated
  auto endTime = std::chrono::steady_clock::now();
  auto duration = std::chrono::duration_cast<std::chrono::microseconds>(endTime - startTime);
  auto durationSecs = (duration.count() / 1e6);

  std::cout << "Expectation value:       " << expec << std::endl;
  std::cout << "Final number of terms:   " << numOutTerms << std::endl;
  std::cout << "Maximum number of terms: " << maxNumTerms << std::endl;
  std::cout << "Runtime:                 " << durationSecs << " seconds\n" << std::endl;


  // Sphinx: #14
  // ========================================================================
  // Clean up
  // ========================================================================

  HANDLE_CUPP_ERROR(cupaulipropDestroyPauliExpansionView(outView));
 
  for (auto & gate : xLayer) {
    HANDLE_CUPP_ERROR(cupaulipropDestroyOperator(gate));
  }
  for (auto & gate : zzAdjointCliffordSequence) {
    HANDLE_CUPP_ERROR(cupaulipropDestroyOperator(gate));
  }

  HANDLE_CUPP_ERROR(cupaulipropDestroyWorkspaceDescriptor(workspace));
  HANDLE_CUPP_ERROR(cupaulipropDestroyWorkspaceDescriptor(minWorkspace));
  HANDLE_CUPP_ERROR(cupaulipropDestroyWorkspaceDescriptor(avgWorkspace));
  HANDLE_CUDA_ERROR(cudaFree(d_workspaceBuffer));

  HANDLE_CUPP_ERROR(cupaulipropDestroyPauliExpansion(inExpansion));
  HANDLE_CUPP_ERROR(cupaulipropDestroyPauliExpansion(outExpansion));

  HANDLE_CUDA_ERROR(cudaFree(d_inExpansionPauliBuffer));
  HANDLE_CUDA_ERROR(cudaFree(d_outExpansionPauliBuffer));
  HANDLE_CUDA_ERROR(cudaFree(d_inExpansionCoefBuffer));
  HANDLE_CUDA_ERROR(cudaFree(d_outExpansionCoefBuffer));

  HANDLE_CUPP_ERROR(cupaulipropDestroy(handle));

  return EXIT_SUCCESS;
}
