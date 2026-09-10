/*
 * Copyright (c) 2025-2026, NVIDIA CORPORATION & AFFILIATES.
 *
 * SPDX-License-Identifier: BSD-3-Clause
 */

/**
 * @file kicked_ising_multiprocess_example.cpp
 * @brief Multiprocess counterpart of kicked_ising_example.cpp, supporting
 *        single-process execution, MPI, and NCCL bootstrapped by MPI.
 *
 * @details The circuit, observable, truncation strategies, and propagation
 * algorithm are unchanged from kicked_ising_example.cpp. This example adds the
 * execution setup and semantics needed for distribution: each rank selects a
 * GPU and owns local expansion buffers, rank 0 supplies the initial observable,
 * operator application and trace evaluation become collective operations, and
 * MPI reductions combine rank-local term-count statistics for reporting.
 *
 * CMake builds a single kicked_ising_multiprocess_example binary. It is
 * single-process when CMake is configured without MPI; with MPI it supports
 * the MPI provider, and with NCCL additionally compiled in it supports both
 * providers. The active provider is selected at runtime from the filename of
 * the plugin in CUPAULIPROP_COMM_LIB (a basename containing "nccl" selects
 * the NCCL provider; anything else selects MPI).
 *
 * Launch a single-process build directly:
 *
 * @code{.sh}
 * <build-dir>/bin/kicked_ising_multiprocess_example
 * @endcode
 *
 * Before a distributed launch, set CUPAULIPROP_COMM_LIB to the desired
 * MPI or NCCL distributed-interface plugin, then launch one process per GPU:
 *
 * @code{.sh}
 * export CUPAULIPROP_COMM_LIB=/path/to/libcupauliprop_distributed_interface_mpi.so
 * mpirun -n <num-ranks> <build-dir>/bin/kicked_ising_multiprocess_example
 *
 * export CUPAULIPROP_COMM_LIB=/path/to/libcupauliprop_distributed_interface_nccl.so
 * mpirun -n <num-ranks> <build-dir>/bin/kicked_ising_multiprocess_example
 * @endcode
 *
 * Under the NCCL provider, MPI is used only to launch processes, bootstrap
 * the NCCL communicator, and reduce reporting statistics; cuPauliProp
 * communication itself uses NCCL.
 */

#include <cupauliprop.h>
#include <cuda_runtime.h>

#include <algorithm>
#include <cassert>
#include <chrono>
#include <cmath>
#include <cstdlib>
#include <iostream>
#include <string>
#include <vector>

// MPI supplies process launch and rank discovery in every distributed build.
// It is also the cuPauliProp communication backend when NCCL is not enabled.
#if defined(MPI_ENABLED)
#include <mpi.h>
#endif

// NCCL types and functions are needed only by the NCCL-backed target. That
// target still includes MPI above because MPI bootstraps the NCCL communicator.
#if defined(NCCL_ENABLED)
#include <nccl.h>
#endif

// This example does not provide a non-MPI mechanism for exchanging the NCCL
// unique ID, so an NCCL build necessarily also has MPI support.
#if defined(NCCL_ENABLED) && !defined(MPI_ENABLED)
#error "This example requires MPI_ENABLED to launch processes and bootstrap NCCL"
#endif


// Sphinx: #1
// ========================================================================
// CUDA, communication, and cuPauliProp error handling
// ========================================================================

[[noreturn]] void abortAllProcesses(int errorCode = EXIT_FAILURE) {
#if defined(MPI_ENABLED)
  MPI_Abort(MPI_COMM_WORLD, errorCode);
#else
  (void)errorCode;
#endif
  std::abort();
}

#define HANDLE_CUPP_ERROR(x)                               \
{                                                          \
  const auto err = x;                                      \
  if (err != CUPAULIPROP_STATUS_SUCCESS)                   \
  {                                                        \
    printf("cuPauliProp error in line %d\n", __LINE__);    \
    fflush(stdout);                                        \
    abortAllProcesses();                                   \
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
    abortAllProcesses();                                    \
  }                                                         \
};

#define HANDLE_NCCL_ERROR(x)                                                  \
{                                                                             \
  const auto err = x;                                                         \
  if (err != ncclSuccess)                                                     \
  {                                                                           \
    printf("NCCL Error: %s in line %d\n", ncclGetErrorString(err), __LINE__); \
    fflush(stdout);                                                           \
    abortAllProcesses();                                                      \
  }                                                                           \
};

#if defined(MPI_ENABLED)
#define HANDLE_MPI_ERROR(x)                                                   \
{                                                                             \
  const auto err = x;                                                         \
  if (err != MPI_SUCCESS)                                                     \
  {                                                                           \
    char error[MPI_MAX_ERROR_STRING];                                         \
    int errorLength = 0;                                                      \
    MPI_Error_string(err, error, &errorLength);                               \
    fprintf(stderr, "MPI Error: %.*s in line %d\n",                           \
            errorLength, error, __LINE__);                                    \
    fflush(stderr);                                                           \
    abortAllProcesses(err);                                                   \
  }                                                                           \
};

// The library dlopens the plugin named by $CUPAULIPROP_COMM_LIB, so that
// filename already identifies the provider; inferring it here keeps a single
// binary usable with either provider without a separate selection flag. A
// basename containing "nccl" selects the NCCL provider; anything else is
// treated as MPI.
bool commLibRequestsNccl() {
  const char* commLibPath = std::getenv("CUPAULIPROP_COMM_LIB");
  if (commLibPath == nullptr) {
    fprintf(stderr,
      "CUPAULIPROP_COMM_LIB must point to a cuPauliProp "
      "distributed-interface plugin for a distributed launch\n");
    abortAllProcesses();
  }
  const std::string path(commLibPath);
  const size_t lastSlash = path.find_last_of('/');
  const std::string baseName =
    (lastSlash == std::string::npos) ? path : path.substr(lastSlash + 1);
  return baseName.find("nccl") != std::string::npos;
}
#endif


// Sphinx: #2
// ========================================================================
// Memory usage
// ========================================================================

bool USE_MAX_VRAM = false;
double MAX_VRAM_PERCENT = 90; // 0-100%

// Process-local fixed capacities. Every process allocates two Pauli buffers,
// two coefficient buffers, and one workspace buffer with these respective sizes.
size_t FIXED_EXPANSION_PAULI_MEM = 16 * (1LLU << 20); // bytes = 16 MiB
size_t FIXED_EXPANSION_COEF_MEM  =  4 * (1LLU << 20); // bytes = 4  MiB
size_t FIXED_WORKSPACE_MEM       = 128 * (1LLU << 20); // bytes = 128 MiB


// Sphinx: #3
// ========================================================================
// Circuit preparation (Trotterised Ising on IBM heavy hex topology)
// ========================================================================

const int NUM_CIRCUIT_QUBITS = 127;
const int NUM_ROTATIONS_PER_LAYER = 48;
const int NUM_PAULIS_PER_X_ROTATION = 1;
const int NUM_PAULIS_PER_Z_ROTATION = 2;

const double PI = 3.14159265358979323846;
const double ZZ_ROTATION_ANGLE = - PI / 2.0;

const int32_t ZZ_QUBITS_RED[NUM_ROTATIONS_PER_LAYER][NUM_PAULIS_PER_Z_ROTATION] = {
  {  2,   1}, { 33,  39}, { 59,  60}, { 66,  67}, { 72,  81}, {118, 119},
  { 21,  20}, { 26,  25}, { 13,  12}, { 31,  32}, { 70,  74}, {122, 123},
  { 96,  97}, { 57,  56}, { 63,  64}, {107, 108}, {103, 104}, { 46,  45},
  { 28,  35}, {  7,   6}, { 79,  78}, {  5,   4}, {109, 114}, { 62,  61},
  { 58,  71}, { 37,  52}, { 76,  77}, {  0,  14}, { 36,  51}, {106, 105},
  { 73,  85}, { 88,  87}, { 68,  55}, {116, 115}, { 94,  95}, {100, 110},
  { 17,  30}, { 92, 102}, { 50,  49}, { 83,  84}, { 48,  47}, { 98,  99},
  {  8,   9}, {121, 120}, { 23,  24}, { 44,  43}, { 22,  15}, { 53,  41}
};

const int32_t ZZ_QUBITS_BLUE[NUM_ROTATIONS_PER_LAYER][NUM_PAULIS_PER_Z_ROTATION] = {
  { 53,  60}, {123, 124}, { 21,  22}, { 11,  12}, { 67,  68}, {  2,   3},
  { 66,  65}, {122, 121}, {110, 118}, {  6,   5}, { 94,  90}, { 28,  29},
  { 14,  18}, { 63,  62}, {111, 104}, {100,  99}, { 45,  44}, {  4,  15},
  { 20,  19}, { 57,  58}, { 77,  71}, { 76,  75}, { 26,  27}, { 16,   8},
  { 35,  47}, { 31,  30}, { 48,  49}, { 69,  70}, {125, 126}, { 89,  74},
  { 80,  79}, {116, 117}, {114, 113}, { 10,   9}, {106,  93}, {101, 102},
  { 92,  83}, { 98,  91}, { 82,  81}, { 54,  64}, { 96, 109}, { 85,  84},
  { 87,  86}, {108, 112}, { 34,  24}, { 42,  43}, { 40,  41}, { 39,  38}
};

const int32_t ZZ_QUBITS_GREEN[NUM_ROTATIONS_PER_LAYER][NUM_PAULIS_PER_Z_ROTATION] = {
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

std::vector<cupaulipropQuantumOperator_t> getXRotationLayer(
  cupaulipropHandle_t handle, double xRotationAngle
) {
  std::vector<cupaulipropQuantumOperator_t> layer(NUM_CIRCUIT_QUBITS);
  const cupaulipropPauliKind_t paulis[NUM_PAULIS_PER_X_ROTATION] = {
    CUPAULIPROP_PAULI_X};

  for (int32_t i = 0; i < NUM_CIRCUIT_QUBITS; i++) {
    HANDLE_CUPP_ERROR(cupaulipropCreatePauliRotationGateOperator(
      handle, xRotationAngle, NUM_PAULIS_PER_X_ROTATION, &i, paulis, &layer[i]));
  }
  return layer;
}

std::vector<cupaulipropQuantumOperator_t> getZZRotationLayer(
  cupaulipropHandle_t handle,
  const int32_t topology[NUM_ROTATIONS_PER_LAYER][NUM_PAULIS_PER_Z_ROTATION]
) {
  std::vector<cupaulipropQuantumOperator_t> layer(NUM_ROTATIONS_PER_LAYER);
  const cupaulipropPauliKind_t paulis[NUM_PAULIS_PER_Z_ROTATION] = {
    CUPAULIPROP_PAULI_Z, CUPAULIPROP_PAULI_Z};

  for (uint32_t i = 0; i < NUM_ROTATIONS_PER_LAYER; i++) {
    HANDLE_CUPP_ERROR(cupaulipropCreatePauliRotationGateOperator(
      handle, ZZ_ROTATION_ANGLE, NUM_PAULIS_PER_Z_ROTATION, topology[i], paulis, &layer[i]));
  }
  return layer;
}

std::vector<cupaulipropQuantumOperator_t> getIBMHeavyHexIsingCircuit(
  cupaulipropHandle_t handle, double xRotationAngle, int numTrotterSteps
) {
  std::vector<cupaulipropQuantumOperator_t> circuit;

  for (int n = 0; n < numTrotterSteps; n++) {
    auto layerX       = getXRotationLayer(handle, xRotationAngle);
    auto layerRedZZ   = getZZRotationLayer(handle, ZZ_QUBITS_RED);
    auto layerBlueZZ  = getZZRotationLayer(handle, ZZ_QUBITS_BLUE);
    auto layerGreenZZ = getZZRotationLayer(handle, ZZ_QUBITS_GREEN);

    circuit.insert(circuit.end(), layerX.begin(),       layerX.end());
    circuit.insert(circuit.end(), layerRedZZ.begin(),   layerRedZZ.end());
    circuit.insert(circuit.end(), layerBlueZZ.begin(),  layerBlueZZ.end());
    circuit.insert(circuit.end(), layerGreenZZ.begin(), layerGreenZZ.end());
  }
  return circuit;
}


// Sphinx: #5
// ========================================================================
// Observable preparation
// ========================================================================

std::vector<cupaulipropPackedIntegerType_t> getPauliStringAsPackedIntegers(
  std::vector<cupaulipropPauliKind_t> paulis,
  std::vector<uint32_t> qubits
) {
  assert(paulis.size() == qubits.size());
  assert(*std::max_element(qubits.begin(), qubits.end()) < NUM_CIRCUIT_QUBITS);

  int32_t numPackedInts;
  HANDLE_CUPP_ERROR(cupaulipropGetNumPackedIntegers(NUM_CIRCUIT_QUBITS, &numPackedInts));

  std::vector<cupaulipropPackedIntegerType_t> out(numPackedInts * 2, 0);
  auto xPtr = &out[0];
  auto zPtr = &out[numPackedInts];

  for (auto i = 0; i < qubits.size(); i++) {
    auto numBitsPerPackedInt = 8 * sizeof(cupaulipropPackedIntegerType_t);
    auto intInd = qubits[i] / numBitsPerPackedInt;
    auto bitInd = qubits[i] % numBitsPerPackedInt;

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

int main(int argc, char** argv) {

  // To be overwritten below
  int rank = 0;
  int numRanks = 1;

  // Both distributed modes use MPI to establish the process topology and map
  // each rank to a GPU. In the NCCL build, MPI performs only launch, bootstrap,
  // and reporting duties; cuPauliProp's data communication uses NCCL instead.
  // Without MPI, the initialized defaults describe one process on device 0.
#if defined(MPI_ENABLED)
  MPI_Init(&argc, &argv);
  HANDLE_MPI_ERROR(MPI_Comm_set_errhandler(MPI_COMM_WORLD, MPI_ERRORS_RETURN));
  HANDLE_MPI_ERROR(MPI_Comm_rank(MPI_COMM_WORLD, &rank));
  HANDLE_MPI_ERROR(MPI_Comm_size(MPI_COMM_WORLD, &numRanks));

  MPI_Comm localComm;
  HANDLE_MPI_ERROR(MPI_Comm_split_type(
    MPI_COMM_WORLD, MPI_COMM_TYPE_SHARED, rank, MPI_INFO_NULL, &localComm));
  int localRank = 0;
  int localSize = 1;
  HANDLE_MPI_ERROR(MPI_Comm_rank(localComm, &localRank));
  HANDLE_MPI_ERROR(MPI_Comm_size(localComm, &localSize));
  HANDLE_MPI_ERROR(MPI_Comm_free(&localComm));

  // Oversubscribed nodes map ranks round-robin onto the visible devices.
  int numDevices = 0;
  HANDLE_CUDA_ERROR(cudaGetDeviceCount(&numDevices));
  if (numDevices == 0) {
    fprintf(stderr, "No visible CUDA devices\n");
    abortAllProcesses();
  }
  HANDLE_CUDA_ERROR(cudaSetDevice(localRank % numDevices));
#else
  HANDLE_CUDA_ERROR(cudaSetDevice(0));
#endif

  if (rank == 0) {
    std::cout << "cuPauliProp IBM Heavy-hex Ising Example" << std::endl;
    std::cout << "========================================================"
              << std::endl << std::endl;
  }

  cudaStream_t stream = 0;
  cupaulipropHandle_t handle;
  HANDLE_CUPP_ERROR(cupaulipropCreate(&handle));

  // Select the provider at runtime from the plugin filename. Both
  // communicator variables exist for the whole run because teardown must
  // free whichever one was created after the handle is destroyed.
#if defined(MPI_ENABLED)
  const bool useNccl = commLibRequestsNccl();
  MPI_Comm mpiCommunicator = MPI_COMM_NULL;
#endif
#if defined(NCCL_ENABLED)
  ncclComm_t ncclCommunicator = nullptr;
#endif

#if defined(NCCL_ENABLED)
  if (useNccl) {
    // NCCL communicators require an out-of-band bootstrap: rank 0 creates a
    // unique ID, MPI broadcasts it, and every rank constructs its NCCL
    // endpoint. The resulting opaque ncclComm_t is passed to the NCCL
    // provider.
    ncclUniqueId ncclId;
    if (rank == 0) {
      HANDLE_NCCL_ERROR(ncclGetUniqueId(&ncclId));
    }
    HANDLE_MPI_ERROR(
      MPI_Bcast(&ncclId, sizeof(ncclId), MPI_BYTE, 0, MPI_COMM_WORLD));

    HANDLE_NCCL_ERROR(
      ncclCommInitRank(&ncclCommunicator, numRanks, ncclId, rank));
    HANDLE_CUPP_ERROR(cupaulipropResetDistributedConfiguration(
      handle,
      CUPAULIPROP_DISTRIBUTED_PROVIDER_NCCL,
      &ncclCommunicator,
      sizeof(ncclCommunicator)));
  }
#endif
#if defined(MPI_ENABLED)
  if (!useNccl) {
    // The MPI provider can consume an MPI communicator directly. Duplicate
    // MPI_COMM_WORLD so the example owns the communicator given to cuPauliProp
    // and can preserve it until the handle has been destroyed.
    HANDLE_MPI_ERROR(MPI_Comm_dup(MPI_COMM_WORLD, &mpiCommunicator));
    HANDLE_CUPP_ERROR(cupaulipropResetDistributedConfiguration(
      handle,
      CUPAULIPROP_DISTRIBUTED_PROVIDER_MPI,
      &mpiCommunicator,
      sizeof(mpiCommunicator)));
  }
#endif
#if defined(MPI_ENABLED) && !defined(NCCL_ENABLED)
  if (useNccl) {
    if (rank == 0) {
      fprintf(stderr,
        "CUPAULIPROP_COMM_LIB names an NCCL plugin, but this binary was "
        "built without NCCL support; point it at the MPI plugin or rebuild "
        "with NCCL.\n");
    }
    abortAllProcesses();
  }
#endif

  if (rank == 0) {
    // Report the selected cuPauliProp communication path.
#if defined(MPI_ENABLED)
    if (useNccl) {
      std::cout << "Execution mode: distributed, NCCL backend (MPI bootstrap)" << std::endl;
    } else {
      std::cout << "Execution mode: distributed, MPI backend" << std::endl;
    }
#else
    std::cout << "Execution mode: single-process" << std::endl;
#endif
    std::cout << "Number of ranks: " << numRanks << "\n" << std::endl;
  }


  // Sphinx: #7
  // ========================================================================
  // Decide memory usage
  // ========================================================================

  // Process-local GPU memory sizes, to be allocated by every process.
  size_t expansionPauliMem;
  size_t expansionCoefMem;
  size_t workspaceMem;
  size_t totalUsedMem;

  if (USE_MAX_VRAM) {
    size_t totalFreeMem, totalGlobalMem;
    HANDLE_CUDA_ERROR(cudaMemGetInfo(&totalFreeMem, &totalGlobalMem));
    size_t totalUsableMem = static_cast<size_t>(totalFreeMem * MAX_VRAM_PERCENT / 100);

    int32_t numPackedInts;
    HANDLE_CUPP_ERROR(cupaulipropGetNumPackedIntegers(NUM_CIRCUIT_QUBITS, &numPackedInts));
    size_t pauliMemPerTerm = 2 * numPackedInts * sizeof(cupaulipropPackedIntegerType_t);
    size_t coefMemPerTerm = sizeof(double);
    size_t totalMemPerTerm = pauliMemPerTerm + coefMemPerTerm;

    // Reserve approximately 20% for each expansion and 60% for workspace.
    // Distributed redistribution and deduplication require overlapping scratch
    // buffers, so they need a larger share than either local expansion.
    size_t perExpansionMem = totalUsableMem / 5;
    size_t expansionTermCapacity = perExpansionMem / totalMemPerTerm;
    expansionPauliMem = expansionTermCapacity * pauliMemPerTerm;
    expansionCoefMem  = expansionTermCapacity * coefMemPerTerm;
    workspaceMem =
      totalUsableMem - 2 * expansionPauliMem - 2 * expansionCoefMem;

    totalUsedMem = 2 * expansionPauliMem + 2 * expansionCoefMem + workspaceMem;
    if (rank == 0) {
      std::cout << "Dedicated memory per process: " << MAX_VRAM_PERCENT << "% of "
                << totalFreeMem << " B free = " << totalUsedMem << " B = "
                << totalUsedMem / (1LLU << 20) << " MiB, divided into..." << std::endl;
    }
  } else {
    expansionPauliMem = FIXED_EXPANSION_PAULI_MEM;
    expansionCoefMem  = FIXED_EXPANSION_COEF_MEM;
    workspaceMem      = FIXED_WORKSPACE_MEM;

    totalUsedMem = 2 * expansionPauliMem + 2 * expansionCoefMem + workspaceMem;
    if (rank == 0) {
      std::cout << "Dedicated memory per process: " << totalUsedMem << " B = "
                << totalUsedMem / (1LLU << 20) << " MiB, divided into..." << std::endl;
    }
  }

  if (rank == 0) {
    const size_t aggregateUsedMem = totalUsedMem * static_cast<size_t>(numRanks);
    std::cout << "  expansion Pauli buffer: " << expansionPauliMem << " B" << std::endl;
    std::cout << "  expansion coef buffer:  " << expansionCoefMem << " B" << std::endl;
    std::cout << "  workspace buffer:       " << workspaceMem << " B" << std::endl;
    std::cout << "Number of processes: " << numRanks << std::endl;
    std::cout << "Aggregate dedicated memory: " << aggregateUsedMem << " B = "
              << aggregateUsedMem / (1LLU << 20) << " MiB\n" << std::endl;
  }


  // Sphinx: #8
  // ========================================================================
  // Pauli expansion preparation
  // ========================================================================

  void* d_inExpansionPauliBuffer;
  void* d_outExpansionPauliBuffer;
  void* d_inExpansionCoefBuffer;
  void* d_outExpansionCoefBuffer;
  HANDLE_CUDA_ERROR(cudaMalloc(&d_inExpansionPauliBuffer,  expansionPauliMem));
  HANDLE_CUDA_ERROR(cudaMalloc(&d_inExpansionCoefBuffer,   expansionCoefMem));
  HANDLE_CUDA_ERROR(cudaMalloc(&d_outExpansionPauliBuffer, expansionPauliMem));
  HANDLE_CUDA_ERROR(cudaMalloc(&d_outExpansionCoefBuffer,  expansionCoefMem));

  if (rank == 0) {
    std::cout << "Observable: Z_62\n" << std::endl;
  }
  std::vector<cupaulipropPauliKind_t> observablePaulis = {CUPAULIPROP_PAULI_Z};
  std::vector<uint32_t> observableQubits = {62};
  double observableCoef = 1.0;

  int64_t numLocalTerms = 0;
  if (rank == 0) {
    numLocalTerms = 1;
    auto observablePackedInts =
      getPauliStringAsPackedIntegers(observablePaulis, observableQubits);
    size_t numObservableBytes =
      observablePackedInts.size() * sizeof(observablePackedInts[0]);
    HANDLE_CUDA_ERROR(cudaMemcpy(
      d_inExpansionPauliBuffer, observablePackedInts.data(),
      numObservableBytes, cudaMemcpyHostToDevice));
    HANDLE_CUDA_ERROR(cudaMemcpy(
      d_inExpansionCoefBuffer, &observableCoef,
      sizeof(observableCoef), cudaMemcpyHostToDevice));
  }

  cupaulipropPauliExpansion_t inExpansion;
  cupaulipropPauliExpansion_t outExpansion;
  cupaulipropSortOrder_t sortOrder = CUPAULIPROP_SORT_ORDER_NONE;
  int32_t hasDuplicates = 0;
  cudaDataType_t dataType = CUDA_R_64F;

  // Each call creates only this rank's local segment of the distributed
  // expansion. The buffers, their capacities, and the local term count may
  // differ between ranks; together, the local segments form the full expansion.
  HANDLE_CUPP_ERROR(cupaulipropCreatePauliExpansion(
    handle, NUM_CIRCUIT_QUBITS,
    d_inExpansionPauliBuffer, expansionPauliMem,
    d_inExpansionCoefBuffer, expansionCoefMem,
    dataType, numLocalTerms, sortOrder, hasDuplicates,
    &inExpansion));
  HANDLE_CUPP_ERROR(cupaulipropCreatePauliExpansion(
    handle, NUM_CIRCUIT_QUBITS,
    d_outExpansionPauliBuffer, expansionPauliMem,
    d_outExpansionCoefBuffer, expansionCoefMem,
    dataType, 0, CUPAULIPROP_SORT_ORDER_NONE, 0,
    &outExpansion));


  // Sphinx: #9
  // ========================================================================
  // Workspace preparation
  // ========================================================================

  cupaulipropWorkspaceDescriptor_t workspace;
  HANDLE_CUPP_ERROR(cupaulipropCreateWorkspaceDescriptor(handle, &workspace));

  void* d_workspaceBuffer;
  HANDLE_CUDA_ERROR(cudaMalloc(&d_workspaceBuffer, workspaceMem));
  HANDLE_CUPP_ERROR(cupaulipropWorkspaceSetMemory(
    handle, workspace,
    CUPAULIPROP_MEMSPACE_DEVICE, CUPAULIPROP_WORKSPACE_SCRATCH,
    d_workspaceBuffer, workspaceMem));


  // Sphinx: #10
  // ========================================================================
  // Truncation parameter preparation
  // ========================================================================

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

  const int numGatesBetweenTruncations = 10;

  if (rank == 0) {
    std::cout << "Coefficient truncation threshold:  "
              << coefTruncParams.cutoff << std::endl;
    std::cout << "Pauli weight truncation threshold: "
              << weightTruncParams.cutoff << std::endl;
    std::cout << "Truncation performed after every:  "
              << numGatesBetweenTruncations << " gates\n" << std::endl;
  }


  // Sphinx: #11
  // ========================================================================
  // Back-propagation of the observable through the circuit
  // ========================================================================

  double xRotationAngle = PI / 4.;
  int numTrotterSteps = 20;
  auto circuit = getIBMHeavyHexIsingCircuit(handle, xRotationAngle, numTrotterSteps);

  if (rank == 0) {
    std::cout << "Circuit: 127 qubit IBM heavy-hex Ising circuit, with..." << std::endl;
    std::cout << "  Trotter steps: " << numTrotterSteps << std::endl;
    std::cout << "  Total gates:   " << circuit.size() << std::endl;
    std::cout << "  Rx angle:      " << xRotationAngle << " (i.e. PI/4)\n" << std::endl;
  }

  uint32_t adjoint = true;
  sortOrder = CUPAULIPROP_SORT_ORDER_NONE;

  // Global deduplication, requested with keepDuplicates=false, can require
  // redistributing Pauli terms between processes. This can significantly
  // increase communication costs and therefore simulation runtime.
  int32_t keepDuplicates = false;

  if (rank == 0) {
    std::cout << "Imposed postconditions:" << std::endl;
    if (sortOrder != CUPAULIPROP_SORT_ORDER_NONE) {
      std::cout << "  Pauli strings will be sorted." << std::endl;
    }
    if (!keepDuplicates) {
      std::cout << "  Pauli strings will be unique." << std::endl;
    }
    if (keepDuplicates && sortOrder == CUPAULIPROP_SORT_ORDER_NONE) {
      std::cout << "No postconditions imposed on Pauli strings." << std::endl;
    }
    std::cout << std::endl;
  }

  // Exclude preceding asynchronous CUDA work and align all ranks before timing.
  HANDLE_CUDA_ERROR(cudaStreamSynchronize(stream));
#if defined(MPI_ENABLED)
  HANDLE_MPI_ERROR(MPI_Barrier(MPI_COMM_WORLD));
#endif
  auto startTime = std::chrono::steady_clock::now();
  int64_t maxNumLocalTerms = 0;

  for (int gateInd = circuit.size() - 1; gateInd >= 0; --gateInd) {
    cupaulipropQuantumOperator_t gate = circuit[gateInd];

    cupaulipropPauliExpansionView_t inView;
    int64_t currentNumLocalTerms;
    HANDLE_CUPP_ERROR(cupaulipropPauliExpansionGetNumTerms(
      handle, inExpansion, &currentNumLocalTerms));
    HANDLE_CUPP_ERROR(cupaulipropPauliExpansionGetContiguousRange(
      handle, inExpansion, 0, currentNumLocalTerms, &inView));

    if (currentNumLocalTerms > maxNumLocalTerms)
      maxNumLocalTerms = currentNumLocalTerms;

    bool applyTruncation = (gateInd % numGatesBetweenTruncations == 0);
    int32_t appliedNumTruncStrats = applyTruncation ? numTruncStrats : 0;
    const cupaulipropTruncationStrategy_t* appliedTruncStrats =
      applyTruncation ? truncStrats : nullptr;
      
    // In multi-process execution, Prepare returns process-local sizing with
    // statistical slack for potential inter-process term imbalances. Unlike
    // in single-process execution, passing workspaces and output expansions
    // of the sizes prescribed by Prepare to the subsequent Compute function
    // does not guarantee it will succeed, though the probability of failure
    // is very small, and vanishes for meaningfully large systems.
    int64_t reqExpansionPauliMem;
    int64_t reqExpansionCoefMem;
    int64_t reqWorkspaceMem;
    HANDLE_CUPP_ERROR(cupaulipropPauliExpansionViewPrepareOperatorApplication(
      handle, inView, gate, sortOrder, keepDuplicates,
      appliedNumTruncStrats, appliedTruncStrats,
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
      abortAllProcesses();
    }

    HANDLE_CUPP_ERROR(cupaulipropWorkspaceSetMemory(
      handle, workspace,
      CUPAULIPROP_MEMSPACE_DEVICE, CUPAULIPROP_WORKSPACE_SCRATCH,
      d_workspaceBuffer, workspaceMem));

    const auto operatorStatus =
      cupaulipropPauliExpansionViewComputeOperatorApplication(
        handle, inView, outExpansion, gate,
        adjoint, sortOrder, keepDuplicates,
        appliedNumTruncStrats, appliedTruncStrats,
        workspace, stream);
    if (operatorStatus == CUPAULIPROP_STATUS_INSUFFICIENT_OUT_EXPANSION
        && numRanks > 1 && rank == 0) {
      std::cerr
        << "The actual inter-process term distribution exceeded the "
           "probabilistic capacity reported by PrepareOperatorApplication. "
           "This outcome is extremely unlikely."
        << std::endl;
    }
    HANDLE_CUPP_ERROR(operatorStatus);

    HANDLE_CUPP_ERROR(cupaulipropDestroyPauliExpansionView(inView));
    std::swap(inExpansion, outExpansion);
  }

  std::swap(inExpansion, outExpansion);

  int64_t numOutTerms;
  HANDLE_CUPP_ERROR(cupaulipropPauliExpansionGetNumTerms(
    handle, outExpansion, &numOutTerms));
  if (numOutTerms > maxNumLocalTerms)
    maxNumLocalTerms = numOutTerms;

  // GetNumTerms reports a rank-local count. Use MPI to find the largest local
  // shard seen in either distributed mode; NCCL builds can also use this reduction
  // because MPI remains active for reporting. In single-process mode the local
  // maximum already is the global maximum.
#if defined(MPI_ENABLED)
  int64_t maxNumLocalTermsAcrossRanks = 0;
  HANDLE_MPI_ERROR(MPI_Allreduce(
    &maxNumLocalTerms, &maxNumLocalTermsAcrossRanks,
    1, MPI_INT64_T, MPI_MAX, MPI_COMM_WORLD));
  maxNumLocalTerms = maxNumLocalTermsAcrossRanks;
#endif


  // Sphinx: #12
  // ========================================================================
  // Evaluation of the expectation value of the observable
  // ========================================================================

  cupaulipropPauliExpansionView_t outView;
  HANDLE_CUPP_ERROR(cupaulipropPauliExpansionGetContiguousRange(
    handle, outExpansion, 0, numOutTerms, &outView));

  // Sum rank-local output sizes for display in either distributed mode. This
  // explicit MPI reduction is only for the statistic: cuPauliProp's trace call
  // below already combines distributed trace contributions through the selected
  // MPI or NCCL provider. No reduction is needed in single-process mode.
#if defined(MPI_ENABLED)
  int64_t totalNumOutTerms = 0;
  HANDLE_MPI_ERROR(MPI_Allreduce(
    &numOutTerms, &totalNumOutTerms, 1, MPI_INT64_T, MPI_SUM, MPI_COMM_WORLD));
  numOutTerms = totalNumOutTerms;
#endif

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
    abortAllProcesses();
  }

  HANDLE_CUPP_ERROR(cupaulipropWorkspaceSetMemory(
    handle, workspace,
    CUPAULIPROP_MEMSPACE_DEVICE, CUPAULIPROP_WORKSPACE_SCRATCH,
    d_workspaceBuffer, workspaceMem));

  double expecSignificand;
  double expecExponent;
  HANDLE_CUPP_ERROR(cupaulipropPauliExpansionViewComputeTraceWithZeroState(
    handle, outView, &expecSignificand, &expecExponent, workspace, stream));

  // Complete local GPU work and align all ranks before ending timing.
  HANDLE_CUDA_ERROR(cudaStreamSynchronize(stream));
#if defined(MPI_ENABLED)
  HANDLE_MPI_ERROR(MPI_Barrier(MPI_COMM_WORLD));
#endif
  auto endTime = std::chrono::steady_clock::now();
  double expec = expecSignificand * std::pow(2.0, expecExponent);

  auto duration =
    std::chrono::duration_cast<std::chrono::microseconds>(endTime - startTime);
  auto durationSecs = duration.count() / 1e6;

  if (rank == 0) {
    std::cout << "Expectation value:                  " << expec << std::endl;
    std::cout << "Final global number of terms:       " << numOutTerms << std::endl;
    std::cout << "Maximum local terms on any rank:    " << maxNumLocalTerms << std::endl;
    std::cout << "Runtime:                            "
              << durationSecs << " seconds\n" << std::endl;
  }


  // Sphinx: #13
  // ========================================================================
  // Clean up
  // ========================================================================

  HANDLE_CUPP_ERROR(cupaulipropDestroyPauliExpansionView(outView));

  for (auto& gate : circuit) {
    HANDLE_CUPP_ERROR(cupaulipropDestroyOperator(gate));
  }

  HANDLE_CUPP_ERROR(cupaulipropDestroyWorkspaceDescriptor(workspace));
  HANDLE_CUDA_ERROR(cudaFree(d_workspaceBuffer));

  HANDLE_CUPP_ERROR(cupaulipropDestroyPauliExpansion(inExpansion));
  HANDLE_CUPP_ERROR(cupaulipropDestroyPauliExpansion(outExpansion));

  HANDLE_CUDA_ERROR(cudaFree(d_inExpansionPauliBuffer));
  HANDLE_CUDA_ERROR(cudaFree(d_outExpansionPauliBuffer));
  HANDLE_CUDA_ERROR(cudaFree(d_inExpansionCoefBuffer));
  HANDLE_CUDA_ERROR(cudaFree(d_outExpansionCoefBuffer));

  HANDLE_CUPP_ERROR(cupaulipropDestroy(handle));

  // The provider communicator had to outlive the cuPauliProp handle. Destroy
  // whichever concrete communicator was created during setup, then finalize
  // the MPI process environment used by both distributed modes. The
  // single-process path owns neither resource and therefore has no
  // corresponding cleanup.
#if defined(NCCL_ENABLED)
  if (useNccl) {
    HANDLE_NCCL_ERROR(ncclCommDestroy(ncclCommunicator));
  }
#endif
#if defined(MPI_ENABLED)
  if (mpiCommunicator != MPI_COMM_NULL) {
    HANDLE_MPI_ERROR(MPI_Comm_free(&mpiCommunicator));
  }
  MPI_Finalize();
#endif

  return EXIT_SUCCESS;
}
