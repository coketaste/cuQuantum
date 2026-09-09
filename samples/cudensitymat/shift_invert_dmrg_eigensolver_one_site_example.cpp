/* Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
 *
 * SPDX-License-Identifier: BSD-3-Clause
 */

// One-site shift-invert DMRG for a three-site transverse-field Ising model.

#include <cudensitymat.h>
#include <cuda_runtime_api.h>

#include <array>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <iomanip>
#include <iostream>
#include <vector>

namespace {

#define HANDLE_CUDA_ERROR(call) checkCuda((call), __LINE__)
#define HANDLE_CUDM_ERROR(call) checkCudm((call), __LINE__)

void checkCuda(cudaError_t status, int line)
{
  if (status != cudaSuccess) {
    std::fprintf(
      stderr, "CUDA error at line %d: %s\n",
      line, cudaGetErrorString(status));
    std::abort();
  }
}

void checkCudm(cudensitymatStatus_t status, int line)
{
  if (status != CUDENSITYMAT_STATUS_SUCCESS) {
    std::fprintf(
      stderr, "cuDensityMat error %d at line %d\n",
      static_cast<int>(status), line);
    std::abort();
  }
}

void * copyToDevice(const std::vector<double> & values)
{
  void * devicePointer = nullptr;
  HANDLE_CUDA_ERROR(
    cudaMalloc(&devicePointer, values.size() * sizeof(double)));
  HANDLE_CUDA_ERROR(cudaMemcpy(
    devicePointer, values.data(), values.size() * sizeof(double),
    cudaMemcpyHostToDevice));
  return devicePointer;
}

constexpr cudaDataType_t DATA_TYPE = CUDA_R_64F;
constexpr int32_t NUM_MPS_SITES = 3;
constexpr int64_t MPO_BOND_EXTENT = 3;
constexpr double ISING_COUPLING = 0.7;
constexpr double TRANSVERSE_FIELD = 0.4;
// The reference level is d1=0.06143845150738322 from sigma; the next
// distinct distance is d2=0.25.
constexpr double TARGET_SIGMA = 0.15;
constexpr double EXPECTED_ENERGY = 0.21143845150738322;
constexpr double ENERGY_ERROR_TOLERANCE = 1e-8;
constexpr double NORM_TOLERANCE = 1e-10;
constexpr double PLUS_AMPLITUDE = 0.70710678118654752440;
constexpr double MPS_NOISE = 1e-2;
constexpr uint64_t MPS_SEED = 314159;
constexpr std::array<int64_t, NUM_MPS_SITES - 1>
  MPS_BOND_EXTENTS = {2, 2};
constexpr std::array<std::size_t, NUM_MPS_SITES>
  MPS_COMPONENT_SIZES = {
    4 * sizeof(double), 8 * sizeof(double), 4 * sizeof(double)};

class SplitMix64
{
public:
  explicit SplitMix64(uint64_t seed) : state_(seed) {}

  double signedUnit()
  {
    uint64_t value = (state_ += 0x9e3779b97f4a7c15ULL);
    value = (value ^ (value >> 30)) * 0xbf58476d1ce4e5b9ULL;
    value = (value ^ (value >> 27)) * 0x94d049bb133111ebULL;
    value ^= value >> 31;
    return 2.0 * static_cast<double>(value >> 11) * 0x1.0p-53 - 1.0;
  }

private:
  uint64_t state_;
};

double denseMPSNorm(
    const std::array<std::vector<double>, NUM_MPS_SITES> & tensors)
{
  double normSquared = 0.0;
  for (int64_t p0 = 0; p0 < 2; ++p0) {
    for (int64_t p1 = 0; p1 < 2; ++p1) {
      for (int64_t p2 = 0; p2 < 2; ++p2) {
        double amplitude = 0.0;
        for (int64_t b0 = 0; b0 < 2; ++b0) {
          for (int64_t b1 = 0; b1 < 2; ++b1) {
            amplitude +=
              tensors[0][p0 + 2 * b0]
              * tensors[1][b0 + 2 * (p1 + 2 * b1)]
              * tensors[2][b1 + 2 * p2];
          }
        }
        normSquared += amplitude * amplitude;
      }
    }
  }
  return std::sqrt(normSquared);
}

struct TFIMMPOStorage
{
  std::array<std::vector<double>, NUM_MPS_SITES> hostTensors;
  std::array<void *, NUM_MPS_SITES> devicePointers {};

  static double identity(int64_t bra, int64_t ket)
  {
    return bra == ket ? 1.0 : 0.0;
  }

  static double pauliX(int64_t bra, int64_t ket)
  {
    return bra != ket ? 1.0 : 0.0;
  }

  static double pauliZ(int64_t bra, int64_t ket)
  {
    if (bra != ket) return 0.0;
    return bra == 0 ? 1.0 : -1.0;
  }

  void build()
  {
    destroy();
    hostTensors = {
      std::vector<double>(12, 0.0),
      std::vector<double>(36, 0.0),
      std::vector<double>(12, 0.0)};

    // Standard bond-3 automaton in the documented Fortran mode order.
    for (int64_t bra = 0; bra < 2; ++bra) {
      for (int64_t ket = 0; ket < 2; ++ket) {
        const double i = identity(bra, ket);
        const double x = pauliX(bra, ket);
        const double z = pauliZ(bra, ket);

        // Site 0: [ket, right, bra].
        hostTensors[0][ket + 2 * (0 + 3 * bra)] =
          -TRANSVERSE_FIELD * x;
        hostTensors[0][ket + 2 * (1 + 3 * bra)] =
          -ISING_COUPLING * z;
        hostTensors[0][ket + 2 * (2 + 3 * bra)] = i;

        // Site 1: [left, ket, right, bra].
        hostTensors[1][0 + 3 * (ket + 2 * (0 + 3 * bra))] = i;
        hostTensors[1][1 + 3 * (ket + 2 * (0 + 3 * bra))] = z;
        hostTensors[1][2 + 3 * (ket + 2 * (0 + 3 * bra))] =
          -TRANSVERSE_FIELD * x;
        hostTensors[1][2 + 3 * (ket + 2 * (1 + 3 * bra))] =
          -ISING_COUPLING * z;
        hostTensors[1][2 + 3 * (ket + 2 * (2 + 3 * bra))] = i;

        // Site 2: [left, ket, bra].
        hostTensors[2][0 + 3 * (ket + 2 * bra)] = i;
        hostTensors[2][1 + 3 * (ket + 2 * bra)] = z;
        hostTensors[2][2 + 3 * (ket + 2 * bra)] =
          -TRANSVERSE_FIELD * x;
      }
    }

    for (int32_t site = 0; site < NUM_MPS_SITES; ++site) {
      devicePointers[site] = copyToDevice(hostTensors[site]);
    }
  }

  void destroy()
  {
    for (auto & pointer : devicePointers) {
      if (pointer != nullptr) {
        HANDLE_CUDA_ERROR(cudaFree(pointer));
        pointer = nullptr;
      }
    }
  }
};

struct NoisyPlusMPSStorage
{
  std::array<std::vector<double>, NUM_MPS_SITES> hostTensors;
  std::array<void *, NUM_MPS_SITES> devicePointers {};

  void build()
  {
    destroy();
    hostTensors = {
      std::vector<double>(4),
      std::vector<double>(8),
      std::vector<double>(4)};
    SplitMix64 generator(MPS_SEED);
    for (auto & tensor : hostTensors) {
      for (auto & value : tensor) {
        value = MPS_NOISE * generator.signedUnit();
      }
    }

    // Add |+>^3 on bond channel zero; noise supplies every symmetry sector.
    for (int64_t physical = 0; physical < 2; ++physical) {
      hostTensors[0][physical + 2 * 0] += PLUS_AMPLITUDE;
      hostTensors[1][0 + 2 * (physical + 2 * 0)] += PLUS_AMPLITUDE;
      hostTensors[2][0 + 2 * physical] += PLUS_AMPLITUDE;
    }

    const double initialNorm = denseMPSNorm(hostTensors);
    if (!std::isfinite(initialNorm) || initialNorm <= 0.0) {
      std::fprintf(stderr, "Invalid initial MPS norm\n");
      std::abort();
    }
    for (auto & value : hostTensors[0]) {
      value /= initialNorm;
    }

    for (int32_t site = 0; site < NUM_MPS_SITES; ++site) {
      devicePointers[site] = copyToDevice(hostTensors[site]);
    }
  }

  void download()
  {
    for (int32_t site = 0; site < NUM_MPS_SITES; ++site) {
      HANDLE_CUDA_ERROR(cudaMemcpy(
        hostTensors[site].data(), devicePointers[site],
        MPS_COMPONENT_SIZES[site], cudaMemcpyDeviceToHost));
    }
  }

  void destroy()
  {
    for (auto & pointer : devicePointers) {
      if (pointer != nullptr) {
        HANDLE_CUDA_ERROR(cudaFree(pointer));
        pointer = nullptr;
      }
    }
  }
};

constexpr int32_t NUM_SITES = 1;
constexpr int32_t MAX_POWER_ITERATIONS = 40;
constexpr int32_t MAX_SWEEPS = 20;
constexpr int32_t LINEAR_MAX_ITERATIONS = 400;
constexpr double ENERGY_TOLERANCE = 1e-12;
constexpr double LOCAL_TOLERANCE = 1e-12;

} // namespace

int main()
{
  HANDLE_CUDA_ERROR(cudaSetDevice(0));

  cudensitymatHandle_t handle = nullptr;
  HANDLE_CUDM_ERROR(cudensitymatCreate(&handle));

  std::cout
    << "One-site shift-invert DMRG eigensolver example\n"
    << "  Initial MPS: deterministic noisy |+>^3, bonds {2, 2}\n"
    << "  DMRG NUM_SITES=" << NUM_SITES
    << ", MAX_POWER_ITERATIONS=" << MAX_POWER_ITERATIONS
    << ", MAX_SWEEPS=" << MAX_SWEEPS << "\n"
    << "  Linear MAX_ITERATIONS=" << LINEAR_MAX_ITERATIONS
    << ", requested local tolerance=" << LOCAL_TOLERANCE << "\n";

  // Build the model MPO and wrap it in a Hamiltonian operator.
  TFIMMPOStorage mpoStorage;
  mpoStorage.build();
  const std::array<int64_t, NUM_MPS_SITES> spaceShape = {2, 2, 2};
  const std::array<int64_t, NUM_MPS_SITES - 1>
    mpoBondExtents = {MPO_BOND_EXTENT, MPO_BOND_EXTENT};
  std::array<cudensitymatWrappedTensorCallback_t, NUM_MPS_SITES>
    mpoCallbacks = {
      cudensitymatTensorCallbackNone,
      cudensitymatTensorCallbackNone,
      cudensitymatTensorCallbackNone};
  std::array<
    cudensitymatWrappedTensorGradientCallback_t, NUM_MPS_SITES>
    mpoGradientCallbacks = {
      cudensitymatTensorGradientCallbackNone,
      cudensitymatTensorGradientCallbackNone,
      cudensitymatTensorGradientCallbackNone};

  cudensitymatMatrixProductOperator_t mpo = nullptr;
  HANDLE_CUDM_ERROR(cudensitymatCreateMatrixProductOperator(
    handle, NUM_MPS_SITES, spaceShape.data(),
    CUDENSITYMAT_BOUNDARY_CONDITION_OPEN, mpoBondExtents.data(),
    DATA_TYPE, mpoStorage.devicePointers.data(), mpoCallbacks.data(),
    mpoGradientCallbacks.data(), &mpo));

  cudensitymatOperatorTerm_t operatorTerm = nullptr;
  HANDLE_CUDM_ERROR(cudensitymatCreateOperatorTerm(
    handle, NUM_MPS_SITES, spaceShape.data(), &operatorTerm));
  std::array<int32_t, NUM_MPS_SITES> modesActedOn = {0, 1, 2};
  std::array<int32_t, NUM_MPS_SITES> modeDuality = {0, 0, 0};
  const int32_t mpoConjugation = 0;
  HANDLE_CUDM_ERROR(cudensitymatOperatorTermAppendMPOProduct(
    handle, operatorTerm, 1, &mpo, &mpoConjugation,
    modesActedOn.data(), modeDuality.data(),
    make_cuDoubleComplex(1.0, 0.0),
    cudensitymatScalarCallbackNone,
    cudensitymatScalarGradientCallbackNone));

  cudensitymatOperator_t hamiltonian = nullptr;
  HANDLE_CUDM_ERROR(cudensitymatCreateOperator(
    handle, NUM_MPS_SITES, spaceShape.data(), &hamiltonian));
  HANDLE_CUDM_ERROR(cudensitymatOperatorAppendTerm(
    handle, hamiltonian, operatorTerm, 0,
    make_cuDoubleComplex(1.0, 0.0),
    cudensitymatScalarCallbackNone,
    cudensitymatScalarGradientCallbackNone));

  // Create and attach the fixed-capacity, normalized noisy MPS.
  cudensitymatState_t mpsState = nullptr;
  HANDLE_CUDM_ERROR(cudensitymatCreateStateMPS(
    handle, CUDENSITYMAT_STATE_PURITY_PURE, NUM_MPS_SITES,
    spaceShape.data(), CUDENSITYMAT_BOUNDARY_CONDITION_OPEN,
    MPS_BOND_EXTENTS.data(), DATA_TYPE, 1, &mpsState));

  NoisyPlusMPSStorage mpsStorage;
  mpsStorage.build();
  HANDLE_CUDM_ERROR(cudensitymatStateAttachComponentStorage(
    handle, mpsState, NUM_MPS_SITES,
    mpsStorage.devicePointers.data(), MPS_COMPONENT_SIZES.data()));

  // Configure shift-invert DMRG and its local linear solver.
  cudensitymatEigenDecomposition_t eigen = nullptr;
  HANDLE_CUDM_ERROR(cudensitymatCreateEigenDecomposition(
    handle, hamiltonian, 1, CUDENSITYMAT_EIGEN_SPECTRUM_SMALLEST,
    CUDENSITYMAT_EIGEN_SCOPE_SPLIT,
    CUDENSITYMAT_EIGEN_APPROACH_LINEAR, &eigen));

  const int32_t splitKind =
    CUDENSITYMAT_EIGEN_SCOPE_SPLIT_SHIFT_INVERT_DMRG;
  HANDLE_CUDM_ERROR(cudensitymatEigenDecompositionConfigure(
    handle, eigen, CUDENSITYMAT_EIGEN_SPLIT_SCOPE_KIND,
    &splitKind, sizeof(splitKind)));

  cudensitymatEigenDecompositionScopeSplitDMRGConfig_t dmrgConfig =
    nullptr;
  HANDLE_CUDM_ERROR(
    cudensitymatCreateEigenDecompositionScopeSplitDMRGConfig(
      handle, &dmrgConfig));
  HANDLE_CUDM_ERROR(
    cudensitymatEigenDecompositionScopeSplitDMRGConfigSetAttribute(
      handle, dmrgConfig,
      CUDENSITYMAT_EIGEN_SPLIT_SCOPE_DMRG_NUM_SITES,
      &NUM_SITES, sizeof(NUM_SITES)));
  HANDLE_CUDM_ERROR(
    cudensitymatEigenDecompositionScopeSplitDMRGConfigSetAttribute(
      handle, dmrgConfig,
      CUDENSITYMAT_EIGEN_SPLIT_SCOPE_DMRG_MAX_POWER_ITERATIONS,
      &MAX_POWER_ITERATIONS, sizeof(MAX_POWER_ITERATIONS)));
  HANDLE_CUDM_ERROR(
    cudensitymatEigenDecompositionScopeSplitDMRGConfigSetAttribute(
      handle, dmrgConfig,
      CUDENSITYMAT_EIGEN_SPLIT_SCOPE_DMRG_MAX_SWEEPS,
      &MAX_SWEEPS, sizeof(MAX_SWEEPS)));
  HANDLE_CUDM_ERROR(
    cudensitymatEigenDecompositionScopeSplitDMRGConfigSetAttribute(
      handle, dmrgConfig,
      CUDENSITYMAT_EIGEN_SPLIT_SCOPE_DMRG_ENERGY_TOLERANCE,
      &ENERGY_TOLERANCE, sizeof(ENERGY_TOLERANCE)));
  HANDLE_CUDM_ERROR(cudensitymatEigenDecompositionConfigure(
    handle, eigen, CUDENSITYMAT_EIGEN_SPLIT_SCOPE_DMRG_CONFIG,
    &dmrgConfig, sizeof(dmrgConfig)));
  HANDLE_CUDM_ERROR(
    cudensitymatDestroyEigenDecompositionScopeSplitDMRGConfig(
      dmrgConfig));

  cudensitymatEigenDecompositionApproachLinearConfig_t linearConfig =
    nullptr;
  HANDLE_CUDM_ERROR(
    cudensitymatCreateEigenDecompositionApproachLinearConfig(
      handle, &linearConfig));
  HANDLE_CUDM_ERROR(
    cudensitymatEigenDecompositionApproachLinearConfigSetAttribute(
      handle, linearConfig,
      CUDENSITYMAT_EIGEN_APPROACH_LINEAR_MAX_ITERATIONS,
      &LINEAR_MAX_ITERATIONS, sizeof(LINEAR_MAX_ITERATIONS)));
  HANDLE_CUDM_ERROR(cudensitymatEigenDecompositionConfigure(
    handle, eigen, CUDENSITYMAT_EIGEN_APPROACH_LINEAR_CONFIG,
    &linearConfig, sizeof(linearConfig)));
  HANDLE_CUDM_ERROR(
    cudensitymatDestroyEigenDecompositionApproachLinearConfig(
      linearConfig));

  // Prepare the computation and attach the requested caller-owned workspace.
  cudensitymatWorkspaceDescriptor_t workspace = nullptr;
  HANDLE_CUDM_ERROR(cudensitymatCreateWorkspace(handle, &workspace));
  std::size_t freeMemory = 0;
  std::size_t totalMemory = 0;
  HANDLE_CUDA_ERROR(cudaMemGetInfo(&freeMemory, &totalMemory));
  const std::size_t workspaceLimit =
    static_cast<std::size_t>(0.8 * static_cast<double>(freeMemory));
  HANDLE_CUDM_ERROR(cudensitymatEigenDecompositionPrepare(
    handle, eigen, 1, mpsState, CUDENSITYMAT_COMPUTE_64F,
    workspaceLimit, workspace, nullptr));

  std::size_t scratchSize = 0;
  HANDLE_CUDM_ERROR(cudensitymatWorkspaceGetMemorySize(
    handle, workspace, CUDENSITYMAT_MEMSPACE_DEVICE,
    CUDENSITYMAT_WORKSPACE_SCRATCH, &scratchSize));
  void * scratchBuffer = nullptr;
  if (scratchSize > 0) {
    HANDLE_CUDA_ERROR(cudaMalloc(&scratchBuffer, scratchSize));
    HANDLE_CUDM_ERROR(cudensitymatWorkspaceSetMemory(
      handle, workspace, CUDENSITYMAT_MEMSPACE_DEVICE,
      CUDENSITYMAT_WORKSPACE_SCRATCH,
      scratchBuffer, scratchSize));
  }
  std::cout << "  Attached scratch workspace: "
            << scratchSize << " bytes\n";

  const std::vector<double> sigmaInput = {TARGET_SIGMA};
  void * eigenvalueBuffer = copyToDevice(sigmaInput);
  double achievedLocalResidual = LOCAL_TOLERANCE;
  cudensitymatState_t eigenstates[] = {mpsState};
  HANDLE_CUDM_ERROR(cudensitymatEigenDecompositionCompute(
    handle, eigen, 0.0, 1, 0, nullptr, 1, eigenstates,
    eigenvalueBuffer, &achievedLocalResidual, workspace, nullptr));
  HANDLE_CUDA_ERROR(cudaDeviceSynchronize());

  double returnedEnergy = 0.0;
  HANDLE_CUDA_ERROR(cudaMemcpy(
    &returnedEnergy, eigenvalueBuffer, sizeof(returnedEnergy),
    cudaMemcpyDeviceToHost));

  // Verify the returned MPS with an independent dense contraction.
  mpsStorage.download();
  const double representedMPSNorm = denseMPSNorm(mpsStorage.hostTensors);
  const double energyError =
    std::abs(returnedEnergy - EXPECTED_ENERGY);
  const bool validOutput =
    std::isfinite(returnedEnergy)
    && std::isfinite(achievedLocalResidual)
    && achievedLocalResidual >= 0.0
    && std::isfinite(representedMPSNorm)
    && std::abs(representedMPSNorm - 1.0) <= NORM_TOLERANCE
    && std::isfinite(energyError)
    && energyError < ENERGY_ERROR_TOLERANCE;

  std::cout << std::scientific << std::setprecision(12)
            << "\nModel                                        : "
            << "N=3 open-boundary TFIM\n"
            << "J                                            : "
            << ISING_COUPLING << "\n"
            << "h                                            : "
            << TRANSVERSE_FIELD << "\n"
            << "Target sigma                                 : "
            << TARGET_SIGMA << "\n"
            << "Returned energy                              : "
            << returnedEnergy << "\n"
            << "Achieved local residual                      : "
            << achievedLocalResidual << "\n"
            << "Independently computed MPS norm              : "
            << representedMPSNorm << "\n"
            << "Energy error |E - E_ref|                     : "
            << energyError << "\n"
            << "Result                                        : "
            << (validOutput ? "PASS" : "FAIL") << "\n";

  // Destroy API objects and caller-owned allocations in reverse order.
  HANDLE_CUDA_ERROR(cudaFree(eigenvalueBuffer));
  if (scratchBuffer != nullptr) {
    HANDLE_CUDA_ERROR(cudaFree(scratchBuffer));
  }
  HANDLE_CUDM_ERROR(cudensitymatDestroyWorkspace(workspace));
  HANDLE_CUDM_ERROR(cudensitymatDestroyEigenDecomposition(eigen));
  HANDLE_CUDM_ERROR(cudensitymatDestroyState(mpsState));
  HANDLE_CUDM_ERROR(cudensitymatDestroyOperator(hamiltonian));
  HANDLE_CUDM_ERROR(cudensitymatDestroyOperatorTerm(operatorTerm));
  HANDLE_CUDM_ERROR(cudensitymatDestroyMatrixProductOperator(mpo));
  mpsStorage.destroy();
  mpoStorage.destroy();
  HANDLE_CUDM_ERROR(cudensitymatDestroy(handle));
  HANDLE_CUDA_ERROR(cudaDeviceReset());

  return validOutput ? 0 : 1;
}
