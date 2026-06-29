/* Copyright (c) 2024-2026, NVIDIA CORPORATION & AFFILIATES.
 *
 * SPDX-License-Identifier: BSD-3-Clause
 */

// DMRG eigensolver example.
//
// Computes the ground state of the 1-D spin-1/2 antiferromagnetic Heisenberg
// chain  H = J * sum_i ( Sx_i Sx_{i+1} + Sy_i Sy_{i+1} + Sz_i Sz_{i+1} )  with
// the 1-site DMRG solver (SCOPE_SPLIT + APPROACH_KRYLOV + SMALLEST_REAL) on an
// MPS state.  The exact N=8 open-boundary energy (J=1) is E0 = -3.374932598688;
// the example checks the DMRG result against it.
//
// Workflow:
//  1. Build the Heisenberg MPO (bond dim 5)
//  2. Create an initial MPS state (noisy Neel product)
//  3. Create the EigenDecomposition (SMALLEST_REAL + SCOPE_SPLIT + KRYLOV)
//  4. Configure the DMRG and Krylov sub-configs
//  5. Prepare + attach workspace + Compute
//  6. Report the ground-state energy and residual vs the exact value
//  7. Clean up

#include <cudensitymat.h>
#include "helpers.h"

#include <algorithm>
#include <cmath>
#include <complex>
#include <vector>
#include <numeric>
#include <random>
#include <iostream>
#include <iomanip>
#include <cassert>


using Complex = std::complex<double>;
constexpr cudaDataType_t kDataType = CUDA_C_64F;

constexpr bool verbose = true;

// --- Simulation parameters ---
constexpr int32_t  NUM_SITES  = 8;
constexpr int64_t  PHYS_DIM   = 2;     // spin-1/2
constexpr int64_t  MPO_BOND   = 5;     // Heisenberg MPO bond dim
constexpr int64_t  MAX_BOND   = 16;    // MPS bond-dim cap (= full Schmidt rank at N=8)
constexpr int32_t  MAX_SWEEPS = 50;
constexpr int32_t  KRYLOV_MAX_DIM = 3; // local Krylov subspace cap (boundary-dim constraint)
constexpr double   ENERGY_TOL = 1e-10;
constexpr double   RESIDUAL   = 1e-6;
constexpr double   NOISE_EPS  = 1e-2;
constexpr uint64_t RNG_SEED   = 1729;

constexpr double   J_COUPLING = 1.0;   // antiferromagnetic

// Exact ground-state energy for the N=8 open-boundary Heisenberg chain (J=1),
// from dense diagonalization of the 2^8 x 2^8 Hamiltonian.  Printed reference.
constexpr double   E0_EXACT   = -3.374932598688;


// ============================================================================
// Heisenberg MPO builder  (spin-1/2, bond dimension 5)
// ============================================================================
//
// Standard W-matrix: column 0 holds the finish operators (I, Sx, Sy, Sz), the
// bottom row the start operators (J*Sx, J*Sy, J*Sz, I); a term J*S^a_i S^a_{i+1}
// starts at site i and finishes at i+1, with no on-site term.  Left boundary is
// the bottom row, right boundary is column 0.
// Site tensor layout (b_L, d, b_R, d): index = aL + bL*(ket + d*(aR + bR*bra)).

struct HeisenbergMPO {

  std::vector<std::vector<Complex>> hostTensors;
  std::vector<void*> gpuPtrs;

  // d x d spin-1/2 operators, entry (bra, ket) = <bra | S | ket>.
  static Complex I_op(int bra, int ket)  { return (bra == ket) ? Complex(1, 0) : Complex(0, 0); }
  static Complex Sx_op(int bra, int ket) { return (bra != ket) ? Complex(0.5, 0) : Complex(0, 0); }
  static Complex Sy_op(int bra, int ket) {
    if (bra == 0 && ket == 1) return Complex(0, -0.5);   // -i/2
    if (bra == 1 && ket == 0) return Complex(0,  0.5);   // +i/2
    return Complex(0, 0);
  }
  static Complex Sz_op(int bra, int ket) {
    if (bra == ket) return Complex(bra == 0 ? 0.5 : -0.5, 0);
    return Complex(0, 0);
  }

  void build() {
    const Complex zero{0.0, 0.0};
    hostTensors.assign(NUM_SITES, {});
    gpuPtrs.assign(NUM_SITES, nullptr);

    auto idx_full = [&](int64_t bL, int64_t bR, int64_t aL, int64_t ket,
                        int64_t aR, int64_t bra) {
      return aL + bL * (ket + PHYS_DIM * (aR + bR * bra));
    };
    // Place a d x d block at MPO corner (aL, aR); M is a callable (bra, ket).
    auto place = [&](std::vector<Complex>& T, int64_t bL, int64_t bR,
                     int64_t aL, int64_t aR, auto M) {
      for (int bra = 0; bra < PHYS_DIM; ++bra)
        for (int ket = 0; ket < PHYS_DIM; ++ket)
          T[idx_full(bL, bR, aL, ket, aR, bra)] += M(bra, ket);
    };

    const double J = J_COUPLING;
    for (int32_t site = 0; site < NUM_SITES; ++site) {
      const int64_t bL  = (site == 0) ? 1 : MPO_BOND;
      const int64_t bR  = (site == NUM_SITES - 1) ? 1 : MPO_BOND;
      const int64_t vol = bL * PHYS_DIM * bR * PHYS_DIM;
      hostTensors[site].assign(vol, zero);
      auto& T = hostTensors[site];

      if (site == 0) {
        // Bottom row of W:  [0, J*Sx, J*Sy, J*Sz, I],  aL = 0.
        place(T, bL, bR, 0, 1, [&](int b, int k){ return Complex(J, 0) * Sx_op(b, k); });
        place(T, bL, bR, 0, 2, [&](int b, int k){ return Complex(J, 0) * Sy_op(b, k); });
        place(T, bL, bR, 0, 3, [&](int b, int k){ return Complex(J, 0) * Sz_op(b, k); });
        place(T, bL, bR, 0, 4, [&](int b, int k){ return I_op(b, k); });
      } else if (site == NUM_SITES - 1) {
        // First column of W:  [I, Sx, Sy, Sz, 0]^T,  aR = 0.
        place(T, bL, bR, 0, 0, [&](int b, int k){ return I_op(b, k); });
        place(T, bL, bR, 1, 0, [&](int b, int k){ return Sx_op(b, k); });
        place(T, bL, bR, 2, 0, [&](int b, int k){ return Sy_op(b, k); });
        place(T, bL, bR, 3, 0, [&](int b, int k){ return Sz_op(b, k); });
      } else {
        // Full bulk W: first column (finish) + bottom row (start).
        place(T, bL, bR, 0, 0, [&](int b, int k){ return I_op(b, k); });
        place(T, bL, bR, 1, 0, [&](int b, int k){ return Sx_op(b, k); });
        place(T, bL, bR, 2, 0, [&](int b, int k){ return Sy_op(b, k); });
        place(T, bL, bR, 3, 0, [&](int b, int k){ return Sz_op(b, k); });
        place(T, bL, bR, 4, 1, [&](int b, int k){ return Complex(J, 0) * Sx_op(b, k); });
        place(T, bL, bR, 4, 2, [&](int b, int k){ return Complex(J, 0) * Sy_op(b, k); });
        place(T, bL, bR, 4, 3, [&](int b, int k){ return Complex(J, 0) * Sz_op(b, k); });
        place(T, bL, bR, 4, 4, [&](int b, int k){ return I_op(b, k); });
      }

      gpuPtrs[site] = createInitializeArrayGPU(hostTensors[site]);
    }
  }

  void destroy() {
    for (auto & ptr : gpuPtrs) {
      if (ptr) { destroyArrayGPU(ptr); ptr = nullptr; }
    }
  }
};


// ============================================================================
// Noisy Neel initial MPS  ( |up,down,up,...> + epsilon * Gaussian noise )
// ============================================================================
//
// Neel lies in the total-Sz = 0 sector of the singlet ground state; the noise
// gives DMRG a generic (non-exact) starting point.

struct NoisyNeelMPS {

  std::vector<std::vector<Complex>> hostTensors;
  std::vector<void*> gpuPtrs;

  void build(const std::vector<int64_t>& bondDims, double epsilon, uint64_t seed) {
    const Complex zero{0.0, 0.0}, one{1.0, 0.0};
    std::mt19937_64 rng(seed);
    std::normal_distribution<double> noise(0.0, epsilon);

    hostTensors.assign(NUM_SITES, {});
    gpuPtrs.assign(NUM_SITES, nullptr);

    for (int32_t site = 0; site < NUM_SITES; ++site) {
      const int64_t bL  = (site == 0) ? 1 : bondDims[site - 1];
      const int64_t bR  = (site == NUM_SITES - 1) ? 1 : bondDims[site];
      const int64_t vol = bL * PHYS_DIM * bR;
      hostTensors[site].assign(vol, zero);

      auto idx = [&](int64_t aL, int64_t sigma, int64_t aR) {
        return aL + bL * (sigma + PHYS_DIM * aR);
      };

      // Neel seed: site i occupies |i mod 2>  (alternating up / down).
      const int64_t spin = site % 2;
      hostTensors[site][idx(0, spin, 0)] = one;
      // Per-element Gaussian noise (Re, Im) iid N(0, epsilon^2).
      for (auto & v : hostTensors[site]) {
        v += Complex(noise(rng), noise(rng));
      }

      gpuPtrs[site] = createInitializeArrayGPU(hostTensors[site]);
    }
  }

  void destroy() {
    for (auto & ptr : gpuPtrs) {
      if (ptr) { destroyArrayGPU(ptr); ptr = nullptr; }
    }
  }
};


// ============================================================================
// Example workflow
// ============================================================================

void exampleWorkflow(cudensitymatHandle_t handle)
{
  if (verbose) {
    std::cout << "DMRG eigensolver example\n"
              << "  Model:           spin-1/2 antiferromagnetic Heisenberg chain\n"
              << "  H = J * sum_i ( Sx_i Sx_{i+1} + Sy_i Sy_{i+1} + Sz_i Sz_{i+1} )\n"
              << "  N (sites)       = " << NUM_SITES  << "\n"
              << "  d (phys dim)    = " << PHYS_DIM   << "\n"
              << "  J (coupling)    = " << J_COUPLING << "\n"
              << "  chi (bond cap)  = " << MAX_BOND   << "\n"
              << "  max sweeps      = " << MAX_SWEEPS << "\n"
              << "  energy tol      = " << ENERGY_TOL << "\n\n";
  }

  // --- 1. Build the Heisenberg MPO ---
  HeisenbergMPO mpo;
  mpo.build();
  if (verbose) std::cout << "Built Heisenberg MPO (bond dim " << MPO_BOND << ")\n";

  const std::vector<int64_t> spaceShape(NUM_SITES, PHYS_DIM);
  std::vector<int64_t> mpoBondDims(NUM_SITES - 1, MPO_BOND);

  cudensitymatMatrixProductOperator_t mpoHandle = nullptr;
  std::vector<cudensitymatWrappedTensorCallback_t> mpoCB(
      NUM_SITES, cudensitymatTensorCallbackNone);
  std::vector<cudensitymatWrappedTensorGradientCallback_t> mpoGCB(
      NUM_SITES, cudensitymatTensorGradientCallbackNone);
  HANDLE_CUDM_ERROR(cudensitymatCreateMatrixProductOperator(handle,
                      NUM_SITES,
                      spaceShape.data(),
                      CUDENSITYMAT_BOUNDARY_CONDITION_OPEN,
                      mpoBondDims.data(),
                      kDataType,
                      mpo.gpuPtrs.data(),
                      mpoCB.data(),
                      mpoGCB.data(),
                      &mpoHandle));
  if (verbose) std::cout << "Created MPO handle\n";

  // --- 2. Wrap the MPO into an Operator (one term, one MPO product) ---
  cudensitymatOperatorTerm_t operatorTerm = nullptr;
  HANDLE_CUDM_ERROR(cudensitymatCreateOperatorTerm(handle,
                      NUM_SITES, spaceShape.data(), &operatorTerm));

  std::vector<int32_t> modesActedOn(NUM_SITES);
  std::iota(modesActedOn.begin(), modesActedOn.end(), 0);
  std::vector<int32_t> modeDuality(NUM_SITES, 0);
  std::vector<int32_t> mpoConj = {0};
  HANDLE_CUDM_ERROR(cudensitymatOperatorTermAppendMPOProduct(handle,
                      operatorTerm,
                      1, &mpoHandle, mpoConj.data(),
                      modesActedOn.data(), modeDuality.data(),
                      make_cuDoubleComplex(1.0, 0.0),
                      cudensitymatScalarCallbackNone,
                      cudensitymatScalarGradientCallbackNone));

  cudensitymatOperator_t hamiltonian = nullptr;
  HANDLE_CUDM_ERROR(cudensitymatCreateOperator(handle,
                      NUM_SITES, spaceShape.data(), &hamiltonian));
  HANDLE_CUDM_ERROR(cudensitymatOperatorAppendTerm(handle,
                      hamiltonian, operatorTerm, 0,
                      make_cuDoubleComplex(1.0, 0.0),
                      cudensitymatScalarCallbackNone,
                      cudensitymatScalarGradientCallbackNone));
  if (verbose) std::cout << "Created Hamiltonian operator from MPO\n";

  // --- 3. Build the initial MPS state ---
  //
  // Bond dims are capped at the natural Schmidt-rank bound min(d^i, d^(N-i)).
  std::vector<int64_t> mpsBondDims(NUM_SITES - 1);
  for (int32_t i = 0; i < NUM_SITES - 1; ++i) {
    int64_t leftDim = 1;
    for (int32_t j = 0; j <= i; ++j) leftDim *= spaceShape[j];
    int64_t rightDim = 1;
    for (int32_t j = i + 1; j < NUM_SITES; ++j) rightDim *= spaceShape[j];
    mpsBondDims[i] = std::min<int64_t>({MAX_BOND, leftDim, rightDim});
  }

  cudensitymatState_t mpsState = nullptr;
  HANDLE_CUDM_ERROR(cudensitymatCreateStateMPS(handle,
                      CUDENSITYMAT_STATE_PURITY_PURE,
                      NUM_SITES,
                      spaceShape.data(),
                      CUDENSITYMAT_BOUNDARY_CONDITION_OPEN,
                      mpsBondDims.data(),
                      kDataType,
                      /*batchSize=*/1,
                      &mpsState));

  NoisyNeelMPS mps;
  mps.build(mpsBondDims, NOISE_EPS, RNG_SEED);

  int32_t numComponents = 0;
  HANDLE_CUDM_ERROR(cudensitymatStateGetNumComponents(handle, mpsState, &numComponents));
  std::vector<std::size_t> componentSizes(numComponents);
  HANDLE_CUDM_ERROR(cudensitymatStateGetComponentStorageSize(handle, mpsState,
                      numComponents, componentSizes.data()));
  HANDLE_CUDM_ERROR(cudensitymatStateAttachComponentStorage(handle, mpsState,
                      numComponents, mps.gpuPtrs.data(), componentSizes.data()));
  if (verbose) std::cout << "Built and attached noisy-Neel MPS (alternating up/down + eps*Gaussian, seed="
                         << RNG_SEED << ")\n";

  // --- 4. Create EigenDecomposition (SCOPE_SPLIT + APPROACH_KRYLOV + SMALLEST_REAL) ---
  cudensitymatEigenDecomposition_t eigen = nullptr;
  HANDLE_CUDM_ERROR(cudensitymatCreateEigenDecomposition(handle,
                      hamiltonian,
                      /*isHermitian=*/1,
                      CUDENSITYMAT_EIGEN_SPECTRUM_SMALLEST_REAL,
                      CUDENSITYMAT_EIGEN_SCOPE_SPLIT,
                      CUDENSITYMAT_EIGEN_APPROACH_KRYLOV,
                      &eigen));
  if (verbose) std::cout << "Created EigenDecomposition object\n";

  // --- 5. Configure DMRG attributes (MAX_SWEEPS, ENERGY_TOLERANCE).
  // Sub-config values are deep-copied at Configure, so the handle can be freed now.
  cudensitymatEigenDecompositionScopeSplitDMRGConfig_t dmrgCfg = nullptr;
  HANDLE_CUDM_ERROR(cudensitymatCreateEigenDecompositionScopeSplitDMRGConfig(handle, &dmrgCfg));
  {
    const int32_t maxSweeps = MAX_SWEEPS;
    HANDLE_CUDM_ERROR(cudensitymatEigenDecompositionScopeSplitDMRGConfigSetAttribute(
        handle, dmrgCfg,
        CUDENSITYMAT_EIGEN_SPLIT_SCOPE_DMRG_MAX_SWEEPS,
        &maxSweeps, sizeof(maxSweeps)));
    const double energyTol = ENERGY_TOL;
    HANDLE_CUDM_ERROR(cudensitymatEigenDecompositionScopeSplitDMRGConfigSetAttribute(
        handle, dmrgCfg,
        CUDENSITYMAT_EIGEN_SPLIT_SCOPE_DMRG_ENERGY_TOLERANCE,
        &energyTol, sizeof(energyTol)));
  }
  HANDLE_CUDM_ERROR(cudensitymatEigenDecompositionConfigure(handle, eigen,
                      CUDENSITYMAT_EIGEN_SPLIT_SCOPE_DMRG_CONFIG,
                      &dmrgCfg, sizeof(dmrgCfg)));
  HANDLE_CUDM_ERROR(cudensitymatDestroyEigenDecompositionScopeSplitDMRGConfig(dmrgCfg));
  dmrgCfg = nullptr;
  if (verbose) std::cout << "Configured DMRG sub-config (max_sweeps="
                         << MAX_SWEEPS << ", energy_tol=" << ENERGY_TOL << ")\n";

  // --- 5b. Configure the Krylov sub-config.
  // The 1-site local problem is only bL*d*bR = 4 at the boundary; the engine
  // needs dim >= (Krylov max_dim + min block size), so cap max_dim at 3.
  cudensitymatEigenDecompositionApproachKrylovConfig_t krylovCfg = nullptr;
  HANDLE_CUDM_ERROR(cudensitymatCreateEigenDecompositionApproachKrylovConfig(handle, &krylovCfg));
  {
    const int32_t krylovMaxDim = KRYLOV_MAX_DIM;
    HANDLE_CUDM_ERROR(cudensitymatEigenDecompositionApproachKrylovConfigSetAttribute(
        handle, krylovCfg,
        CUDENSITYMAT_EIGEN_APPROACH_KRYLOV_MAX_DIM,
        &krylovMaxDim, sizeof(krylovMaxDim)));
  }
  HANDLE_CUDM_ERROR(cudensitymatEigenDecompositionConfigure(handle, eigen,
                      CUDENSITYMAT_EIGEN_APPROACH_KRYLOV_CONFIG,
                      &krylovCfg, sizeof(krylovCfg)));
  HANDLE_CUDM_ERROR(cudensitymatDestroyEigenDecompositionApproachKrylovConfig(krylovCfg));
  krylovCfg = nullptr;
  if (verbose) std::cout << "Configured Krylov sub-config (max_dim=" << KRYLOV_MAX_DIM << ")\n";

  // --- 6. Prepare + attach workspace ---
  cudensitymatWorkspaceDescriptor_t workspaceDescr = nullptr;
  HANDLE_CUDM_ERROR(cudensitymatCreateWorkspace(handle, &workspaceDescr));

  std::size_t freeMem = 0, totalMem = 0;
  HANDLE_CUDA_ERROR(cudaMemGetInfo(&freeMem, &totalMem));
  freeMem = static_cast<std::size_t>(static_cast<double>(freeMem) * 0.85);
  if (verbose) std::cout << "Available workspace memory (bytes) = " << freeMem << "\n";

  HANDLE_CUDM_ERROR(cudensitymatEigenDecompositionPrepare(handle, eigen,
                      /*maxEigenStates=*/1, mpsState, CUDENSITYMAT_COMPUTE_64F,
                      freeMem, workspaceDescr, /*stream=*/0x0));

  std::size_t scratchSize = 0;
  HANDLE_CUDM_ERROR(cudensitymatWorkspaceGetMemorySize(handle, workspaceDescr,
                      CUDENSITYMAT_MEMSPACE_DEVICE,
                      CUDENSITYMAT_WORKSPACE_SCRATCH,
                      &scratchSize));
  void * scratchBuf = nullptr;
  if (scratchSize > 0) {
    HANDLE_CUDA_ERROR(cudaMalloc(&scratchBuf, scratchSize));
    HANDLE_CUDM_ERROR(cudensitymatWorkspaceSetMemory(handle, workspaceDescr,
                        CUDENSITYMAT_MEMSPACE_DEVICE,
                        CUDENSITYMAT_WORKSPACE_SCRATCH,
                        scratchBuf, scratchSize));
  }
  if (verbose) std::cout << "Prepared DMRG plan; scratch workspace = " << scratchSize << " bytes\n";

  // --- 7. Compute the ground-state pair (eigenvalue, MPS) ---
  void * eigenvalueGpu = createArrayGPU<Complex>(1);
  double residual = RESIDUAL;
  cudensitymatState_t eigenstatesArr[1] = { mpsState };

  HANDLE_CUDA_ERROR(cudaDeviceSynchronize());
  HANDLE_CUDM_ERROR(cudensitymatEigenDecompositionCompute(handle, eigen,
                      /*time=*/0.0,
                      /*batchSize=*/1,
                      /*numParams=*/0, /*params=*/nullptr,
                      /*numEigenStates=*/1, eigenstatesArr,
                      eigenvalueGpu, &residual,
                      workspaceDescr, /*stream=*/0x0));
  HANDLE_CUDA_ERROR(cudaDeviceSynchronize());

  cuDoubleComplex eigenvalueHost{0.0, 0.0};
  HANDLE_CUDA_ERROR(cudaMemcpy(&eigenvalueHost, eigenvalueGpu,
                               sizeof(cuDoubleComplex), cudaMemcpyDeviceToHost));

  const double eDmrg     = eigenvalueHost.x;
  const double energyErr = std::abs(eDmrg - E0_EXACT);
  const bool   converged = (energyErr < 1e-6) && (residual < RESIDUAL);

  std::cout << "\n=================================================================\n"
            << "DMRG ground-state energy : " << std::scientific << std::setprecision(12)
            << eDmrg << "\n"
            << "Exact (dense ED, N=" << NUM_SITES << ")  : " << E0_EXACT << "\n"
            << "Energy error             : " << std::scientific << std::setprecision(3)
            << energyErr << "\n"
            << "Final residual           : " << residual << "\n"
            << "Result                   : " << (converged ? "PASS" : "FAIL") << "\n"
            << "=================================================================\n\n";

  // --- 8. Cleanup (reverse Create order) ---
  destroyArrayGPU(eigenvalueGpu);
  if (scratchBuf) HANDLE_CUDA_ERROR(cudaFree(scratchBuf));
  HANDLE_CUDM_ERROR(cudensitymatDestroyWorkspace(workspaceDescr));
  HANDLE_CUDM_ERROR(cudensitymatDestroyEigenDecomposition(eigen));
  HANDLE_CUDM_ERROR(cudensitymatDestroyState(mpsState));
  HANDLE_CUDM_ERROR(cudensitymatDestroyOperator(hamiltonian));
  HANDLE_CUDM_ERROR(cudensitymatDestroyOperatorTerm(operatorTerm));
  HANDLE_CUDM_ERROR(cudensitymatDestroyMatrixProductOperator(mpoHandle));
  mps.destroy();
  mpo.destroy();

  if (verbose) std::cout << "Destroyed all resources\n";
}


int main(int /*argc*/, char ** /*argv*/)
{
  HANDLE_CUDA_ERROR(cudaSetDevice(0));
  if (verbose) std::cout << "Set active CUDA device 0\n";

  cudensitymatHandle_t handle;
  HANDLE_CUDM_ERROR(cudensitymatCreate(&handle));
  if (verbose) std::cout << "Created cuDensityMat library handle\n";

  exampleWorkflow(handle);

  HANDLE_CUDM_ERROR(cudensitymatDestroy(handle));
  if (verbose) std::cout << "Destroyed cuDensityMat library handle\n";

  HANDLE_CUDA_ERROR(cudaDeviceReset());
  return 0;
}
