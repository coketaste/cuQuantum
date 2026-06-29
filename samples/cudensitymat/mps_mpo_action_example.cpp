/* Copyright (c) 2024-2026, NVIDIA CORPORATION & AFFILIATES.
 *
 * SPDX-License-Identifier: BSD-3-Clause
 */

// MPS-MPO operator-action example.
//
// Demonstrates applying a 1-D transverse-field Ising MPO to a pure MPS state
// via the split-scope variational ALS fitting path:
//
//   stateOut += alpha * H * stateIn
//
// All quantum-state inputs are matrix product states (MPS); the Hamiltonian
// is encoded as a matrix product operator (MPO). The output MPS is fitted
// via 1-site alternating least squares (ALS), so the user controls the
// truncation budget by choosing the output MPS bond dimensions.
//
// Workflow:
//  1. Build Hamiltonian as MPO (nearest-neighbor ZZ + transverse X field)
//  2. Build single-term Operator wrapping the MPO
//  3. Create input and output MPS states (cudensitymatCreateStateMPS)
//  4. Initialize input MPS to a Neel state |0101...> and zero-initialise
//     stateOut via cudensitymatStateInitializeZero
//  5. Create OperatorAction (FITTING_SCOPE_SPLIT + FITTING_APPROACH_LINSOLVE)
//  6. (Optional) Configure ALS sub-config (num_sites, max_sweeps, tolerance)
//  7. Prepare action and allocate workspace
//  8. Compute one operator action at t = 0 (stateOut <- alpha * H * stateIn)
//  9. Report the output MPS bond extents
// 10. Clean up all resources
//
// Assignment semantics. The split-scope operator action fits stateOut to
// alpha * H * stateIn accumulated on top of the prior contents of stateOut.
// Zero-initialising stateOut with cudensitymatStateInitializeZero before the
// Compute call therefore makes the result a plain assignment
// stateOut = alpha * H * stateIn (up to FP64 fitting accuracy). The example
// reports the allocated output MPS component shapes; a schedule-agnostic norm
// check should use the public MPS norm API once it is available.

#include <cudensitymat.h>
#include "helpers.h"

#include <algorithm>
#include <cmath>
#include <complex>
#include <vector>
#include <numeric>
#include <iostream>
#include <cassert>


using Complex = std::complex<double>;
constexpr cudaDataType_t DATA_TYPE = CUDA_C_64F;

constexpr bool verbose = true;

// --- Simulation parameters ---
// 8 qubits, TFIM (J = 1.0, h = 0.5). Output MPS capped at bond dim 16 which
// saturates the natural max bond ([2,4,8,16,8,4,2]) for an 8-site qubit chain
// and is comfortably above chi_MPO * chi_MPS_in = 3 * 1 = 3.
constexpr int32_t  NUM_SITES        = 8;
constexpr int64_t  PHYS_DIM         = 2;
constexpr int64_t  MAX_OUT_BOND_DIM = 16;
constexpr int64_t  MPO_BOND_DIM     = 3;
constexpr double   J_COUPLING       = 1.0;
constexpr double   H_FIELD          = 0.5;


// ============================================================================
// Transverse-field Ising MPO builder (bond dim 3)
// ============================================================================
//
// H = -J * sum_{i} Z_i Z_{i+1}  +  h * sum_{i} X_i
//
// The bulk MPO matrix (indexed by bond dimensions aL, aR) is:
//
//        |  I      0     0  |
//   W =  |  Z      0     0  |
//        | h*X   -J*Z    I  |
//
// Library mode ordering (column-major):
//   Left boundary (site 0):     [phys_ket(d), right_bond(bR), phys_bra(d)]   (bL omitted, treated as 1)
//   Interior sites:             [left_bond(bL), phys_ket(d), right_bond(bR), phys_bra(d)]
//   Right boundary (site N-1):  [left_bond(bL), phys_ket(d), phys_bra(d)]    (bR omitted, treated as 1)

struct IsingMPO {

  std::vector<std::vector<Complex>> hostTensors;
  std::vector<void*> gpuPtrs;

  void build() {
    const Complex zero{0.0, 0.0}, one{1.0, 0.0}, mone{-1.0, 0.0};
    const Complex jc{-J_COUPLING, 0.0};
    const Complex hc{H_FIELD, 0.0};

    auto I_mat = [&](int r, int c) -> Complex { return (r == c) ? one : zero; };
    auto X_mat = [&](int r, int c) -> Complex { return (r != c) ? one : zero; };
    auto Z_mat = [&](int r, int c) -> Complex { return (r == c) ? ((r == 0) ? one : mone) : zero; };

    hostTensors.resize(NUM_SITES);
    gpuPtrs.resize(NUM_SITES, nullptr);

    for (int32_t site = 0; site < NUM_SITES; ++site) {
      const int64_t bL = (site == 0) ? 1 : MPO_BOND_DIM;
      const int64_t bR = (site == NUM_SITES - 1) ? 1 : MPO_BOND_DIM;
      const int64_t vol = bL * PHYS_DIM * PHYS_DIM * bR;
      hostTensors[site].assign(vol, zero);

      auto idx = [&](int64_t aL, int64_t ket, int64_t aR, int64_t bra) -> int64_t {
        return aL + bL * (ket + PHYS_DIM * (aR + bR * bra));
      };

      if (NUM_SITES == 1) {
        for (int bra = 0; bra < PHYS_DIM; ++bra)
          for (int ket = 0; ket < PHYS_DIM; ++ket)
            hostTensors[site][idx(0, ket, 0, bra)] = hc * X_mat(bra, ket);
      } else if (site == 0) {
        for (int bra = 0; bra < PHYS_DIM; ++bra)
          for (int ket = 0; ket < PHYS_DIM; ++ket) {
            hostTensors[site][idx(0, ket, 0, bra)] = hc * X_mat(bra, ket);
            hostTensors[site][idx(0, ket, 1, bra)] = jc * Z_mat(bra, ket);
            hostTensors[site][idx(0, ket, 2, bra)] = I_mat(bra, ket);
          }
      } else if (site == NUM_SITES - 1) {
        for (int bra = 0; bra < PHYS_DIM; ++bra)
          for (int ket = 0; ket < PHYS_DIM; ++ket) {
            hostTensors[site][idx(0, ket, 0, bra)] = I_mat(bra, ket);
            hostTensors[site][idx(1, ket, 0, bra)] = Z_mat(bra, ket);
            hostTensors[site][idx(2, ket, 0, bra)] = hc * X_mat(bra, ket);
          }
      } else {
        for (int bra = 0; bra < PHYS_DIM; ++bra)
          for (int ket = 0; ket < PHYS_DIM; ++ket) {
            hostTensors[site][idx(0, ket, 0, bra)] = I_mat(bra, ket);
            hostTensors[site][idx(1, ket, 0, bra)] = Z_mat(bra, ket);
            hostTensors[site][idx(2, ket, 0, bra)] = hc * X_mat(bra, ket);
            hostTensors[site][idx(2, ket, 1, bra)] = jc * Z_mat(bra, ket);
            hostTensors[site][idx(2, ket, 2, bra)] = I_mat(bra, ket);
          }
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
// Build a Neel-state MPS  |0,1,0,1,...>  with given bond dimensions
// ============================================================================
// Each MPS tensor A[site] has shape (bondL, phys, bondR) in column-major.
// For a product state, only the (0, site%2, 0) slice is nonzero; remaining
// bond indices are zero-padded.

struct NeelMPS {

  std::vector<std::vector<Complex>> hostTensors;
  std::vector<void*> gpuPtrs;

  void build(const std::vector<int64_t>& bondDims) {
    const Complex zero{0.0, 0.0}, one{1.0, 0.0};

    hostTensors.resize(NUM_SITES);
    gpuPtrs.resize(NUM_SITES, nullptr);

    for (int32_t site = 0; site < NUM_SITES; ++site) {
      const int64_t bL = (site == 0) ? 1 : bondDims[site - 1];
      const int64_t bR = (site == NUM_SITES - 1) ? 1 : bondDims[site];
      const int64_t vol = bL * PHYS_DIM * bR;
      hostTensors[site].assign(vol, zero);

      auto idx = [&](int64_t aL, int64_t sigma, int64_t aR) -> int64_t {
        return aL + bL * (sigma + PHYS_DIM * aR);
      };

      const int64_t neelSpin = site % 2;
      hostTensors[site][idx(0, neelSpin, 0)] = one;

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
// Helper: cap MPS bond dimensions by the exact Hilbert-space dimension on
// either side. For an OBC qubit chain, the max admissible bond at site i is
// min(2^(i+1), 2^(N-i-1)).
// ============================================================================
static std::vector<int64_t> makeMaxBondDims(int64_t maxBondCap)
{
  std::vector<int64_t> dims(NUM_SITES - 1);
  for (int32_t i = 0; i < NUM_SITES - 1; ++i) {
    int64_t leftDim = 1;
    for (int32_t j = 0; j <= i; ++j) leftDim *= PHYS_DIM;
    int64_t rightDim = 1;
    for (int32_t j = i + 1; j < NUM_SITES; ++j) rightDim *= PHYS_DIM;
    dims[i] = std::min({maxBondCap, leftDim, rightDim});
  }
  return dims;
}


// ============================================================================
// Example workflow
// ============================================================================

void exampleWorkflow(cudensitymatHandle_t handle)
{
  // --- 1. Build the transverse-field Ising MPO ---
  IsingMPO mpo;
  mpo.build();
  if (verbose)
    std::cout << "Built transverse-field Ising MPO (bond dim " << MPO_BOND_DIM << ")\n";

  const std::vector<int64_t> spaceShape(NUM_SITES, PHYS_DIM);
  std::vector<int64_t> mpoBondDims(NUM_SITES - 1, MPO_BOND_DIM);

  cudensitymatMatrixProductOperator_t mpoHandle;
  std::vector<cudensitymatWrappedTensorCallback_t> mpoCallbacks(NUM_SITES, cudensitymatTensorCallbackNone);
  std::vector<cudensitymatWrappedTensorGradientCallback_t> mpoGradCallbacks(NUM_SITES, cudensitymatTensorGradientCallbackNone);
  HANDLE_CUDM_ERROR(cudensitymatCreateMatrixProductOperator(handle,
                      NUM_SITES,
                      spaceShape.data(),
                      CUDENSITYMAT_BOUNDARY_CONDITION_OPEN,
                      mpoBondDims.data(),
                      DATA_TYPE,
                      mpo.gpuPtrs.data(),
                      mpoCallbacks.data(),
                      mpoGradCallbacks.data(),
                      &mpoHandle));
  if (verbose)
    std::cout << "Created MPO handle\n";

  // --- 2. Wrap the MPO in a single-term Operator ---
  // Split-scope operator actions require the operator to wrap exactly one
  // term containing a single MPO product.
  cudensitymatOperatorTerm_t operatorTerm;
  HANDLE_CUDM_ERROR(cudensitymatCreateOperatorTerm(handle,
                      NUM_SITES,
                      spaceShape.data(),
                      &operatorTerm));

  std::vector<int32_t> modesActedOn(NUM_SITES);
  std::iota(modesActedOn.begin(), modesActedOn.end(), 0);
  std::vector<int32_t> modeDuality(NUM_SITES, 0);
  std::vector<int32_t> mpoConjugation = {0};

  HANDLE_CUDM_ERROR(cudensitymatOperatorTermAppendMPOProduct(handle,
                      operatorTerm,
                      1,
                      &mpoHandle,
                      mpoConjugation.data(),
                      modesActedOn.data(),
                      modeDuality.data(),
                      make_cuDoubleComplex(1.0, 0.0),
                      cudensitymatScalarCallbackNone,
                      cudensitymatScalarGradientCallbackNone));

  cudensitymatOperator_t hamiltonian;
  HANDLE_CUDM_ERROR(cudensitymatCreateOperator(handle,
                      NUM_SITES,
                      spaceShape.data(),
                      &hamiltonian));
  HANDLE_CUDM_ERROR(cudensitymatOperatorAppendTerm(handle,
                      hamiltonian,
                      operatorTerm,
                      0,
                      make_cuDoubleComplex(1.0, 0.0),
                      cudensitymatScalarCallbackNone,
                      cudensitymatScalarGradientCallbackNone));
  if (verbose)
    std::cout << "Constructed Hamiltonian operator from MPO\n";

  // --- 3. Create input and output MPS states ---
  const int64_t batchSize = 1;

  // Both stateIn and stateOut are sized to the maximum admissible bond
  // dimension for the given physical dimensions. The Neel input is
  // mathematically a bond-1 state; we zero-pad it into the maximum-bond
  // tensors. The output is sized to MAX_OUT_BOND_DIM = 16, which is well
  // above the bond dimension needed to represent H * stateIn exactly
  // (chi_MPO * chi_MPS_in = 3).
  std::vector<int64_t> mpsBondDims = makeMaxBondDims(MAX_OUT_BOND_DIM);

  cudensitymatState_t stateIn, stateOut;
  HANDLE_CUDM_ERROR(cudensitymatCreateStateMPS(handle,
                      CUDENSITYMAT_STATE_PURITY_PURE,
                      NUM_SITES,
                      spaceShape.data(),
                      CUDENSITYMAT_BOUNDARY_CONDITION_OPEN,
                      mpsBondDims.data(),
                      DATA_TYPE,
                      batchSize,
                      &stateIn));
  HANDLE_CUDM_ERROR(cudensitymatCreateStateMPS(handle,
                      CUDENSITYMAT_STATE_PURITY_PURE,
                      NUM_SITES,
                      spaceShape.data(),
                      CUDENSITYMAT_BOUNDARY_CONDITION_OPEN,
                      mpsBondDims.data(),
                      DATA_TYPE,
                      batchSize,
                      &stateOut));

  int32_t numComponents = 0;
  HANDLE_CUDM_ERROR(cudensitymatStateGetNumComponents(handle, stateIn, &numComponents));
  assert(numComponents == NUM_SITES);

  std::vector<std::size_t> componentSizes(numComponents);
  HANDLE_CUDM_ERROR(cudensitymatStateGetComponentStorageSize(handle,
                      stateIn, numComponents, componentSizes.data()));

  if (verbose) {
    std::cout << "MPS state has " << numComponents << " components, sizes (bytes):";
    for (auto s : componentSizes) std::cout << " " << s;
    std::cout << "\n";
    std::cout << "MPS bond dims:";
    for (auto b : mpsBondDims) std::cout << " " << b;
    std::cout << "\n";
  }

  // --- 4. Allocate GPU storage; initialise stateIn to Neel, stateOut to 0 ---
  //
  // The per-site GPU buffers for stateIn must be allocated AND populated with
  // their initial data before attaching them to the state, so we upload the
  // Neel data on the host first via NeelMPS::build. For stateOut a plain
  // cudaMalloc is enough; cudensitymatStateInitializeZero (called below)
  // writes zeros into the attached buffers so the operator action behaves as
  // an assignment.
  NeelMPS neelMps;
  neelMps.build(mpsBondDims);

  std::vector<void *> stateOutBuffers(numComponents, nullptr);
  for (int32_t c = 0; c < numComponents; ++c) {
    HANDLE_CUDA_ERROR(cudaMalloc(&stateOutBuffers[c], componentSizes[c]));
  }

  HANDLE_CUDM_ERROR(cudensitymatStateAttachComponentStorage(handle,
                      stateIn, numComponents,
                      neelMps.gpuPtrs.data(), componentSizes.data()));
  HANDLE_CUDM_ERROR(cudensitymatStateAttachComponentStorage(handle,
                      stateOut, numComponents,
                      stateOutBuffers.data(), componentSizes.data()));

  // Zero-initialise stateOut so that the subsequent operator action assigns
  // alpha * H * stateIn into it (rather than accumulating onto prior data).
  HANDLE_CUDM_ERROR(cudensitymatStateInitializeZero(handle, stateOut, /*stream=*/0));

  if (verbose)
    std::cout << "Initialized MPS states (input = Neel |0101...>, "
                 "output = zero MPS via cudensitymatStateInitializeZero)\n";

  // --- 5. Create the OperatorAction (split scope + LinSolve approach) ---
  cudensitymatOperator_t operators[] = {hamiltonian};
  cudensitymatOperatorAction_t operatorAction;
  HANDLE_CUDM_ERROR(cudensitymatCreateOperatorAction(handle,
                      /*numOperators=*/1,
                      operators,
                      CUDENSITYMAT_FITTING_SCOPE_SPLIT,
                      CUDENSITYMAT_FITTING_APPROACH_LINSOLVE,
                      &operatorAction));
  if (verbose)
    std::cout << "Created OperatorAction (scope = SPLIT, approach = LINSOLVE)\n";

  // --- 6. (Optional) Configure ALS sub-config ---
  // The defaults (num_sites = 1, max_sweeps = 20, tolerance = 1e-10) already
  // suffice for a TFIM demo. We pass an explicit configuration here only to
  // show the wiring; tightening max_sweeps below 20 may degrade convergence.
  cudensitymatStateFittingScopeSplitALSConfig_t alsConfig{nullptr};
  HANDLE_CUDM_ERROR(cudensitymatCreateStateFittingScopeSplitALSConfig(handle, &alsConfig));
  {
    const int32_t numSites = 1;
    HANDLE_CUDM_ERROR(cudensitymatStateFittingScopeSplitALSConfigSetAttribute(handle,
                        alsConfig,
                        CUDENSITYMAT_FITTING_SPLIT_SCOPE_ALS_NUM_SITES,
                        &numSites, sizeof(numSites)));
    const int32_t maxSweeps = 20;
    HANDLE_CUDM_ERROR(cudensitymatStateFittingScopeSplitALSConfigSetAttribute(handle,
                        alsConfig,
                        CUDENSITYMAT_FITTING_SPLIT_SCOPE_ALS_MAX_SWEEPS,
                        &maxSweeps, sizeof(maxSweeps)));
    const double tolerance = 1e-10;
    HANDLE_CUDM_ERROR(cudensitymatStateFittingScopeSplitALSConfigSetAttribute(handle,
                        alsConfig,
                        CUDENSITYMAT_FITTING_SPLIT_SCOPE_ALS_TOLERANCE,
                        &tolerance, sizeof(tolerance)));
  }
  HANDLE_CUDM_ERROR(cudensitymatOperatorActionConfigure(handle,
                      operatorAction,
                      CUDENSITYMAT_FITTING_SPLIT_SCOPE_ALS_CONFIG,
                      &alsConfig, sizeof(alsConfig)));
  // The sub-config is deep-copied at Configure; it is safe to destroy now.
  HANDLE_CUDM_ERROR(cudensitymatDestroyStateFittingScopeSplitALSConfig(alsConfig));
  if (verbose)
    std::cout << "Configured ALS (num_sites = 1, max_sweeps = 20, tolerance = 1e-10)\n";

  // --- 7. Prepare action and allocate workspace ---
  cudensitymatWorkspaceDescriptor_t workspaceDescr;
  HANDLE_CUDM_ERROR(cudensitymatCreateWorkspace(handle, &workspaceDescr));

  std::size_t freeMem = 0, totalMem = 0;
  HANDLE_CUDA_ERROR(cudaMemGetInfo(&freeMem, &totalMem));
  freeMem = static_cast<std::size_t>(static_cast<double>(freeMem) * 0.95);
  if (verbose)
    std::cout << "Available workspace memory (bytes) = " << freeMem << "\n";

  HANDLE_CUDM_ERROR(cudensitymatOperatorActionPrepare(handle,
                      operatorAction,
                      &stateIn,
                      stateOut,
                      CUDENSITYMAT_COMPUTE_64F,
                      freeMem,
                      workspaceDescr,
                      /*stream=*/0));
  if (verbose)
    std::cout << "Prepared OperatorAction\n";

  std::size_t scratchSize = 0;
  HANDLE_CUDM_ERROR(cudensitymatWorkspaceGetMemorySize(handle,
                      workspaceDescr,
                      CUDENSITYMAT_MEMSPACE_DEVICE,
                      CUDENSITYMAT_WORKSPACE_SCRATCH,
                      &scratchSize));
  void * scratchBuf = nullptr;
  if (scratchSize > 0) {
    HANDLE_CUDA_ERROR(cudaMalloc(&scratchBuf, scratchSize));
    HANDLE_CUDM_ERROR(cudensitymatWorkspaceSetMemory(handle,
                        workspaceDescr,
                        CUDENSITYMAT_MEMSPACE_DEVICE,
                        CUDENSITYMAT_WORKSPACE_SCRATCH,
                        scratchBuf, scratchSize));
  }
  if (verbose)
    std::cout << "Scratch workspace (bytes) = " << scratchSize << "\n";

  // --- 8. Compute one operator action (stateOut <- alpha * H * stateIn) ---
  // Because stateOut was zero-initialised above, the split-scope fit assigns
  // alpha * H * stateIn into stateOut rather than accumulating onto prior
  // contents, so the result equals H |Neel> for this fixture (alpha = 1).
  if (verbose)
    std::cout << "\nApplying H to Neel state via OperatorActionCompute (t = 0)\n";

  HANDLE_CUDM_ERROR(cudensitymatOperatorActionCompute(handle,
                      operatorAction,
                      /*time=*/0.0,
                      batchSize,
                      /*numParams=*/0,
                      /*params=*/nullptr,
                      &stateIn,
                      stateOut,
                      workspaceDescr,
                      /*stream=*/0));
  HANDLE_CUDA_ERROR(cudaStreamSynchronize(0));
  if (verbose)
    std::cout << "OperatorActionCompute completed\n";

  // --- 9. Report the output MPS bond extents ---
  // Bond extents reported here are the as-allocated ones; the ALS fit cannot
  // exceed them but may use a subset internally.
  if (verbose) {
    std::cout << "\nOutput MPS component shapes (column-major modes):\n";
    for (int32_t c = 0; c < numComponents; ++c) {
      int32_t globalId = 0;
      int32_t numModes = 0;
      int32_t batchModeLocation = 0;
      HANDLE_CUDM_ERROR(cudensitymatStateGetComponentNumModes(handle,
                          stateOut, c, &globalId, &numModes, &batchModeLocation));
      std::vector<int64_t> extents(numModes), offsets(numModes);
      HANDLE_CUDM_ERROR(cudensitymatStateGetComponentInfo(handle,
                          stateOut, c, &globalId,
                          &numModes,
                          extents.data(),
                          offsets.data()));
      std::cout << "  site " << c << " : [";
      for (int32_t m = 0; m < numModes; ++m) {
        std::cout << extents[m] << (m + 1 < numModes ? ", " : "");
      }
      std::cout << "]\n";
    }
  }

  // --- 10. Clean up ---
  if (scratchBuf)
    HANDLE_CUDA_ERROR(cudaFree(scratchBuf));
  HANDLE_CUDM_ERROR(cudensitymatDestroyWorkspace(workspaceDescr));
  HANDLE_CUDM_ERROR(cudensitymatDestroyOperatorAction(operatorAction));
  HANDLE_CUDM_ERROR(cudensitymatDestroyOperator(hamiltonian));
  HANDLE_CUDM_ERROR(cudensitymatDestroyOperatorTerm(operatorTerm));
  HANDLE_CUDM_ERROR(cudensitymatDestroyMatrixProductOperator(mpoHandle));
  HANDLE_CUDM_ERROR(cudensitymatDestroyState(stateOut));
  HANDLE_CUDM_ERROR(cudensitymatDestroyState(stateIn));

  for (auto * buf : stateOutBuffers) {
    if (buf) HANDLE_CUDA_ERROR(cudaFree(buf));
  }
  neelMps.destroy();
  mpo.destroy();

  if (verbose)
    std::cout << "\nDestroyed all resources\n";
}


int main(int argc, char ** argv)
{
  HANDLE_CUDA_ERROR(cudaSetDevice(0));
  if (verbose)
    std::cout << "Set active device\n";

  cudensitymatHandle_t handle;
  HANDLE_CUDM_ERROR(cudensitymatCreate(&handle));
  if (verbose)
    std::cout << "Created library handle\n";

  exampleWorkflow(handle);

  HANDLE_CUDM_ERROR(cudensitymatDestroy(handle));
  if (verbose)
    std::cout << "Destroyed library handle\n";

  HANDLE_CUDA_ERROR(cudaDeviceReset());
  return 0;
}
