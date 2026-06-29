/* Copyright (c) 2024-2026, NVIDIA CORPORATION & AFFILIATES.
 *
 * SPDX-License-Identifier: BSD-3-Clause
 */

// MPS-MPO operator-action performance benchmark.
//
// Companion of `mps_mpo_action_example.cpp`. Where the example targets a small
// transverse-field Ising problem with an analytical norm reference, this
// benchmark targets a larger workload: a superconducting-chip Hamiltonian on
// 40 sites with mixed mode dimensions, next-nearest-neighbor coupling, and a
// 128 output bond dimension budget.
//
// The benchmark measures and prints:
//   * Prepare CUDA-event elapsed time (cudensitymatOperatorActionPrepare).
//   * Average and minimum Compute CUDA-event elapsed time over repeated
//     cudensitymatOperatorActionCompute calls on a single prepared plan.
//
// Hamiltonian (superconducting transmon chain):
//
//   H = Σᵢ [εᵢ Nᵢ + (αᵢ/2) Nᵢ (Nᵢ−1)]
//     + Σᵢ J  (aᵢ† aᵢ₊₁ + aᵢ aᵢ₊₁†)
//     + Σᵢ K  Nᵢ Nᵢ₊₂                       (next-nearest-neighbor)
//
//   Nᵢ = diag(0,1,…,dᵢ−1)               (number operator)
//   aᵢ |n⟩ = √n |n−1⟩                   (lowering)
//   aᵢ† |n⟩ = √(n+1) |n+1⟩              (raising)
//
// Mode dimensions alternate between d=2 and d=4 to exercise both qubit-like
// and qudit-like local Hilbert spaces (mixed mode dimensions with some d > 2).
//
// The MPO is encoded as a 6-state finite-state automaton (bond dimension 6).
// The state numbering places DONE at index 0 and START at the highest index
// (state 5), so that the implicit open-boundary contractions naturally select
// the "done" column at the right boundary (bulk column 0) and the "start" row
// at the left boundary (bulk row 5) with minimal index remapping. The state
// transitions are:
//
//   start  →ⁱ      start                        (transmit nothing)
//   start  →ᵃ†     after-a†                     (open NN pair, partner = a)
//   start  →ᵃ      after-a                      (open NN pair, partner = a†)
//   start  →ᴺ      just-placed-N                (open NNN pair, awaiting K·N two sites later)
//   start  →ᵒⁿˢⁱᵗᵉ done                         (place on-site εN + α/2 N(N−1))
//   after-a†       →ʲᵃ   done                   (close NN pair: J·a)
//   after-a        →ʲᵃ†  done                   (close NN pair: J·a†)
//   just-placed-N  →ⁱ    pass-N                 (pass through middle site)
//   pass-N         →ᴷᴺ   done                   (close NNN pair: K·N)
//   done           →ⁱ    done                   (transmit "done")
//
// State index assignments:
//   0 = done, 1 = after-a†, 2 = after-a, 3 = just-placed-N, 4 = pass-N, 5 = start.

#include <cudensitymat.h>
#include "helpers.h"

#include <algorithm>
#include <cassert>
#include <chrono>
#include <cmath>
#include <complex>
#include <cstdint>
#include <cstdlib>
#include <iomanip>
#include <iostream>
#include <limits>
#include <numeric>
#include <string>
#include <vector>


using Complex = std::complex<double>;
constexpr cudaDataType_t DATA_TYPE = CUDA_C_64F;

constexpr bool verbose = true;

// ---------------------------------------------------------------------------
// Problem parameters.
//
// The defaults are a 40-site / chi=128 workload. The benchmark prepares the
// operator action once and then times a loop of repeated Compute calls on the
// same prepared plan.
// ---------------------------------------------------------------------------
constexpr int32_t NUM_SITES     = 40;     // number of chain sites
constexpr int64_t MAX_BOND_DIM  = 128;    // output MPS bond dimension budget
constexpr int32_t MPO_BOND_DIM  = 6;      // fixed by the 6-state automaton above
constexpr int     NUM_REPEATS   = 5;      // timed repeated-Compute iterations (after a warm-up)

// Hamiltonian coupling strengths. Values are illustrative and chosen so
// that all three families of terms (on-site, NN, NNN) contribute non-trivially
// to the cost function.
constexpr double EPS_BASE   = 1.0;   // on-site detuning ε
constexpr double ALPHA_BASE = -0.2;  // anharmonicity α (only acts when d > 2)
constexpr double J_NN       = 0.1;   // nearest-neighbor exchange coefficient
constexpr double K_NNN      = 0.05;  // next-nearest-neighbor diagonal coupling

// Mixed local mode dimensions: even sites = 2-level (qubit), odd sites =
// 4-level (transmon truncated to 4 levels). This exercises both qubit-like
// and qudit-like local Hilbert spaces on heterogeneous bonds.
constexpr int64_t physDim(int32_t site) {
  return (site % 2 == 0) ? 2 : 4;
}


// ---------------------------------------------------------------------------
// Local-operator helpers (column-major d × d).
//
// Indexing convention: M[row + col * d] where row = "bra" (output index of
// |out⟩) and col = "ket" (input index of |in⟩), matching the convention used
// in `mps_mpo_action_example.cpp` (cf. its X_mat(bra, ket) usage).
// ---------------------------------------------------------------------------
using Mat = std::vector<Complex>;

static Mat zeroMat(int64_t d)
{
  return Mat(d * d, Complex{0.0, 0.0});
}

static Mat identityMat(int64_t d)
{
  Mat M = zeroMat(d);
  for (int64_t i = 0; i < d; ++i) M[i + i * d] = Complex{1.0, 0.0};
  return M;
}

static Mat numberMat(int64_t d)
{
  Mat M = zeroMat(d);
  for (int64_t n = 0; n < d; ++n) M[n + n * d] = Complex{static_cast<double>(n), 0.0};
  return M;
}

static Mat loweringMat(int64_t d)
{
  // a |n⟩ = √n |n−1⟩ ⇒ ⟨n−1|a|n⟩ = √n. Set M[n−1 (row), n (col)] = √n.
  Mat M = zeroMat(d);
  for (int64_t n = 1; n < d; ++n) {
    M[(n - 1) + n * d] = Complex{std::sqrt(static_cast<double>(n)), 0.0};
  }
  return M;
}

static Mat raisingMat(int64_t d)
{
  // a† |n−1⟩ = √n |n⟩ ⇒ ⟨n|a†|n−1⟩ = √n. Set M[n (row), n−1 (col)] = √n.
  Mat M = zeroMat(d);
  for (int64_t n = 1; n < d; ++n) {
    M[n + (n - 1) * d] = Complex{std::sqrt(static_cast<double>(n)), 0.0};
  }
  return M;
}

// On-site term: εN + (α/2) N(N−1). Diagonal in the number basis.
static Mat onSiteMat(int64_t d, double eps, double alpha)
{
  Mat M = zeroMat(d);
  for (int64_t n = 0; n < d; ++n) {
    const double val = eps * static_cast<double>(n)
                     + 0.5 * alpha * static_cast<double>(n) * static_cast<double>(n - 1);
    M[n + n * d] = Complex{val, 0.0};
  }
  return M;
}


// ---------------------------------------------------------------------------
// Superconducting-chip MPO.
//
// Each site holds a (bL × d × bR × d) column-major tensor whose (sL, ket, sR,
// bra) entry equals coef · M[bra, ket], summed over all state-machine
// transitions sL → sR active at this site. Boundary sites have bL = 1 or
// bR = 1; the put() helper applies explicit boundary-collapse remapping so
// that the START row of bulk W lands at the left boundary's only storage
// slot, and the DONE column of bulk W lands at the right boundary's only
// storage slot. With the state numbering used here (state 0 = done at the
// lowest bulk index, state 5 = start at the highest), the right boundary
// needs no index remap (bulk col 0 = done col is already at storage sR=0),
// while the left boundary remaps bulk row 5 (start) onto storage sL=0.
// ---------------------------------------------------------------------------
struct ChipMPO {

  std::vector<std::vector<Complex>> hostTensors;
  std::vector<void *> gpuPtrs;
  std::vector<int64_t> mpoBondDims;

  void build() {
    hostTensors.resize(NUM_SITES);
    gpuPtrs.resize(NUM_SITES, nullptr);
    mpoBondDims.assign(NUM_SITES - 1, MPO_BOND_DIM);

    for (int32_t site = 0; site < NUM_SITES; ++site) {
      const int64_t d  = physDim(site);
      const int64_t bL = (site == 0)              ? 1 : MPO_BOND_DIM;
      const int64_t bR = (site == NUM_SITES - 1)  ? 1 : MPO_BOND_DIM;
      const int64_t vol = bL * d * bR * d;
      hostTensors[site].assign(vol, Complex{0.0, 0.0});

      auto idx = [&](int64_t sL, int64_t ket, int64_t sR, int64_t bra) -> int64_t {
        return sL + bL * (ket + d * (sR + bR * bra));
      };

      const Mat I_mat  = identityMat(d);
      const Mat N_mat  = numberMat(d);
      const Mat a_mat  = loweringMat(d);
      const Mat ad_mat = raisingMat(d);
      const Mat on_mat = onSiteMat(d, EPS_BASE, ALPHA_BASE);

      // Boundary state-collapse (state 0 = done, state 5 = start).
      //
      // The MPO storage layout has bL=1 at the left boundary and bR=1 at the
      // right boundary (the missing bond is omitted from storage). Storage
      // bond indices at non-boundary edges directly match bulk-W bond indices,
      // so:
      //   - LEFT  boundary (bL=1): storage's sL=0 is the only surviving slot;
      //     the row of bulk-W placed there must be the START row. Because we
      //     chose state 5 = start (bulk row 5), the put(*, ...) calls with
      //     sL=5 must be REMAPPED onto storage sL=0; all other left-boundary
      //     writes (sL != 5) must be dropped.
      //   - RIGHT boundary (bR=1): storage's sR=0 is the only surviving slot;
      //     the column of bulk-W placed there must be the DONE column. Because
      //     we chose state 0 = done (bulk column 0), dropping every put(*,
      //     sR!=0, ...) at the right boundary naturally leaves the done column
      //     at sR=0 — no remap needed.
      auto put = [&](int64_t sL, int64_t sR, const Mat & M, double coef) {
        int64_t sL_storage = sL;
        if (bL == 1) {                         // left-boundary: keep start row, remap to storage 0
          if (sL != 5) return;
          sL_storage = 0;
        }
        if (bR == 1 && sR != 0) return;        // right-boundary: drop non-done columns
        for (int64_t bra = 0; bra < d; ++bra) {
          for (int64_t ket = 0; ket < d; ++ket) {
            hostTensors[site][idx(sL_storage, ket, sR, bra)] += coef * M[bra + ket * d];
          }
        }
      };

      // State 5 (start) outgoing transitions. At the left boundary the put()
      // lambda remaps sL=5 onto storage sL=0.
      put(5, 5, I_mat,  1.0);             // start → start, pass-through (stay at start)
      put(5, 1, ad_mat, 1.0);             // open NN pair: place a†, advance to "after-a†"
      put(5, 2, a_mat,  1.0);             // open NN pair: place a, advance to "after-a"
      put(5, 3, N_mat,  1.0);             // open NNN pair: place N, advance to "just-placed-N"
      put(5, 0, on_mat, 1.0);             // start → done via on-site εN + α/2 N(N−1)

      // Closing / middle transitions (no remap; interior bulk indices).
      put(1, 0, a_mat,  J_NN);            // after-a†   → done: close NN with J·a
      put(2, 0, ad_mat, J_NN);            // after-a    → done: close NN with J·a†
      put(3, 4, I_mat,  1.0);             // just-placed-N → pass-N: pass through middle of NNN
      put(4, 0, N_mat,  K_NNN);           // pass-N     → done: close NNN with K·N

      // State 0 (done) outgoing pass-through. At the right boundary (bR=1)
      // the put() lambda keeps this (sR=0 ⇒ matches storage's only sR slot);
      // at site 0 (bL=1, sL=0 ≠ 5) the lambda drops this (no incoming-`done`
      // left state to transmit from at the very first site).
      put(0, 0, I_mat, 1.0);              // done → done, pass-through

      gpuPtrs[site] = createInitializeArrayGPU(hostTensors[site]);
    }
  }

  void destroy() {
    for (auto & ptr : gpuPtrs) {
      if (ptr) { destroyArrayGPU(ptr); ptr = nullptr; }
    }
  }
};


// ---------------------------------------------------------------------------
// MPS input builder.
//
// Deterministic Neel-style product input |0, 1, 0, 1, ...>. At d=2 sites we use
// occupation 0, at d=4 sites occupation 1. The product state is bond-1 in
// effect but stored in the same (bL x d x bR) tensor shape as a general MPS
// (one non-zero entry per site, rest zero-padded).
// ---------------------------------------------------------------------------
struct InputMPS {

  std::vector<std::vector<Complex>> hostTensors;
  std::vector<void *> gpuPtrs;

  void build(const std::vector<int64_t> & bondDims) {
    hostTensors.resize(NUM_SITES);
    gpuPtrs.resize(NUM_SITES, nullptr);

    for (int32_t site = 0; site < NUM_SITES; ++site) {
      const int64_t d  = physDim(site);
      const int64_t bL = (site == 0)             ? 1 : bondDims[site - 1];
      const int64_t bR = (site == NUM_SITES - 1) ? 1 : bondDims[site];
      const int64_t vol = bL * d * bR;
      hostTensors[site].assign(vol, Complex{0.0, 0.0});

      const int64_t occupation = (site % 2 == 0) ? 0 : 1;
      const int64_t idx = 0 + bL * (occupation + d * 0);
      hostTensors[site][idx] = Complex{1.0, 0.0};

      gpuPtrs[site] = createInitializeArrayGPU(hostTensors[site]);
    }
  }

  void destroy() {
    for (auto & ptr : gpuPtrs) {
      if (ptr) { destroyArrayGPU(ptr); ptr = nullptr; }
    }
  }
};


// ---------------------------------------------------------------------------
// Cap MPS bond dimensions by the exact Hilbert-space dimension on either side.
// For a mixed-dim OBC chain, the max admissible bond at link i is
//   min(maxBondCap, ∏_{j ≤ i} d_j, ∏_{j > i} d_j).
// Cumulative products grow quickly (40-site chain with 20 × d=2 + 20 × d=4 has
// a Hilbert space of 2²⁰ · 4²⁰ = 2⁶⁰ ≈ 1.15 × 10¹⁸), so we clamp inside the
// loop to avoid overflow and stay at int64.
// ---------------------------------------------------------------------------
static std::vector<int64_t> makeMaxBondDims(int64_t maxBondCap)
{
  std::vector<int64_t> dims(NUM_SITES - 1);
  for (int32_t i = 0; i < NUM_SITES - 1; ++i) {
    int64_t leftDim = 1;
    for (int32_t j = 0; j <= i; ++j) {
      leftDim = std::min<int64_t>(leftDim * physDim(j), maxBondCap);
    }
    int64_t rightDim = 1;
    for (int32_t j = i + 1; j < NUM_SITES; ++j) {
      rightDim = std::min<int64_t>(rightDim * physDim(j), maxBondCap);
    }
    dims[i] = std::min({maxBondCap, leftDim, rightDim});
  }
  return dims;
}

// ---------------------------------------------------------------------------
// Benchmark workflow.
// ---------------------------------------------------------------------------
void benchmarkWorkflow(cudensitymatHandle_t handle)
{
  // --- 1. Space shape (per-site mode dimensions) ---
  std::vector<int64_t> spaceShape(NUM_SITES);
  for (int32_t s = 0; s < NUM_SITES; ++s) spaceShape[s] = physDim(s);

  if (verbose) {
    std::cout << "MPS-MPO performance benchmark configuration:\n";
    std::cout << "  NUM_SITES        = " << NUM_SITES        << "\n";
    std::cout << "  MAX_BOND_DIM     = " << MAX_BOND_DIM     << "\n";
    std::cout << "  MPO_BOND_DIM     = " << MPO_BOND_DIM     << "\n";
    std::cout << "  mode dims        = [";
    for (int32_t s = 0; s < NUM_SITES; ++s) {
      std::cout << spaceShape[s] << (s + 1 < NUM_SITES ? ", " : "");
    }
    std::cout << "]\n";
  }

  // --- 2. Build the superconducting-chip MPO ---
  ChipMPO mpo;
  mpo.build();
  if (verbose) std::cout << "Built superconducting-chip MPO (bond dim "
                         << MPO_BOND_DIM << ")\n";

  cudensitymatMatrixProductOperator_t mpoHandle;
  std::vector<cudensitymatWrappedTensorCallback_t> mpoCallbacks(NUM_SITES, cudensitymatTensorCallbackNone);
  std::vector<cudensitymatWrappedTensorGradientCallback_t> mpoGradCallbacks(NUM_SITES, cudensitymatTensorGradientCallbackNone);
  HANDLE_CUDM_ERROR(cudensitymatCreateMatrixProductOperator(handle,
                      NUM_SITES,
                      spaceShape.data(),
                      CUDENSITYMAT_BOUNDARY_CONDITION_OPEN,
                      mpo.mpoBondDims.data(),
                      DATA_TYPE,
                      mpo.gpuPtrs.data(),
                      mpoCallbacks.data(),
                      mpoGradCallbacks.data(),
                      &mpoHandle));

  // --- 3. Wrap MPO in single-term Operator (the Create-time gate requires
  // exactly one MPO product / one term for split-scope operator actions) ---
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

  // --- 4. Create input and output MPS states (matched bond extents) ---
  const std::vector<int64_t> mpsBondDims = makeMaxBondDims(MAX_BOND_DIM);

  if (verbose) {
    std::cout << "  MPS bond extents = [";
    for (size_t i = 0; i < mpsBondDims.size(); ++i) {
      std::cout << mpsBondDims[i] << (i + 1 < mpsBondDims.size() ? ", " : "");
    }
    std::cout << "]\n";
  }

  const int64_t batchSize = 1;
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

  std::size_t totalStateBytes = 0;
  for (auto s : componentSizes) totalStateBytes += s;
  if (verbose) {
    std::cout << "  Per-MPS storage  = " << (totalStateBytes / (1024 * 1024))
              << " MiB across " << numComponents << " components\n";
  }

  // --- 5. Allocate GPU storage for both states ---
  // For stateIn we upload deterministic Neel-style data via InputMPS::build
  // (host -> GPU during the build call). For stateOut a plain cudaMalloc is
  // enough — we zero it via cudensitymatStateInitializeZero so the operator
  // action behaves as an assignment.
  InputMPS inputMps;
  inputMps.build(mpsBondDims);

  std::vector<void *> stateOutBuffers(numComponents, nullptr);
  for (int32_t c = 0; c < numComponents; ++c) {
    HANDLE_CUDA_ERROR(cudaMalloc(&stateOutBuffers[c], componentSizes[c]));
  }

  HANDLE_CUDM_ERROR(cudensitymatStateAttachComponentStorage(handle,
                      stateIn, numComponents,
                      inputMps.gpuPtrs.data(), componentSizes.data()));
  HANDLE_CUDM_ERROR(cudensitymatStateAttachComponentStorage(handle,
                      stateOut, numComponents,
                      stateOutBuffers.data(), componentSizes.data()));

  // Zero-initialise stateOut so the first Compute call below performs an
  // assignment rather than an accumulation.
  HANDLE_CUDM_ERROR(cudensitymatStateInitializeZero(handle, stateOut, /*stream=*/0));

  // --- 6. Create OperatorAction (split scope + LinSolve approach) ---
  cudensitymatOperator_t operators[] = {hamiltonian};
  cudensitymatOperatorAction_t operatorAction;
  HANDLE_CUDM_ERROR(cudensitymatCreateOperatorAction(handle,
                      /*numOperators=*/1,
                      operators,
                      CUDENSITYMAT_FITTING_SCOPE_SPLIT,
                      CUDENSITYMAT_FITTING_APPROACH_LINSOLVE,
                      &operatorAction));

  // --- 7. Configure ALS sub-config (explicit values shown for discoverability;
  // these match the defaults — raise max_sweeps to increase the per-Compute
  // work if a heavier benchmark is desired) ---
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
  HANDLE_CUDM_ERROR(cudensitymatDestroyStateFittingScopeSplitALSConfig(alsConfig));

  // --- 8. Prepare (timed via cudaEvent_t) ---
  cudensitymatWorkspaceDescriptor_t workspaceDescr;
  HANDLE_CUDM_ERROR(cudensitymatCreateWorkspace(handle, &workspaceDescr));

  std::size_t freeMem = 0, totalMem = 0;
  HANDLE_CUDA_ERROR(cudaMemGetInfo(&freeMem, &totalMem));
  freeMem = static_cast<std::size_t>(static_cast<double>(freeMem) * 0.95);
  if (verbose) {
    std::cout << "  Free device mem  = " << (freeMem / (1024 * 1024))
              << " MiB (95% headroom budget for the workspace)\n";
  }

  cudaEvent_t prepStart, prepEnd;
  HANDLE_CUDA_ERROR(cudaEventCreate(&prepStart));
  HANDLE_CUDA_ERROR(cudaEventCreate(&prepEnd));
  HANDLE_CUDA_ERROR(cudaEventRecord(prepStart, /*stream=*/0));
  HANDLE_CUDM_ERROR(cudensitymatOperatorActionPrepare(handle,
                      operatorAction,
                      &stateIn,
                      stateOut,
                      CUDENSITYMAT_COMPUTE_64F,
                      freeMem,
                      workspaceDescr,
                      /*stream=*/0));
  HANDLE_CUDA_ERROR(cudaEventRecord(prepEnd, /*stream=*/0));
  HANDLE_CUDA_ERROR(cudaEventSynchronize(prepEnd));
  float prepMs = 0.0f;
  HANDLE_CUDA_ERROR(cudaEventElapsedTime(&prepMs, prepStart, prepEnd));
  HANDLE_CUDA_ERROR(cudaEventDestroy(prepStart));
  HANDLE_CUDA_ERROR(cudaEventDestroy(prepEnd));

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

  if (verbose) {
    std::cout << "\n=== Prepare summary ===\n";
    std::cout << "  Prepare CUDA-event time : " << prepMs << " ms\n";
    std::cout << "  Scratch workspace size  : " << (scratchSize / (1024 * 1024))
              << " MiB\n";
  }

  // --- 9. Compute (timed via cudaEvent_t) ---
  //
  // The first Compute after Prepare is an untimed warm-up that primes the
  // prepared plan. We then run a timed loop of NUM_REPEATS Compute calls on
  // the SAME prepared plan, re-zeroing stateOut before each one so every
  // iteration performs an identical assignment-mode workload, to measure
  // repeated-Compute throughput.
  cudaEvent_t cStart, cEnd;
  HANDLE_CUDA_ERROR(cudaEventCreate(&cStart));
  HANDLE_CUDA_ERROR(cudaEventCreate(&cEnd));

  // Warm-up assignment Compute (untimed; primes plan/caches).
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

  // Timed repeated-Compute loop. Re-zero stateOut before each so the work is
  // identical and repeatable (assignment, not accumulation).
  float totalComputeMs = 0.0f;
  float minComputeMs = std::numeric_limits<float>::max();
  for (int rep = 0; rep < NUM_REPEATS; ++rep) {
    HANDLE_CUDM_ERROR(cudensitymatStateInitializeZero(handle, stateOut, /*stream=*/0));
    HANDLE_CUDA_ERROR(cudaEventRecord(cStart, /*stream=*/0));
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
    HANDLE_CUDA_ERROR(cudaEventRecord(cEnd, /*stream=*/0));
    HANDLE_CUDA_ERROR(cudaEventSynchronize(cEnd));
    float iterMs = 0.0f;
    HANDLE_CUDA_ERROR(cudaEventElapsedTime(&iterMs, cStart, cEnd));
    totalComputeMs += iterMs;
    minComputeMs = std::min(minComputeMs, iterMs);
  }
  HANDLE_CUDA_ERROR(cudaEventDestroy(cStart));
  HANDLE_CUDA_ERROR(cudaEventDestroy(cEnd));

  const float computeMs = (NUM_REPEATS > 0) ? (totalComputeMs / NUM_REPEATS) : 0.0f;
  const double callsPerSec = (computeMs > 0.0f) ? (1000.0 / computeMs) : 0.0;
  const double stateGiB    = static_cast<double>(totalStateBytes)
                             / (1024.0 * 1024.0 * 1024.0);

  if (verbose) {
    std::cout << "\n=== Compute summary (" << NUM_REPEATS
              << " timed repeats after warm-up) ===\n";
    std::cout << std::fixed << std::setprecision(3);
    std::cout << "  avg time / call  : " << computeMs    << " ms\n";
    std::cout << "  min time / call  : " << minComputeMs << " ms\n";
    std::cout << "  Compute calls/s  : " << callsPerSec  << "\n";
    std::cout << "  Per-MPS GiB      : " << stateGiB     << "\n";
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
  inputMps.destroy();
  mpo.destroy();

  if (verbose) std::cout << "\nDestroyed all resources\n";
}


int main(int argc, char ** argv)
{
  (void)argc; (void)argv;

  HANDLE_CUDA_ERROR(cudaSetDevice(0));
  if (verbose) std::cout << "Set active device\n";

  cudensitymatHandle_t handle;
  HANDLE_CUDM_ERROR(cudensitymatCreate(&handle));
  if (verbose) std::cout << "Created library handle\n";

  const auto t0 = std::chrono::steady_clock::now();
  benchmarkWorkflow(handle);
  const auto t1 = std::chrono::steady_clock::now();
  const double totalSec = std::chrono::duration<double>(t1 - t0).count();

  if (verbose) {
    std::cout << "\nTotal benchmark wall-clock time : "
              << std::fixed << std::setprecision(2) << totalSec << " s\n";
  }

  HANDLE_CUDM_ERROR(cudensitymatDestroy(handle));
  if (verbose) std::cout << "Destroyed library handle\n";

  HANDLE_CUDA_ERROR(cudaDeviceReset());
  return 0;
}
