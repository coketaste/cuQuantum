/*
 * Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
 *
 * SPDX-License-Identifier: BSD-3-Clause
 */

// Sphinx: #1
/*
 * Distributed truncated tensor SVD.
 *
 * This example demonstrates the distributed truncated SVD entry points on
 * the bond-swapping step of the anisotropic tensor renormalization group
 * (ATRG) [Adachi, Okubo, and Todo, Phys. Rev. B 102, 054432 (2020)], where
 * two half-tensors are contracted over i and the result is decomposed:
 *
 *     Theta[u, v, a, x, y, b] = sum_i B[i, u, v, a] * C[i, x, y, b]
 *     Theta ~= sum_g X[a, x, y, g] * s[g] * Y[g, u, v, b]
 *
 * Theta is built on the host on every rank purely to produce input data;
 * the SVD below is the sample's only distributed cuTensorNet call.
 *
 * The input and output factors use independent block-cyclic distributions.
 * Along each distributed mode, a positive block size assigns consecutive
 * blocks round-robin across process-grid coordinates, while block size zero
 * selects one near-even contiguous slab per coordinate. A process-grid
 * extent of one leaves that mode undistributed.
 *
 * Run with: mpirun -n 4 decompose_example_mpi_nccl
 * Use one GPU per process. CUTENSORNET_COMM_LIB must point at the MPI
 * distributed interface library.
 */

#include <stdlib.h>
#include <stdio.h>

#include <algorithm>
#include <cmath>
#include <vector>

#include <mpi.h>
#include <cuda_runtime.h>
#include <cutensornet.h>

#define HANDLE_ERROR(x)                                                          \
{ const auto err = x;                                                            \
  if( err != CUTENSORNET_STATUS_SUCCESS )                                        \
  { printf("Error: %s in line %d\n", cutensornetGetErrorString(err), __LINE__);  \
    fflush(stdout); MPI_Abort(MPI_COMM_WORLD, err); }                            \
};

#define HANDLE_CUDA_ERROR(x)                                                     \
{ const auto err = x;                                                            \
  if( err != cudaSuccess )                                                       \
  { printf("CUDA Error: %s in line %d\n", cudaGetErrorString(err), __LINE__);    \
    fflush(stdout); MPI_Abort(MPI_COMM_WORLD, err); }                            \
};

#define HANDLE_MPI_ERROR(x)                                                      \
{ const auto err = x;                                                            \
  if( err != MPI_SUCCESS )                                                       \
  { char error[MPI_MAX_ERROR_STRING]; int len;                                   \
    MPI_Error_string(err, error, &len);                                          \
    printf("MPI Error: %s in line %d\n", error, __LINE__);                       \
    fflush(stdout); MPI_Abort(MPI_COMM_WORLD, err); }                            \
};

/** One contiguous run of global indices owned by this rank, and where it
 *  sits in the rank's packed local shard. */
struct OwnedRun
{
    int64_t globalStart;
    int64_t localStart;
    int64_t count;
};

/** Return this rank's owned runs of one mode.
 *
 *  Block-cyclic ownership: a mode of global extent E with block size bs
 *  over p ranks is cut into ceil(E / bs) blocks of bs consecutive indices
 *  (the last possibly short), dealt round-robin -- block k lives on rank
 *  k mod p as that rank's local block k / p. Hence global index g sits on
 *  rank (g / bs) mod p at local offset (g / bs) / p * bs + g % bs.
 *
 *  Block size 0 selects one contiguous near-even slab per rank; ranks
 *  r < E mod p own one extra element. Runs are globally ascending and
 *  locally consecutive. */
std::vector<OwnedRun> ownedRuns(int64_t extent, int64_t blockSize,
                                int64_t nranksForMode, int rank)
{
    const int64_t r = rank % nranksForMode;
    std::vector<OwnedRun> runs;
    if (blockSize == 0)
    {
        const int64_t base = extent / nranksForMode;
        const int64_t rem = extent % nranksForMode;
        const int64_t count = base + (r < rem ? 1 : 0);
        if (count > 0)
            runs.push_back({r * base + std::min(r, rem), 0, count});
        return runs;
    }
    const int64_t numBlocks = (extent + blockSize - 1) / blockSize;
    int64_t localStart = 0;
    for (int64_t k = r; k < numBlocks; k += nranksForMode)
    {
        const int64_t start = k * blockSize;
        const int64_t count = std::min(blockSize, extent - start);
        runs.push_back({start, localStart, count});
        localStart += count;
    }
    return runs;
}

/** Total number of owned elements of one mode (the local extent). */
int64_t ownedCount(const std::vector<OwnedRun>& runs)
{
    int64_t count = 0;
    for (const auto& run: runs) count += run.count;
    return count;
}

/** Deterministic values in [-1, 1] so that every rank generates the same
 *  global tensors without communication. */
double pseudoRandom(uint64_t& state)
{
    state = state * 6364136223846793005ULL + 1442695040888963407ULL;
    return 2.0 * static_cast<double>(state >> 11) / 9007199254740992.0 - 1.0;
}

int main(int argc, char** argv)
{
    // Sphinx: #2
    /*******************************************
    * MPI setup: one GPU per process
    ********************************************/

    HANDLE_MPI_ERROR( MPI_Init(&argc, &argv) );

    int rank{-1}, numRanks{0};
    HANDLE_MPI_ERROR( MPI_Comm_rank(MPI_COMM_WORLD, &rank) );
    HANDLE_MPI_ERROR( MPI_Comm_size(MPI_COMM_WORLD, &numRanks) );
    const bool verbose = (rank == 0);

    // The device must be selected before cutensornetCreate().
    int numDevices{0};
    HANDLE_CUDA_ERROR( cudaGetDeviceCount(&numDevices) );
    HANDLE_CUDA_ERROR( cudaSetDevice(rank % numDevices) );

    if (verbose)
        printf("cuTensorNet-vers:%ld\n", cutensornetGetVersion());

    cutensornetHandle_t handle;
    HANDLE_ERROR( cutensornetCreate(&handle) );

    // Bind a duplicated MPI communicator; it must stay alive until the next
    // reset. All distributed calls below are collective over it.
    MPI_Comm cutnComm;
    HANDLE_MPI_ERROR( MPI_Comm_dup(MPI_COMM_WORLD, &cutnComm) );
    HANDLE_ERROR( cutensornetDistributedResetConfiguration(handle, &cutnComm, sizeof(cutnComm)) );

    // Sphinx: #3
    /*******************************************************
    * Build Theta = sum_i B[i,u,v,a] C[i,x,y,b] on the host
    ********************************************************/

    const int64_t D = 7;      // spatial degrees of freedom (u, v, x, y); kept extent of g
    const int64_t CHI = 13;   // vertical bond dimension (a, b)
    const int64_t CHI_T = 6;  // temporal bond dimension (i), contracted away below

    std::vector<double> bHalf(CHI_T * D * D * CHI);
    std::vector<double> cHalf(CHI_T * D * D * CHI);
    uint64_t rngState = 42;
    for (auto& value: bHalf) value = pseudoRandom(rngState);
    for (auto& value: cHalf) value = pseudoRandom(rngState);

    // Theta in Fortran order over modes (u, v, a, x, y, b); B and C in
    // Fortran order over (i, u, v, a) and (i, x, y, b).
    const size_t elementsTheta = D * D * CHI * D * D * CHI;
    std::vector<double> theta(elementsTheta, 0.0);
    for (int64_t b = 0; b < CHI; ++b)
    for (int64_t y = 0; y < D; ++y)
    for (int64_t x = 0; x < D; ++x)
    for (int64_t a = 0; a < CHI; ++a)
    for (int64_t v = 0; v < D; ++v)
    for (int64_t u = 0; u < D; ++u)
    {
        double sum = 0.0;
        for (int64_t i = 0; i < CHI_T; ++i)
            sum += bHalf[i + CHI_T * (u + D * (v + D * a))]
                 * cHalf[i + CHI_T * (x + D * (y + D * b))];
        theta[u + D * (v + D * (a + CHI * (x + D * (y + D * b))))] = sum;
    }

    // Sphinx: #4
    /*******************************************************
    * Create the distributed tensor descriptors (collective)
    ********************************************************/

    const cudaDataType_t typeData = CUDA_R_64F;

    std::vector<int32_t> modesTheta{'u','v','a','x','y','b'};
    std::vector<int32_t> modesX{'a','x','y','g'};
    std::vector<int32_t> modesY{'g','u','v','b'};

    std::vector<int64_t> extentsTheta{D, D, CHI, D, D, CHI};
    // The shared extent of the outputs at creation time is the truncation
    // cap: the bond swap keeps g = D of the D*D*CHI available singular
    // values. This example truncates to that cap; a value-based cutoff
    // (CUTENSORNET_TENSOR_SVD_CONFIG_ABS_CUTOFF / _REL_CUTOFF /
    // _DISCARDED_WEIGHT_CUTOFF) would instead let the retained extent be
    // decided by the spectrum, bounded above by this cap.
    std::vector<int64_t> extentsX{CHI, D, D, D};
    std::vector<int64_t> extentsY{D, D, D, CHI};

    // Theta is split over u, X over x, and Y over u. The shared mode g
    // remains undistributed. Positive block sizes select round-robin blocks;
    // zero selects slabs. NULL strides select compact Fortran-order shards.
    const int64_t BS = 2;
    std::vector<int64_t> gridTheta{numRanks, 1, 1, 1, 1, 1};
    std::vector<int64_t> gridX{1, numRanks, 1, 1};
    std::vector<int64_t> gridY{1, numRanks, 1, 1};
    std::vector<int64_t> blocksTheta{BS, 0, 0, 0, 0, 0};
    std::vector<int64_t> blocksX{0, BS, 0, 0};
    std::vector<int64_t> blocksY{0, BS, 0, 0};

    cutensornetTensorDescriptor_t descTheta, descX, descY;
    // Every rank creates the same descriptors in the same order.
    HANDLE_ERROR( cutensornetCreateDistributedTensorDescriptor(handle,
                    modesTheta.size(), extentsTheta.data(), NULL, blocksTheta.data(), NULL,
                    gridTheta.data(), modesTheta.data(), typeData, &descTheta) );
    HANDLE_ERROR( cutensornetCreateDistributedTensorDescriptor(handle,
                    modesX.size(), extentsX.data(), NULL, blocksX.data(), NULL,
                    gridX.data(), modesX.data(), typeData, &descX) );
    HANDLE_ERROR( cutensornetCreateDistributedTensorDescriptor(handle,
                    modesY.size(), extentsY.data(), NULL, blocksY.data(), NULL,
                    gridY.data(), modesY.data(), typeData, &descY) );

    if (verbose)
        printf("Created distributed descriptors: Theta split over u across %d rank(s); "
               "bond swap keeps g = %ld of %ld singular values\n",
               numRanks, D, D * D * CHI);

    // Sphinx: #5
    /*******************************************************
    * Allocate the local shards and scatter Theta
    ********************************************************/

    // Rank-local queries return each compact Fortran-order shard's size.
    size_t thetaLocalBytes{0}, xLocalBytes{0}, yLocalBytes{0};
    HANDLE_ERROR( cutensornetTensorDescriptorGetAttribute(handle, descTheta,
                    CUTENSORNET_TENSOR_DESCRIPTOR_LOCAL_DATA_SIZE,
                    &thetaLocalBytes, sizeof(thetaLocalBytes)) );
    HANDLE_ERROR( cutensornetTensorDescriptorGetAttribute(handle, descX,
                    CUTENSORNET_TENSOR_DESCRIPTOR_LOCAL_DATA_SIZE,
                    &xLocalBytes, sizeof(xLocalBytes)) );
    HANDLE_ERROR( cutensornetTensorDescriptorGetAttribute(handle, descY,
                    CUTENSORNET_TENSOR_DESCRIPTOR_LOCAL_DATA_SIZE,
                    &yLocalBytes, sizeof(yLocalBytes)) );

    // This rank's shard of Theta: the block-cyclically owned runs of the u
    // mode (with BS=2 and two ranks, mode u of extent 7 has blocks [0:2)
    // [2:4) [4:6) [6:7); rank 0 owns blocks 0 and 2, rank 1 owns 1 and 3),
    // packed consecutively in Fortran order over the local extents.
    const std::vector<OwnedRun> uRuns = ownedRuns(D, BS, numRanks, rank);
    const int64_t uCount = ownedCount(uRuns);
    std::vector<double> thetaLocal(thetaLocalBytes / sizeof(double));
    for (int64_t b = 0; b < CHI; ++b)
    for (int64_t y = 0; y < D; ++y)
    for (int64_t x = 0; x < D; ++x)
    for (int64_t a = 0; a < CHI; ++a)
    for (int64_t v = 0; v < D; ++v)
    for (const auto& run: uRuns)
    for (int64_t o = 0; o < run.count; ++o)
    {
        const int64_t u = run.globalStart + o;
        const int64_t uLoc = run.localStart + o;
        thetaLocal[uLoc + uCount * (v + D * (a + CHI * (x + D * (y + D * b))))] =
            theta[u + D * (v + D * (a + CHI * (x + D * (y + D * b))))];
    }

    void *dTheta{nullptr}, *dX{nullptr}, *dS{nullptr}, *dY{nullptr};
    const size_t sBytes = D * sizeof(double);  // replicated on every rank
    if (thetaLocalBytes > 0)
        HANDLE_CUDA_ERROR( cudaMalloc(&dTheta, thetaLocalBytes) );
    if (xLocalBytes > 0)
        HANDLE_CUDA_ERROR( cudaMalloc(&dX, xLocalBytes) );
    HANDLE_CUDA_ERROR( cudaMalloc(&dS, sBytes) );
    if (yLocalBytes > 0)
        HANDLE_CUDA_ERROR( cudaMalloc(&dY, yLocalBytes) );
    if (thetaLocalBytes > 0)
        HANDLE_CUDA_ERROR( cudaMemcpy(dTheta, thetaLocal.data(), thetaLocalBytes,
                                      cudaMemcpyHostToDevice) );

    // Initialize the output buffers before the SVD overwrites them.
    {
        std::vector<double> init(std::max(xLocalBytes, yLocalBytes) / sizeof(double));
        for (auto& value: init) value = pseudoRandom(rngState);
        if (xLocalBytes > 0)
            HANDLE_CUDA_ERROR( cudaMemcpy(dX, init.data(), xLocalBytes,
                                          cudaMemcpyHostToDevice) );
        if (yLocalBytes > 0)
            HANDLE_CUDA_ERROR( cudaMemcpy(dY, init.data(), yLocalBytes,
                                          cudaMemcpyHostToDevice) );
    }

    // Sphinx: #6
    /*******************************************************
    * SVD config/info and workspace (same calls as local)
    ********************************************************/

    // No cutoffs are set here, so the SVD truncates to the cap the output
    // descriptors were created with. Distributed SVD requires GESVDP; the
    // config default (GESVD) returns CUTENSORNET_STATUS_NOT_SUPPORTED.
    cutensornetTensorSVDConfig_t svdConfig;
    HANDLE_ERROR( cutensornetCreateTensorSVDConfig(handle, &svdConfig) );
    const cutensornetTensorSVDAlgo_t svdAlgo = CUTENSORNET_TENSOR_SVD_ALGO_GESVDP;
    HANDLE_ERROR( cutensornetTensorSVDConfigSetAttribute(handle, svdConfig,
                    CUTENSORNET_TENSOR_SVD_CONFIG_ALGO, &svdAlgo, sizeof(svdAlgo)) );
    cutensornetTensorSVDInfo_t svdInfo;
    HANDLE_ERROR( cutensornetCreateTensorSVDInfo(handle, &svdInfo) );

    cutensornetWorkspaceDescriptor_t workDesc;
    HANDLE_ERROR( cutensornetCreateWorkspaceDescriptor(handle, &workDesc) );
    // Collective: every rank queries with the same descriptor trio.
    HANDLE_ERROR( cutensornetWorkspaceComputeSVDSizes(handle, descTheta, descX, descY,
                                                      svdConfig, workDesc) );
    int64_t deviceWorkspaceSize{0}, hostWorkspaceSize{0};
    HANDLE_ERROR( cutensornetWorkspaceGetMemorySize(handle, workDesc,
                    CUTENSORNET_WORKSIZE_PREF_MIN, CUTENSORNET_MEMSPACE_DEVICE,
                    CUTENSORNET_WORKSPACE_SCRATCH, &deviceWorkspaceSize) );
    HANDLE_ERROR( cutensornetWorkspaceGetMemorySize(handle, workDesc,
                    CUTENSORNET_WORKSIZE_PREF_MIN, CUTENSORNET_MEMSPACE_HOST,
                    CUTENSORNET_WORKSPACE_SCRATCH, &hostWorkspaceSize) );

    void* devWork{nullptr};
    void* hostWork{nullptr};
    if (deviceWorkspaceSize > 0)
        HANDLE_CUDA_ERROR( cudaMalloc(&devWork, deviceWorkspaceSize) );
    if (hostWorkspaceSize > 0)
        hostWork = malloc(hostWorkspaceSize);
    HANDLE_ERROR( cutensornetWorkspaceSetMemory(handle, workDesc,
                    CUTENSORNET_MEMSPACE_DEVICE, CUTENSORNET_WORKSPACE_SCRATCH,
                    devWork, deviceWorkspaceSize) );
    HANDLE_ERROR( cutensornetWorkspaceSetMemory(handle, workDesc,
                    CUTENSORNET_MEMSPACE_HOST, CUTENSORNET_WORKSPACE_SCRATCH,
                    hostWork, hostWorkspaceSize) );

    // Sphinx: #7
    /*******************************************************
    * Execution: the distributed bond-swap SVD
    ********************************************************/

    cudaStream_t stream;
    HANDLE_CUDA_ERROR( cudaStreamCreate(&stream) );

    // Fixed-extent truncation: Kred == Kcap. After value-based truncation,
    // descX/descY describe Kred; recreate them at Kcap before calling again.
    HANDLE_ERROR( cutensornetTensorSVD(handle,
                    descTheta, dTheta,
                    descX, dX,
                    dS,
                    descY, dY,
                    svdConfig, svdInfo, workDesc, stream) );

    // Outputs become visible to work enqueued on `stream` after the call.
    std::vector<double> xLocal(xLocalBytes / sizeof(double));
    std::vector<double> sHost(D);
    std::vector<double> yLocal(yLocalBytes / sizeof(double));
    if (xLocalBytes > 0)
        HANDLE_CUDA_ERROR( cudaMemcpyAsync(xLocal.data(), dX, xLocalBytes,
                                           cudaMemcpyDeviceToHost, stream) );
    HANDLE_CUDA_ERROR( cudaMemcpyAsync(sHost.data(), dS, sBytes,
                                       cudaMemcpyDeviceToHost, stream) );
    if (yLocalBytes > 0)
        HANDLE_CUDA_ERROR( cudaMemcpyAsync(yLocal.data(), dY, yLocalBytes,
                                           cudaMemcpyDeviceToHost, stream) );
    HANDLE_CUDA_ERROR( cudaStreamSynchronize(stream) );

    int64_t reducedExtent{0};
    double discardedWeight{0.0};
    HANDLE_ERROR( cutensornetTensorSVDInfoGetAttribute(handle, svdInfo,
                    CUTENSORNET_TENSOR_SVD_INFO_REDUCED_EXTENT,
                    &reducedExtent, sizeof(reducedExtent)) );
    HANDLE_ERROR( cutensornetTensorSVDInfoGetAttribute(handle, svdInfo,
                    CUTENSORNET_TENSOR_SVD_INFO_DISCARDED_WEIGHT,
                    &discardedWeight, sizeof(discardedWeight)) );

    // Sphinx: #8
    /*******************************************************
    * Verification
    ********************************************************/

    // The singular values are replicated: every rank must hold the values
    // rank 0 holds.
    std::vector<double> sRoot(sHost);
    HANDLE_MPI_ERROR( MPI_Bcast(sRoot.data(), D, MPI_DOUBLE, 0, MPI_COMM_WORLD) );
    double sReplicationError = 0.0;
    for (int64_t g = 0; g < D; ++g)
        sReplicationError = std::max(sReplicationError, std::fabs(sHost[g] - sRoot[g]));

    // Reassemble X and Y on every rank (zero-fill + allreduce over the
    // block-cyclically owned entries), then reconstruct the truncated
    // Theta. X is split over its x mode, Y over its u mode, both with the
    // same block size, so both reuse the u-mode ownership runs.
    const size_t elementsX = CHI * D * D * D;
    const size_t elementsY = D * D * D * CHI;
    std::vector<double> xFull(elementsX, 0.0);
    std::vector<double> yFull(elementsY, 0.0);
    for (int64_t g = 0; g < D; ++g)
    for (int64_t y = 0; y < D; ++y)
    for (const auto& run: uRuns)
    for (int64_t o = 0; o < run.count; ++o)
    for (int64_t a = 0; a < CHI; ++a)
    {
        const int64_t x = run.globalStart + o;
        const int64_t xLoc = run.localStart + o;
        xFull[a + CHI * (x + D * (y + D * g))] =
            xLocal[a + CHI * (xLoc + uCount * (y + D * g))];
    }
    for (int64_t b = 0; b < CHI; ++b)
    for (int64_t v = 0; v < D; ++v)
    for (const auto& run: uRuns)
    for (int64_t o = 0; o < run.count; ++o)
    for (int64_t g = 0; g < D; ++g)
    {
        const int64_t u = run.globalStart + o;
        const int64_t uLoc = run.localStart + o;
        yFull[g + D * (u + D * (v + D * b))] =
            yLocal[g + D * (uLoc + uCount * (v + D * b))];
    }
    HANDLE_MPI_ERROR( MPI_Allreduce(MPI_IN_PLACE, xFull.data(), elementsX,
                                    MPI_DOUBLE, MPI_SUM, MPI_COMM_WORLD) );
    HANDLE_MPI_ERROR( MPI_Allreduce(MPI_IN_PLACE, yFull.data(), elementsY,
                                    MPI_DOUBLE, MPI_SUM, MPI_COMM_WORLD) );

    // The relative squared residual of the reconstruction must equal the
    // reported discarded weight: both are the weight of the singular values
    // beyond the cap.
    double residual2 = 0.0, thetaNorm2 = 0.0;
    for (int64_t b = 0; b < CHI; ++b)
    for (int64_t y = 0; y < D; ++y)
    for (int64_t x = 0; x < D; ++x)
    for (int64_t a = 0; a < CHI; ++a)
    for (int64_t v = 0; v < D; ++v)
    for (int64_t u = 0; u < D; ++u)
    {
        double rec = 0.0;
        for (int64_t g = 0; g < D; ++g)
            rec += xFull[a + CHI * (x + D * (y + D * g))] * sHost[g]
                 * yFull[g + D * (u + D * (v + D * b))];
        const double ref = theta[u + D * (v + D * (a + CHI * (x + D * (y + D * b))))];
        residual2 += (rec - ref) * (rec - ref);
        thetaNorm2 += ref * ref;
    }
    const double relativeResidual2 = residual2 / thetaNorm2;

    if (verbose)
    {
        printf("reduced extent: %ld (cap %ld)\n", reducedExtent, D);
        printf("leading singular values:");
        for (int64_t g = 0; g < std::min<int64_t>(D, 4); ++g)
            printf(" %.6f", sHost[g]);
        printf(" ...\n");
        printf("truncation: relative residual^2 = %.6f, reported discarded weight = %.6f\n",
               relativeResidual2, discardedWeight);
    }
    if (sReplicationError > 1e-12)
    {
        printf("Error: singular values differ across ranks (%e)\n", sReplicationError);
        MPI_Abort(MPI_COMM_WORLD, 1);
    }
    if (std::fabs(relativeResidual2 - discardedWeight) > 1e-10)
    {
        printf("Error: reconstruction inconsistent with discarded weight\n");
        MPI_Abort(MPI_COMM_WORLD, 1);
    }
    if (verbose)
        printf("Distributed bond-swap SVD verified.\n");

    // Sphinx: #9
    /*******************************************************
    * Free resources
    ********************************************************/

    HANDLE_CUDA_ERROR( cudaStreamDestroy(stream) );
    HANDLE_ERROR( cutensornetDestroyTensorDescriptor(descTheta) );
    HANDLE_ERROR( cutensornetDestroyTensorDescriptor(descX) );
    HANDLE_ERROR( cutensornetDestroyTensorDescriptor(descY) );
    HANDLE_ERROR( cutensornetDestroyTensorSVDConfig(svdConfig) );
    HANDLE_ERROR( cutensornetDestroyTensorSVDInfo(svdInfo) );
    HANDLE_ERROR( cutensornetDestroyWorkspaceDescriptor(workDesc) );
    HANDLE_ERROR( cutensornetDestroy(handle) );

    if (dTheta) HANDLE_CUDA_ERROR( cudaFree(dTheta) );
    if (dX) HANDLE_CUDA_ERROR( cudaFree(dX) );
    HANDLE_CUDA_ERROR( cudaFree(dS) );
    if (dY) HANDLE_CUDA_ERROR( cudaFree(dY) );
    if (devWork) HANDLE_CUDA_ERROR( cudaFree(devWork) );
    if (hostWork) free(hostWork);

    HANDLE_MPI_ERROR( MPI_Comm_free(&cutnComm) );
    HANDLE_MPI_ERROR( MPI_Finalize() );

    if (verbose)
        printf("Free resource and exit.\n");

    return 0;
}
