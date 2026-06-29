/*
 * Copyright (c) 2023-2026, NVIDIA CORPORATION & AFFILIATES.
 *
 * SPDX-License-Identifier: BSD-3-Clause
 */

#include <cuda_runtime_api.h> // cudaMalloc, cudaMemcpy, etc.
#include <cuComplex.h>        // cuDoubleComplex
#include <custatevec.h>       // custatevecSubSVMigratorMigrate
#include <stdio.h>            // printf
#include <stdlib.h>           // EXIT_FAILURE

#include "helper.hpp"         // HANDLE_ERROR, HANDLE_CUDA_ERROR

int main(void) {

    const cudaDataType_t svDataType = CUDA_C_64F;

    const int nSliceLocalIndexBits = 3;
    const int64_t subSvSize = int64_t(1) << nSliceLocalIndexBits;

    // allocate host memory
    const int nSubSvs = 2;
    cuDoubleComplex* subSvs[nSubSvs];
    const size_t subSvSizeInBytes = subSvSize * sizeof(cuDoubleComplex);
    for (int iSv = 0; iSv < nSubSvs; iSv++) {
        HANDLE_CUDA_ERROR( cudaMallocHost(&subSvs[iSv], subSvSizeInBytes) );
    }

    // fill subSvs[0]
    for (int i = 0; i < subSvSize; i++) {
        subSvs[0][i] = {0.25, 0.00};
    }

    // allocate device memory
    const int nDeviceSlices = 1;
    cuDoubleComplex* deviceSlices;
    const size_t deviceSlicesSizeInBytes = nDeviceSlices * subSvSize * sizeof(cuDoubleComplex);
    HANDLE_CUDA_ERROR( cudaMalloc(&deviceSlices, deviceSlicesSizeInBytes) );

    //----------------------------------------------------------------------------------------------

    // initialize custatevec handle
    custatevecHandle_t handle;
    HANDLE_ERROR( custatevecCreate(&handle) );

    // create migrator
    custatevecSubSVMigratorDescriptor_t migrator;
    HANDLE_ERROR( custatevecSubSVMigratorCreate(handle, &migrator, deviceSlices, svDataType,
                                                nDeviceSlices, nSliceLocalIndexBits) );

    int deviceSliceIndex = 0;
    cuDoubleComplex* srcSubSvSlice = subSvs[0];
    cuDoubleComplex* dstSubSvSlice = subSvs[1];

    // migrate subSvs[0] into deviceSlices
    HANDLE_ERROR( custatevecSubSVMigratorMigrate(handle, migrator, deviceSliceIndex, srcSubSvSlice,
                                                 nullptr, 0, subSvSize) );

    // migrate deviceSlices into subSvs[1]
    HANDLE_ERROR( custatevecSubSVMigratorMigrate(handle, migrator, deviceSliceIndex, nullptr,
                                                 dstSubSvSlice, 0, subSvSize) );

    // destroy migrator
    HANDLE_ERROR( custatevecSubSVMigratorDestroy(handle, migrator));

    // destroy custatevec handle
    HANDLE_ERROR( custatevecDestroy(handle) );

    //----------------------------------------------------------------------------------------------

    // check if subSvs[1] has expected values
    bool correct = true;
    for (int i = 0; i < subSvSize; i++) {
        cuDoubleComplex expectedValue = {0.25, 0.00};
        if (!almost_equal(subSvs[1][i], expectedValue)) {
            correct = false;
            break;
        }
    }

    // free host memory
    for (int iSv = 0; iSv < nSubSvs; iSv++) {
        HANDLE_CUDA_ERROR( cudaFreeHost(subSvs[iSv]) );
    }

    // free device memory
    HANDLE_CUDA_ERROR( cudaFree(deviceSlices) );

    if (correct) {
        printf("subsv_migration example PASSED\n");
        return EXIT_SUCCESS;
    }
    else {
        printf("subsv_migration example FAILED: wrong result\n");
        return EXIT_FAILURE;
    }
}