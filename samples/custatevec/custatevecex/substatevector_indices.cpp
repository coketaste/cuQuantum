/*
 * Copyright (c) 2025-2026, NVIDIA CORPORATION & AFFILIATES.
 *
 * SPDX-License-Identifier: BSD-3-Clause
 */

//
// This example clarifies the meaning and utility of sub-statevector (subSV)
// indices, queriable from the cuStateVec Ex API. It can be run in various
// deployment configurations (single-device, multi-device, multi-process, 
// single-device + migration, multi-process + migration) to preview the
// subSV indices, illustrating how the statevector is distributed.
//
// Note the ordering of printed subSVs in multi-process deployment is
// arbitrary and can vary between executions. Furthermore, the subSV
// indices within a process are affected by the global wire ordering,
// and ergo the location of the migration wires.
//
// Example outputs:
//
// ./substatevector_indices -d 1
// 
//     Configure state vector: Single-device, Qubits: 20, MigrationWires: 0, DataType: float
//     process 0 subSVs: [0]
//     PASSED
//
// ./substatevector_indices -d 4
//
//     Configure state vector: Multi-device, Qubits: 20, DataType: float, Devices: 4
//     process 0 subSVs: [0 1 2 3]
//     PASSED
//
// ./substatevector_indices -d 1 -m 3
// 
//     Configure state vector: Single-device, Qubits: 20, MigrationWires: 3, DataType: float
//     process 0 subSVs: [0 1 2 3 4 5 6 7]
//     PASSED
//
// mpirun -np 4 ./substatevector_indices -p -k 6
//
//     Configure state vector: Multi-process, Qubits: 20, MigrationWires: 0, DataType: float, NetworkType: 6
//     process 0 subSVs: [0]
//     process 2 subSVs: [2]
//     process 1 subSVs: [1]
//     process 3 subSVs: [3]
//     PASSED
//
// mpirun -np 4 ./substatevector_indices -p -k 6 -m 3
//
//     Configure state vector: Multi-process, Qubits: 20, MigrationWires: 3, DataType: float, NetworkType: 6
//     process 0 subSVs: [0 1 2 3 4 5 6 7]
//     process 1 subSVs: [8 9 10 11 12 13 14 15]
//     process 3 subSVs: [24 25 26 27 28 29 30 31]
//     process 2 subSVs: [16 17 18 19 20 21 22 23]
//     PASSED
//

#include <custatevecEx.h>              // custatevecEx API
#include <stdlib.h>                    // exit()
#include <cstdio>                      // printf
#include <vector>                      // std::vector<>
#include <string>                      // std::to_string
#include "stateVectorConstruction.hpp" // Shared state vector construction module
#include "common.hpp"                  // Error checking utilities

int main(int argc, char* argv[])
{
    const int numWires = 20;

    // Bootstrap multi-process environment (silences non-root nodes)
    bootstrapMultiProcessEnvironment(&argc, &argv);
    
    // Configure and create state vector
    auto svConfig = configureStateVector(argc, argv, numWires);
    auto sv = createStateVector(svConfig);
    
    // Obtain substatevector indices
    int32_t numSubSVs;
    ERRCHK(custatevecExStateVectorGetProperty(sv, CUSTATEVEC_EX_SV_PROP_NUM_SUBSVS,
                                              &numSubSVs, sizeof(numSubSVs)));
    std::vector<int32_t> subSVIndices(numSubSVs);
    ERRCHK(custatevecExStateVectorGetProperty(sv, CUSTATEVEC_EX_SV_PROP_SUBSV_INDICES,
                                              subSVIndices.data(),
                                              numSubSVs * sizeof(int32_t)));

    // Un-silence non-root nodes (unless in quiet mode)
    setOutputEnabled(!isQuietModeEnabled());

    // Every process prints their local substatevector indices
    int rank = getMultiProcessRank();
    auto label = "process " + std::to_string(rank) + " subSVs";
    outputVectorDump(label.c_str(), subSVIndices);

    // Cleanup
    ERRCHK(custatevecExDictionaryDestroy(svConfig));
    ERRCHK(custatevecExStateVectorDestroy(sv));
    finalizeMultiProcessEnvironment();

    // Root process unconditionally prints success status
    if (rank == 0)
        printf("PASSED\n");
    return EXIT_SUCCESS;
}