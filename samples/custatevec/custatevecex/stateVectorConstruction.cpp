/*
 * Copyright (c) 2025-2026, NVIDIA CORPORATION & AFFILIATES.
 *
 * SPDX-License-Identifier: BSD-3-Clause
 */

/**
 * @file stateVectorConstruction.cpp
 * @brief State Vector Construction Factory for cuStateVec Ex API
 *
 * This module provides a unified factory interface for creating state vector
 * configurations across different distribution types:
 * - Single Device: One GPU
 * - Multi Device: Multiple GPUs with P2P
 * - Multi Process: Distributed across processes with MPI
 * - Host Memory: Hybrid usage of both GPU and host memory
 * - Multi Process + Host Memory: Distributed across processes with MPI, where
 *                                each process uses both GPU and host memory
 */

#include <custatevecEx.h>
#include <custatevecEx_ext.h>
#include "stateVectorConstruction.hpp"
#include "networkStructure.hpp"
#include <cstdio>
#include <cstdlib>
#include <vector>
#include <numeric>  // std::iota
#include <cstring>  // strcmp
#include <cstdarg>  // va_list, va_start, va_end
#include <unistd.h> // getopt

//
// Module-level variables
//

// Module-level flag to track multi-process environment state
static bool isMultiProcess_ = false;

// Module-level flag to track whether the user requested quiet mode (-q).
static bool isQuietModeEnabled_ = false;

// True when custatevecExCommunicatorInitialize() succeeded (Finalize must be called on shutdown)
static bool communicatorLibraryInitialized_ = false;

// Module-level communicator for multi-process operations (non-null when creation succeeded)
static custatevecExCommunicatorDescriptor_t exCommunicator_ = nullptr;

// Rank of module-level communicator, permanently bound after exCommunicator
// initialisation, persisting even after exCommunicator destruction
static int exCommunicatorRank_ = 0;

// Module-level data type for state vector
static cudaDataType_t svDataType_ = CUDA_C_32F;

//
// Helper functions for gracefully exiting
//

static void gracefullyEndMultiProcessEnvironment()
{
    // Destroy communicator first (required before Finalize per API contract)
    if (exCommunicator_ != nullptr)
    {
        custatevecExCommunicatorDestroy(exCommunicator_);
        exCommunicator_ = nullptr;
    }

    // Finalize IPC library if it was successfully initialized
    if (communicatorLibraryInitialized_)
    {
        custatevecExCommunicatorStatus_t status;
        custatevecExCommunicatorFinalize(&status);
        communicatorLibraryInitialized_ = false;
    }

    // Mark multi-process as disabled
    isMultiProcess_ = false;
}

static void errorThenExit(const char* format, ...)
{
    // Only root prints (though all processes think they're root during init),
    // and always does so regardless of whether user has enabled quiet mode
    if (exCommunicatorRank_ == 0)
    {
        printf("Error: ");
        va_list args;
        va_start(args, format);
        vprintf(format, args);
        va_end(args);
        fflush(stdout);
    }

    // Always safe to call, even when not in a multi-process env
    gracefullyEndMultiProcessEnvironment();
    exit(EXIT_FAILURE);
}

//
// Cmd-line arg parsing
//

struct CmdLineArgs
{

    bool showHelpMessage;      // h cmd line arg
    bool suppressOutput;       // q ...
    bool useMultiProcess;      // p
    int numDevices;            // d
    int numMigrationWires;     // m
    cudaDataType_t svDataType; // t
    int networkFlag;           // k

    // one of below will be determined by networkFlag (k)
    custatevecDeviceNetworkType_t multiDeviceNetwork;
    NetworkLayers multiProcessNetwork;
};

/**
 * @brief Show usage information
 */
static void showHelpMessage(const char* programName)
{
    output("Usage: %s [options]\n", programName);
    output("\n");
    output("Options:\n");
    output("  -h          Show this help message and exit\n");
    output("  -q          Quiet mode (suppress output, except errors)\n");
    output("  -p          Multi-process mode\n");
    output("  -d <num>    Number of devices for multi-device (default is 1)\n");
    output("  -m <num>    Number of migration wires for host-memory usage (default: 0)\n");
    output("  -t <type>   Data type - 'f'/'float' or 'd'/'double' (default: float)\n");
    output("  -k <net>    Device network topology for multi-device: 1=SWITCH, 2=FULLMESH "
           "(default: SWITCH, indicated by =0)\n");
    output("              Network structure for multi-process: 3=SuperPOD, 4=GB200NVL, "
           "5=SwitchTree, 6=Communicator (default: SuperPOD, indicated by =0)\n");
    output("\n");
    output("Examples:\n");
    output("  %s                          # Use default settings\n", programName);
    output("  %s -d 2                     # Use 2 GPUs in multi-device mode\n", programName);
    output("  %s -m 2                     # Use 2 migration wires, storing 75%% of state in host "
           "memory\n",
           programName);
    output("  %s -t double                # Use double precision\n", programName);
    output("  mpirun -np 4 %s -p          # Use 4 processes\n", programName);
    output("  mpirun -np 4 %s -p -k 4     # Use 4 processes with GB200NVL network structure\n",
           programName);
    output("\n");
}

/**
 * @brief Extract, validate and effect cmd-line options
 *
 * Note this is deliberately called twice within the multi-process examples; first within
 * bootstrapMultiProcessEnvironment(), and then in configureStateVector().
 * This is because the non-multi-process examples only call configureStateVector().
 */
static CmdLineArgs processCmdLineArgs(int argc, char* argv[])
{
    CmdLineArgs out;

    // Default parameters
    out.showHelpMessage = false; // Don't show help msg
    out.suppressOutput = false;  // Show all output
    out.useMultiProcess = false; // Use single-process, potentially duplicating processes
    out.numDevices = 1;          // Use single device
    out.numMigrationWires = 0;   // Don't use host-memory
    out.svDataType = CUDA_C_32F; // Use float
    out.networkFlag = 0;         // Use default specific to multi-device or multi-process

    // Parse command line options with getopt
    int opt;

    // Reset getopt state for proper parsing
    optind = 1;

    // Collect and validate (in isolation) each command line option
    while ((opt = getopt(argc, argv, "hpqd:m:t:k:")) != -1)
    {
        switch (opt)
        {
        case 'h':
            out.showHelpMessage = true; // validate all args before showing help msg
            break;
        case 'q':
            out.suppressOutput = true; // never suppress error messages
            break;
        case 'p':
            out.useMultiProcess = true;
            break;
        case 'd':
            out.numDevices = atoi(optarg);
            if (out.numDevices < 1)
            {
                errorThenExit("Number of devices (%d) must be positive\n", out.numDevices);
            }
            if (out.numDevices & (out.numDevices - 1))
            {
                errorThenExit("Number of devices (%d) must be a power of 2\n", out.numDevices);
            }
            break;
        case 'm':
            out.numMigrationWires = atoi(optarg);
            if (out.numMigrationWires < 0)
            {
                errorThenExit("Number of migration wires must be positive or zero\n");
            }
            break;
        case 't':
            if (strcmp(optarg, "f") == 0 || strcmp(optarg, "float") == 0)
            {
                out.svDataType = CUDA_C_32F;
            }
            else if (strcmp(optarg, "d") == 0 || strcmp(optarg, "double") == 0)
            {
                out.svDataType = CUDA_C_64F;
            }
            else
            {
                errorThenExit("Data type (%s) must be 'f'/'float' or 'd'/'double'\n", optarg);
            }
            break;
        case 'k':
            out.networkFlag = atoi(optarg);
            if (out.networkFlag < 0 || out.networkFlag > 6)
            {
                errorThenExit("Network type (%d) must be in range 0-6\n", out.networkFlag);
            }
            break;
        case '?':
            // getopt already printed error message for unknown option
            break;
        default:
            break;
        }
    }

    // Validate combinations of arguments
    if (out.numDevices > 1 && out.useMultiProcess)
    {
        errorThenExit("Only one device can be utilised per-process in multi-process mode\n");
    }
    if (out.numDevices > 1 && out.numMigrationWires > 0)
    {
        errorThenExit("Cannot use host memory (via migration wires) in multi-device mode\n");
    }
    if (out.networkFlag != 0 && out.numDevices == 1 && !out.useMultiProcess)
    {
        errorThenExit("Cannot specify network flag in single-device and single-process mode\n");
    }

    // Validate and set network in multi-device mode
    // (Ignore out.multiProcessNetwork which will never be consulted)
    if (out.numDevices > 1)
    {
        switch (out.networkFlag)
        {
        case 0: // default
        case 1:
            out.multiDeviceNetwork = CUSTATEVEC_DEVICE_NETWORK_TYPE_SWITCH;
            break;
        case 2:
            out.multiDeviceNetwork = CUSTATEVEC_DEVICE_NETWORK_TYPE_FULLMESH;
            break;
        case 3:
        case 4:
        case 5:
        case 6:
            errorThenExit("Cannot use multi-process network layers (3-6) in multi-device mode\n");
            break;
        default:
            errorThenExit("Unknown network topology flag (%d)\n", out.networkFlag);
            break;
        }
    }

    // Validate and set network in multi-process mode
    // (Ignore out.multiDeviceNetwork which will never be consulted)
    if (out.useMultiProcess)
    {
        switch (out.networkFlag)
        {
        case 0: // default
        case 3:
            out.multiProcessNetwork = createSuperPODNetworkConfig();
            break;
        case 4:
            out.multiProcessNetwork = createGB200NVLNetworkConfig();
            break;
        case 5:
            out.multiProcessNetwork = createSwitchTreeNetworkConfig();
            break;
        case 6:
            out.multiProcessNetwork = createCommunicatorNetwork();
            break;
        case 1:
        case 2:
            errorThenExit(
                "Cannot use a multi-device network topology (1-2) in multi-process mode\n");
            break;
        default:
            errorThenExit("Unknown network layer flag (%d)\n", out.networkFlag);
            break;
        }
    }

    // Handle help message (will be duplicated across MPI processes, for simplicity)
    if (out.showHelpMessage)
    {
        showHelpMessage(argv[0]);
        gracefullyEndMultiProcessEnvironment();
        exit(EXIT_SUCCESS);
    }

    // Handle output suppression
    isQuietModeEnabled_ = out.suppressOutput;
    if (out.suppressOutput)
    {
        setOutputEnabled(false); // silence all processes, including root
    }
    // Caller may subsequently silence non-root processes

    return out;
}

//
// Multi-process Environment Preparation
//

/**
 * @brief Bootstrap multi-process environment.
 *
 * This function consults the command-line arguments and prepares the communicator as
 * necessary for a multi-process environment, overwriting the module-level vars above.
 */
void bootstrapMultiProcessEnvironment(int* argc, char*** argv)
{
    // Validate and effect cmd-line args (such as suppress-output)
    CmdLineArgs args = processCmdLineArgs(*argc, *argv);

    // Users must specify -p to attempt to initialize a multi-process environment,
    // otherwise all processes will run independently, in single-process mode
    if (!args.useMultiProcess)
    {
        return;
    }

    // Communicator configuration - choose one:
    // Option 1: Use built-in OPENMPI (default)
    custatevecCommunicatorType_t communicatorType = CUSTATEVEC_COMMUNICATOR_TYPE_OPENMPI;
    const char* libraryPath = nullptr;

    // Option 2: Use external communicator plugin (uncomment to enable)
    // custatevecCommunicatorType_t communicatorType = CUSTATEVEC_COMMUNICATOR_TYPE_EXTERNAL;
    // const char* libraryPath = "./libexMpiCommunicator.so";  // or nullptr to search in process

    // Try to initialize communicator
    custatevecExCommunicatorStatus_t commStatus;
    // clang-format off
    custatevecStatus_t status = custatevecExCommunicatorInitialize(
        communicatorType,       // communicatorType
        libraryPath,            // libraryPath
        argc,                   // argc
        argv,                   // argv
        &commStatus             // output status
    );
    // clang-format on

    // Check initialization succeeded
    if (status != CUSTATEVEC_STATUS_SUCCESS ||
        commStatus != CUSTATEVEC_EX_COMMUNICATOR_STATUS_SUCCESS)
    {
        errorThenExit("custatevecExCommunicatorInitialize failed (status=%d, commStatus=%d)\n",
                      static_cast<int>(status), static_cast<int>(commStatus));
    }

    // Mark initialization as successful
    communicatorLibraryInitialized_ = true;

    // Try to create a communicator
    status = custatevecExCommunicatorCreate(&exCommunicator_);

    // Check communicator creation succeeded
    if (status != CUSTATEVEC_STATUS_SUCCESS)
    {
        errorThenExit("custatevecExCommunicatorCreate failed (status=%d)\n",
                      static_cast<int>(status));
    }

    // We are now in a multi-process environment; obtain the number of processes and rank
    int numProcesses = -1;
    ERRCHK_EXCOMM(exCommunicator_->intf->getSize(exCommunicator_, &numProcesses));
    ERRCHK_EXCOMM(exCommunicator_->intf->getRank(exCommunicator_, &exCommunicatorRank_));

    // Check the number of processes is valid
    if (numProcesses == 1)
    {
        errorThenExit("Must use more than 1 process in multi-process mode.");
    }
    if (numProcesses < 1 || (numProcesses & (numProcesses - 1)) != 0)
    {
        errorThenExit("number of processes (%d) must be a positive power of 2.\n", numProcesses);
    }

    // Suppress output on non-root processes (unless user suppresses ALL output)
    setOutputEnabled(exCommunicatorRank_ == 0 && !args.suppressOutput);

    // Officiate multi-process environment
    isMultiProcess_ = true;
}

/**
 * @brief Finalize multi-process environment
 */
void finalizeMultiProcessEnvironment()
{
    gracefullyEndMultiProcessEnvironment();
}

/**
 * @brief Get the multi-process communicator
 */
custatevecExCommunicatorDescriptor_t getMultiProcessCommunicator()
{
    return exCommunicator_; // may be nullptr
}

/**
 * @brief Get rank in the multi-process communicator
 */
int getMultiProcessRank()
{
    // Can be non-zero even after exCommunicator_ destruction (isMultiProcess_=0)
    return exCommunicatorRank_;
}

/**
 * @brief Report whether the user requested quiet mode (-q).
 *
 * This is distinct to whether output is enabled, since when quiet mode is NOT
 * enabled, non-root output may still be disabled, to suppress duplicate output
 */
bool isQuietModeEnabled()
{
    return isQuietModeEnabled_;
}

//
// State Vector Configuration Factory
//

/**
 * @brief Create single-device state vector configuration (internal)
 *
 * @param svDataType State vector data type (CUDA_C_32F or CUDA_C_64F)
 * @param numWires Number of qubits
 * @param numMigrationWires Number of migration wires, informing portion of
 *   the state vector stored in host memory
 * @return Dictionary containing state vector configuration
 */
static custatevecExDictionaryDescriptor_t
createSingleDeviceConfig(cudaDataType_t svDataType, int32_t numWires, int32_t numMigrationWires)
{
    custatevecExDictionaryDescriptor_t svConfig{nullptr};

    // clang-format off
    ERRCHK(custatevecExConfigureStateVectorSingleDevice(
        &svConfig,
        svDataType,
        numWires,                     // numWires
        numWires - numMigrationWires, // numDeviceWires
        -1,                           // deviceId, -1 specifies the current device.
        0                             // capability
    ));
    // clang-format on

    return svConfig;
}

/**
 * @brief Create multi-device state vector configuration
 */
static custatevecExDictionaryDescriptor_t
createMultiDeviceConfig(cudaDataType_t svDataType, int numWires, int numDevices,
                        custatevecDeviceNetworkType_t networkType)
{
    custatevecExDictionaryDescriptor_t svConfig;

    // Generate device IDs (0, 1, 2, ..., numDevices-1)
    std::vector<int32_t> deviceIds(numDevices);
    std::iota(deviceIds.begin(), deviceIds.end(), 0);

    // Calculate device wires: numWires = numGlobalBits + numDeviceWires
    // numGlobalBits = log2(numDevices) for inter-device communication
    int32_t numGlobalBits = 0;
    int32_t tempDevices = numDevices;
    while (tempDevices > 1)
    {
        numGlobalBits++;
        tempDevices >>= 1;
    }
    int32_t numDeviceWires = numWires - numGlobalBits;

    // clang-format off
    ERRCHK(custatevecExConfigureStateVectorMultiDevice(&svConfig,
        svDataType,                              // data type
        numWires,                                // total number of qubits
        numDeviceWires,                          // qubits per device
        deviceIds.data(),                        // device IDs
        numDevices,                              // number of devices
        networkType,                             // network type (configurable)
        0                                        // capability flags
    ));
    // clang-format on

    return svConfig;
}

/**
 * @brief Create multi-process state vector configuration (internal)
 *
 * @param svDataType State vector data type (CUDA_C_32F or CUDA_C_64F)
 * @param numWires Total number of qubits
 * @param numMigrationWires Number of migration wires, informing portion of
 *   the state vector stored in host memory
 * @param networkLayers Network topology configuration
 * @param exCommDesc Communicator descriptor
 * @return Dictionary containing state vector configuration
 */
static custatevecExDictionaryDescriptor_t createMultiProcessConfig(
    cudaDataType_t svDataType, int32_t numWires, int32_t numMigrationWires,
    const NetworkLayers& networkLayers, custatevecExCommunicatorDescriptor_t exCommDesc)
{
    custatevecExDictionaryDescriptor_t svConfig{nullptr};

    // Get rank and size from communicator
    int32_t rank, numProcesses;
    ERRCHK_EXCOMM(exCommDesc->intf->getRank(exCommDesc, &rank));
    ERRCHK_EXCOMM(exCommDesc->intf->getSize(exCommDesc, &numProcesses));

    // Find numInterProcWires = log2(numProcesses)
    int32_t numInterProcBits = 0;
    int32_t temp = numProcesses;
    while (temp > 1)
    {
        numInterProcBits++;
        temp >>= 1;
    }

    // We will collect and group global bits, which include inter-process and migration wires
    std::vector<custatevecExGlobalIndexBitClass_t> globalIndexBitClasses;
    std::vector<int32_t> numGlobalIndexBitsPerLayer;

    // Here, we treat migration wires as the least significant global bits, but note that the
    // optimal layer position depends on the targeted network topology. For instance, host-device
    // migration could be faster than IB data transfer, and the ideal layer ordering would then be
    // NVLink layer -> Migration layer -> IB layer.
    if (numMigrationWires > 0)
    {
        globalIndexBitClasses.push_back(CUSTATEVEC_EX_GLOBAL_INDEX_BIT_CLASS_MIGRATION);
        numGlobalIndexBitsPerLayer.push_back(numMigrationWires);
    }

    // Collect non-empty inter-process global bit groups and count total assigned bits
    int32_t numAccumulatedGlobalIndexBits = 0;
    for (const auto& layer : networkLayers)
    {
        globalIndexBitClasses.push_back(layer.globalIndexBitClass);
        int numGlobalBitsPerLayer;
        if (layer.numGlobalIndexBits == 0)
        {
            numGlobalBitsPerLayer = numInterProcBits - numAccumulatedGlobalIndexBits;
            numGlobalIndexBitsPerLayer.push_back(numGlobalBitsPerLayer);
            numAccumulatedGlobalIndexBits = numInterProcBits;
            break;
        }
        auto maxNumGlobalIndexBitsPerLayer = numInterProcBits - numAccumulatedGlobalIndexBits;
        if (maxNumGlobalIndexBitsPerLayer <= layer.numGlobalIndexBits)
        {
            numGlobalIndexBitsPerLayer.push_back(maxNumGlobalIndexBitsPerLayer);
            numAccumulatedGlobalIndexBits += maxNumGlobalIndexBitsPerLayer;
            break;
        }
        numGlobalIndexBitsPerLayer.push_back(layer.numGlobalIndexBits);
        numAccumulatedGlobalIndexBits += layer.numGlobalIndexBits;
    }
    if (numAccumulatedGlobalIndexBits < numInterProcBits)
    {
        errorThenExit("NetworkLayers is too thin to build numWires state vector.\n");
    }

    // All remaining wires are local, spanned within a single device
    int32_t numDeviceWires = numWires - numInterProcBits - numMigrationWires;
    int32_t deviceId = -1; // Dynamic device assignment on creating state vector

    // Specify only exchanges between P2P processes can leverage shared memory
    custatevecExMemorySharingMethod_t memorySharingMethod =
        CUSTATEVEC_EX_MEMORY_SHARING_METHOD_NONE;
    for (const auto& layer : networkLayers)
    {
        if (layer.globalIndexBitClass == CUSTATEVEC_EX_GLOBAL_INDEX_BIT_CLASS_INTERPROC_P2P)
        {
            memorySharingMethod = CUSTATEVEC_EX_MEMORY_SHARING_METHOD_AUTODETECT;
            break;
        }
    }

    // Transfer workspace size (16 MB default)
    const size_t transferWorkspaceSizeInBytes = 16 * 1024 * 1024;
    // clang-format off
    ERRCHK(custatevecExConfigureStateVectorMultiProcess(
        &svConfig,
        svDataType,
        numWires,                                           // numWires
        numDeviceWires,                                     // numDeviceWires
        deviceId,                                           // deviceId
        memorySharingMethod,                                // memorySharingMethod (auto-detect if P2P used)
        globalIndexBitClasses.data(),                       // globalIndexBitClasses
        numGlobalIndexBitsPerLayer.data(),                  // numGlobalIndexBitsPerLayer
        static_cast<int32_t>(globalIndexBitClasses.size()), // numGlobalIndexBitLayers
        transferWorkspaceSizeInBytes,                       // transferWorkspaceSizeInBytes
        nullptr,                                            // auxConfig (reserved)
        0                                                   // capability (reserved)
    ));
    // clang-format on

    return svConfig;
}

/**
 * @brief Configure state vector from command line arguments
 */
custatevecExDictionaryDescriptor_t configureStateVector(int argc, char* argv[], int numWires)
{
    // Validate and effect cmd-line args (such as suppress-output)
    // (This is being called for the second time after bootstrapMP(), which is fine)
    CmdLineArgs args = processCmdLineArgs(argc, argv);

    // Multi-process is only possible if enabled in prior bootstrapMultiProcessEnvironment()
    if (args.useMultiProcess && !isMultiProcess_)
    {
        errorThenExit(
            "StateVector config args requested multi-process but env was not initialized\n");
    }

    // Store the configured data type for later retrieval
    svDataType_ = args.svDataType;
    auto dataTypeStr = (args.svDataType == CUDA_C_32F) ? "float" : "double";

    // Dispatch to deployment-specific factory
    if (args.numDevices > 1 && !args.useMultiProcess && args.numMigrationWires == 0)
    {
        output("Configure state vector: Multi-device, Qubits: %d, DataType: %s, Devices: %d\n",
               numWires, dataTypeStr, args.numDevices);
        return createMultiDeviceConfig(args.svDataType, numWires, args.numDevices,
                                       args.multiDeviceNetwork);
    }
    if (args.numDevices == 1 && !args.useMultiProcess)
    {
        output(
            "Configure state vector: Single-device, Qubits: %d, MigrationWires: %d, DataType: %s\n",
            numWires, args.numMigrationWires, dataTypeStr);
        return createSingleDeviceConfig(args.svDataType, numWires, args.numMigrationWires);
    }
    if (args.numDevices == 1 && args.useMultiProcess)
    {
        int numProcesses = -1;
        ERRCHK_EXCOMM(exCommunicator_->intf->getSize(exCommunicator_, &numProcesses));
        output("Configure state vector: Multi-process (%d processes), Qubits: %d, MigrationWires: "
               "%d, DataType: %s, NetworkType: %d\n",
               numProcesses, numWires, args.numMigrationWires, dataTypeStr, args.networkFlag);
        return createMultiProcessConfig(args.svDataType, numWires, args.numMigrationWires,
                                        args.multiProcessNetwork, exCommunicator_);
    }

    errorThenExit("Unknown or unsupported configuration");
    return nullptr;
}

/**
 * @brief Get the configured state vector data type
 *
 * Must not be called before configureStateVector()
 */
cudaDataType_t getStateVectorDataType()
{
    return svDataType_;
}

//
// State Vector Creation
//

/**
 * @brief Create state vector from configuration
 */
custatevecExStateVectorDescriptor_t createStateVector(custatevecExDictionaryDescriptor_t config)
{
    custatevecExStateVectorDescriptor_t stateVector;

    // Branch based on detected environment
    if (isMultiProcess_)
    {
        // Use the stored communicator from bootstrapMultiProcessEnvironment
        // clang-format off
        ERRCHK(custatevecExStateVectorCreateMultiProcess(&stateVector, config,
                                                         nullptr,           // stream
                                                         exCommunicator_,   // communicator
                                                         nullptr));         // resourceManager
        // clang-format on
    }
    else
    {
        ERRCHK(
            custatevecExStateVectorCreateSingleProcess(&stateVector, config, nullptr, 0, nullptr));
    }

    return stateVector;
}
