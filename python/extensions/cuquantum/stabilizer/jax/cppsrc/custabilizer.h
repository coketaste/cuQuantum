/*
 * Copyright 2025-2026 NVIDIA CORPORATION & AFFILIATES.  All rights reserved.
 *
 * NOTICE TO LICENSEE:
 *
 * This source code and/or documentation ("Licensed Deliverables") are
 * subject to NVIDIA intellectual property rights under U.S. and
 * international Copyright laws.
 *
 * These Licensed Deliverables contained herein is PROPRIETARY and
 * CONFIDENTIAL to NVIDIA and is being provided under the terms and
 * conditions of a form of NVIDIA software license agreement by and
 * between NVIDIA and Licensee ("License Agreement") or electronically
 * accepted by Licensee.  Notwithstanding any terms or conditions to
 * the contrary in the License Agreement, reproduction or disclosure
 * of the Licensed Deliverables to any third party without the express
 * written consent of NVIDIA is prohibited.
 *
 * NOTWITHSTANDING ANY TERMS OR CONDITIONS TO THE CONTRARY IN THE
 * LICENSE AGREEMENT, NVIDIA MAKES NO REPRESENTATION ABOUT THE
 * SUITABILITY OF THESE LICENSED DELIVERABLES FOR ANY PURPOSE.  IT IS
 * PROVIDED "AS IS" WITHOUT EXPRESS OR IMPLIED WARRANTY OF ANY KIND.
 * NVIDIA DISCLAIMS ALL WARRANTIES WITH REGARD TO THESE LICENSED
 * DELIVERABLES, INCLUDING ALL IMPLIED WARRANTIES OF MERCHANTABILITY,
 * NONINFRINGEMENT, AND FITNESS FOR A PARTICULAR PURPOSE.
 * NOTWITHSTANDING ANY TERMS OR CONDITIONS TO THE CONTRARY IN THE
 * LICENSE AGREEMENT, IN NO EVENT SHALL NVIDIA BE LIABLE FOR ANY
 * SPECIAL, INDIRECT, INCIDENTAL, OR CONSEQUENTIAL DAMAGES, OR ANY
 * DAMAGES WHATSOEVER RESULTING FROM LOSS OF USE, DATA OR PROFITS,
 * WHETHER IN AN ACTION OF CONTRACT, NEGLIGENCE OR OTHER TORTIOUS
 * ACTION, ARISING OUT OF OR IN CONNECTION WITH THE USE OR PERFORMANCE
 * OF THESE LICENSED DELIVERABLES.
 *
 * U.S. Government End Users.  These Licensed Deliverables are a
 * "commercial item" as that term is defined at 48 C.F.R. 2.101 (OCT
 * 1995), consisting of "commercial computer software" and "commercial
 * computer software documentation" as such terms are used in 48
 * C.F.R. 12.212 (SEPT 1995) and is provided to the U.S. Government
 * only as a commercial end item.  Consistent with 48 C.F.R.12.212 and
 * 48 C.F.R. 227.7202-1 through 227.7202-4 (JUNE 1995), all
 * U.S. Government End Users acquire the Licensed Deliverables with
 * only those rights set forth herein.
 *
 * Any use of the Licensed Deliverables in individual and commercial
 * software must include, in the user documentation and internal
 * comments to the code, the above Disclaimer and U.S. Government End
 * Users Notice.
 * </blockquote>}
 */

/**
 * @file
 * @brief This file contains all public function declarations of the
 * cuStabilizer library.
 */

#pragma once

#include <cuda_runtime_api.h>

// LIBRARY VERSION

#define CUSTABILIZER_MAJOR 0 //!< cuStabilizer major version.
#define CUSTABILIZER_MINOR 4 //!< cuStabilizer minor version.
#define CUSTABILIZER_PATCH 0 //!< cuStabilizer patch version.
#define CUSTABILIZER_VERSION \
  (CUSTABILIZER_MAJOR * 10000 + CUSTABILIZER_MINOR * 100 + CUSTABILIZER_PATCH) //!< cuStabilizer version

#if defined(__cplusplus)
#include <cstdint>
#include <cstdio>

extern "C" {
#else
#include <stdint.h>
#include <stdio.h>

#endif // defined(__cplusplus)

// -- < Constants and Enums > --

/**
 * \brief Status codes returned by the cuStabilizer library.
 */
typedef enum {
  /** The operation has completed successfully. */
  CUSTABILIZER_STATUS_SUCCESS = 0,
  /** Unspecified failure. */
  CUSTABILIZER_STATUS_ERROR = 1,
  /** cuStabilizer is not initialized. */
  CUSTABILIZER_STATUS_NOT_INITIALIZED = 2,
  /** One or more arguments are invalid. */
  CUSTABILIZER_STATUS_INVALID_VALUE = 3,
  /** Operation or feature is not supported. */
  CUSTABILIZER_STATUS_NOT_SUPPORTED = 4,
  /** Device memory allocation failed. */
  CUSTABILIZER_STATUS_ALLOC_FAILED = 5,
  /** Internal error occurred. */
  CUSTABILIZER_STATUS_INTERNAL_ERROR = 6,
  /** Insufficient workspace provided. */
  CUSTABILIZER_STATUS_INSUFFICIENT_WORKSPACE = 7,
  /** CUDA error occurred. */
  CUSTABILIZER_STATUS_CUDA_ERROR = 8,
  /** Insufficient storage for sparse output; resize and retry. */
  CUSTABILIZER_STATUS_INSUFFICIENT_SPARSE_STORAGE = 9,
} custabilizerStatus_t;

// -- </ Constants and Enums > --


// -- < Types and Structures > --

/**
 * \ingroup Circuit
 * \brief Opaque data structure holding the Circuit.
 *
 */
typedef void* custabilizerCircuit_t;

/**
 * \ingroup FrameSimulator
 * \brief Opaque data structure holding the simulator state.
 *
 */
typedef void* custabilizerFrameSimulator_t;

/**
 * \brief Opaque data structure holding the library context.
 */
typedef void* custabilizerHandle_t;

/**
 * \brief Integer type for specifying bit-packed tables.
 */
typedef uint32_t custabilizerBitInt_t;

// -- </ Types and Structures > --


/**
 * \brief Returns the semantic version number of the cuStabilizer library.
 * \return Combined version number in format 10000 * major + 100 * minor +
 * patch.
 */
int custabilizerGetVersion();

/**
 * \brief Get the description string for a given cuStabilizer status code.
 *
 * \param[in] status The status code.
 * \return A null-terminated string describing the status code.
 */
const char* custabilizerGetErrorString(custabilizerStatus_t status);

/**
 * \brief Create and initialize the library context.
 *
 * \param[out] handle Library handle.
 * \return custabilizerStatus_t
 */
custabilizerStatus_t custabilizerCreate(custabilizerHandle_t* handle);

/**
 * \brief Destroy the library context.
 *
 * \param[in] handle Library handle.
 * \return custabilizerStatus_t
 */
custabilizerStatus_t custabilizerDestroy(custabilizerHandle_t handle);


// -- < Circuit methods > --

/**
 * \defgroup Circuit Circuit
 * \{
 */

/**
 * \brief Returns the size of the device buffer required for a circuit.
 *
 * \param[in] handle Library handle.
 * \param[in] circuitString String representation of the circuit.
 * \param[out] bufferSize Size of the buffer in bytes.
 * \return custabilizerStatus_t
 */
custabilizerStatus_t custabilizerCircuitSizeFromString(const custabilizerHandle_t handle,
                                                       const char* circuitString,
                                                       int64_t* bufferSize);

/**
 * \brief Create a new circuit from a string representation.
 *
 * The string format is compatible with
 * [Stim](https://github.com/quantumlib/Stim) circuit string.
 *
 * \param[in] handle Library handle.
 * \param[in] circuitString String representation of the circuit.
 * \param[in,out] bufferDevice Device buffer to store the circuit.
 * \param[in] bufferSize Size of the device buffer in bytes.
 * \param[out] circuit Pointer to the created circuit.
 * \return custabilizerStatus_t
 *
 * Example:
 *
 * \code
 * custabilizerHandle_t handle;
 * custabilizerCreate(&handle);
 *
 * char circuitString[] =
 *    "H 0\n"
 *    "X_ERROR(0.5) 1\n"
 *    "CNOT 0 1\n";
 * int64_t bufferSize;
 * custabilizerCircuit_t circuit;
 * custabilizerCircuitSizeFromString(handle, circuitString, &bufferSize);
 * void *buffer;
 * cudaMalloc(&buffer, bufferSize);
 * custabilizerCreateCircuitFromString(handle, circuitString, buffer,
 *                                     bufferSize, &circuit);
 * \endcode
 *
 * Use \ref custabilizerFrameSimulatorApplyCircuit to run the circuit.
 *
 */
custabilizerStatus_t custabilizerCreateCircuitFromString(const custabilizerHandle_t handle,
                                                         const char* circuitString,
                                                         void* bufferDevice,
                                                         int64_t bufferSize,
                                                         custabilizerCircuit_t* circuit);

/**
 * \brief Destroy a circuit.
 *
 * \param[in] circuit Circuit to destroy.
 * \return custabilizerStatus_t
 */
custabilizerStatus_t custabilizerDestroyCircuit(custabilizerCircuit_t circuit);

/** \} */
// -- </ Circuit methods > --


// -- < Frame simulator methods > --

/**
 * \defgroup FrameSimulator FrameSimulator
 * \{
 */

/**
 * \brief Create a FrameSimulator
 *
 * \param[in] handle Library handle.
 * \param[in] numQubits Number of qubits in the Pauli frame.
 * \param[in] numShots Number of samples to simulate.
 * \param[in] numMeasurements Number of measurements in the measurement table
 * \param[in] tableStrideMajor Stride over the major axis for all input bit
 *            tables. Specified in bytes and must be a multiple of 4.
 * \param[out] frameSimulator Pointer to the created frame simulator.
 * \return custabilizerStatus_t
 *
 * \details
 * The stride is specified by the `tableStrideMajor` parameter, which is
 * usually `(numShots + 7)/8` padded to the next multiple of 4.
 *
 * The data is updated by calling \ref custabilizerFrameSimulatorApplyCircuit.
 *
 */
custabilizerStatus_t custabilizerCreateFrameSimulator(const custabilizerHandle_t handle,
                                                      int64_t numQubits,
                                                      int64_t numShots,
                                                      int64_t numMeasurements,
                                                      int64_t tableStrideMajor,
                                                      custabilizerFrameSimulator_t* frameSimulator);

/**
 * \brief Destroy the FrameSimulator
 *
 * \param[in] frameSimulator Frame simulator to destroy.
 * \return custabilizerStatus_t
 */
custabilizerStatus_t custabilizerDestroyFrameSimulator(custabilizerFrameSimulator_t frameSimulator);

/**
 * \brief Run Pauli frame simulation using the circuit
 *
 * \param[in] handle Library handle.
 * \param[in] frameSimulator An instance of FrameSimulator with parameters
 *            consistent with the bit tables
 * \param[in] circuit A circuit that acts on at most `numQubits`
 *            and contains at most `numMeasurements` measurements
 * \param[in] randomizeFrameAfterMeasurement Disabling the randomization is
 *            helpful in some cases to focus on the error frame propagation.
 * \param[in] seed Random seed.
 * \param[in,out] xTableDevice Device buffer of the X bit table in qubit-major order.
 *            Must be of size at least `numQubits` * `tableStrideMajor`
 * \param[in,out] zTableDevice Device buffer of the Z bit table in qubit-major order.
 *            Must be of size at least `numQubits` * `tableStrideMajor`
 * \param[in,out] mTableDevice Device buffer of the measurement bit table in
 *            measurement-major order. Must be of size at least
 *            `numMeasurements` * `tableStrideMajor`
 * \param[in] stream CUDA stream.
 * \return custabilizerStatus_t
 *
 * \details
 * Use \ref custabilizerCreateFrameSimulator to create a frame simulator with 
 * appropriate parameters for this call.
 * The method accepts an initial state in the form of bit tables.
 * All bit tables assume LSB ordering. That is, the bit for the first shot is
 * stored at mask 0x1. If the buffers are smaller than required minimum size,
 * the behavior is undefined.
 *
 * The `xTableDevice` and `zTableDevice` specify the initial Pauli frame in a qubit-major format.
 * The operator on Pauli string `I` and qubit `J` is encoded by
 * bits `I` on row `J` in x_table and z_table.
 *
 * \code
 *  (x_table[J][I], z_table[J][I])   Pauli operator
 *               0, 0                     I
 *               0, 1                     Z
 *               1, 0                     X
 *               1, 1                     Y
 * \endcode
 *
 * Here is an illustrative example of bit tables with 4 paulis on 3 qubits:
 * `XYZ`, `IIZ`, `XII`, `IIY`.
 *
 * \code
 * int64_t numQubits = 3;
 * int64_t numShots = 32;
 * int64_t numMeasurements = 2;
 * int64_t stride = (numShots + 7) / 8;
 * int bit_table_bytes = numQubits * stride;
 * int m_table_bytes = numMeasurements * stride;
 * int bit_int_bytes = sizeof(custabilizerBitInt_t);
 * custabilizerBitInt_t x_table[bit_table_bytes / bit_int_bytes] = {
 *     //    IXIX 
 *     //    IIIY
 *     //    YIZZ 
 * // pauli  4321
 *     0x00000101, // Qubit 0
 *     0x00000001, // Qubit 1
 *     0x00001000  // Qubit 2
 * };
 * custabilizerBitInt_t z_table[bit_table_bytes / bit_int_bytes] = {
 * // pauli  4321
 *     0x00000000, // Qubit 0
 *     0x00000001, // Qubit 1
 *     0x00001011  // Qubit 2
 * };
 * custabilizerBitInt_t m_table[m_table_bytes / bit_int_bytes] = {
 *     0x00000000,
 *     0x00000000
 * };
 * custabilizerBitInt_t *xTableDevice, *zTableDevice, *mTableDevice;
 * cudaMalloc(&xTableDevice, bit_table_bytes);
 * cudaMalloc(&zTableDevice, bit_table_bytes);
 * cudaMalloc(&mTableDevice, m_table_bytes);
 * cudaMemcpy(xTableDevice, x_table, bit_table_bytes, cudaMemcpyHostToDevice);
 * cudaMemcpy(zTableDevice, z_table, bit_table_bytes, cudaMemcpyHostToDevice);
 * cudaMemcpy(mTableDevice, m_table, m_table_bytes, cudaMemcpyHostToDevice);
 *
 * custabilizerHandle_t handle;
 * custabilizerCreate(&handle);
 * custabilizerFrameSimulator_t frameSimulator;
 * custabilizerStatus_t status = custabilizerCreateFrameSimulator(handle,
 *                               numQubits, numShots, numMeasurements,
 *                               stride, &frameSimulator);
 *
 * int seed = 5;
 * cudaStream_t stream = 0;
 * int rnd_frame = 0;
 * // Assuming `circuit` is defined earlier
 * status = custabilizerFrameSimulatorApplyCircuit(
 *                               handle, frameSimulator, circuit, rnd_frame,
 *                               seed, xTableDevice, zTableDevice, mTableDevice, stream);
 *
 * custabilizerDestroyFrameSimulator(frameSimulator);
 * cudaFree(xTableDevice);
 * cudaFree(zTableDevice);
 * cudaFree(mTableDevice);
 * custabilizerDestroy(handle);
 *
 * \endcode
 */
custabilizerStatus_t custabilizerFrameSimulatorApplyCircuit(const custabilizerHandle_t handle,
                                                            custabilizerFrameSimulator_t frameSimulator,
                                                            const custabilizerCircuit_t circuit,
                                                            int randomizeFrameAfterMeasurement,
                                                            uint64_t seed,
                                                            custabilizerBitInt_t* xTableDevice,
                                                            custabilizerBitInt_t* zTableDevice,
                                                            custabilizerBitInt_t* mTableDevice,
                                                            cudaStream_t stream);

/** \} */
// -- </ Frame simulator methods > --

// -- < RNG methods > --

/**
 * \defgroup Rng Rng
 * \{
 */

/**
 * \brief Sample Bernoulli random bits from a probability array.
 *
 * Generates a dense bit-packed matrix of shape (numProbs, numSamples) where
 * row \p i contains independent Bernoulli(\p probs[i]) samples.
 *
 * The output is bit-packed using \ref custabilizerBitInt_t. Row \p i occupies
 * words \p [i * numSamples/32, (i+1) * numSamples/32). Within each word,
 * bit \p b corresponds to sample (word_index * 32 + b).
 *
 * \param[in] handle Library handle.
 * \param[in] numSamples Number of samples (minor dimension). Must be a multiple of 32.
 * \param[in] numProbs Number of probabilities (major dimension).
 * \param[in] probs Probability array of length \p numProbs (device-accessible pointer).
 *            Values should be in [0, 1]; out-of-range values are clamped, NaN is treated as 0.
 * \param[in] seed Random seed.
 * \param[out] samples Output buffer for bit-packed samples (device-accessible pointer). Must have at least
 *             (numProbs * numSamples / 32) words.
 * \param[in] stream CUDA stream.
 * \return custabilizerStatus_t
 */
custabilizerStatus_t custabilizerSampleProbArray(custabilizerHandle_t handle,
                                                 int64_t numSamples,
                                                 int64_t numProbs,
                                                 const double* probs,
                                                 uint64_t seed,
                                                 custabilizerBitInt_t* samples,
                                                 cudaStream_t stream);

/**
 * \brief Query the device workspace size required for sparse Bernoulli sampling.
 *
 * The returned size depends on \p numProbs and \p numSamples. A single
 * workspace allocation can be reused across calls with different seeds,
 * probability values, or output buffers, as long as numProbs and numSamples
 * do not exceed those used in the Prepare call.
 *
 * \param[in] handle Library handle.
 * \param[in] numSamples Number of samples.
 * \param[in] numProbs Number of probabilities.
 * \param[out] workspaceSize Required device workspace size in bytes.
 * \return custabilizerStatus_t
 */
custabilizerStatus_t custabilizerSampleProbArraySparsePrepare(custabilizerHandle_t handle,
                                                              int64_t numSamples,
                                                              int64_t numProbs,
                                                              size_t* workspaceSize);

/**
 * \brief Sample Bernoulli random bits from a probability array and return a CSR matrix.
 *
 * Generates a CSR matrix of shape (numSamples, numProbs), where each row \p s
 * contains column indices \p e such that an independent Bernoulli(\p probs[e]) trial succeeded.
 *
 * The CSR data is returned via \p rowOffsets (length \p numSamples+1) and \p columnIndices
 * (length \p *nnz on input).
 *
 * Capacity handling:
 * - On input, \p *nnz is the capacity of \p columnIndices (number of entries).
 * - On successful return, \p *nnz is the number of non-zeros written.
 * - If capacity is insufficient, returns \ref CUSTABILIZER_STATUS_INSUFFICIENT_SPARSE_STORAGE and sets
 *   \p *nnz to the required number of non-zeros.
 *
 * Column indices within each row are NOT sorted; they follow probability-sort
 * order from the tiled sampling kernel.
 *
 * \param[in] handle Library handle.
 * \param[in] numSamples Number of samples (rows / shots).
 * \param[in] numProbs Number of probabilities (columns).
 * \param[in] probs Probability array of length \p numProbs (device-accessible pointer).
 *            Values should be in [0, 1]; out-of-range values are clamped, NaN is treated as 0.
 * \param[in] seed Random seed.
 * \param[in,out] nnz On input, capacity of \p columnIndices. On output, number of non-zeros used (or required).
 * \param[out] columnIndices Output CSR column indices (device-accessible pointer), length at least \p *nnz on input.
 * \param[out] rowOffsets Output CSR row offsets (device-accessible pointer), length \p numSamples+1.
 * \param[in] workspace Device-accessible workspace of at least the size returned by Prepare.
 * \param[in] workspaceSize Size of workspace in bytes.
 * \param[in] stream CUDA stream.
 * \return custabilizerStatus_t
 */
custabilizerStatus_t custabilizerSampleProbArraySparseCompute(custabilizerHandle_t handle,
                                                              int64_t numSamples,
                                                              int64_t numProbs,
                                                              const double* probs,
                                                              uint64_t seed,
                                                              uint64_t* nnz,
                                                              uint64_t* columnIndices,
                                                              uint64_t* rowOffsets,
                                                              void* workspace,
                                                              size_t workspaceSize,
                                                              cudaStream_t stream);

/** \} */

// -- </ RNG methods > --


// -- < Linear algebra methods > --

/**
 * \defgroup Linalg Linalg
 * \{
 */

/**
 * \brief Compute GF(2) sparse-dense matrix multiplication \p C = A @ B.
 *
 * \p A is sparse (CSR), \p B is bit-packed dense.
 *
 * Shapes:
 * - \p A is CSR with shape (m, k)
 * - \p B is bit-packed dense with shape (k, n)
 * - \p C is bit-packed dense with shape (m, n)
 *
 * Packing requirement:
 * - \p n must be a multiple of 32 so that rows are word-aligned in the bit-packed storage.
 *
 * The multiplication is over GF(2): \( C = A \otimes B + \beta C \)
 *   - beta == 0: \( C  = A \otimes B \)  (assign, C is not read)
 *   - beta == 1: \( C ^= A \otimes B \)  (XOR-accumulate)
 *
 * \param[in] handle Library handle.
 * \param[in] m Number of rows of \p A and \p C.
 * \param[in] n Number of columns of \p B and \p C (must be a multiple of 32).
 * \param[in] k Number of columns of \p A and rows of \p B.
 * \param[in] nnz Number of non-zeros in \p A and length of \p columnIndices.
 * \param[in] columnIndices CSR column indices of \p A (device-accessible pointer), length \p nnz.
 * \param[in] rowOffsets CSR row offsets of \p A (device-accessible pointer), length \p m+1.
 * \param[in] B Bit-packed dense input matrix \p B (device-accessible pointer).
 * \param[in] beta 0 for assign (C not read), 1 for XOR-accumulate.
 * \param[in,out] C Bit-packed dense output matrix \p C (device-accessible pointer).
 * \param[in] stream CUDA stream.
 * \return custabilizerStatus_t
 */
custabilizerStatus_t custabilizerGF2SparseDenseMatrixMultiply(custabilizerHandle_t handle,
                                                              uint64_t m,
                                                              uint64_t n,
                                                              uint64_t k,
                                                              uint64_t nnz,
                                                              const uint64_t* columnIndices,
                                                              const uint64_t* rowOffsets,
                                                              const custabilizerBitInt_t* B,
                                                              int32_t beta,
                                                              custabilizerBitInt_t* C,
                                                              cudaStream_t stream);

/**
 * \brief Compute GF(2) sparse-sparse matrix multiplication \p C = A @ B.
 *
 * Both \p A and \p B are in CSR format with element-level sparsity.
 * \p B stores individual column indices (bit positions), not bit-packed words.
 *
 * Shapes:
 * - \p A is CSR with shape (m, k)
 * - \p B is CSR with shape (k, n), column indices are bit positions in [0, n)
 * - \p C is bit-packed dense with shape (m, n)
 *
 * Packing requirement:
 * - \p n must be a multiple of 32 so that rows are word-aligned in the bit-packed storage.
 *
 *   - beta == 0: \( C  = A \otimes B \)  (assign, C is not read)
 *   - beta == 1: \( C ^= A \otimes B \)  (XOR-accumulate)
 *
 * \param[in] handle Library handle.
 * \param[in] m Number of rows of \p A and \p C.
 * \param[in] n Number of columns of \p B and \p C (must be a multiple of 32).
 * \param[in] k Number of columns of \p A and rows of \p B.
 * \param[in] aColumnIndices CSR column indices of \p A (device-accessible pointer).
 * \param[in] aRowOffsets CSR row offsets of \p A (device-accessible pointer), length \p m+1.
 * \param[in] bNNZ Number of non-zeros in \p B.
 * \param[in] bColumnIndices CSR column indices of \p B (device-accessible pointer), length \p bNNZ.
 *            Column indices within each row must be sorted in ascending order.
 * \param[in] bRowOffsets CSR row offsets of \p B (device-accessible pointer), length \p k+1.
 * \param[in] beta 0 for assign (C not read), 1 for XOR-accumulate.
 * \param[in,out] C Bit-packed dense output matrix \p C (device-accessible pointer).
 * \param[in] stream CUDA stream.
 * \return custabilizerStatus_t
 */
custabilizerStatus_t custabilizerGF2SparseSparseMatrixMultiply(custabilizerHandle_t handle,
                                                               uint64_t m,
                                                               uint64_t n,
                                                               uint64_t k,
                                                               const uint64_t* aColumnIndices,
                                                               const uint64_t* aRowOffsets,
                                                               uint64_t bNNZ,
                                                               const uint64_t* bColumnIndices,
                                                               const uint64_t* bRowOffsets,
                                                               int32_t beta,
                                                               custabilizerBitInt_t* C,
                                                               cudaStream_t stream);

/** \} */

// -- </ Linear algebra methods > --


#if defined(__cplusplus)
} // extern "C"
#endif // defined(__cplusplus)
