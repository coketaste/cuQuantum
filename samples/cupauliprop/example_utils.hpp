/*
 * Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
 *
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES.
 * SPDX-License-Identifier: BSD-3-Clause
 */

#pragma once

#include <cupauliprop.h>
#include <cuda_runtime.h>

#include <cstddef>
#include <cstdint>
#include <cstdlib>
#include <iostream>
#include <random>


/*
 * ERROR HANDLING
 */

#define HANDLE_CUPP_ERROR(x)                                             \
{                                                                        \
  const auto err = x;                                                    \
  if (err != CUPAULIPROP_STATUS_SUCCESS) {                               \
    std::cerr << "cuPauliProp error at line " << __LINE__ << std::endl;  \
    std::abort();                                                        \
  }                                                                      \
}

#define HANDLE_CUDA_ERROR(x)                                             \
{                                                                        \
  const auto err = x;                                                    \
  if (err != cudaSuccess) {                                              \
    std::cerr << "CUDA error at line " << __LINE__ << ": "               \
              << cudaGetErrorString(err) << std::endl;                   \
    std::abort();                                                        \
  }                                                                      \
}


/*
 * RANDOM OPERATOR PREPARATION
 */

cupaulipropQuantumOperator_t createRandomPauliRotationGate(
  cupaulipropHandle_t handle,
  std::mt19937_64& rng,
  int32_t numQubits);

cupaulipropQuantumOperator_t createRandomPauliNoiseChannel(
  cupaulipropHandle_t handle,
  std::mt19937_64& rng,
  int32_t numQubits);

cupaulipropQuantumOperator_t createRandomCliffordGate(
  cupaulipropHandle_t handle,
  std::mt19937_64& rng,
  int32_t numQubits);


/*
 * RANDOM EXPANSION PREPARATION
 */

void growExpansionViaRandomRotations(
  cupaulipropHandle_t handle,
  cudaStream_t stream,
  cupaulipropPauliExpansion_t& inOutExpansion,
  cupaulipropPauliExpansion_t& tempExpansion,
  int32_t numQubits,
  int64_t targetNumTerms,
  size_t pauliBufferSize,
  size_t coefBufferSize,
  void* d_workspaceBuffer,
  size_t workspaceBufferSize);
