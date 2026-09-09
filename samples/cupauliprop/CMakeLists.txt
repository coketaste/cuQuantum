##
## SPDX-FileCopyrightText: Copyright (c) 2024-2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
## SPDX-License-Identifier: BSD-3-Clause
##
## Redistribution and use in source and binary forms, with or without
## modification, are permitted provided that the following conditions are met:
##
## 1. Redistributions of source code must retain the above copyright notice, this
## list of conditions and the following disclaimer.
##
## 2. Redistributions in binary form must reproduce the above copyright notice,
## this list of conditions and the following disclaimer in the documentation
## and/or other materials provided with the distribution.
##
## 3. Neither the name of the copyright holder nor the names of its
## contributors may be used to endorse or promote products derived from
## this software without specific prior written permission.
##
## THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
## AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
## IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE
## DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE LIABLE
## FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL
## DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR
## SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER
## CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY,
## OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
## OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
##

cmake_minimum_required(VERSION 3.22.0 FATAL_ERROR)

# Guard against automatically setting CMAKE_CUDA_ARCHITECTURES by CMP0104
if(DEFINED CMAKE_CUDA_ARCHITECTURES AND CMAKE_CUDA_ARCHITECTURES)
  set(CMAKE_CUDA_ARCHITECTURES_DEFINED TRUE)
else()
  set(CMAKE_CUDA_ARCHITECTURES_DEFINED FALSE)
endif()

project(example_cuPauliProp LANGUAGES C CXX CUDA)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CUDA_STANDARD 17)

enable_language(CUDA)
find_package(CUDAToolkit REQUIRED)

if(NOT CMAKE_CUDA_ARCHITECTURES_DEFINED)
  if((${CUDAToolkit_VERSION_MAJOR} LESS 12) OR
     (${CUDAToolkit_VERSION_MAJOR} EQUAL 12 AND ${CUDAToolkit_VERSION_MINOR} LESS 8))
    set(CMAKE_CUDA_ARCHITECTURES 75-real;80-real;86-real;90)
  else()
    set(CMAKE_CUDA_ARCHITECTURES 75-real;80-real;86-real;90-real;100-real;120-real;100-virtual)
  endif()
endif()
message(STATUS "Target CUDA architectures: ${CMAKE_CUDA_ARCHITECTURES}")

# Find cuPauliProp
# CUPAULIPROP_ROOT takes precedence over CUQUANTUM_ROOT
if(DEFINED CUPAULIPROP_ROOT OR DEFINED ENV{CUPAULIPROP_ROOT})
  if(DEFINED ENV{CUPAULIPROP_ROOT})
    set(CUPAULIPROP_ROOT "$ENV{CUPAULIPROP_ROOT}")
  endif()
  message(STATUS "Using CUPAULIPROP_ROOT = ${CUPAULIPROP_ROOT}")
elseif(DEFINED CUQUANTUM_ROOT OR DEFINED ENV{CUQUANTUM_ROOT})
  if(DEFINED ENV{CUQUANTUM_ROOT})
    set(CUPAULIPROP_ROOT "$ENV{CUQUANTUM_ROOT}")
  else()
    set(CUPAULIPROP_ROOT "${CUQUANTUM_ROOT}")
  endif()
  message(STATUS "Using CUQUANTUM_ROOT = ${CUPAULIPROP_ROOT}")
else()
  message(FATAL_ERROR "Please set the environment variable CUPAULIPROP_ROOT or CUQUANTUM_ROOT to the path of the cuQuantum/cuPauliProp installation.")
endif()

message(STATUS "Looking for cuPauliProp in ${CUPAULIPROP_ROOT}")
if(NOT EXISTS ${CUPAULIPROP_ROOT})
  message(FATAL_ERROR "Cannot find CUPAULIPROP_ROOT at: ${CUPAULIPROP_ROOT}")
endif()
set(CUPAULIPROP_INC_DIR ${CUPAULIPROP_ROOT}/include)
# Auto-detect lib or lib64 directory for cuPauliProp
if(EXISTS "${CUPAULIPROP_ROOT}/lib64")
  set(CUPAULIPROP_LIB_DIR "${CUPAULIPROP_ROOT}/lib64")
elseif(EXISTS "${CUPAULIPROP_ROOT}/lib")
  set(CUPAULIPROP_LIB_DIR "${CUPAULIPROP_ROOT}/lib")
else()
  message(FATAL_ERROR "Could not find 'lib' or 'lib64' in CUPAULIPROP_ROOT: ${CUPAULIPROP_ROOT}")
endif()
message(STATUS "Found cuPauliProp library directory: ${CUPAULIPROP_LIB_DIR}")

# Kicked Ising example on heavy-hexagonal lattice
add_executable(kicked_ising_example
  kicked_ising_example.cpp
)
set_target_properties(kicked_ising_example
  PROPERTIES
    CXX_STANDARD ${CMAKE_CXX_STANDARD}
    CUDA_STANDARD ${CMAKE_CUDA_STANDARD}
    POSITION_INDEPENDENT_CODE ON
)
target_include_directories(kicked_ising_example
  PUBLIC
    ${CUPAULIPROP_INC_DIR}
    ${CMAKE_CUDA_TOOLKIT_INCLUDE_DIRECTORIES}
)
target_link_directories(kicked_ising_example
  PUBLIC
    ${CUPAULIPROP_LIB_DIR}
)
target_link_libraries(kicked_ising_example
  PUBLIC
    cupauliprop
    CUDA::cudart
)

# Kicked Ising example using fused Clifford application
add_executable(kicked_ising_fused_example
  kicked_ising_fused_example.cpp
)
set_target_properties(kicked_ising_fused_example
  PROPERTIES
    CXX_STANDARD ${CMAKE_CXX_STANDARD}
    CUDA_STANDARD ${CMAKE_CUDA_STANDARD}
    POSITION_INDEPENDENT_CODE ON
)
target_include_directories(kicked_ising_fused_example
  PUBLIC
    ${CUPAULIPROP_INC_DIR}
    ${CMAKE_CUDA_TOOLKIT_INCLUDE_DIRECTORIES}
)
target_link_directories(kicked_ising_fused_example
  PUBLIC
    ${CUPAULIPROP_LIB_DIR}
)
target_link_libraries(kicked_ising_fused_example
  PUBLIC
    cupauliprop
    CUDA::cudart
)

# Kicked Ising backward-diff example on heavy-hexagonal lattice
add_executable(kicked_ising_backward_diff_example
  kicked_ising_backward_diff_example.cpp
)
set_target_properties(kicked_ising_backward_diff_example
  PROPERTIES
    CXX_STANDARD ${CMAKE_CXX_STANDARD}
    CUDA_STANDARD ${CMAKE_CUDA_STANDARD}
    POSITION_INDEPENDENT_CODE ON
)
target_include_directories(kicked_ising_backward_diff_example
  PUBLIC
    ${CUPAULIPROP_INC_DIR}
    ${CMAKE_CUDA_TOOLKIT_INCLUDE_DIRECTORIES}
)
target_link_directories(kicked_ising_backward_diff_example
  PUBLIC
    ${CUPAULIPROP_LIB_DIR}
)
target_link_libraries(kicked_ising_backward_diff_example
  PUBLIC
    cupauliprop
    CUDA::cudart
)

# Optional distributed-provider support for the multiprocess example.
option(ENABLE_MPI "Enable MPI provider support in the multiprocess example" OFF)
option(ENABLE_NCCL "Enable NCCL provider support in the multiprocess example (requires MPI)" OFF)

if(ENABLE_MPI)
  find_package(MPI REQUIRED)
endif()

if(ENABLE_NCCL)
  if(NOT ENABLE_MPI)
    message(FATAL_ERROR "ENABLE_NCCL requires ENABLE_MPI: the NCCL provider is bootstrapped over MPI.")
  endif()
  if(DEFINED ENV{NCCL_ROOT} AND NOT DEFINED NCCL_ROOT)
    set(NCCL_ROOT "$ENV{NCCL_ROOT}")
  endif()
  if(NOT DEFINED NCCL_ROOT)
    message(FATAL_ERROR "ENABLE_NCCL requires NCCL_ROOT to point to an NCCL installation.")
  endif()
  set(NCCL_INCLUDE_DIR ${NCCL_ROOT}/include)
  if(EXISTS "${NCCL_ROOT}/lib")
    set(NCCL_LIB_DIR "${NCCL_ROOT}/lib")
  elseif(EXISTS "${NCCL_ROOT}/lib64")
    set(NCCL_LIB_DIR "${NCCL_ROOT}/lib64")
  endif()
  find_library(NCCL_LIBRARY
    NAMES nccl libnccl.so libnccl.so.2
    PATHS ${NCCL_LIB_DIR}
    NO_DEFAULT_PATH
  )
  if(NOT EXISTS "${NCCL_INCLUDE_DIR}/nccl.h" OR NOT NCCL_LIBRARY)
    message(FATAL_ERROR "Could not find NCCL under NCCL_ROOT: ${NCCL_ROOT}")
  endif()
  message(STATUS "Found NCCL: ${NCCL_LIBRARY}")
endif()

# Multiprocess kicked Ising example: one binary supporting every provider
# compiled in (single-process always; MPI and NCCL when enabled). The provider
# is selected at runtime from the plugin filename in $CUPAULIPROP_COMM_LIB.
add_executable(kicked_ising_multiprocess_example
  kicked_ising_multiprocess_example.cpp
)
set_target_properties(kicked_ising_multiprocess_example
  PROPERTIES
    CXX_STANDARD ${CMAKE_CXX_STANDARD}
    CUDA_STANDARD ${CMAKE_CUDA_STANDARD}
    POSITION_INDEPENDENT_CODE ON
)
target_include_directories(kicked_ising_multiprocess_example
  PUBLIC
    ${CUPAULIPROP_INC_DIR}
    ${CMAKE_CUDA_TOOLKIT_INCLUDE_DIRECTORIES}
)
target_link_directories(kicked_ising_multiprocess_example
  PUBLIC
    ${CUPAULIPROP_LIB_DIR}
)
target_link_libraries(kicked_ising_multiprocess_example
  PUBLIC
    cupauliprop
    CUDA::cudart
)
if(ENABLE_MPI)
  target_compile_definitions(kicked_ising_multiprocess_example PRIVATE MPI_ENABLED)
  target_link_libraries(kicked_ising_multiprocess_example PRIVATE MPI::MPI_CXX)
endif()
if(ENABLE_NCCL)
  target_compile_definitions(kicked_ising_multiprocess_example PRIVATE NCCL_ENABLED)
  target_include_directories(kicked_ising_multiprocess_example PRIVATE ${NCCL_INCLUDE_DIR})
  target_link_libraries(kicked_ising_multiprocess_example PRIVATE ${NCCL_LIBRARY})
endif()

# fused API demo
add_executable(fused_operators_example
  fused_operators_example.cpp
  example_utils.cpp
)
set_target_properties(fused_operators_example
  PROPERTIES
    CXX_STANDARD ${CMAKE_CXX_STANDARD}
    CUDA_STANDARD ${CMAKE_CUDA_STANDARD}
    POSITION_INDEPENDENT_CODE ON
)
target_include_directories(fused_operators_example
  PUBLIC
    ${CUPAULIPROP_INC_DIR}
    ${CMAKE_CUDA_TOOLKIT_INCLUDE_DIRECTORIES}
)
target_link_directories(fused_operators_example
  PUBLIC
    ${CUPAULIPROP_LIB_DIR}
)
target_link_libraries(fused_operators_example
  PUBLIC
    cupauliprop
    CUDA::cudart
)
