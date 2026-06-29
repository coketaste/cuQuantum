# Copyright (c) 2022-2026, NVIDIA CORPORATION & AFFILIATES.
#
# SPDX-License-Identifier: BSD-3-Clause

SHELL            := /bin/bash
CUDA_PATH        := ${CUDA_PATH}
CUDENSITYMAT_ROOT ?= ${CUQUANTUM_ROOT}
CUTENSORNET_ROOT  ?= ${CUQUANTUM_ROOT}
MPI_ROOT         := ${MPI_ROOT}
NCCL_ROOT        := ${NCCL_ROOT}
CUDA_MAJOR_VERSION := $(shell "${CUDA_PATH}/bin/"nvcc --version | egrep -o "V[0-9]+.[0-9]+.[0-9]" | cut -d. -f1 | cut -c2-)
CUDA_MINOR_VERSION := $(shell "${CUDA_PATH}/bin/"nvcc --version | egrep -o "V[0-9]+.[0-9]+.[0-9]" | cut -d. -f2)

# Auto-detect cuTENSOR library directory (same logic as CMakeLists.txt)
ifeq ($(shell test -d "${CUTENSOR_ROOT}/lib/x86_64-linux-gnu" && echo "exists"),exists)
  CUTENSOR_LIB_DIR := ${CUTENSOR_ROOT}/lib/x86_64-linux-gnu
else ifeq ($(shell test -d "${CUTENSOR_ROOT}/lib/aarch64-linux-gnu" && echo "exists"),exists)
  CUTENSOR_LIB_DIR := ${CUTENSOR_ROOT}/lib/aarch64-linux-gnu
else
  CUTENSOR_LIB_DIR := ${CUTENSOR_ROOT}/lib
endif

# Auto-detect NCCL library (handles pip packages that only have versioned .so.2)
ifdef NCCL_ROOT
  ifeq ($(shell test -f "${NCCL_ROOT}/lib/libnccl.so" && echo "exists"),exists)
    NCCL_LIB := ${NCCL_ROOT}/lib/libnccl.so
  else ifeq ($(shell test -f "${NCCL_ROOT}/lib64/libnccl.so" && echo "exists"),exists)
    NCCL_LIB := ${NCCL_ROOT}/lib64/libnccl.so
  else ifeq ($(shell test -f "${NCCL_ROOT}/lib/libnccl.so.2" && echo "exists"),exists)
    NCCL_LIB := ${NCCL_ROOT}/lib/libnccl.so.2
  else ifeq ($(shell test -f "${NCCL_ROOT}/lib64/libnccl.so.2" && echo "exists"),exists)
    NCCL_LIB := ${NCCL_ROOT}/lib64/libnccl.so.2
  endif
endif

ifeq ($(shell test -d "${CUDENSITYMAT_ROOT}/lib64" && echo "exists"),exists)
  CUDENSITYMAT_LIB_DIR := ${CUDENSITYMAT_ROOT}/lib64
else
  CUDENSITYMAT_LIB_DIR := ${CUDENSITYMAT_ROOT}/lib
endif

ifeq ($(shell test -d "${CUTENSORNET_ROOT}/lib64" && echo "exists"),exists)
  CUTENSORNET_LIB_DIR := ${CUTENSORNET_ROOT}/lib64
else
  CUTENSORNET_LIB_DIR := ${CUTENSORNET_ROOT}/lib
endif

INCLUDE_DIRS     := -I${CUDENSITYMAT_ROOT}/include -I${CUTENSORNET_ROOT}/include -I${CUTENSOR_ROOT}/include
LIBRARY_DIRS     := -L${CUDENSITYMAT_LIB_DIR} -lcudensitymat -L${CUTENSORNET_LIB_DIR} -lcutensornet -L${CUTENSOR_LIB_DIR} -lcutensor
ARCH_FLAGS_SM75  = -gencode arch=compute_75,code=sm_75
ARCH_FLAGS_SM80  = -gencode arch=compute_80,code=sm_80
ARCH_FLAGS_SM86  = -gencode arch=compute_86,code=sm_86
ARCH_FLAGS_SM90  = -gencode arch=compute_90,code=sm_90   -gencode arch=compute_90,code=compute_90
ARCH_FLAGS_SM100 = -gencode arch=compute_100,code=sm_100 -gencode arch=compute_100,code=compute_100
ARCH_FLAGS_SM120 = -gencode arch=compute_120,code=sm_120

ifeq ($(shell [ $(CUDA_MAJOR_VERSION) -lt 12 ] || ( [ $(CUDA_MAJOR_VERSION) -eq 12 ] && [ $(CUDA_MINOR_VERSION) -lt 8 ] ); echo $$?),0)
	ARCH_FLAGS = $(ARCH_FLAGS_SM75) $(ARCH_FLAGS_SM80) $(ARCH_FLAGS_SM86) $(ARCH_FLAGS_SM90)
else
	ARCH_FLAGS = $(ARCH_FLAGS_SM75) $(ARCH_FLAGS_SM80) $(ARCH_FLAGS_SM86) $(ARCH_FLAGS_SM90) \
	             $(ARCH_FLAGS_SM100) $(ARCH_FLAGS_SM120)
endif

CXX_FLAGS        = -std=c++17 $(INCLUDE_DIRS) $(LIBRARY_DIRS) $(ARCH_FLAGS)

all: check-env
	${CUDA_PATH}/bin/nvcc operator_action_example.cpp ${CXX_FLAGS} -o operator_action_example
	${CUDA_PATH}/bin/nvcc operator_action_gradient_example.cpp ${CXX_FLAGS} -o operator_action_gradient_example
	${CUDA_PATH}/bin/nvcc operator_action_batched_gradient_example.cpp ${CXX_FLAGS} -o operator_action_batched_gradient_example
	${CUDA_PATH}/bin/nvcc operator_eigenspectrum_example.cpp ${CXX_FLAGS} -o operator_eigenspectrum_example
  ifdef MPI_ROOT
	${CUDA_PATH}/bin/nvcc operator_action_mpi_example.cpp ${CXX_FLAGS} -DMPI_ENABLED -L${MPI_ROOT}/lib -lmpi -I${MPI_ROOT}/include -o operator_action_mpi_example
  endif
  ifdef NCCL_ROOT
    ifdef MPI_ROOT
      ifdef NCCL_LIB
	${CUDA_PATH}/bin/nvcc operator_action_nccl_example.cpp ${CXX_FLAGS} -DNCCL_ENABLED -DMPI_ENABLED -Xlinker ${NCCL_LIB} -I${NCCL_ROOT}/include -L${MPI_ROOT}/lib -lmpi -I${MPI_ROOT}/include -o operator_action_nccl_example
      endif
    endif
  endif

operator_action_example:
	${CUDA_PATH}/bin/nvcc operator_action_example.cpp ${CXX_FLAGS} -o operator_action_example

operator_action_gradient_example:
	${CUDA_PATH}/bin/nvcc operator_action_gradient_example.cpp ${CXX_FLAGS} -o operator_action_gradient_example

operator_action_batched_gradient_example:
	${CUDA_PATH}/bin/nvcc operator_action_batched_gradient_example.cpp ${CXX_FLAGS} -o operator_action_batched_gradient_example

operator_eigenspectrum_example:
	${CUDA_PATH}/bin/nvcc operator_eigenspectrum_example.cpp ${CXX_FLAGS} -o operator_eigenspectrum_example

ifdef MPI_ROOT
operator_action_mpi_example:
	${CUDA_PATH}/bin/nvcc operator_action_mpi_example.cpp ${CXX_FLAGS} -DMPI_ENABLED -L${MPI_ROOT}/lib -lmpi -I${MPI_ROOT}/include -o operator_action_mpi_example
endif

ifdef NCCL_ROOT
ifdef MPI_ROOT
ifdef NCCL_LIB
operator_action_nccl_example:
	${CUDA_PATH}/bin/nvcc operator_action_nccl_example.cpp ${CXX_FLAGS} -DNCCL_ENABLED -DMPI_ENABLED -Xlinker ${NCCL_LIB} -I${NCCL_ROOT}/include -L${MPI_ROOT}/lib -lmpi -I${MPI_ROOT}/include -o operator_action_nccl_example
endif
endif
endif

check-env:
	@ echo "" && \
	echo "CUDA_PATH=${CUDA_PATH}"; \
	echo "CUDENSITYMAT_ROOT=${CUDENSITYMAT_ROOT}"; \
	echo "CUTENSOR_ROOT=${CUTENSOR_ROOT}"; \
	echo "CUTENSOR_LIB_DIR=${CUTENSOR_LIB_DIR}"; \
	echo "MPI_ROOT=${MPI_ROOT}"; \
	echo "NCCL_ROOT=${NCCL_ROOT}"; \
	echo "NCCL_LIB=${NCCL_LIB}"; \
	echo ""; \
	if [[ -z "${CUDENSITYMAT_ROOT}" ]]; \
	then \
		echo "" && \
		echo "Neither CUDENSITYMAT_ROOT nor CUQUANTUM_ROOT is set." && \
		exit 1; \
	fi; \
	if [[ -z "${MPI_ROOT}" ]]; \
	then \
		echo "" && \
		echo "MPI_ROOT is not set. Please set MPI_ROOT if you would like to build the distributed example(s)." && \
		echo ""; \
	fi; \
	if [[ -z "${NCCL_ROOT}" ]]; \
	then \
		echo "" && \
		echo "NCCL_ROOT is not set. Please set NCCL_ROOT if you would like to build the NCCL example." && \
		echo ""; \
	elif [[ -z "${NCCL_LIB}" ]]; \
	then \
		echo "" && \
		echo "NCCL_ROOT is set but libnccl.so or libnccl.so.2 was not found in ${NCCL_ROOT}/lib or ${NCCL_ROOT}/lib64." && \
		echo ""; \
	fi

clean:
	rm -f operator_action_example operator_action_example.o
	rm -f operator_action_gradient_example operator_action_gradient_example.o
	rm -f operator_action_batched_gradient_example operator_action_batched_gradient_example.o
	rm -f operator_eigenspectrum_example operator_eigenspectrum_example.o
ifdef MPI_ROOT
	rm -f operator_action_mpi_example operator_action_mpi_example.o
endif
ifdef NCCL_ROOT
	rm -f operator_action_nccl_example operator_action_nccl_example.o
endif
