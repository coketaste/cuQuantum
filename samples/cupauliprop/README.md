# cuPauliProp - Samples

* [Documentation](https://docs.nvidia.com/cuda/cuquantum/latest/cupauliprop/index.html)

## Install

### Linux

You can use `make` or `cmake` to compile the cuPauliProp samples. The environment variables `CUDA_PATH` and `CUQUANTUM_ROOT`  or `CUPAULIPROP_ROOT` need to be defined to point to the CUDA Toolkit and cuPauliProp locations, respectively.

Using `make`:
```bash
export CUDA_PATH=<path_to_cuda_root>
export CUPAULIPROP_ROOT=<path_to_cupauliprop_root>
make
```

To build the multiprocess example with MPI provider support:
```bash
export MPI_ROOT=<path_to_mpi_root>
make
```
To additionally enable the NCCL provider (which requires MPI for
bootstrapping):
```bash
export MPI_ROOT=<path_to_mpi_root>
export NCCL_ROOT=<path_to_nccl_root>
make
```
Without `MPI_ROOT`, the multiprocess example builds in its single-process
mode.
or `cmake`:
```bash
export CUDA_PATH=<path_to_cuda_root>
export CUPAULIPROP_ROOT=<path_to_cupauliprop_root>
cmake . && make
```
With `cmake`, distributed-provider support in the multiprocess example is
enabled via options instead of `MPI_ROOT`/`NCCL_ROOT`. For MPI only:
```bash
cmake . -DENABLE_MPI=TRUE && make
```
For MPI and NCCL:
```bash
cmake . -DENABLE_MPI=TRUE -DENABLE_NCCL=TRUE -DNCCL_ROOT=<path_to_nccl_root> && make
```

## Run

To run the standard kicked Ising example:
```bash
./kicked_ising_example
```

The fused variant rewrites the adjoints of the fixed `Rzz(-pi/2)` gates as
Clifford `CX-S-CX` sequences and applies each complete entangling sequence with
the fused operator API:
```bash
./kicked_ising_fused_example
```

To run the fused operators example, which benchmarks applying a sequence of operators via the fused API against the single-operator API, use:
```bash
./fused_operators_example
```

The multiprocess example is a single binary that supports single-process
execution and, when built with the corresponding support, the MPI and NCCL
distributed providers. Run it single-process:
```bash
./kicked_ising_multiprocess_example
```

A distributed run additionally requires `CUPAULIPROP_COMM_LIB` to point to a
built distributed-interface plugin, which the library loads at runtime. The
`activate_mpi_cupp.sh` and `activate_nccl_cupp.sh` scripts in
`<path_to_cupauliprop_root>/distributed_interfaces` compile the corresponding
plugin against your MPI/NCCL installation and export `CUPAULIPROP_COMM_LIB`
for you; alternatively, export it manually to an already-built plugin. The
active provider is selected at runtime from the plugin filename (a name
containing `nccl` selects the NCCL provider; anything else selects MPI).
Launch one process per GPU:
```bash
export CUPAULIPROP_COMM_LIB=<path_to_plugins>/libcupauliprop_distributed_interface_mpi.so
mpirun -n <num_ranks> ./kicked_ising_multiprocess_example

export CUPAULIPROP_COMM_LIB=<path_to_plugins>/libcupauliprop_distributed_interface_nccl.so
mpirun -n <num_ranks> ./kicked_ising_multiprocess_example
```
Under the MPI provider, ranks may share GPUs (devices are assigned
round-robin); NCCL requires one GPU per rank by default (NCCL 2.30+ can lift
this via the experimental `NCCL_MULTI_RANK_GPU_ENABLE=1`).

**Note**: Depending on how CUDA Toolkit is installed,
you might need to add it to `LD_LIBRARY_PATH` like this:
```bash
export LD_LIBRARY_PATH=$CUDA_PATH/lib64:$LD_LIBRARY_PATH
```

## Support

* **Supported SM Architectures:** SM 7.5, SM 8.0, SM 8.6, SM 9.0, SM 10.0, SM 12.0
* **Supported OSes:** Linux
* **Supported CPU Architectures**: x86_64, aarch64-sbsa
* **Language**: C++11 or above

## Prerequisites

* [CUDA Toolkit 12.x](https://developer.nvidia.com/cuda-downloads) or higher and compatible driver
(see [CUDA Driver Release Notes](https://docs.nvidia.com/cuda/cuda-toolkit-release-notes/index.html#cuda-major-component-versions)).
* CMake 3.22+ if using `cmake`.
