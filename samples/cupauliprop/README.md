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
or `cmake`:
```bash
export CUDA_PATH=<path_to_cuda_root>
export CUPAULIPROP_ROOT=<path_to_cupauliprop_root>
cmake . && make
```

## Run

To execute the kicked ising circuit example in a command shell, simply use:
```bash
./kicked_ising_example
```

To run the fused operators example, which benchmarks applying a sequence of operators via the fused API against the single-operator API, use:
```bash
./fused_operators_example
```

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
