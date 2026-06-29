/* Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
 *
 * SPDX-License-Identifier: BSD-3-Clause
 *
 * JAX FFI handler for cuStabilizer's GF(2) sparse-dense matmul.
 *
 * The dense -> CSR conversion of the A operand is the caller's responsibility
 * (see cuquantum.stabilizer.jax.matmul_gf2_spdn's docstring). A wrapping FFI
 * for the pack is intentionally not provided -- jax.experimental.sparse,
 * cusparse, and cupy all expose dense->CSR primitives that compose with this
 * handler at the JAX-trace level.
 */

#include <cstdio>
#include <cstdint>
#include <cuda_runtime.h>
#include <xla/ffi/api/c_api.h>
#include <xla/ffi/api/ffi.h>
#include "custabilizer.h"

#include "custabilizer_jax.h"

// --- Sparse-dense GF(2) matmul --------------------------------------------
//
// Maps matmul_gf2(a (G, T, P), b (B, P)) -> (B, G, T) onto cuStabilizer's
// custabilizerGF2SparseDenseMatrixMultiply with m=B, n=G*T_pad, k=P.
//
// Inputs (all device-resident):
//   A_rowoff  uint64 (B+1)        CSR row offsets of b
//   A_colidx  uint64 (max_nnz)    CSR column indices of b (cap, kernel reads nnz)
//   B_packed  uint32 (P, n_pad/32) bit-packed dense matrix params^T
//
// Static attrs:
//   m, n, k, handle (library handle as intptr_t)
//
// Output:
//   C  uint32 (B, n_pad/32)       bit-packed result
//
// The 8-byte d2h memcpy of rowoff[m] is the cost of using the SpDn C ABI
// (which takes an explicit nnz); the SpSp ABI reads it internally but
// requires sorted CSR for B, which is overkill for this caller.

xla::ffi::Error SpDnMatmulGf2Impl(cudaStream_t stream,
                                  xla::ffi::Buffer<xla::ffi::U64> A_rowoff,
                                  xla::ffi::Buffer<xla::ffi::U64> A_colidx,
                                  xla::ffi::Buffer<xla::ffi::U32> B_packed,
                                  xla::ffi::Result<xla::ffi::Buffer<xla::ffi::U32>> C,
                                  int64_t m,
                                  int64_t n,
                                  int64_t k,
                                  int64_t handle_intptr)
{
    auto handle = reinterpret_cast<custabilizerHandle_t>(handle_intptr);

    // Read A's nnz = rowoff[m] from device.
    uint64_t a_nnz = 0;
    cudaError_t cerr =
        cudaMemcpyAsync(&a_nnz, A_rowoff.typed_data() + m, sizeof(uint64_t), cudaMemcpyDeviceToHost, stream);
    if (cerr != cudaSuccess)
    {
        char msg[160];
        std::snprintf(msg, sizeof(msg), "cudaMemcpyAsync(nnz) failed: %s", cudaGetErrorString(cerr));
        return xla::ffi::Error(xla::ffi::ErrorCode::kInternal, msg);
    }
    cerr = cudaStreamSynchronize(stream);
    if (cerr != cudaSuccess)
    {
        return xla::ffi::Error(xla::ffi::ErrorCode::kInternal, cudaGetErrorString(cerr));
    }

    custabilizerStatus_t st = custabilizerGF2SparseDenseMatrixMultiply(
        handle, static_cast<uint64_t>(m), static_cast<uint64_t>(n), static_cast<uint64_t>(k), a_nnz,
        A_colidx.typed_data(), A_rowoff.typed_data(),
        reinterpret_cast<const custabilizerBitInt_t*>(B_packed.typed_data()),
        /*beta=*/0, reinterpret_cast<custabilizerBitInt_t*>(C->untyped_data()), stream);
    if (st != CUSTABILIZER_STATUS_SUCCESS)
    {
        char msg[160];
        std::snprintf(msg, sizeof(msg), "custabilizerGF2SparseDenseMatrixMultiply returned %d", (int)st);
        return xla::ffi::Error(xla::ffi::ErrorCode::kInternal, msg);
    }
    return xla::ffi::Error::Success();
}

XLA_FFI_DEFINE_HANDLER_SYMBOL(SpDnMatmulGf2Handler,
                              SpDnMatmulGf2Impl,
                              xla::ffi::Ffi::Bind()
                                  .Ctx<xla::ffi::PlatformStream<cudaStream_t>>()
                                  .Arg<xla::ffi::Buffer<xla::ffi::U64>>() // A_rowoff
                                  .Arg<xla::ffi::Buffer<xla::ffi::U64>>() // A_colidx
                                  .Arg<xla::ffi::Buffer<xla::ffi::U32>>() // B_packed
                                  .Ret<xla::ffi::Buffer<xla::ffi::U32>>() // C (bit-packed)
                                  .Attr<int64_t>("m")
                                  .Attr<int64_t>("n")
                                  .Attr<int64_t>("k")
                                  .Attr<int64_t>("handle"));
