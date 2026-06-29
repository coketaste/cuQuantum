/* Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
 *
 * SPDX-License-Identifier: BSD-3-Clause
 */

#include "pybind11/pybind11.h"

#include "custabilizer_jax.h"

namespace py = pybind11;

template <typename T>
py::capsule EncapsulateFfiHandler(T* func)
{
    static_assert(std::is_invocable_r_v<XLA_FFI_Error*, T, XLA_FFI_CallFrame*>,
                  "Encapsulated function must be an XLA FFI handler");
    return py::capsule(reinterpret_cast<void*>(func));
}

py::dict Registrations()
{
    py::dict dict;
    dict["cust_spdn_matmul_gf2"] = EncapsulateFfiHandler(SpDnMatmulGf2Handler);
    return dict;
}

PYBIND11_MODULE(custabilizer_jax, m) { m.def("registrations", &Registrations); }
