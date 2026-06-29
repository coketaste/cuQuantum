# Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES
#
# SPDX-License-Identifier: BSD-3-Clause

"""Tests for the shared 2-site MPS Python surface: ``SVDConfig.max_extent`` and
``MPSPureState`` current bond extents."""

import numpy as np
import pytest

from cuquantum.bindings import cudensitymat as cudm
from cuquantum.densitymat import MPSPureState, SVDConfig, WorkStream
from cuquantum.densitymat.svd import _build_svd_config_handle


@pytest.fixture
def ctx():
    return WorkStream()


def _read_svd_attr(handle, ptr, attr, dtype):
    out = np.zeros(1, dtype=dtype)
    cudm.svd_config_get_attribute(handle, ptr, attr, out.ctypes.data, out.dtype.itemsize)
    return out[0]


def test_svd_config_max_extent_roundtrip(ctx):
    """``max_extent`` is wired through ``_build_svd_config_handle`` to the C SVD config."""
    handle = ctx._handle._validated_ptr
    # rel_cutoff > 1 is intentionally accepted (degenerate, not invalid).
    ptr = _build_svd_config_handle(handle, SVDConfig(max_extent=256, rel_cutoff=2.0))
    try:
        assert _read_svd_attr(handle, ptr, cudm.SVDConfigAttribute.MAX_EXTENT, np.int64) == 256
        assert _read_svd_attr(handle, ptr, cudm.SVDConfigAttribute.REL_CUTOFF, np.float64) == 2.0
    finally:
        cudm.destroy_svd_config(ptr)


def test_svd_config_max_extent_default(ctx):
    """An unset ``max_extent`` leaves the C default (0 = no limit)."""
    handle = ctx._handle._validated_ptr
    ptr = _build_svd_config_handle(handle, SVDConfig())
    try:
        assert _read_svd_attr(handle, ptr, cudm.SVDConfigAttribute.MAX_EXTENT, np.int64) == 0
    finally:
        cudm.destroy_svd_config(ptr)


def _make_mps(ctx, hilbert_space_dims, bond_dims):
    psi = MPSPureState(ctx, hilbert_space_dims, bond_dims, 1, "complex128")
    psi.allocate_storage()
    return psi


def test_current_bond_extents_default_equals_max(ctx):
    psi = _make_mps(ctx, (2, 2, 2), (4, 4))
    assert psi.current_bond_extents == (4, 4)


def test_current_bond_extents_roundtrip(ctx):
    psi = _make_mps(ctx, (2, 2, 2), (4, 4))
    psi.set_current_bond_extents([2, 3])
    assert psi.current_bond_extents == (2, 3)


def test_current_bond_extents_single_bond(ctx):
    psi = _make_mps(ctx, (2, 2), (4,))
    assert psi.current_bond_extents == (4,)
    psi.set_current_bond_extents([2])
    assert psi.current_bond_extents == (2,)


@pytest.mark.parametrize("bad", [[0, 3], [2, 5], [-1, 3]])
def test_current_bond_extents_rejects_invalid(ctx, bad):
    """0 < extent <= maximum; a rejected set leaves the extents unchanged."""
    psi = _make_mps(ctx, (2, 2, 2), (4, 4))
    psi.set_current_bond_extents([2, 3])
    with pytest.raises(cudm.cuDensityMatError):
        psi.set_current_bond_extents(bad)
    assert psi.current_bond_extents == (2, 3)


def test_current_bond_extents_set_before_storage(ctx):
    """Legal call ordering: SetCurrentBondExtents may be called *before*
    storage is allocated (it records metadata only) and the value survives the
    subsequent allocation (Get reads back exactly what was Set)."""
    psi = MPSPureState(ctx, (2, 2, 2), (4, 4), 1, "complex128")
    psi.set_current_bond_extents([2, 3])
    assert psi.current_bond_extents == (2, 3)
    psi.allocate_storage()
    # Allocation must not clobber the previously-recorded current extents.
    assert psi.current_bond_extents == (2, 3)
    # ... and Set still works after allocation, the other legal ordering.
    psi.set_current_bond_extents([4, 4])
    assert psi.current_bond_extents == (4, 4)


def test_current_bond_extents_boundary_values(ctx):
    """The full legal range 0 < extent <= maximum is accepted at both ends:
    the minimum (all 1, a product-state seed) and the maximum (== buffer)."""
    psi = _make_mps(ctx, (2, 2, 2, 2), (2, 4, 2))
    psi.set_current_bond_extents([1, 1, 1])
    assert psi.current_bond_extents == (1, 1, 1)
    psi.set_current_bond_extents([2, 4, 2])
    assert psi.current_bond_extents == (2, 4, 2)


# ---------------------------------------------------------------------------
# Eager (SetAttribute-time) validation of the SVD / TDVP config attributes
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "field,bad_value",
    [
        ("abs_cutoff", -1e-3),
        ("rel_cutoff", -1e-3),
        ("discarded_weight_cutoff", -1e-3),
        ("max_extent", -1),
    ],
    ids=["abs", "rel", "discarded_weight", "max_extent"],
)
def test_svd_config_rejects_negative(ctx, field, bad_value):
    """A negative cutoff or a negative ``max_extent`` is rejected eagerly when the
    SVD config handle is built (the C SVDConfigSetAttribute returns INVALID_VALUE)."""
    handle = ctx._handle._validated_ptr
    with pytest.raises(cudm.cuDensityMatError):
        _build_svd_config_handle(handle, SVDConfig(**{field: bad_value}))


def _set_tdvp_attr(handle, ptr, attr, value):
    dtype = cudm.get_time_propagation_scope_split_tdvp_config_attribute_dtype(attr)
    arr = np.array([value], dtype=dtype)
    cudm.time_propagation_scope_split_tdvp_config_set_attribute(
        handle, ptr, attr, arr.ctypes.data, arr.dtype.itemsize
    )


@pytest.mark.parametrize("good", [1, 2])
def test_tdvp_num_sites_accepts_one_and_two(ctx, good):
    """``TDVP_NUM_SITES`` accepts exactly the supported local-update sizes {1, 2}."""
    handle = ctx._handle._validated_ptr
    ptr = cudm.create_time_propagation_scope_split_tdvp_config(handle)
    try:
        _set_tdvp_attr(
            handle, ptr,
            cudm.TimePropagationScopeSplitTDVPConfigAttribute.PROPAGATION_SPLIT_SCOPE_TDVP_NUM_SITES,
            good,
        )
    finally:
        cudm.destroy_time_propagation_scope_split_tdvp_config(ptr)


@pytest.mark.parametrize("bad", [0, 3, -1, 4])
def test_tdvp_num_sites_rejects_out_of_range(ctx, bad):
    """``TDVP_NUM_SITES`` outside {1, 2} is rejected eagerly at SetAttribute
    (>=3 -> NOT_SUPPORTED, non-positive -> INVALID_VALUE; both surface as the
    Python ``cuDensityMatError``)."""
    handle = ctx._handle._validated_ptr
    ptr = cudm.create_time_propagation_scope_split_tdvp_config(handle)
    try:
        with pytest.raises(cudm.cuDensityMatError):
            _set_tdvp_attr(
                handle, ptr,
                cudm.TimePropagationScopeSplitTDVPConfigAttribute.PROPAGATION_SPLIT_SCOPE_TDVP_NUM_SITES,
                bad,
            )
    finally:
        cudm.destroy_time_propagation_scope_split_tdvp_config(ptr)


def test_svd_config_max_extent_above_buffer_accepted(ctx):
    """A global ``max_extent`` larger than any per-bond buffer is accepted by the
    config object itself (it is only "inert" per bond at Prepare; the cap value
    is still recorded faithfully)."""
    handle = ctx._handle._validated_ptr
    ptr = _build_svd_config_handle(handle, SVDConfig(max_extent=1 << 20))
    try:
        assert _read_svd_attr(
            handle, ptr, cudm.SVDConfigAttribute.MAX_EXTENT, np.int64
        ) == (1 << 20)
    finally:
        cudm.destroy_svd_config(ptr)


@pytest.mark.parametrize("bad", [[2], [2, 3, 4], None, np.array([[2, 3]])])
def test_current_bond_extents_rejects_bad_shape(ctx, bad):
    """Python validates shape/length before passing a raw pointer to the C API."""
    psi = _make_mps(ctx, (2, 2, 2), (4, 4))
    with pytest.raises((TypeError, ValueError)):
        psi.set_current_bond_extents(bad)
