# Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES
#
# SPDX-License-Identifier: BSD-3-Clause

"""Experimental N-dimensional distributed tensor layouts."""

from __future__ import annotations

import math
from collections.abc import Sequence
from dataclasses import dataclass, field

from nvmath.distributed import distribution as _nvmath_distribution

from ._internal import runtime as _runtime


__all__ = ["ProcessGrid", "BlockCyclic", "LocalTensorLayout", "get_local_layout"]


class ProcessGrid(_nvmath_distribution.ProcessGrid):
    """An N-dimensional regular process grid with complete replica layers.

    Ranks fill the grid in ``layout`` order (column- or row-major). When the
    communicator size exceeds the grid size, it must be an integer multiple
    of it, and the surplus ranks form complete replica layers: rank ``r``
    occupies the same coordinates as rank ``r % size``, in replica layer
    ``r // size``. A mode whose grid extent is one is not partitioned.

    Args:
        shape: Per-mode extents of the process grid.
        layout: ``ProcessGrid.Layout.COL_MAJOR`` or ``ROW_MAJOR`` rank
            ordering; optional when at most one mode has extent greater
            than one.
        process_array: Custom process arrangements are not supported; must
            be ``None``.

    .. warning:: This API is experimental and subject to future changes.
    """

    def __init__(
        self,
        *,
        shape: Sequence[int] | None = None,
        layout: ProcessGrid.Layout | None = None,
        process_array=None,
    ):
        self._nranks = _runtime.get_process_group().nranks

        if process_array is not None:
            raise NotImplementedError("custom process_array grids are not supported")
        if shape is None:
            raise ValueError("shape must be provided when process_array=None")

        self._shape = tuple(shape)
        if not self._shape:
            raise ValueError("shape must contain at least one dimension")
        if any(type(extent) is not int for extent in self._shape):
            raise TypeError(f"shape must be a sequence of integers, got {shape}")
        if any(extent <= 0 for extent in self._shape):
            raise ValueError(f"shape dimensions must be positive, got {shape}")

        self._size = math.prod(self._shape)
        if self._nranks % self._size != 0:
            raise ValueError(
                f"Number of processes ({self._nranks}) must be an integer "
                f"multiple of the process-grid size ({self._size})"
            )

        if layout is None:
            if self._is_1d_distribution():
                layout = ProcessGrid.Layout.ROW_MAJOR
            else:
                raise ValueError(
                    "layout must be provided when partitioning multiple dimensions"
                )
        if not isinstance(layout, ProcessGrid.Layout):
            raise TypeError(
                f"layout must be of type ProcessGrid.Layout, got {layout}"
            )

        self._layout = layout
        self._process_array = None
        self._replication_factor = self._nranks // self._size

    @property
    def size(self) -> int:
        """Number of coordinates in one active process grid."""
        return self._size

    @property
    def replication_factor(self) -> int:
        """Number of complete process-grid replica layers."""
        return self._replication_factor

    def rank_to_coords(self, rank: int) -> tuple[int, ...]:
        """Return a rank's coordinates within its active-grid replica."""
        rank = self._validate_rank(rank)
        active_rank = rank % self._size
        coords = [0] * len(self._shape)

        if self._layout == ProcessGrid.Layout.COL_MAJOR:
            indices = range(len(self._shape))
        else:
            indices = reversed(range(len(self._shape)))

        for index in indices:
            coords[index] = active_rank % self._shape[index]
            active_rank //= self._shape[index]
        return tuple(coords)

    def replica_id(self, rank: int) -> int:
        """Return a rank's replica-layer index."""
        return self._validate_rank(rank) // self._size

    def coords_to_rank(
        self,
        coords: Sequence[int],
        replica_id: int = 0,
    ) -> int:
        """Return the rank for active-grid coordinates and a replica layer."""
        validated_coords = self._validate_coords(coords)
        replica_id = self._validate_replica_id(replica_id)

        if self._layout == ProcessGrid.Layout.COL_MAJOR:
            active_rank = 0
            stride = 1
            for coord, extent in zip(validated_coords, self._shape):
                active_rank += coord * stride
                stride *= extent
        else:
            active_rank = 0
            for coord, extent in zip(validated_coords, self._shape):
                active_rank = active_rank * extent + coord

        return replica_id * self._size + active_rank

    def _validate_rank(self, rank: int) -> int:
        if type(rank) is not int:
            raise TypeError(f"rank must be an integer, got {rank}")
        if rank < 0 or rank >= self._nranks:
            raise ValueError(
                f"rank must be in [0, {self._nranks}), got {rank}"
            )
        return rank

    def _validate_coords(self, coords: Sequence[int]) -> tuple[int, ...]:
        try:
            coords = tuple(coords)
        except TypeError as exc:
            raise TypeError(
                f"coords must be a sequence of integers, got {coords}"
            ) from exc

        if len(coords) != len(self._shape):
            raise ValueError(
                f"coords dimensionality ({len(coords)}) does not match "
                f"process-grid dimensionality ({len(self._shape)})"
            )
        if any(type(coord) is not int for coord in coords):
            raise TypeError(f"coords must be a sequence of integers, got {coords}")
        if any(
            coord < 0 or coord >= extent
            for coord, extent in zip(coords, self._shape)
        ):
            raise ValueError(
                f"coords {coords} are not valid for process-grid shape {self._shape}"
            )
        return coords

    def _validate_replica_id(self, replica_id: int) -> int:
        if type(replica_id) is not int:
            raise TypeError(
                f"replica_id must be an integer, got {replica_id}"
            )
        if replica_id < 0 or replica_id >= self._replication_factor:
            raise ValueError(
                f"replica_id must be in [0, {self._replication_factor}), "
                f"got {replica_id}"
            )
        return replica_id

    def _is_1d_distribution(self) -> bool:
        """True if the grid partitions at most one tensor mode."""
        return sum(extent > 1 for extent in self._shape) <= 1

    def _is_row_wise(self) -> bool:
        """True if a 2D active grid partitions only its row dimension."""
        return len(self._shape) == 2 and self._shape[1] == 1

    def _is_col_wise(self) -> bool:
        """True if a 2D active grid partitions only its column dimension."""
        return len(self._shape) == 2 and self._shape[0] == 1

    def __str__(self) -> str:
        return (
            f"{self.__class__.__name__}(shape={self._shape}, "
            f"layout={self._layout.name}, process_array=None, "
            f"replication_factor={self._replication_factor})"
        )

    def __hash__(self) -> int:
        # Match the nvmath parent hash so equal parent/subclass grids remain
        # valid dictionary keys. Unequal layouts may intentionally collide.
        return hash(self._shape)

    def __eq__(self, other) -> bool:
        if not isinstance(other, _nvmath_distribution.ProcessGrid):
            return False
        if self._process_array is not None or other._process_array is not None:
            return False
        if self._shape != other._shape or self._nranks != other._nranks:
            return False
        if self._is_1d_distribution():
            return True
        return self._layout == other._layout


def _slab_local_extent(
    global_extent: int,
    process_count: int,
    relative_coord: int,
) -> int:
    base, remainder = divmod(global_extent, process_count)
    return base + int(relative_coord < remainder)


def _cyclic_local_extent(
    global_extent: int,
    block_size: int,
    process_count: int,
    relative_coord: int,
) -> int:
    full_blocks, tail = divmod(global_extent, block_size)
    blocks_per_process, extra_blocks = divmod(full_blocks, process_count)

    local_extent = blocks_per_process * block_size
    if relative_coord < extra_blocks:
        local_extent += block_size
    elif relative_coord == extra_blocks:
        local_extent += tail
    return local_extent


class BlockCyclic(_nvmath_distribution.BlockCyclic):
    """N-D block-cyclic distribution with per-mode slab support.

    Each tensor mode is partitioned across the corresponding process-grid
    dimension. An integer block size assigns consecutive blocks round-robin
    across grid coordinates (the final block may be shorter), while ``None``
    assigns one near-even contiguous slab per coordinate. A grid extent of
    one leaves that mode undistributed.

    Args:
        process_grid: The :class:`ProcessGrid` of this subpackage (the
            nvmath parent class is not accepted).
        block_sizes: Per-mode entries, one per process-grid dimension: a
            positive integer for block-cyclic partitioning, or ``None`` for
            slab partitioning.
        first_process: Optional grid coordinates of the process owning the
            first block of each mode; defaults to the grid origin.

    .. warning:: This API is experimental and subject to future changes.
    """

    def __init__(
        self,
        process_grid: ProcessGrid,
        block_sizes: Sequence[int | None],
        *,
        first_process: Sequence[int] | None = None,
    ):
        if not isinstance(process_grid, ProcessGrid):
            raise TypeError(
                "process_grid must be an extended ProcessGrid from "
                "cuquantum.tensornet.experimental.distributed"
            )

        try:
            validated_block_sizes = tuple(block_sizes)
        except TypeError as exc:
            raise TypeError(
                f"block_sizes must be a sequence, got {block_sizes}"
            ) from exc

        if len(validated_block_sizes) != len(process_grid.shape):
            raise ValueError(
                f"Number of block sizes ({len(validated_block_sizes)}) does not "
                f"match process-grid dimensionality ({len(process_grid.shape)})"
            )
        for block_size in validated_block_sizes:
            if block_size is not None and type(block_size) is not int:
                raise TypeError(
                    "block_sizes entries must be None or positive integers, "
                    f"got {validated_block_sizes}"
                )
            if block_size is not None and block_size <= 0:
                raise ValueError(
                    "integer block_sizes entries must be positive, "
                    f"got {validated_block_sizes}"
                )

        validated_first_process = self._validate_first_process(
            process_grid,
            first_process,
        )

        # The nvmath parent accepts integer entries only and uses
        # _block_sizes for equality. Keep its state in this injective
        # sanitized form; _slab_block_sizes retains the None-aware values.
        parent_block_sizes = tuple(
            0 if block_size is None else block_size
            for block_size in validated_block_sizes
        )
        super().__init__(
            process_grid,
            parent_block_sizes,
            first_process=validated_first_process,
        )
        self._slab_block_sizes = validated_block_sizes

    @staticmethod
    def _validate_first_process(
        process_grid: ProcessGrid,
        first_process: Sequence[int] | None,
    ) -> tuple[int, ...]:
        if first_process is None:
            return (0,) * len(process_grid.shape)

        try:
            first_process = tuple(first_process)
        except TypeError as exc:
            raise TypeError(
                f"first_process must be a sequence, got {first_process}"
            ) from exc

        if len(first_process) != len(process_grid.shape):
            raise ValueError(
                f"first_process dimensionality ({len(first_process)}) does not "
                f"match process-grid dimensionality ({len(process_grid.shape)})"
            )
        if any(type(coord) is not int for coord in first_process):
            raise TypeError(
                f"first_process must be a sequence of integers, got {first_process}"
            )
        if any(
            coord < 0 or coord >= extent
            for coord, extent in zip(first_process, process_grid.shape)
        ):
            raise ValueError(
                f"first_process {first_process} is not a valid index into the "
                f"process grid of shape {process_grid.shape}"
            )
        return first_process

    @property
    def block_sizes(self) -> tuple[int | None, ...]:
        """Per-mode block size; ``None`` denotes contiguous slab partitioning."""
        return self._slab_block_sizes

    def shape(
        self, rank: int, global_shape: Sequence[int] | None = None
    ) -> tuple[int, ...]:
        """Return the local shard shape for ``rank`` and ``global_shape``.

        When ``global_shape`` is omitted, use the bound global shape.
        """
        # The parent implementation uses its integer-only _block_sizes;
        # calculate with the None-aware slab representation instead.
        nranks = _runtime.get_process_group().nranks
        if type(rank) is not int or rank < 0 or rank >= nranks:
            raise ValueError(f"rank must be in [0, {nranks}), got {rank}")
        if global_shape is None:
            if not self._bound:
                raise RuntimeError(
                    "global_shape is required for an unbound distribution"
                )
            global_shape = self._data_global_shape
        elif self._bound and tuple(global_shape) != self._data_global_shape:
            raise ValueError(
                "This distribution is already bound to a different global "
                f"shape: provided {tuple(global_shape)}, bound to "
                f"{self._data_global_shape}"
            )
        return self._calc_local_shape(rank, self._slab_block_sizes, global_shape)

    def __repr__(self) -> str:
        # The parent would print its sanitized integer block sizes (0 for
        # slab); show the user's None-aware tuple instead.
        return (
            f"{self.__class__.__name__}(process_grid={self._process_grid}, "
            f"block_sizes={self._slab_block_sizes})"
        )

    def _validate_global_shape(
        self,
        global_shape: Sequence[int],
    ) -> tuple[int, ...]:
        try:
            global_shape = tuple(global_shape)
        except TypeError as exc:
            raise TypeError(
                f"global_shape must be a sequence, got {global_shape}"
            ) from exc

        if len(global_shape) != self.ndim:
            raise ValueError(
                f"Global-shape dimensionality ({len(global_shape)}) does not "
                f"match distribution dimensionality ({self.ndim})"
            )
        if any(type(extent) is not int for extent in global_shape):
            raise TypeError(
                f"global_shape must be a sequence of integers, got {global_shape}"
            )
        if any(extent < 0 for extent in global_shape):
            raise ValueError(
                f"global_shape dimensions must be nonnegative, got {global_shape}"
            )
        return global_shape

    def _calc_local_shape(
        self,
        rank: int,
        block_sizes: Sequence[int | None],
        global_shape: Sequence[int],
    ) -> tuple[int, ...]:
        global_shape = self._validate_global_shape(global_shape)
        coords = self._process_grid.rank_to_coords(rank)

        local_shape = []
        for extent, process_count, coord, first, block_size in zip(
            global_shape,
            self._process_grid.shape,
            coords,
            self._first_process,
            block_sizes,
        ):
            if process_count == 1:
                local_extent = extent
            else:
                relative_coord = (coord - first) % process_count
                if block_size is None:
                    local_extent = _slab_local_extent(
                        extent,
                        process_count,
                        relative_coord,
                    )
                else:
                    local_extent = _cyclic_local_extent(
                        extent,
                        block_size,
                        process_count,
                        relative_coord,
                    )
            local_shape.append(local_extent)
        return tuple(local_shape)

    def _bind(
        self,
        global_shape: Sequence[int],
        *,
        shape: Sequence[int] | None = None,
    ) -> BlockCyclic:
        if self._bound:
            raise _nvmath_distribution.BindDistributionError(
                f"{self} is already bound"
            )

        global_shape = self._validate_global_shape(global_shape)
        rank = _runtime.get_process_group().rank
        local_shape = self._calc_local_shape(
            rank,
            self._slab_block_sizes,
            global_shape,
        )
        if shape is not None and tuple(shape) != local_shape:
            raise _nvmath_distribution.BindDistributionError(
                f"The local shape {shape} on process {rank} is not the expected "
                f"one based on global shape {global_shape}, process grid "
                f"{self._process_grid}, and block sizes {self._slab_block_sizes}: "
                f"expected shape is {local_shape}"
            )

        self._data_global_shape = global_shape
        self._data_shape = local_shape
        self._bound = True
        return self

    def to(self, cls, /, *, ndim=None, copy=False):
        """Convert to a compatible nvmath distribution type.

        Block-cyclic conversions return this distribution (or a copy).
        Conversion to ``Slab`` requires exactly one unshifted slab-distributed
        mode and no replica layers. Other conversions are not supported.
        """
        super()._to_checks(cls, ndim)

        if issubclass(cls, _nvmath_distribution.BlockCyclic):
            return self.copy() if copy else self

        if cls is _nvmath_distribution.Slab:
            distributed_modes = [
                mode
                for mode, process_count in enumerate(self._process_grid.shape)
                if process_count > 1
            ]
            if (
                len(distributed_modes) != 1
                or self._process_grid.replication_factor != 1
            ):
                raise _nvmath_distribution.ConvertDistributionError(
                    "BlockCyclic can be converted to Slab only when exactly one "
                    "mode is distributed and there are no replica layers"
                )

            partition_dim = distributed_modes[0]
            if (
                self._slab_block_sizes[partition_dim] is not None
                or self._first_process[partition_dim] != 0
            ):
                raise _nvmath_distribution.ConvertDistributionError(
                    "BlockCyclic can be converted to Slab only when the "
                    "distributed mode is an unshifted slab"
                )

            slab = _nvmath_distribution.Slab(partition_dim, ndim=self.ndim)
            if self._bound:
                slab._bind(self._data_global_shape, shape=self._data_shape)
            return slab

        raise _nvmath_distribution.ConvertDistributionError(
            f"Conversion from {self.__class__.__name__} to {cls.__name__} "
            "is not supported"
        )


def _validate_coordinate(
    coord: Sequence[int],
    shape: Sequence[int],
    name: str,
) -> tuple[int, ...]:
    try:
        coord = tuple(coord)
    except TypeError as exc:
        raise TypeError(f"{name} must be a sequence of integers") from exc
    if len(coord) != len(shape):
        raise ValueError(
            f"{name} dimensionality ({len(coord)}) does not match "
            f"tensor dimensionality ({len(shape)})"
        )
    if any(type(index) is not int for index in coord):
        raise TypeError(f"{name} must be a sequence of integers, got {coord}")
    if any(index < 0 or index >= extent for index, extent in zip(coord, shape)):
        raise ValueError(f"{name} {coord} is outside shape {shape}")
    return coord


@dataclass(frozen=True)
class LocalTensorLayout:
    """Compact rank-local layout for a block-cyclic distribution."""

    rank: int
    process_coords: tuple[int, ...]
    global_shape: tuple[int, ...]
    local_shape: tuple[int, ...]
    element_strides: tuple[int, ...]
    storage_size_elements: int
    _process_grid_shape: tuple[int, ...] = field(repr=False)
    _block_sizes: tuple[int | None, ...] = field(repr=False)
    _first_process: tuple[int, ...] = field(repr=False)

    @property
    def owned_global_segments(
        self,
    ) -> tuple[tuple[tuple[int, int], ...], ...]:
        """Per-mode owned global index segments as ``(start, extent)`` pairs.

        Element ownership is the Cartesian product of these mode-wise
        segments. Cyclic modes may contribute multiple disjoint segments;
        slab and replicated modes contribute at most one.
        """
        segments_per_mode = []
        for (
            global_extent,
            local_extent,
            process_count,
            process_coord,
            first,
            block_size,
        ) in zip(
            self.global_shape,
            self.local_shape,
            self._process_grid_shape,
            self.process_coords,
            self._first_process,
            self._block_sizes,
        ):
            relative_coord = (process_coord - first) % process_count
            if local_extent == 0 or global_extent == 0:
                segments_per_mode.append(())
            elif process_count == 1:
                segments_per_mode.append(((0, global_extent),))
            elif block_size is None:
                base, remainder = divmod(global_extent, process_count)
                start = relative_coord * base + min(relative_coord, remainder)
                segments_per_mode.append(((start, local_extent),))
            else:
                segments = []
                num_blocks = (global_extent + block_size - 1) // block_size
                for block in range(relative_coord, num_blocks, process_count):
                    start = block * block_size
                    extent = min(block_size, global_extent - start)
                    if extent > 0:
                        segments.append((start, extent))
                segments_per_mode.append(tuple(segments))
        return tuple(segments_per_mode)

    def physical_offset(self, local_coord: Sequence[int]) -> int:
        """Return the dense element offset of a local coordinate."""
        local_coord = _validate_coordinate(
            local_coord, self.local_shape, "local_coord"
        )
        return sum(
            index * stride
            for index, stride in zip(local_coord, self.element_strides)
        )

    def local_to_global(self, local_coord: Sequence[int]) -> tuple[int, ...]:
        """Map a local coordinate owned by this rank to a global coordinate."""
        local_coord = _validate_coordinate(
            local_coord, self.local_shape, "local_coord"
        )
        global_coord = []
        for (
            local_index,
            global_extent,
            process_count,
            process_coord,
            first,
            block_size,
        ) in zip(
            local_coord,
            self.global_shape,
            self._process_grid_shape,
            self.process_coords,
            self._first_process,
            self._block_sizes,
        ):
            relative_coord = (process_coord - first) % process_count
            if process_count == 1:
                global_index = local_index
            elif block_size is None:
                base, remainder = divmod(global_extent, process_count)
                start = relative_coord * base + min(relative_coord, remainder)
                global_index = start + local_index
            else:
                local_block, intra = divmod(local_index, block_size)
                global_block = local_block * process_count + relative_coord
                global_index = global_block * block_size + intra
            global_coord.append(global_index)
        return tuple(global_coord)

    def global_to_local(
        self, global_coord: Sequence[int]
    ) -> tuple[int, ...] | None:
        """Map a global coordinate to this rank's local coordinate, if owned."""
        global_coord = _validate_coordinate(
            global_coord, self.global_shape, "global_coord"
        )
        local_coord = []
        for (
            global_index,
            global_extent,
            local_extent,
            process_count,
            process_coord,
            first,
            block_size,
        ) in zip(
            global_coord,
            self.global_shape,
            self.local_shape,
            self._process_grid_shape,
            self.process_coords,
            self._first_process,
            self._block_sizes,
        ):
            relative_coord = (process_coord - first) % process_count
            if process_count == 1:
                local_index = global_index
            elif block_size is None:
                base, remainder = divmod(global_extent, process_count)
                start = relative_coord * base + min(relative_coord, remainder)
                if global_index < start or global_index >= start + local_extent:
                    return None
                local_index = global_index - start
            else:
                global_block, intra = divmod(global_index, block_size)
                if global_block % process_count != relative_coord:
                    return None
                local_index = (global_block // process_count) * block_size + intra
            local_coord.append(local_index)
        return tuple(local_coord)


def get_local_layout(
    distribution: BlockCyclic,
    global_shape: Sequence[int],
    *,
    rank: int | None = None,
) -> LocalTensorLayout:
    """Return the compact local layout for one rank of a distribution.

    When ``rank`` is omitted, use the calling process's rank.
    """
    if not isinstance(distribution, BlockCyclic):
        raise TypeError(
            "distribution must be an experimental distributed.BlockCyclic"
        )

    global_shape = distribution._validate_global_shape(global_shape)
    if rank is None:
        rank = _runtime.get_process_group().rank

    process_coords = distribution._process_grid.rank_to_coords(rank)
    local_shape = distribution._calc_local_shape(
        rank,
        distribution.block_sizes,
        global_shape,
    )
    element_strides = []
    stride = 1
    for extent in local_shape:
        element_strides.append(stride)
        stride *= extent

    return LocalTensorLayout(
        rank=rank,
        process_coords=process_coords,
        global_shape=global_shape,
        local_shape=local_shape,
        element_strides=tuple(element_strides),
        storage_size_elements=math.prod(local_shape),
        _process_grid_shape=distribution._process_grid.shape,
        _block_sizes=distribution.block_sizes,
        _first_process=distribution._first_process,
    )
