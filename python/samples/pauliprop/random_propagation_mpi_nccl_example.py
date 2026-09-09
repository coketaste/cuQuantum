#!/usr/bin/env python3
# Copyright (c) 2025-2026, NVIDIA CORPORATION & AFFILIATES
#
# SPDX-License-Identifier: BSD-3-Clause

"""
Example: distributed Pauli propagation over a large random expansion (NCCL provider).

Each process seeds its GPU with a fixed number of random Pauli terms,
deduplicates the expansion across all processes once (a duplicate-free
expansion admits compactly sized per-process buffers for the following
operations), and then repeatedly applies Pauli rotation gates. Every
application merges duplicate terms across all processes, which redistributes
the expansion — the communication-heavy phase that multi-process execution
accelerates. After
each layer, the input of the next layer is capped to the per-process term
budget through a view, so the per-GPU working set stays constant while the
global expansion size scales with the number of processes (weak scaling).
The run ends with a collective trace against the |0...0> state.

The workload is synthetic: capping through a view drops surplus terms
arbitrarily, so the trace is a checksum of the distributed pipeline rather
than a physical observable. A quarter of the seeded terms are restricted to
{I, Z} so the final trace has support on <0|P|0>.

MPI is used only to launch processes and broadcast the NCCL unique ID;
cuPauliProp communication itself uses NCCL, which requires one GPU per rank.

Requirements:
    - mpi4py and nvmath-python (which provides the NCCL communicator)
    - CUPAULIPROP_COMM_LIB environment variable pointing to the NCCL
      distributed-interface library

Launch one process per GPU:

.. code-block:: bash

   export CUPAULIPROP_COMM_LIB=<path>/libcupauliprop_distributed_interface_nccl.so
   mpirun -n <num_ranks> python3 random_propagation_mpi_nccl_example.py
"""

import os

import cupy as cp
import numpy as np
from mpi4py import MPI
import nvmath.distributed

from cuquantum.pauliprop.experimental import (
    LibraryHandle,
    PauliExpansion,
    PauliExpansionOptions,
    PauliRotationGate,
    get_num_packed_integers,
)


NUM_QUBITS = 48
TERMS_PER_PROCESS = 4 * 1024 * 1024
NUM_LAYERS = 8
ROTATION_ANGLE = 0.2
SEED = 20260820
# Fraction of seeded terms restricted to {I, Z} so <0|P|0> has support.
ZI_ONLY_FRACTION = 4


def random_terms(rank, num_terms):
    """Generate this process's random Pauli terms as packed XZ bits + coefficients."""
    rng = cp.random.default_rng(SEED + 7919 * rank)
    ints_per_mask = get_num_packed_integers(NUM_QUBITS)

    def random_words(count):
        # cupy's integer-generator kernels take 32-bit signed bounds, so
        # compose each 64-bit word from four independent 16-bit draws.
        words = cp.zeros(count, dtype=cp.uint64)
        for shift in (0, 16, 32, 48):
            words |= rng.integers(0, 1 << 16, size=count, dtype=cp.uint64) << cp.uint64(shift)
        return words

    xz_bits = cp.empty((num_terms, 2 * ints_per_mask), dtype=cp.uint64)
    for column in range(2 * ints_per_mask):
        xz_bits[:, column] = random_words(num_terms)

    # Packed Pauli strings must not carry bits beyond the qubit count; zero
    # the tail bits of the last packed word of the X and Z masks.
    tail_qubits = NUM_QUBITS - 64 * (ints_per_mask - 1)
    tail_mask = cp.uint64((1 << tail_qubits) - 1) if tail_qubits < 64 else cp.uint64(2**64 - 1)
    xz_bits[:, ints_per_mask - 1] &= tail_mask
    xz_bits[:, 2 * ints_per_mask - 1] &= tail_mask

    # Clearing the X masks makes the leading slice of terms {I, Z}-only.
    xz_bits[: num_terms // ZI_ONLY_FRACTION, :ints_per_mask] = 0

    coefs = rng.standard_normal(num_terms) / np.sqrt(num_terms)
    return xz_bits, coefs.astype(cp.float64)


def propagate(handle, mpi_comm, provider_label):
    """Run the layered propagation loop and report timings and the final trace."""
    rank, size = mpi_comm.Get_rank(), mpi_comm.Get_size()

    # XX rotations anticommute with roughly half of the random terms, so
    # every layer branches terms and feeds the cross-process duplicate merge.
    gates = [
        PauliRotationGate(
            angle=ROTATION_ANGLE + 0.01 * layer,
            pauli_string="XX",
            qubit_indices=[(2 * layer) % NUM_QUBITS, (2 * layer + 1) % NUM_QUBITS],
        )
        for layer in range(NUM_LAYERS)
    ]

    # Rehearse one application at the per-process term budget to size the
    # ping-pong output buffers. The rehearsal declares the duplicate-free
    # state the loop input holds, which is prescribed compact per-process
    # buffers; capping each layer's input view at the same budget keeps the
    # rehearsed bound valid for every layer.
    options = PauliExpansionOptions(memory_limit="80%", blocking=True)
    rehearsal = PauliExpansion.empty(
        handle, NUM_QUBITS, TERMS_PER_PROCESS, dtype="float64",
        has_duplicates=False, options=options,
    )
    info = rehearsal.apply_gate(gates[0], keep_duplicates=False)
    capacity = info.num_terms_required
    if rank == 0:
        print(f"Distributed random Pauli propagation on {size} process(es)")
        print(f"  provider:          {provider_label}")
        print(f"  qubits:            {NUM_QUBITS}")
        print(f"  terms per process: {TERMS_PER_PROCESS:,}")
        print(f"  rehearsed output capacity per process: {capacity:,}")
        print()

    ints_per_mask = get_num_packed_integers(NUM_QUBITS)

    # Deduplicate the seeded terms across all processes. Twice the local term
    # count covers the distributed output-capacity requirement for roughly
    # balanced inputs; an insufficient output would be reported cleanly.
    xz_bits, coefs = random_terms(rank, TERMS_PER_PROCESS)
    seeded = PauliExpansion(
        handle, NUM_QUBITS, TERMS_PER_PROCESS, xz_bits, coefs,
        has_duplicates=True, options=options,
    )
    dedup_buffer = PauliExpansion(
        handle, NUM_QUBITS, 0,
        cp.empty((2 * TERMS_PER_PROCESS, 2 * ints_per_mask), dtype=cp.uint64),
        cp.empty(2 * TERMS_PER_PROCESS, dtype=cp.float64),
        options=options,
    )
    current = seeded.view().deduplicate(expansion_out=dedup_buffer)
    del seeded, xz_bits, coefs

    buffers = [
        rehearsal.from_empty(
            cp.empty((capacity, 2 * ints_per_mask), dtype=cp.uint64),
            cp.empty(capacity, dtype=cp.float64),
            options=options,
        )
        for _ in range(2)
    ]

    # Time each layer with CUDA events; the barrier aligns ranks before the
    # measured region and the reported figure is the slowest rank's time.
    start_event = cp.cuda.Event()
    end_event = cp.cuda.Event()
    layer_seconds = []
    for layer, gate in enumerate(gates):
        view = current.view(0, min(current.num_terms, TERMS_PER_PROCESS))
        output = buffers[layer % 2]
        cp.cuda.get_current_stream().synchronize()
        mpi_comm.Barrier()
        start_event.record()
        view.apply_gate(gate, expansion_out=output, keep_duplicates=False)
        end_event.record()
        end_event.synchronize()
        seconds = cp.cuda.get_elapsed_time(start_event, end_event) / 1e3
        layer_seconds.append(mpi_comm.allreduce(seconds, op=MPI.MAX))
        current = output

    significand, exponent = current.view().trace_with_zero_state()
    cp.cuda.get_current_stream().synchronize()
    trace = significand * np.exp2(exponent)

    if rank == 0:
        terms_per_layer = size * TERMS_PER_PROCESS
        for layer, seconds in enumerate(layer_seconds):
            print(
                f"layer {layer}: {seconds:.4f} s "
                f"({terms_per_layer / seconds / 1e6:.1f} M input terms/s global)"
            )
        print()
        print(f"final terms per process:  {current.num_terms:,}")
        print(f"trace against |0...0>:    {trace}")


def main():
    if "nccl" not in os.path.basename(os.environ.get("CUPAULIPROP_COMM_LIB", "")).lower():
        raise RuntimeError(
            "Set CUPAULIPROP_COMM_LIB to the NCCL distributed-interface library."
        )
    mpi_comm = MPI.COMM_WORLD.Dup()
    device_id = mpi_comm.Get_rank() % cp.cuda.runtime.getDeviceCount()

    with cp.cuda.Device(device_id):
        # NCCL communicator initialized via nvmath.distributed; MPI performs
        # process launch and bootstrap only.
        nvmath.distributed.initialize(device_id, mpi_comm, backends=["nccl"])

        handle = LibraryHandle(device_id=device_id)
        handle.set_communicator(nvmath.distributed.get_context())
        propagate(handle, mpi_comm, provider_label="NCCL (MPI bootstrap)")


main()
