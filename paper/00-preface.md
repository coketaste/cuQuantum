# 00 Preface

## Why this paper

Quantum simulation on classical hardware is a balancing act between three pressures that often pull in opposite directions:

1. **Faithfulness.** A faithful simulator must respect the linear, unitary, and probabilistic structure of quantum mechanics down to the precision required by the problem.
2. **Scale.** "Useful" circuits today already routinely involve dozens of qubits, hundreds of gate layers, and thousands of measurement shots; tomorrow's circuits will be larger.
3. **Productivity.** A working physicist or ML engineer wants to prototype in Python, dispatch onto whatever hardware is available, and only think about pointers and streams when performance demands it.

The cuQuantum SDK is NVIDIA's answer to those three pressures. It is a family of five C libraries (with Python bindings) that share a common engineering philosophy:

- thin, opaque handles managing GPU resources;
- explicit *describe-configure-execute* phases so heavy work can be amortised and reused;
- pluggable workspaces and memory handlers so users keep control of allocation;
- first-class multi-GPU and multi-node primitives via NCCL and MPI;
- a Python layer that hides almost all of this for everyday use, while leaving the C entry points reachable when needed.

This paper explains, library by library, what each piece does, why it does it that way, and how to use it on real hardware. It is grounded in the actual code shipped in this repository: every concept is paired with a runnable example from `samples/`, `python/samples/`, or the unit tests catalogued in [TESTS.md](../TESTS.md).

## Audience and prerequisites

We assume:

- enough linear algebra to be comfortable with complex matrices, eigenvalues, tensor products, and the basics of matrix factorisations (QR, SVD);
- working knowledge of Python 3 and either C or C++;
- familiarity with the idea of a GPU; we provide a refresher in [02-gpu-primer.md](02-gpu-primer.md) for what is specifically relevant.

We do **not** assume prior exposure to quantum mechanics: chapter [01-quantum-primer.md](01-quantum-primer.md) introduces the four pieces of formalism we lean on (states, gates, measurements, channels) in seven pages. Readers with a physics background should skim it for notation.

## Scope and non-goals

In scope:

- the mathematical content of each library: what objects it represents, what operations it provides, what the complexity of those operations is;
- the GPU engineering: where the heavy work happens, how memory and streams are managed, how distributed runs are organised;
- the public C and Python APIs;
- the test and benchmark coverage already in this repository.

Out of scope:

- a tutorial on quantum information theory beyond what is needed to read the API descriptions;
- comprehensive performance numbers: we cite the apparatus to gather them and a few illustrative measurements, but a full performance study deserves its own document.
- algorithm research: we describe the algorithms cuQuantum implements but do not propose new ones.

## How to read it

Three useful paths:

- **Beginner linear:** read 00, 01, 02, 03 in order, then 10 and 20 in depth, then skim the rest.
- **Practitioner:** start at [03-cuquantum-overview.md](03-cuquantum-overview.md), pick the library that matches your problem, end with the relevant section of [70-advanced-applications.md](70-advanced-applications.md).
- **Architect:** chapters 10, 20, 21, 60 give you the resource-cost picture at full scale.

## Conventions

### Notation

- $|\psi\rangle$ is a column vector in $\mathbb{C}^{2^n}$ (a *ket*).
- $\langle\psi|$ is its conjugate transpose (a *bra*); the inner product is $\langle\phi|\psi\rangle$.
- $U, V$ are unitary matrices: $U^\dagger U = I$.
- $H$ is reserved for Hamiltonians (Hermitian operators), not the Hadamard gate; we write the Hadamard as $\mathsf{H}$ in math and `H` in code.
- $\rho$ is a density matrix: $\rho \succeq 0$, $\mathrm{Tr}(\rho) = 1$.
- $P_i \in \{I, X, Y, Z\}$ is a single-qubit Pauli; an $n$-qubit *Pauli string* is $P = P_1 \otimes \cdots \otimes P_n$.
- $\otimes$ is the Kronecker product; we drop it when the meaning is unambiguous.

### Code citations

Files are cited with full repository paths and (where useful) a line range, e.g. `samples/custatevec/custatevec/gate_application.cu` lines 60-90. We never paste a complete listing; the source files in the repo are the reference.

### "Run it yourself" boxes

Every library chapter ends with a small block of shell commands that work in this repository's `.venv` after the install steps below. We test these on the development H100 PCIe node.

## Setting up to run the examples

The repository ships a `python/setup.py` that builds the cuQuantum Python bindings against NVIDIA-provided runtime wheels. The shortest path to a working environment is:

```bash
cd /home/ysha/cuQuantum
python3.12 -m venv .venv && source .venv/bin/activate
pip install -U pip setuptools wheel        # setuptools is required by CFFI on Py >= 3.12
pip install -e python                      # cuquantum-python (Cython bindings)
pip install -e benchmarks                  # nv-quantum-benchmarks CLI
```

The runtime shared libraries (`libcustatevec.so.1`, `libcutensornet.so.2`, `libcudensitymat.so.0`, `libcustabilizer.so.0`, `libcupauliprop.so.0`) come from the `cuquantum-cu13` and `nvidia-*` wheels pulled in transitively.

Two pitfalls worth flagging up front (both encountered while preparing this paper):

- **Source-tree shadowing.** If you run `pytest` from `python/`, Python may import the partially built sources in `python/cuquantum/` instead of the installed package, which leads to `Failed to dlopen libcustatevec`. Either run pytest from a sibling directory (e.g. `python/tests/`) or remove stale in-place build artefacts from the source tree.
- **`setuptools` on Python 3.12+.** The `cffi` API mode used by some `-cffi` test parametrisations needs `setuptools` at runtime. Install it explicitly into the venv.

These are both consequences of running editable installs against pre-built runtime libraries; they are not unique to cuQuantum, but they bite often enough that the paper mentions them in every chapter that uses pytest.

## Hardware assumed

The development node has:

- one NVIDIA H100 PCIe (80 GB)
- CUDA 13.0
- Python 3.12 with `cuquantum-python-cu13`, `cuquantum-cu13`, `cupy-cuda13x`, `mpi4py`, `qiskit`, `cirq`, `stim`, `nbformat`, `setuptools`.

Multi-GPU and multi-node sections describe what is required and how the code changes; they are not exercised on the single-GPU development node.

## What's next

Continue to [01-quantum-primer.md](01-quantum-primer.md) for the math runway, or jump to [03-cuquantum-overview.md](03-cuquantum-overview.md) if you already know quantum mechanics and just want the library map.
