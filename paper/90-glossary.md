# 90 Glossary

A working glossary for the terms used in this paper. Definitions are pragmatic, not formal.

**Adjoint.** The Hermitian conjugate $U^\dagger$. The "undo" of a unitary. cuStateVec's `applyMatrix` takes an `adjoint` flag; cuPauliProp's backward propagation is built on adjoint application of every gate.

**Amplitude.** A complex number $c_x = \langle x|\psi\rangle$. The state-vector representation stores all $2^n$ amplitudes.

**Autotuning.** cuTensorNet primitive that times candidate kernel implementations for each contraction step in a plan and remembers the winner. Implemented by `cutensornetContractionAutotune`.

**Bond dimension ($\chi$).** In an MPS, the maximum extent of an inter-site bond. Memory $O(n \chi^2)$, gate cost $O(\chi^3)$. The library knob is `MPSConfig.max_extent`.

**Bra, ket.** $\langle\phi|$ and $|\psi\rangle$. The bra is the conjugate-transpose of the ket.

**CFFI.** C-foreign-function-interface library used by some unit-test parametrisations to call into the cuQuantum C entry points without going through Cython. Requires `setuptools` at runtime on Python 3.12+.

**CFTP / CPTP.** Completely positive trace-preserving map. The defining property of a quantum channel.

**Channel.** A CPTP map; equivalently a list of Kraus operators $\{K_k\}$ with $\sum_k K_k^\dagger K_k = I$.

**Clifford gate.** A unitary $U$ such that $U^\dagger P U$ is a Pauli string for every Pauli string $P$. Hadamard, phase, CNOT and their compositions are Cliffords.

**Compute type.** The numerical precision used inside a kernel, distinct from the storage dtype. cuTensorNet's `compute_type=COMPUTE_32F` enables tensor cores for FLOP-heavy contractions.

**Contraction (path / plan).** A pair (a) topological order of pairwise tensor contractions, (b) per-step CUTENSOR algorithm choice. The product of `cutensornetContractionOptimize` and `cutensornetCreateContractionPlan` respectively.

**CSR.** Compressed sparse row, a sparse-matrix format used to represent DEMs and parity-check matrices in cuStabilizer.

**Cumulative-distribution sampling.** The standard inverse-CDF method for drawing samples from a discrete probability distribution. cuStateVec's sampler uses this on $|c_x|^2$.

**Density matrix ($\rho$).** Positive semidefinite, trace-1 operator on $\mathbb{C}^D$. The state of a possibly mixed quantum system.

**DEM (Detector Error Model).** A description of the noise affecting a QEC circuit as a list of *detectors* and *error mechanisms*; format is the de-facto standard set by `stim`. Sampled by cuStabilizer's `DEMSampler`.

**Dual.** In cuDensityMat, the bra-side dual of an operator: where the original acts on $|\psi\rangle$, the dual acts on $\langle\psi|$. Built into the `OperatorTerm` algebra.

**Effective bandwidth.** Bytes touched divided by GPU time, used as a roofline check. The `apply_matrix` benchmark prints this.

**Einstein summation (einsum).** A notation that names tensor axes with letters and contracts repeated letters: `'ij,jk->ik'` is matrix multiplication.

**Frame simulator.** A stabiliser simulator that maintains many independent stabiliser states ("frames") in lock-step under one Clifford circuit, used to sample many shots in parallel.

**Gauge.** Equivalence under change of internal basis on inter-site bonds in an MPS. cuTensorNet maintains a canonical gauge to keep truncations meaningful.

**Gradient (of a contraction).** The reverse-mode adjoint through a tensor-network contraction. Implemented in cuTensorNet's `cutensornetContractionGradient` and the `compute_expectation_gradient` method on `NetworkState`.

**Handle.** Long-lived opaque per-device GPU resource owner; one per library, e.g. `custatevecHandle_t`.

**HBM.** High-bandwidth memory, the GPU's main DRAM. Bandwidth typically 2-4 TB/s on H100.

**Heisenberg picture.** Operator-evolution picture: states stand still, observables transform as $O \mapsto U^\dagger O U$. cuPauliProp's native picture.

**Index bit.** In cuStateVec, the bit position of a qubit in the linear amplitude index. *High-order* bits index between sub-states across GPUs; *low-order* bits index within one sub-state.

**Jump operator ($L_j$).** An operator in the Lindblad master equation describing one decoherence channel.

**Kraus operator ($K_k$).** A summand in a channel's Kraus representation. $\sum_k K_k^\dagger K_k = I$.

**Liouvillian ($\mathcal{L}$).** The superoperator on the right-hand side of a Lindblad master equation.

**MemHandler.** A pair of allocate/deallocate callbacks that cuQuantum can use for scratch. Lets you share memory pools with CuPy / PyTorch.

**MPI.** Message-passing interface. The portable abstraction for multi-node communication; cuQuantum integrates via mpi4py (Python) or directly (C).

**MPO (Matrix Product Operator).** The operator analogue of an MPS - a chain of rank-4 tensors representing an operator on $n$ sites with bounded bond dimension.

**MPS (Matrix Product State).** A 1D tensor-network state representation with bond dimension $\chi$. Approximate for $\chi$ small, exact for $\chi = 2^{n/2}$. cuTensorNet's `MPSConfig` controls truncation.

**Multidiagonal.** A banded matrix. cuDensityMat's `MultidiagonalOperator` stores only the non-zero diagonals.

**NCCL.** NVIDIA Collective Communications Library. Used by cuTensorNet, cuStateVec multi-GPU, cuDensityMat distributed.

**NetworkState.** The high-level cuTensorNet object that maintains a state-as-network. Method calls translate to contractions on demand.

**OperatorTerm / Operator.** cuDensityMat's symbolic representation of a single tensor-product term and a sum of terms, respectively.

**Path finder.** The cuTensorNet optimiser that searches for an efficient contraction order. Heuristic; runs in seconds and saves minutes-to-hours.

**Pauli expansion.** A representation of an observable as a sum of Pauli strings with complex coefficients. Native data structure in cuPauliProp.

**Pauli string.** $P_1 \otimes \cdots \otimes P_n$ with each $P_i \in \{I, X, Y, Z\}$, possibly with a $\pm 1, \pm i$ overall phase.

**Permutation matrix.** A matrix with exactly one 1 per row and column. cuStateVec's `applyGeneralizedPermutationMatrix` is the optimised path for diagonal-and-permuted gates.

**Phased decompose-and-contract.** cuTensorNet idiom: decompose your operands into a network, run path/autotune once, contract many times with `reset_operands`.

**Quantum trajectory / unravelling.** Stochastic pure-state evolution whose ensemble average reproduces a Lindblad evolution. Used by cuTensorNet's noise extension and as an alternative to cuDensityMat at large $n$.

**RDM (Reduced Density Matrix).** $\rho_S = \mathrm{Tr}_{\bar S} \rho$, the marginal state on a subsystem $S$. Computed by `NetworkState.compute_reduced_density_matrix`.

**Slicing.** Decomposing a contraction by partitioning the values of one or more modes and summing over slices, to fit a memory budget or to parallelise. cuTensorNet's `Slicer*` knobs.

**State vector.** The full $2^n$-amplitude representation of a pure state. cuStateVec's data object.

**Stabiliser tableau.** Binary $2n \times (2n+1)$ representation of a stabiliser state.

**Stim.** Reference open-source stabiliser simulator (`arXiv:2103.02202`). cuStabilizer is the GPU counterpart and validates against it.

**Stream.** A CUDA in-order queue of GPU work. Almost every cuQuantum compute call takes a `cudaStream_t`.

**SubSV (sub-state-vector).** A partition of the full state vector across multiple GPUs (and optionally pinned host memory) keyed on high-order index bits.

**Superoperator.** A linear map from operators to operators. The Liouvillian $\mathcal{L}$ is a superoperator.

**TDVP.** Time-Dependent Variational Principle. An MPS-friendly integrator for time evolution; demonstrated in cuDensityMat's TDVP sample.

**Tensor core.** Mixed-precision matrix-multiply unit on NVIDIA GPUs. Used by cuTensorNet when `compute_type` allows it.

**Tensor network.** A finite collection of tensors with shared mode labels representing a single big tensor by Einstein-style contraction.

**Truncation.** In MPS: discarding small singular values to bound bond dimension. In Pauli expansions: discarding small-coefficient or high-weight Paulis.

**Unitary.** A matrix $U$ with $U^\dagger U = I$. Quantum gates are unitaries.

**Workspace.** Externally provided scratch memory bound to a cuQuantum descriptor. Reused across calls.

**Z-basis measurement.** Measurement in the computational basis $\{|0\rangle, |1\rangle\}$.

Continue to [99-references.md](99-references.md).
