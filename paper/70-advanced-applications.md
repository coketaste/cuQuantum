# 70 Advanced Applications

So far each chapter has been about a single library. This chapter shows what composing them looks like for five end-to-end applications you would actually do research with. The point is not to introduce new APIs - everything here is built from material already covered - but to show how the pieces snap together in production.

Each section below ends with a "How to run a smaller version" pointer so you can validate the scaffolding before scaling up.

## 70.1 Variational quantum eigensolver with cuTensorNet gradients

**Goal.** Find the ground-state energy of a molecule by minimising $\langle \psi(\theta) | H | \psi(\theta)\rangle$ over a parametrised circuit.

**Pipeline:**

1. Express $H$ as a sum of Pauli strings (e.g. via OpenFermion, qiskit-nature). Many hundreds of terms is normal.
2. Build the parametrised ansatz circuit on Qiskit/Cirq.
3. Use `cuquantum.tensornet.experimental.NetworkState.from_circuit(circuit, backend='torch', config=TNConfig(...))`.
4. `compute_expectation({pauli_string: coef, ...})` for the forward pass, `compute_expectation_gradient(...)` for the backward pass; PyTorch autograd handles the rest.
5. Optimise with Adam or L-BFGS.

**Why cuTensorNet?** The gradient is reverse-mode through the contraction tree (chapter 20), which is asymptotically optimal for circuits with hundreds of two-qubit gates. State-vector backprop would either need finite differences (cost: number of parameters) or bespoke adjoint code; cuTensorNet's gradient API is automatic.

**Run a smaller version:**
```bash
python python/samples/tensornet/contraction/coarse/example23_torch_grad.py
```
This tutorialises the autograd interop on a 4-tensor contraction. Once that works, swap in your `NetworkState` and a real Hamiltonian.

## 70.2 DMRG-style ground state via approximate MPS

**Goal.** Approximate the ground state of a 1D or quasi-1D Hamiltonian with bond dimension $\chi$.

**Pipeline:**

1. Build $H$ as a `NetworkOperator` (chapter 21) - typically as an MPO of bond dimension a few.
2. Initialise a random MPS via `NetworkState` with `MPSConfig(max_extent=chi)`.
3. Sweep through sites; at each site, locally minimise the energy by solving an effective $\chi^2 \cdot d^2$ eigenvalue problem (DMRG's *site update*). cuTensorNet's tensor decomposition (`cutensornetTensorSVD`) is the workhorse.
4. Truncate, repeat until $\langle E\rangle$ converges.

The sample [python/samples/tensornet/experimental/network_state/circuits_qiskit/example06_mps_approx.py](../python/samples/tensornet/experimental/network_state/circuits_qiskit/example06_mps_approx.py) demonstrates the *forward* MPS sweep in cuTensorNet style; the C++ counterpart [samples/cutensornet/approxTN/mps_example.cu](../samples/cutensornet/approxTN/mps_example.cu) shows the explicit two-site update with truncation. A DMRG driver wraps this loop with a Lanczos eigensolver (which can also be expressed using `cutensornetStateExpectation` for the local matrix-vector products).

**Run a smaller version:** start with the MPS sample above; replace its random circuit with a Trotterised imaginary-time evolution of $H$.

## 70.3 Trajectory-noise simulation of a deep noisy circuit

**Goal.** Estimate $\langle O\rangle$ for a noisy circuit on $n$ qubits at depth $d$, where the full $\rho$ does not fit but Monte-Carlo unravelling does.

**Pipeline:**

1. Build the noisy circuit as a Cirq/Qiskit circuit augmented with channels (depolarising, amplitude damping, etc.).
2. Use the experimental trajectory API in `cuquantum.tensornet.experimental` - see [python/samples/tensornet/experimental/network_state/generic_states/example05_noisy_unitary_channels.py](../python/samples/tensornet/experimental/network_state/generic_states/example05_noisy_unitary_channels.py) and [example06_noisy_unitary_channels_mps.py](../python/samples/tensornet/experimental/network_state/generic_states/example06_noisy_unitary_channels_mps.py).
3. For each trajectory, sample a Kraus operator at each channel, simulate the resulting unitary circuit (exact or MPS), measure $O$.
4. Average over trajectories. Standard error scales as $1/\sqrt{N}$.

The [trajectories_noise/](../python/tests/cuquantum_tests/tensornet/experimental/trajectories_noise/) test directory exercises this pipeline at small scale: `test_onequbit_channel.py` validates against analytical Lindblad solutions; `test_quantum_volume_mid_circuit.py` adds parity-corrected mid-circuit measurement; `test_large_circuits.py` runs MaxCut on random graphs. These tests are the production-quality scaffolds you can adapt.

**Run a smaller version:**
```bash
python -m pytest python/tests/cuquantum_tests/tensornet/experimental/trajectories_noise/test_onequbit_channel.py -v
```

## 70.4 Lindbladian dynamics on a many-body lattice with cuDensityMat

**Goal.** Simulate the Lindblad evolution $\dot\rho = \mathcal{L}(\rho)$ on a lattice with dissipation, sweep parameters, observe a phase transition.

**Pipeline:**

1. Build local operators with `MultidiagonalOperator` for ladder operators and `DenseOperator` for the rest.
2. Assemble $\mathcal{L}$ via the algebra in [python/samples/densitymat/lindbladian_example.py](../python/samples/densitymat/lindbladian_example.py).
3. Use a Krylov or Runge-Kutta integrator (the `operator_action_*` examples are the per-step primitive); cuDensityMat is responsible for `compute_action`, your code does the time-stepping.
4. For batched parameter sweeps, batch the coefficients (chapter 30 §30.5) - one `Operator` evaluation handles all batch entries.
5. Distribute across MPI ranks via `WorkStream(mpi_comm=...)` once memory exceeds one GPU.

The TDVP sample [python/samples/densitymat/mps_tdvp_example.py](../python/samples/densitymat/mps_tdvp_example.py) is a real working integrator that combines MPS state representation with cuDensityMat operator algebra; it is the natural starting point for an MPS-based dissipative simulator.

**Run a smaller version:**
```bash
python python/samples/densitymat/lindbladian_example.py
python python/samples/densitymat/mps_tdvp_example.py
```

## 70.5 Surface-code decoder benchmarking with cuStabilizer

**Goal.** Measure the logical error rate of a Minimum-Weight Perfect-Matching (MWPM) or belief-propagation decoder on a $d \times d$ surface code at noise probability $p$, sweep $d$ and $p$.

**Pipeline:**

1. Use `stim` to generate a noisy surface-code circuit (chapter 40 §40.5).
2. Extract the DEM. Hand it to `DEMSampler`.
3. Sample $N$ shots; each shot is a vector of detector outcomes.
4. Run the decoder on each shot to predict the logical observable; compare against the ground truth (`stim` provides this in the same DEM).
5. Logical error rate = fraction of mismatches; aggregate over $N$.

cuStabilizer's contribution is *throughput*: the DEM sampling runs on the GPU and outputs millions of shots per second, removing what is usually the bottleneck in decoder benchmarks. Decoders themselves remain CPU-side (or use a separate GPU library); the harness pattern is "GPU produces shots, CPU runs decoder".

**Run a smaller version:**
```bash
python python/samples/stabilizer/dem_sampling_example.py --distance 5 --rounds 5 --shots 10000
```

## 70.6 A note on combining libraries

The libraries are designed to coexist. Two patterns we have seen pay off:

- **cuStateVec preconditioning + cuTensorNet exact.** Run a few cheap exact validations with cuStateVec at small $n$ to confirm the cuTensorNet pipeline behaves correctly, then scale up only on cuTensorNet.
- **cuPauliProp + cuStateVec.** When cuPauliProp's truncation introduces concerns, cross-check on a small subspace by running cuStateVec on the same circuit; the two libraries naturally agree on small instances.

The unifying ingredient is the **memory pool**: a single CuPy mempool can be the allocator for cuStateVec, cuTensorNet, and your own tensors (chapter 03 §3.3). Once that is plumbed, switching between libraries within one process is essentially free.

## 70.7 What we did not cover

- **Quantum machine learning pipelines** that wrap circuits in PyTorch / JAX modules. The `backend='torch'` flag in `NetworkState` is the entry point and the rest is conventional ML.
- **Hybrid HPC integrations**, e.g. driving cuQuantum from a CUDA-Q kernel or a Numba-cuda harness. The `nv_quantum_benchmarks` `cudaq` backend is the entry point.
- **Custom Hamiltonians via `NetworkOperator`** beyond simple Pauli sums. The API is straightforward but problem-specific.

These are good follow-ups for a second pass.

## 70.8 Where to read next

- [60-benchmarks.md](60-benchmarks.md) - reproducibility for the experiments above.
- [90-glossary.md](90-glossary.md) - quick reference for any term that snuck in.
- [99-references.md](99-references.md) - external papers and documentation.
