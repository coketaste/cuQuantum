# 99 References

## NVIDIA documentation

- cuQuantum SDK landing page: <https://developer.nvidia.com/cuquantum-sdk>
- cuQuantum SDK documentation: <https://docs.nvidia.com/cuda/cuquantum/latest/>
  - cuStateVec: <https://docs.nvidia.com/cuda/cuquantum/latest/custatevec>
  - cuTensorNet: <https://docs.nvidia.com/cuda/cuquantum/latest/cutensornet>
  - cuDensityMat: <https://docs.nvidia.com/cuda/cuquantum/latest/cudensitymat>
  - cuStabilizer: <https://docs.nvidia.com/cuda/cuquantum/latest/custabilizer>
  - cuPauliProp: <https://docs.nvidia.com/cuda/cuquantum/latest/cupauliprop>
- CUTENSOR documentation: <https://docs.nvidia.com/cuda/cutensor>
- NCCL documentation: <https://docs.nvidia.com/deeplearning/nccl>
- CUDA-Q: <https://nvidia.github.io/cuda-quantum>

## Foundational textbooks

- M. A. Nielsen and I. L. Chuang, *Quantum Computation and Quantum Information*, 10th anniversary ed., Cambridge University Press, 2010.
- H.-P. Breuer and F. Petruccione, *The Theory of Open Quantum Systems*, Oxford University Press, 2007.
- R. Penrose, *The Road to Reality*, Knopf, 2004 - chapters on Hilbert space and the density matrix.

## Foundational papers

### State vector and gate application

- M. Smelyanskiy, N. P. Sawaya, A. Aspuru-Guzik, "qHiPSTER: The Quantum High Performance Software Testing Environment", `arXiv:1601.07195`. Background on full-state-vector simulation at scale.
- T. Haener and D. S. Steiger, "0.5 Petabyte Simulation of a 45-Qubit Quantum Circuit", `arXiv:1704.01127`.
- E. Pednault et al., "Pareto-Efficient Quantum Circuit Simulation Using Tensor Contraction Deferral", `arXiv:1710.05867`.

### Tensor networks

- I. L. Markov and Y. Shi, "Simulating quantum computation by contracting tensor networks", `arXiv:quant-ph/0511069`.
- J. Gray and S. Kourtis, "Hyper-optimized tensor network contraction", `arXiv:2002.01935`. The path-finding methodology cuTensorNet's optimiser is heir to.
- F. Pan, K. Chen and P. Zhang, "Solving the sampling problem of the Sycamore quantum supremacy circuits", `arXiv:2111.03011`.
- U. Schollwoeck, "The density-matrix renormalization group in the age of matrix product states", `arXiv:1008.3477`. The standard reference for MPS, MPO, and DMRG.
- G. Vidal, "Efficient classical simulation of slightly entangled quantum computations", `arXiv:quant-ph/0301063`.

### Open systems

- A. J. Daley, "Quantum trajectories and open many-body quantum systems", `arXiv:1405.6694`.
- F. Verstraete, J. J. Garcia-Ripoll and J. I. Cirac, "Matrix Product Density Operators: Simulation of finite-temperature and dissipative systems", `arXiv:cond-mat/0406426`.
- J. Haegeman et al., "Time-dependent variational principle for quantum lattices", `arXiv:1103.0936`.

### Stabiliser formalism

- D. Gottesman, "The Heisenberg representation of quantum computers", `arXiv:quant-ph/9807006`.
- S. Aaronson and D. Gottesman, "Improved simulation of stabilizer circuits", `arXiv:quant-ph/0406196`. The algorithm cuStabilizer implements.
- C. Gidney, "Stim: a fast stabilizer circuit simulator", `arXiv:2103.02202`. Reference implementation; cuStabilizer cross-validates against it.
- A. G. Fowler, M. Mariantoni, J. M. Martinis and A. N. Cleland, "Surface codes: Towards practical large-scale quantum computation", `arXiv:1208.0928`.

### Heisenberg-picture / Pauli propagation

- T. Begusic, J. Hejazi and G. K.-L. Chan, "Fast and converged classical simulations of evidence for the utility of quantum computing before fault tolerance via tensor networks", `arXiv:2308.05077`.
- M. C. Caro et al., "Out-of-distribution generalization for learning quantum dynamics", `arXiv:2204.10268` - background on truncation as an effective regularizer.
- Y. Kim, A. Eddins, S. Anand, K. X. Wei, E. van den Berg, S. Rosenblatt, H. Nayfeh, Y. Wu, M. Zaletel, K. Temme, A. Kandala, "Evidence for the utility of quantum computing before fault tolerance", *Nature* 618, 500-505 (2023). The IBM heavy-hex experiment that the cuPauliProp `kicked_ising_example.py` reproduces classically.

## GPU computing

- D. Kirk and W. Hwu, *Programming Massively Parallel Processors*, 4th ed., Morgan Kaufmann, 2022.
- M. Harris, "An Easy Introduction to CUDA C and C++", NVIDIA Developer Blog. Useful for the streams / events / kernels primer.
- NVIDIA H100 architecture whitepaper: <https://resources.nvidia.com/en-us-tensor-core/gtc22-whitepaper-hopper>.

## Frameworks cuQuantum interoperates with

- Qiskit: <https://qiskit.org>
- Cirq: <https://quantumai.google/cirq>
- PennyLane: <https://pennylane.ai>
- Qulacs: <https://github.com/qulacs/qulacs>
- CUDA-Q: <https://nvidia.github.io/cuda-quantum>
- Stim: <https://github.com/quantumlib/Stim>
- CuPy: <https://cupy.dev>
- PyTorch: <https://pytorch.org>

## Repository internal references

- [TESTS.md](../TESTS.md) - the unit-test catalogue used throughout this paper.
- [paper/README.md](README.md) - paper TOC and reading paths.
- [paper/figures/](figures/) - mermaid sources for the diagrams in chapters 03, 10, 20, 21, 30, 50, 70.

## Versions

This paper targets the cuQuantum SDK as installed in the development environment (May 2026): `cuquantum-cu13`, `cuquantum-python-cu13` built from this source tree, CUDA 13.0, Python 3.12. Subsequent SDK versions may rename APIs or add functionality; the structure of the library should remain stable.
