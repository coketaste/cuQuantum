"""Capture cuDensityMat oracle data on H100.

Two GPU-backed APIs are exercised:

- ``cudensitymat.compute_action``: builds a Hamiltonian H = sum_i H_i on a
  multi-mode Hilbert space (each H_i is a single-mode Hermitian DenseOperator),
  wraps it as an ``Operator``, attaches it to a ``DenseMixedState`` rho, and
  calls ``compute_action``. The output rho_out = H @ rho can be cross-checked
  analytically.
- ``cudensitymat.eigenspectrum``: builds the same kind of Hamiltonian and runs
  ``OperatorSpectrumSolver`` (Krylov) to extract the k smallest eigenvalues.

Both runners go through the real cuDensityMat GPU path on the H100. The
inputs are saved in a form that a NumPy/CPU consumer can re-derive the
reference: per-mode Hermitians plus the global rho. The replay script's
NumPy fallback reconstructs the expected output from the same inputs.
"""
from __future__ import annotations

import sys
from pathlib import Path
from typing import Iterable

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from capture._common import (  # noqa: E402
    OracleCase, common_argparser, get_environment_metadata, make_rng,
    stable_hash, write_case,
)

LIBRARY = "cudensitymat"


# ---------------------------------------------------------------------------
# Generators
# ---------------------------------------------------------------------------

def _random_hermitian(rng: np.random.Generator, d: int) -> np.ndarray:
    h = rng.standard_normal((d, d)) + 1j * rng.standard_normal((d, d))
    h = 0.5 * (h + h.conj().T)
    return h.astype("complex128")


def _random_density_matrix(rng: np.random.Generator, d: int) -> np.ndarray:
    a = rng.standard_normal((d, d)) + 1j * rng.standard_normal((d, d))
    rho = a @ a.conj().T
    rho /= np.trace(rho).real
    return rho.astype("complex128")


def gen_compute_action(rng: np.random.Generator, *, small: bool):
    """Hilbert spaces (2,2) and optionally (2,2,2)."""
    dim_sets: Iterable[tuple[int, ...]] = (
        [(2, 2)] if small else [(2, 2), (2, 2, 2)]
    )
    for dims in dim_sets:
        d_total = int(np.prod(dims))
        h_per_mode = [_random_hermitian(rng, d) for d in dims]
        rho = _random_density_matrix(rng, d_total)
        pid = f"dims{'x'.join(map(str, dims))}_{stable_hash(rho, *h_per_mode)}"
        inputs = {
            "hilbert_space_dims": np.asarray(dims, dtype=np.int64),
            "rho": rho,
        }
        for i, h in enumerate(h_per_mode):
            inputs[f"h_{i}"] = h
        yield pid, inputs


def gen_eigenspectrum(rng: np.random.Generator, *, small: bool):
    # The Krylov-subspace bound is conservative; small Hilbert spaces tend
    # to be rejected by the solver. We use (4, 4, 4) for the small set and
    # add (4, 5, 6) for the larger set.
    dim_sets: Iterable[tuple[int, ...]] = (
        [(4, 4, 4)] if small else [(4, 4, 4), (4, 5, 6)]
    )
    for dims in dim_sets:
        d_total = int(np.prod(dims))
        h_per_mode = [_random_hermitian(rng, d) for d in dims]
        k = 3
        pid = f"dims{'x'.join(map(str, dims))}_k{k}_{stable_hash(*h_per_mode)}"
        inputs = {
            "hilbert_space_dims": np.asarray(dims, dtype=np.int64),
            "k": np.asarray(k, dtype=np.int64),
        }
        for i, h in enumerate(h_per_mode):
            inputs[f"h_{i}"] = h
        yield pid, inputs


# ---------------------------------------------------------------------------
# Helpers shared by the runners
# ---------------------------------------------------------------------------

def _build_hamiltonian(dims: tuple[int, ...], h_per_mode: list[np.ndarray]):
    """Construct cuDensityMat Operator H = sum_i (H_i acting on mode i)."""
    import cupy as cp
    from cuquantum.densitymat import (
        DenseOperator, Operator, tensor_product,
    )
    batch_size = 1
    terms = []
    for i, h in enumerate(h_per_mode):
        h_arr = cp.asarray(h, dtype="complex128").reshape(*h.shape, batch_size)
        terms.append(tensor_product((DenseOperator(h_arr), [i])))
    H_term = terms[0]
    for t in terms[1:]:
        H_term = H_term + t
    return Operator(dims, (H_term,)), batch_size


# ---------------------------------------------------------------------------
# Runners
# ---------------------------------------------------------------------------

def run_compute_action(inputs: dict) -> dict:
    import cupy as cp
    from cuquantum.densitymat import DenseMixedState, WorkStream

    dims = tuple(int(x) for x in np.asarray(inputs["hilbert_space_dims"]).tolist())
    rho_in = np.asarray(inputs["rho"], dtype="complex128")
    h_per_mode = [
        np.asarray(inputs[f"h_{i}"], dtype="complex128") for i in range(len(dims))
    ]
    num_modes = len(dims)

    ctx = WorkStream()
    hamiltonian, batch_size = _build_hamiltonian(dims, h_per_mode)

    rho = DenseMixedState(ctx, dims, batch_size, "complex128")
    rho.attach_storage(cp.empty(rho.storage_size, dtype="complex128"))
    rho_view = rho.view()
    # The view tensor has shape (*dims, *dims, batch_size). Flatten the
    # (D, D) input rho into that layout.
    rho_view[:] = cp.asarray(rho_in).reshape(*dims, *dims, batch_size)
    rho_out_state = rho.clone(cp.zeros_like(rho.storage, order="F"))

    hamiltonian.prepare_action(ctx, rho)
    hamiltonian.compute_action(0.0, [], rho, rho_out_state)
    cp.cuda.Stream.null.synchronize()

    rho_out_view = rho_out_state.view()
    d_total = int(np.prod(dims))
    rho_out = cp.asnumpy(rho_out_view).reshape(d_total, d_total)
    return {"rho_out": rho_out.astype("complex128")}


def run_eigenspectrum(inputs: dict) -> dict:
    import cupy as cp
    from cuquantum.densitymat import (
        DensePureState, OperatorSpectrumConfig, OperatorSpectrumSolver, WorkStream,
    )

    dims = tuple(int(x) for x in np.asarray(inputs["hilbert_space_dims"]).tolist())
    k = int(np.asarray(inputs["k"]))
    h_per_mode = [
        np.asarray(inputs[f"h_{i}"], dtype="complex128") for i in range(len(dims))
    ]

    ctx = WorkStream()
    hamiltonian, batch_size = _build_hamiltonian(dims, h_per_mode)
    hilbert_vol = int(np.prod(dims))

    rng = np.random.default_rng(0)
    states = []
    for _ in range(k):
        state = DensePureState(ctx, dims, batch_size, "complex128")
        state.allocate_storage()
        seed_vec = (rng.standard_normal(hilbert_vol * batch_size)
                    + 1j * rng.standard_normal(hilbert_vol * batch_size))
        state.storage[:] = cp.asarray(seed_vec, dtype="complex128")
        norm = state.norm()
        state.inplace_scale(1.0 / cp.sqrt(norm))
        states.append(state)
    states = tuple(states)

    # The Krylov subspace size is bounded by min_block_size * max_buffer_ratio,
    # and that times 2 plus min_block_size must fit inside the Hilbert space.
    # For our small test systems we keep both parameters as small as the
    # checker allows.
    config = OperatorSpectrumConfig(
        min_krylov_block_size=2,
        max_buffer_ratio=2,
        max_restarts=40,
    )
    spectrum = OperatorSpectrumSolver(hamiltonian, "SA", True, config)
    spectrum.prepare(ctx, states[0], max_num_eigvals=k)
    result = spectrum.compute(t=0.0, params=[], states=states, tol=1e-10)
    cp.cuda.Stream.null.synchronize()

    evals_arr = result.evals
    evals = (evals_arr.get() if hasattr(evals_arr, "get")
             else np.asarray(evals_arr))
    evals = evals.flatten().astype("float64")
    # Sort ascending so the comparison is order-invariant.
    evals = np.sort(evals)[:k]
    return {"eigvals": evals}


# ---------------------------------------------------------------------------
# Driver
# ---------------------------------------------------------------------------

API_TABLE = [
    ("cudensitymat.compute_action", gen_compute_action, run_compute_action),
    ("cudensitymat.eigenspectrum",  gen_eigenspectrum,  run_eigenspectrum),
]


def main(argv=None) -> int:
    p = common_argparser(LIBRARY)
    args = p.parse_args(argv)
    rng = make_rng(args.seed)
    env = get_environment_metadata()
    n = 0
    for api, gen, runner in API_TABLE:
        for pid, inputs in gen(rng, small=args.small):
            if args.filter and args.filter not in pid:
                continue
            print(f"[capture] {api} {pid}")
            if args.dry_run:
                continue
            outputs = runner(inputs)
            case = OracleCase(
                api=api, param_id=pid, inputs=inputs, outputs=outputs,
                params={"seed": args.seed},
                metadata={
                    "library": LIBRARY,
                    "kind": "deterministic",
                    "backend": "cuquantum-gpu",
                },
            )
            write_case(args.out, LIBRARY, case, env)
            n += 1
    print(f"[capture] wrote {n} cases")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
