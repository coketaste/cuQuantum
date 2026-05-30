# Figure: Liouvillian batched evaluation in cuDensityMat

Referenced from [30-cudensitymat.md](../30-cudensitymat.md).

```mermaid
flowchart TB
    Hamiltonian["Hamiltonian H = sum_a c_a O_a"] --> Lio["Build Liouvillian L = -i [H, .] + dissipators"]
    Jumps["Jump operators L_j with rates gamma_j"] --> Lio
    Lio --> Action["Operator.compute_action(rho_in, rho_out)"]
    rho_in["Batched state rho_in"] --> Action
    Action --> rho_out["Batched state rho_out"]
    rho_out --> Integrator["External RK / Krylov integrator"]
    Integrator --> rho_in
```

The same `Operator` instance is reused across all timesteps and all batch entries. Coefficients can be host-side numbers, device arrays, or callbacks evaluated at each step.
