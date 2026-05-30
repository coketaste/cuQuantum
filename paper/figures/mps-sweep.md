# Figure: MPS gate-application sweep

Referenced from [21-cutensornet-mps.md](../21-cutensornet-mps.md).

```mermaid
flowchart LR
    A1["Site 1 tensor A1"] --- A2["Site 2 tensor A2"]
    A2 --- A3["Site 3 tensor A3"]
    A3 --- A4["Site 4 tensor A4"]
    A4 --- A5["Site 5 tensor A5"]
```

Two-site update for a gate G acting on sites k and k+1:

```mermaid
flowchart LR
    Pair["Block = A_k contracted with A_(k+1)"] --> Apply["Multiply by gate G"]
    Apply --> SVD["Truncated SVD with bond dim chi"]
    SVD --> NewK["Updated A_k"]
    SVD --> NewK1["Updated A_(k+1)"]
```

Sweeping left-to-right and right-to-left applies all gates while keeping the state in canonical gauge.
