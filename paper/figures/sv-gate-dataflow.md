# Figure: cuStateVec gate-application data flow

Referenced from [10-custatevec.md](../10-custatevec.md).

```mermaid
flowchart LR
    Start["Amplitude array, length 2^n"] --> Reshape["Logical reshape to 2^k by 2^(n-k)"]
    Reshape --> Read["Stream-load 2^k slabs at outer index"]
    Read --> GEMM["Multiply by 2^k by 2^k gate matrix G"]
    GEMM --> Write["Stream-store back to amplitude array"]
    Write --> Next["Next outer index"]
    Next --> Read
    Write --> End["Updated amplitude array"]
```

For controlled gates, only outer indices whose control bits match are processed; the rest are passed through. Memory traffic per gate: 2 * 2^n * b bytes regardless of k.
