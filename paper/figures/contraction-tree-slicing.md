# Figure: Contraction tree and slicing

Referenced from [20-cutensornet.md](../20-cutensornet.md).

```mermaid
flowchart TB
    Inputs["Input tensors A, B, C, D, E"] --> AB["Pairwise contraction A*B"]
    Inputs --> CD["Pairwise contraction C*D"]
    AB --> ABE["A*B then with E"]
    CD --> ABCD["A*B*C*D"]
    ABE --> Final["Final tensor"]
    ABCD --> Final
```

If the largest intermediate (e.g. A*B*C*D) does not fit in memory, the optimiser picks one or more *slice modes* whose values are looped over:

```mermaid
flowchart LR
    Slice["For slice s = 0..S-1"] --> Sub["Contract subnetwork at slice s"]
    Sub --> Accum["Accumulate into output"]
    Accum --> Slice
```

Slices are independent and can be assigned to different streams or different GPUs in a distributed run.
