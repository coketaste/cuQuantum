# Figure: cuStateVec multi-GPU and migration

Referenced from [10-custatevec.md](../10-custatevec.md).

```mermaid
flowchart TB
    Full["Full state vector |psi>, length 2^n"] --> Split{"Index-bit partition by P high-order bits"}
    Split --> SubA["Sub-SV on GPU 0"]
    Split --> SubB["Sub-SV on GPU 1"]
    Split --> SubC["..."]
    Split --> SubP["Sub-SV on GPU P-1"]
    SubA -->|"low-order gate"| LocalA["Local kernel"]
    SubB -->|"low-order gate"| LocalB["Local kernel"]
    SubA -->|"high-order swap"| NCCL["NCCL exchange / index-bit swap"]
    SubB -->|"high-order swap"| NCCL
    NCCL --> SubA
    NCCL --> SubB
    SubA -->|"page-out at n>P*log2(GPU mem)"| Host["Pinned host memory"]
    Host -->|"page-in"| SubA
```

The migrator API (`custatevecSubSVMigrator*`) lets the user schedule which sub-SVs reside on which device for each gate; PCIe bandwidth becomes the bottleneck once paging is active.
