# Figure: Pauli propagation tree

Referenced from [50-cupauliprop.md](../50-cupauliprop.md).

```mermaid
flowchart TB
    Start["Observable Z_k, single Pauli"] --> Step1["Apply U_L^dagger"]
    Step1 --> Branch1["P -> P or commuting case"]
    Step1 --> Branch2["P -> cos theta P + i sin theta P P_g"]
    Branch1 --> Step2["Apply U_(L-1)^dagger"]
    Branch2 --> Step2
    Step2 --> Trunc["Truncate: drop |c| < eps or weight > w"]
    Trunc --> Step3["...continue backwards..."]
    Step3 --> Final["tilde O = sum c_i P_i"]
    Final --> Inner["Inner product with |0...0>"]
    Inner --> Result["Expectation value"]
```

Without truncation the tree branches at every non-commuting rotation and grows exponentially. The library's truncation step prunes coefficients below a threshold and Paulis above an operator-weight cap.
