# Research Figures Pack (Mermaid)

This file centralizes reusable Mermaid diagrams for your paper and slides.

## Figure 1: Hybrid CVAE Full Pipeline

```mermaid
flowchart LR
    A[Tokens + Controls] --> B[Conditioning Fusion]
    B --> C[Transformer Encoder]
    C --> D[mu and logvar]
    D --> E[Reparameterize z]
    E --> F[Transformer Decoder]
    F --> G[Token Logits]
    F --> H[Expression Head optional]
```

## Figure 2: Transformer-Flow Full Pipeline

```mermaid
flowchart LR
    A[Token Prefix + Controls] --> B[Embeddings + Taal Position]
    B --> C[Decoder-only Transformer Backbone]
    C --> D[Token Head]
    C --> E[Expression Head]
    C --> F[Flow Velocity Head]
    D --> G[Autoregressive CE]
    E --> H[Expression MSE]
    F --> I[Flow MSE]
    D --> J[Grammar Term]
```

## Figure 3: Flow-Matching Geometry

```mermaid
flowchart LR
    A[Noise epsilon] --> C[x_t interpolation]
    B[Data expression x] --> C
    C --> D[Model predicts velocity v_theta]
    E[Target x - epsilon] --> F[MSE between predicted and target velocity]
    D --> F
```

## Figure 4: Loss Composition (V2)

```mermaid
flowchart TD
    A[Total loss] --> B[Token CE]
    A --> C[Flow MSE]
    A --> D[Expr MSE]
    A --> E[Grammar loss]
```

## Figure 5: KL Annealing Concept (Hybrid CVAE)

```mermaid
flowchart LR
    A[Epoch index] --> B[Cyclic position]
    B --> C[KL weight ramp]
    C --> D[Lower-bounded KL coefficient]
    D --> E[Total CVAE objective]
```

## Figure 6: Evaluation Protocol

```mermaid
flowchart TD
    A[Generated samples] --> B[Automatic metrics]
    A --> C[Expert listening]
    B --> D[Compliance and coherence scores]
    C --> E[Perceptual quality scores]
    D --> F[Unified result table]
    E --> F
```

## Figure 7: Patent Claim Decomposition

```mermaid
flowchart LR
    A[System claim] --> B[Architecture claim]
    A --> C[Objective claim]
    A --> D[Domain constraint claim]
    C --> C1[Flow term]
    C --> C2[Grammar term]
    D --> D1[Raga constraints]
    D --> D2[Taal cycle encoding]
```
