# Hybrid CVAE: Architecture and Pipeline

## 1. Model Intent

The Hybrid CVAE learns controllable Indian classical symbolic music generation by:
- Encoding token sequences into a latent distribution
- Conditioning generation on mood, raga, taal, tempo, and duration
- Decoding autoregressively from sampled latent variables
- Optionally integrating expression features as an auxiliary conditioning and prediction stream

## 2. Full Architecture

```mermaid
flowchart LR
    A[Token sequence] --> B[Token Embedding]
    C[Conditioning inputs] --> D[Conditioning Module]
    E[Expression optional] --> F[Expression Encoder]

    B --> G[Encoder input fusion]
    D --> G
    F --> G

    G --> H[Transformer Encoder stack]
    H --> I[Pooling]
    I --> J[mu head]
    I --> K[logvar head]
    J --> L[Reparameterization]
    K --> L

    L --> M[Latent z]
    M --> N[Latent projection to memory]

    A --> O[Decoder token embedding path]
    D --> O
    O --> P[Transformer Decoder stack]
    N --> P

    P --> Q[Token output head]
    P --> R[Expression head optional]
```

## 3. Encoder-Decoder Internals

```mermaid
flowchart TD
    A[PreNorm input] --> B[RoPE self-attention]
    B --> C[Residual add]
    C --> D[FFN GELU block]
    D --> E[Residual add]
```

Decoder extends this with cross-attention to latent memory.

## 4. Conditioning Mechanism

The conditioning module embeds and fuses:
- Mood index
- Raga index
- Taal index
- Tempo scalar
- Duration scalar

The fused vector is projected to model embedding dimension and added to token representations.

## 5. Latent Variable Pathway

- Encoder output is mean-pooled (mask-aware if padding exists).
- Two linear heads produce $\mu$ and $\log\sigma^2$.
- Latent sample uses reparameterization trick.
- Decoder consumes latent memory via cross-attention during autoregressive reconstruction/generation.

## 6. Training Flow

```mermaid
flowchart LR
    A[Batch tokens + condition labels] --> B[Forward pass]
    B --> C1[Reconstruction CE]
    B --> C2[KL divergence]
    C1 --> D[Total loss]
    C2 --> D
    D --> E[Backward and optimizer]
```

## 7. Generation Flow

```mermaid
sequenceDiagram
    participant U as User Controls
    participant C as Conditioning Module
    participant P as Prior Sampler
    participant D as Decoder

    U->>C: mood/raga/taal/tempo/duration
    C->>P: conditioning vector
    P->>D: sample z from N(0,I)
    loop token generation
        D->>D: causal decode
        D->>D: top-k/top-p sample next token
    end
    D-->>U: generated token sequence and optional expression
```

## 8. Why This Hybrid Design Is Useful

- Latent space captures global compositional intent.
- Autoregressive decoder preserves sequential structure.
- Conditioning provides direct control for user-facing generation.
- Expression branch allows bridging symbolic and performance-level characteristics.
