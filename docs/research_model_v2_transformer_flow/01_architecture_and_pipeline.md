# V2 Transformer-Flow: Architecture and Pipeline

## 1. High-Level Concept

The V2 model combines:
- Discrete autoregressive symbolic modeling (BPE MIDI token stream)
- Continuous flow-matching for musical expression contours
- Domain-aware grammar shaping for raga compliance
- Taal-aware rhythmic positional structure

This creates a dual-output generator that predicts both symbolic note events and performance expression dynamics.

## 2. End-to-End Architecture

```mermaid
flowchart LR
    A[Input tokens x1..xT] --> B[Token Embedding]
    C[Conditioning: mood raga taal tempo duration] --> D[Condition Projection]
    E[Expression frames optional] --> F[Expression Encoder]
    F --> G[Frame-to-model projection]
    B --> H[Backbone input sum]
    D --> H
    G --> H
    I[Taal Position Encoder] --> H

    H --> J[Decoder-only Transformer stack]
    J --> K[Final RMSNorm]

    K --> L[Token Head]
    K --> M[Expression Head]
    K --> N[Flow-Matching Head]

    L --> O[Next-token logits]
    M --> P[Predicted expression]
    N --> Q[Velocity field v_theta]
```

## 3. Internal Block Design

```mermaid
flowchart TD
    X[Hidden state h] --> N1[RMSNorm]
    N1 --> A1[Causal Self Attention with RoPE]
    A1 --> R1[Residual add]
    R1 --> N2[RMSNorm]
    N2 --> F1[SwiGLU-style FFN]
    F1 --> R2[Residual add]
    R2 --> Y[Next hidden state]
```

## 4. Conditioning Path

The conditioning vector is formed by concatenating:
- Mood embedding
- Raga embedding
- Taal embedding
- Tempo scalar projection
- Duration scalar projection

Then projected to model dimension and broadcast across sequence length.

## 5. Rhythmic (Taal) Structure Injection

Taal position is modeled with:
- Cycle-position embedding: position modulo taal cycle length
- Beat-strength embedding: sam and other beat strengths

This adds an explicit rhythmic inductive bias so token generation can track cyclic rhythmic structure.

## 6. Training Pipeline

```mermaid
flowchart LR
    A[Batch: tokens expression labels] --> B[Sample t from Uniform 0..1]
    B --> C[Interpolate expression x_t]
    C --> D[Forward pass]
    D --> E1[Token CE loss]
    D --> E2[Flow velocity MSE]
    D --> E3[Expression MSE]
    D --> E4[Grammar loss]
    E1 --> F[Weighted sum]
    E2 --> F
    E3 --> F
    E4 --> F
    F --> G[Backward optimizer step]
```

## 7. Inference Pipeline

```mermaid
sequenceDiagram
    participant U as User Prompt/Control
    participant C as Conditioning Module
    participant M as Transformer-Flow Model
    participant D as Decoder Loop

    U->>C: mood, raga, taal, tempo, duration
    C->>M: conditioning vector
    loop until EOS or max length
        D->>M: current token prefix
        M->>D: next-token logits + optional expression
        D->>D: top-k/top-p sampling and append token
    end
```

## 8. Why This Architecture Matters

- Symbolic branch preserves grammar and long-form structure.
- Continuous branch captures expressive rendering behavior.
- Shared backbone allows interaction between symbolic and expressive cues.
- Grammar term biases the model toward musically valid raga pitch behavior.
