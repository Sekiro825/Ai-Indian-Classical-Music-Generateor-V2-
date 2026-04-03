# Research Paper Master Draft

## Title

Domain-Constrained Hybrid Symbolic-Expressive Generative Modeling for Indian Classical Music: A Comparative Study of Hybrid CVAE and Transformer-Flow Architectures

## Abstract

This work presents and compares two controllable generative modeling paradigms for Indian classical music: (1) a Hybrid Conditional Variational Autoencoder (Hybrid CVAE) and (2) a Transformer-Flow architecture (V2) that combines autoregressive token modeling with flow-matching expression dynamics. Both models condition generation on musically meaningful controls including raga, taal, mood, tempo, and duration. We provide architecture-level analysis, objective-level decomposition, and a domain-focused novelty framing centered on grammar-aware tonal constraints and rhythmic cyclic encoding. The Hybrid CVAE emphasizes latent global structure and controllability, while the Transformer-Flow model emphasizes sharp local expression dynamics with multi-term objective coupling. We propose a reproducible evaluation protocol with grammar compliance, rhythmic coherence, expression fidelity, and expert listening assessment, and provide a claim-ready technical decomposition useful for future IP and translational research planning.

## 1. Introduction

Indian classical music generation requires simultaneous control of:
- Long-horizon structure (phrase and progression)
- Raga-specific tonal compliance
- Taal-specific cyclic rhythmic behavior
- Performance-level expression contours

Purely symbolic autoregressive models often under-represent expressive dynamics, while purely continuous models can struggle with explicit rule-level symbolic constraints. This motivates hybrid symbolic-continuous modeling approaches.

## 2. Contributions

1. A unified documentation-backed comparison of two implemented architectures in one repository.
2. A domain-aware training decomposition with grammar and rhythm constraints.
3. A practical reproducibility blueprint for publication and downstream patent strategy preparation.
4. A comparative perspective on latent-variable vs flow-matching trade-offs in culturally grounded music generation.

## 3. Model Family A: Hybrid CVAE

### 3.1 Core idea

The Hybrid CVAE encodes conditioned token sequences into a latent posterior and reconstructs/generates autoregressively through a Transformer decoder with latent cross-attention.

### 3.2 Objective

$$
\mathcal{L}_{A} = \mathcal{L}_{recon} + \lambda_{KL}(e)\mathcal{L}_{KL}^{fb}
$$

Optional expression supervision term:

$$
\mathcal{L}_{A}' = \mathcal{L}_{A} + \lambda_{expr}\mathcal{L}_{expr}
$$

## 4. Model Family B: Transformer-Flow (V2)

### 4.1 Core idea

A decoder-only Transformer produces symbolic tokens while jointly learning expression via flow-matching velocity fields and direct expression regression.

### 4.2 Objective

$$
\mathcal{L}_{B} = \mathcal{L}_{token} + \lambda_{flow}\mathcal{L}_{flow} + \lambda_{expr}\mathcal{L}_{expr} + \lambda_{grammar}\mathcal{L}_{grammar}
$$

## 5. Comparative Hypotheses

1. Hybrid CVAE may improve macro-structural coherence via latent global control.
2. Transformer-Flow may improve local expressive sharpness and grammar-aware compliance under richer objective shaping.
3. Taal-aware rhythmic injection should improve cycle consistency in both families when enabled.

## 6. Experimental Protocol

### 6.1 Quantitative metrics

- Token-level NLL / perplexity
- Raga grammar compliance score
- Vadi/samvadi emphasis score
- Taal cycle coherence score
- Expression fidelity (MSE, correlation)

### 6.2 Human evaluation

- Expert blind rating: raga identity
- Expert blind rating: rhythmic validity
- Expert blind rating: expressive musicality

### 6.3 Ablations

- Remove grammar term
- Remove flow term
- Remove expression path
- Remove taal position encoding
- Hybrid CVAE without cyclical KL

## 7. Results Template (to fill)

| Model | Grammar Compliance | Rhythm Coherence | Expression Fidelity | Human Musicality |
|---|---:|---:|---:|---:|
| Hybrid CVAE |  |  |  |  |
| Transformer-Flow V2 |  |  |  |  |

## 8. Discussion

The two families target complementary modeling behavior: latent-variable modeling offers compact global variation control, while flow-enhanced training can provide stronger local expressive transitions and sharper conditioning-response dynamics. The domain-specific grammar and taal controls appear central for culturally valid generation.

## 9. Conclusion

This repository demonstrates a practical progression from latent-conditioned symbolic generation toward objective-coupled symbolic-expressive flow modeling, creating a strong foundation for both academic publication and translational IP strategy.

## 10. Reproducibility Checklist

- Model configs and training commands archived
- Dataset curation protocol documented
- Evaluation scripts and random seeds preserved
- Checkpoint selection criteria explicitly stated
