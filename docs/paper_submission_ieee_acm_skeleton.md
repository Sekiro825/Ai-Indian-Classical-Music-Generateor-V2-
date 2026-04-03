# IEEE/ACM Submission Skeleton (Research Paper Template)

This is a publication-ready writing scaffold for your Indian classical music generation work.
It is designed to be adapted into either IEEE or ACM style with minimal edits.

## 0. Venue Mode

Select one mode before drafting:
- Mode A: IEEE conference/journal style
- Mode B: ACM conference style

Use consistent section naming and reference format once selected.

## 1. Title Page Block

### 1.1 Candidate Title

Domain-Constrained Hybrid Symbolic-Expressive Generative Modeling for Indian Classical Music: Comparative Analysis of Hybrid CVAE and Transformer-Flow Architectures

### 1.2 Author Block Placeholder

- First Author, Affiliation, Email
- Second Author, Affiliation, Email
- Third Author, Affiliation, Email

### 1.3 Keywords

Indian classical music generation; controllable generation; conditional VAE; flow matching; symbolic music modeling; raga grammar; taal conditioning

## 2. Abstract (150-250 words template)

Write one paragraph covering:
1. Problem and domain challenge.
2. Proposed method families (Hybrid CVAE and Transformer-Flow).
3. Core novelty (symbolic + expressive hybridization, grammar-aware and rhythmic constraints).
4. Evaluation protocol (automatic + expert listening).
5. Main quantitative/qualitative outcome statement.

Template:

This paper studies controllable Indian classical music generation under structural and expressive constraints. We investigate two implemented model families: a Hybrid Conditional Variational Autoencoder and a Transformer-Flow model that jointly optimizes autoregressive token prediction, expression dynamics, and grammar-aware tonal regularization. Both architectures condition on mood, raga, taal, tempo, and duration, and incorporate rhythmic/tonal inductive biases for culturally grounded generation. We evaluate model behavior using token likelihood, raga compliance, rhythmic coherence, expression fidelity, and expert listening studies. Results show [insert key findings], with trade-offs between global latent structure and local expressive sharpness. These findings provide a reproducible benchmark and practical design guidance for domain-specific symbolic-expressive music generation systems.

## 3. Introduction

### 3.1 Problem Statement

- Why Indian classical generation is difficult:
  - Long-term structural coherence
  - Raga grammar constraints
  - Taal cycle constraints
  - Expression realism

### 3.2 Gaps in Prior Work

- General music language models may ignore domain grammar.
- Symbolic-only models can under-model expressive dynamics.
- Continuous-only methods can weaken explicit symbolic control.

### 3.3 Contributions (bullet-ready)

1. A dual-model comparative framework (Hybrid CVAE vs Transformer-Flow).
2. Domain-aware objective/control formulation for raga-taal conditioning.
3. Reproducible evaluation stack and research artifacts.
4. Practical guidance for translational and IP-oriented research paths.

## 4. Related Work

Structure suggested:
1. Symbolic music generation with Transformers.
2. VAE/CVAE sequence generation methods.
3. Diffusion/flow methods in music/audio domains.
4. Rule-constrained and domain-conditioned generation.
5. Indian classical computational music literature.

Conclude with a positioning paragraph that states exactly what your implementation contributes beyond these clusters.

## 5. Method

### 5.1 Notation

Define:
- Token sequence: $x_{1:T}$
- Controls: $c = (mood, raga, taal, tempo, duration)$
- Expression: $e_{1:T}$

### 5.2 Hybrid CVAE Method

Include architecture summary and objective:

$$
\mathcal{L}_{CVAE} = \mathcal{L}_{recon} + \lambda_{KL}(e)\mathcal{L}_{KL}^{fb}
$$

Optional extension term:

$$
\mathcal{L}_{CVAE}^{+} = \mathcal{L}_{CVAE} + \lambda_{expr}\mathcal{L}_{expr}
$$

### 5.3 Transformer-Flow Method

Include objective:

$$
\mathcal{L}_{TF} = \mathcal{L}_{token} + \lambda_{flow}\mathcal{L}_{flow} + \lambda_{expr}\mathcal{L}_{expr} + \lambda_{grammar}\mathcal{L}_{grammar}
$$

Flow construction block:

$$
t \sim \mathcal{U}(0,1), \quad x_t = (1-t)\epsilon + t x, \quad v^* = x - \epsilon
$$

### 5.4 Domain Constraints

- Raga grammar: vivadi penalty, vadi/samvadi rewards.
- Taal cyclic encoding: cycle position + beat strength.

### 5.5 Complexity and Scaling

Report:
- Parameter count
- Context length
- Memory strategy (checkpointing, mixed precision, accumulation)

## 6. Experimental Setup

### 6.1 Data

Report:
- Source datasets and curation rules
- Train/val/test split strategy
- Tokenization settings
- Expression feature pipeline

### 6.2 Training Configs

Provide table:

| Field | Hybrid CVAE | Transformer-Flow |
|---|---:|---:|
| Params |  |  |
| Seq length |  |  |
| Batch/micro-batch |  |  |
| LR/scheduler |  |  |
| Epochs |  |  |
| Hardware |  |  |

### 6.3 Baselines

Minimum baselines:
1. Symbolic-only autoregressive model.
2. CVAE without expression branch.
3. Transformer-Flow without grammar term.
4. Transformer-Flow without flow head.

### 6.4 Metrics

- Perplexity / NLL
- Grammar compliance
- Taal coherence
- Expression fidelity
- Expert listening scores

## 7. Results

### 7.1 Main Results Table

| Model | NLL | Grammar Compliance | Rhythm Coherence | Expr Fidelity | Human Score |
|---|---:|---:|---:|---:|---:|
| Hybrid CVAE |  |  |  |  |  |
| Transformer-Flow |  |  |  |  |  |

### 7.2 Ablation Table

| Variant | Grammar term | Flow term | Expr term | Taal position | Score |
|---|---|---|---|---|---:|
| Full | on | on | on | on |  |
| A1 | off | on | on | on |  |
| A2 | on | off | on | on |  |
| A3 | on | on | off | on |  |
| A4 | on | on | on | off |  |

### 7.3 Qualitative Analysis

Include:
- Case studies by raga
- Failure cases
- Rhythm drift and grammar violations
- Expression artifacts

## 8. Discussion

### 8.1 Comparative Insights

- Where Hybrid CVAE is stronger
- Where Transformer-Flow is stronger
- Practical deployment trade-offs

### 8.2 Limitations

- Dataset bias
- Subjectivity of evaluation
- Generalization across underrepresented ragas

### 8.3 Broader Impact and Ethics

- Cultural representation considerations
- Responsible use of generated music
- Attribution and dataset provenance concerns

## 9. Conclusion

One-paragraph summary plus 2-3 concrete future work points.

## 10. Reproducibility Appendix Checklist

1. Seed and deterministic settings.
2. Exact command lines.
3. Config dumps and commit hash.
4. Model selection criteria.
5. Inference decoding parameters.
6. Evaluation script details.

## 11. Figure and Table Placeholders

### Figure F1. Full architecture comparison
Use: docs/research_figures_mermaid_pack.md

### Figure F2. Loss decomposition chart
Use: docs/research_figures_mermaid_pack.md

### Figure F3. Flow interpolation schematic
Use: docs/research_figures_mermaid_pack.md

### Table T1. Dataset and split summary
### Table T2. Hyperparameter summary
### Table T3. Main results
### Table T4. Ablations

## 12. Citation/Bibliography Skeleton

### 12.1 BibTeX Keys Placeholder

- @article{music_transformer_xxxx, ...}
- @inproceedings{cvae_seq_xxxx, ...}
- @article{flow_matching_xxxx, ...}
- @article{domain_constrained_music_xxxx, ...}

### 12.2 Reference Quality Rules

- Prefer peer-reviewed venues.
- Separate background citations from directly compared baselines.
- Clearly distinguish adopted components vs novel integration.

## 13. Claims Language (Safe Scientific Framing)

Recommended wording style:
- "We propose"
- "We observe"
- "Our results suggest"

Avoid absolute wording unless fully proven:
- "state-of-the-art" without broad benchmark evidence
- "first ever" without exhaustive prior-art verification

## 14. Submission Packaging Checklist

1. Anonymous version prepared (if double-blind).
2. Artifact appendix prepared.
3. Figure readability checked at single-column width.
4. Statistical significance reporting finalized.
5. Supplemental audio/demo links prepared if allowed by venue.
