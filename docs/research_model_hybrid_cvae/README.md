# Hybrid CVAE Model Documentation Suite

This folder contains research-paper-grade technical documentation for the Hybrid CVAE model in this repository.

## Document Map

1. 01_architecture_and_pipeline.md
- Full model architecture
- Conditioning and latent pathway
- Training and generation diagrams

2. 02_objective_functions_and_formulas.md
- CVAE objective and KL formulation
- Free-bits and cyclical KL annealing equations
- Expression branch formulation

3. 03_novelty_uniqueness_and_patentability.md
- Novelty framing
- Hybrid structure explanation
- Prior art and patentability strategy

4. 04_research_paper_ready_sections.md
- Method text for your paper
- Suggested ablations and evaluation template
- Reproducibility and limitations structure

## Model Snapshot (Code-Aligned)

- Core: Conditional Variational Autoencoder with Transformer encoder-decoder
- Positional strategy: Rotary positional embeddings in attention
- Conditioning: mood, raga, taal, tempo, duration
- Latent process: encoder to (mu, logvar), reparameterization, decoder cross-attention to latent
- Loss: reconstruction CE + KL divergence with free-bits and KL scheduling

## Source Alignment

Aligned to:
- src/sekiro_ai/hybrid/models/hybrid_cvae.py
- src/sekiro_ai/hybrid/config/hybrid_config.py
- scripts/train_hybrid.py
