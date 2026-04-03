# V2 Transformer-Flow Model Documentation Suite

This folder contains research-paper-grade technical documentation for the V2 Transformer-Flow model implemented in this repository.

## Document Map

1. 01_architecture_and_pipeline.md
- End-to-end architecture
- Data and conditioning pipeline
- Training and inference flow diagrams

2. 02_objective_functions_and_formulas.md
- Full mathematical objective
- Flow-matching equations
- Grammar-aware objective decomposition
- Optimization details

3. 03_novelty_uniqueness_and_patentability.md
- Novelty analysis
- Prior-art comparison framing
- Patentability checklist and claim strategy

4. 04_research_paper_ready_sections.md
- Ready-to-adapt paper text blocks
- Suggested figures, tables, and ablations
- Reproducibility and threat-to-validity notes

## Model Snapshot (Code-Aligned)

- Backbone: Decoder-only Transformer with RoPE, RMSNorm, SwiGLU-style feed-forward
- Heads: Token prediction head + expression prediction head + flow-matching velocity head
- Conditioning: Mood, raga, taal, tempo, duration
- Rhythmic inductive bias: Taal cycle position encoder
- Objective: Weighted sum of token CE, flow MSE, expression MSE, grammar loss

## Source Alignment

The documentation here is aligned to:
- scripts/train_v2_flow.py
- src/sekiro_ai/v2/transformer_flow_model.py
- src/sekiro_ai/v2/config.py

Use this suite as the main source for method and appendix sections in your paper.
