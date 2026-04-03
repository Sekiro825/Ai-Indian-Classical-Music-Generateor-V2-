# V2 Grammar Upgrade Plan (Post-Hybrid Baseline)

This plan is for upgrading V2 so it can generate Indian classical music with:

- raga grammar correctness,
- phrase-level variation,
- expressive nuance,
- robust long-form continuity.

Use this only after we establish a strong hybrid baseline on curated data.

## Goals

- Reduce bland/generic outputs.
- Enforce raga-safe decoding (avoid vivadi drift).
- Preserve expressive variation (meend/gamak-like contours via expression channel).
- Improve long-context form development (>=180s coherent structure).

## Current Gaps (Why V2 Sounds Bland Now)

- Cached training path zeros expression and marks `has_expression=False`, so flow/expression losses are effectively off.
- Cache currently forces weak metadata defaults (`taal=unknown`, fixed tempo/duration patterns).
- Generation uses plain top-k sampling and no grammar-aware token masking.
- Checkpoint context and training duration are still in calibration territory.

## Phase 1: Data/Cache Integrity (Must Do First)

### 1.1 Patch cache builder to preserve expressive supervision

Target:

- `scripts/pre_tokenize_v2.py`

Required changes:

- load companion `.expr.npy` for each MIDI when available,
- store `expression` and `has_expression`,
- preserve or infer `taal`, `tempo`, `duration` instead of hardcoded defaults,
- write these fields into cache entries.

### 1.2 Patch cached dataset loader

Target:

- `scripts/train_v2_flow.py` (`V2CachedDataset`)

Required changes:

- read cached `expression` tensor if present,
- set `has_expression=True` when expression exists,
- avoid zeroing expression by default,
- use cached `taal`, `tempo`, `duration` when available.

### 1.3 Rebuild cache

```bash
python scripts/pre_tokenize_v2.py \
  --midi_dir data/midi_v2_curated \
  --bpe_path models/tokenizer_v2/bpe_tokenizer.json \
  --metadata src/sekiro_ai/config/raga_metadata.json \
  --output data/v2_tokenized_cache_curated_expr.pt \
  --workers 16
```

## Phase 2: Grammar-Aware Decoding

### 2.1 Add raga token mask at inference

Target:

- `scripts/generate_v2_sample.py`

Required changes:

- load grammar from metadata (`allowed/vivadi/vadi/samvadi`),
- block vivadi `NOTE_ON` tokens during sampling,
- add optional chalan prefix prompt tokenization,
- expose CLI flags:
  - `--enforce_grammar`
  - `--use_chalan_prefix`
  - `--top_p`
  - `--repetition_penalty`
  - `--no_repeat_ngram_size`

### 2.2 Recommended decoding defaults (music-first)

- `temperature=1.05`
- `top_k=80`
- `top_p=0.92`
- repetition penalty `1.08`
- no-repeat ngram size `6`

### 2.3 Long-Form Generation Defaults (3 Minutes)

Target a 3-minute output and block early EOS:

```bash
python scripts/generate_v2_sample.py \
  --checkpoint checkpoints_v2_1b_13h_longctx/best.pt \
  --bpe_path models/tokenizer_v2/bpe_tokenizer.json \
  --vocab_path checkpoints_v2_1b_13h_longctx/vocabularies.json \
  --mood calm \
  --raga yaman \
  --taal trital \
  --tempo 84 \
  --duration 180 \
  --max_new_tokens 8192 \
  --min_new_tokens 4096 \
  --temperature 1.05 \
  --top_k 80 \
  --output outputs/v2_long_yaman_3min.mid
```

## Phase 3: Grammar-Weighted Training Objective

### 3.1 Add auxiliary grammar loss

Target:

- `scripts/train_v2_flow.py`

Required behavior:

- penalize vivadi pitch probability mass,
- mildly reward vadi/samvadi support,
- keep weight modest to avoid rigid/robotic output.

Start with:

- `grammar_loss_weight = 0.15`

## Phase 4: Long-Context Curriculum

Train in 3 stages:

1. `seq_length=512` stabilization (token + expression + grammar all active)
2. `seq_length=1024` consolidation
3. optional `seq_length=1536` polish if memory allows

Do not move stages unless listening quality improves, not just val loss.

## Phase 5: Evaluation Gates (Pass/Fail)

For each target raga (e.g., yaman, bhairavi, darbari):

- Grammar gate:
  - low vivadi rate,
  - strong vadi/samvadi presence.
- Variation gate:
  - multiple distinct phrase families across 5 samples.
- Nuance gate:
  - dynamic contour variation, non-flat velocity/timing behavior.
- Long-form gate:
  - coherent development for >=90s without early collapse.

If a gate fails, do not proceed to larger training scale.

## Phase 6: Suggested Training Command (After Patches)

```bash
python scripts/train_v2_flow.py \
  --midi_dir data/midi_v2_curated \
  --metadata src/sekiro_ai/config/raga_metadata.json \
  --bpe_path models/tokenizer_v2/bpe_tokenizer.json \
  --config configs/1.0b_mamba_flow_budget.json \
  --cache_pt data/v2_tokenized_cache_curated_expr.pt \
  --checkpoint_dir checkpoints_v2_1b_grammar_v1 \
  --epochs 40 \
  --batch_size 4 \
  --micro_batch_size 1 \
  --grad_accum 12 \
  --lr 1.5e-5 \
  --warmup_epochs 2 \
  --seq_length 1024 \
  --workers 8 \
  --max_train_steps_per_epoch 160 \
  --max_val_steps 30 \
  --max_disk_gb 220 \
  --device cuda
```

## Final Notes

- Optimize for musical listening outcomes, not only token loss.
- Keep hybrid as the quality reference while V2 is being upgraded.
- Lock every improvement behind A/B comparisons using identical prompts.
