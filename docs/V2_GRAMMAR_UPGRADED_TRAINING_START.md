# V2 Grammar-Upgraded Model: Training Start Guide

This guide starts the upgraded V2 training flow that is already implemented in code:

- expression-preserving cache building,
- cached dataset loading with expression support,
- grammar-weighted training loss,
- grammar-aware decoding with chalan-prefix support.

## 1) Build the curated dataset

```bash
python scripts/prepare_curated_v2_dataset.py \
  --manifest data/midi_v2/manifest_processed.json \
  --output_dir data/midi_v2_curated \
  --report_path data/midi_v2_curated_report.json \
  --min_confidence 0.8 \
  --min_tracks_per_raga 5 \
  --min_duration_s 8.0 \
  --link_mode hardlink \
  --overwrite
```

## 2) Train the BPE tokenizer if it is missing

Skip this step if `models/tokenizer_v2/bpe_tokenizer.json` already exists.

```bash
python scripts/train_bpe_tokenizer.py \
  --midi_dir data/midi_v2_curated \
  --extra_midi_dir data/midi_v2_curated \
  --output_path models/tokenizer_v2/bpe_tokenizer.json \
  --vocab_size 10000 \
  --sample_ratio 0.05 \
  --workers 32
```

## 3) Build the expression-preserving cache

```bash
python scripts/pre_tokenize_v2.py \
  --midi_dir data/midi_v2_curated \
  --bpe_path models/tokenizer_v2/bpe_tokenizer.json \
  --metadata src/sekiro_ai/config/raga_metadata.json \
  --output data/v2_tokenized_cache_curated_expr.pt \
  --workers 32
```

## 4) Run a 10-Epoch Smoke Test

Use this first to measure speed, memory use, and stability before the longer run.

```bash
python scripts/train_v2_flow.py \
  --midi_dir data/midi_v2_curated \
  --metadata src/sekiro_ai/config/raga_metadata.json \
  --bpe_path models/tokenizer_v2/bpe_tokenizer.json \
  --config configs/1.0b_mamba_flow_budget.json \
  --cache_pt data/v2_tokenized_cache_curated_expr.pt \
  --checkpoint_dir checkpoints_v2_1b_grammar_smoke10 \
  --epochs 10 \
  --batch_size 4 \
  --micro_batch_size 1 \
  --grad_accum 12 \
  --lr 1.5e-5 \
  --warmup_epochs 1 \
  --seq_length 1024 \
  --workers 8 \
  --max_train_steps_per_epoch 160 \
  --max_val_steps 30 \
  --max_disk_gb 200 \
  --device cuda
```

If the smoke test is too slow, lower `--max_train_steps_per_epoch` first. If memory is tight, reduce `--micro_batch_size`.

Expected speed:

- Roughly `6-10 minutes per epoch` on this RTX 6000 Pro class machine for this smoke setup.
- The planning target for the longer 13-hour style run is about `7-8 minutes per epoch`.
- If epoch time is much slower than that, reduce `--max_train_steps_per_epoch` before touching other settings.

## 5) Start upgraded training

At the current observed smoke-test pace, expect the 40-epoch main run to land around `10-11.5 hours` total, depending on validation and checkpoint overhead.

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
  --max_disk_gb 200 \
  --device cuda
```

## 6) Resume training after interruption

If you stop the run, close the session, or Lightning interrupts the machine, resume from the latest checkpoint in the same directory. The trainer now refreshes `latest.pt` every 20 optimizer steps, so recovery is no longer limited to epoch boundaries.

```bash
python scripts/train_v2_flow.py \
  --midi_dir data/midi_v2_curated \
  --metadata src/sekiro_ai/config/raga_metadata.json \
  --bpe_path models/tokenizer_v2/bpe_tokenizer.json \
  --config configs/1.0b_mamba_flow_budget.json \
  --cache_pt data/v2_tokenized_cache_curated_expr.pt \
  --checkpoint_dir checkpoints_v2_1b_grammar_v1 \
  --resume checkpoints_v2_1b_grammar_v1/latest.pt \
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
  --max_disk_gb 200 \
  --device cuda
```

Resume rules:

- Use `latest.pt` from the same checkpoint directory.
- Keep `--epochs` set to the final target epoch count, not the remaining count.
- Do not change the checkpoint directory when resuming.
- You may lose up to about 20 optimizer steps of progress if the stop happens between checkpoint writes.

## 7) Generate a grammar-aware sample after training

```bash
python scripts/generate_v2_sample.py \
  --checkpoint checkpoints_v2_1b_grammar_v1/best.pt \
  --bpe_path models/tokenizer_v2/bpe_tokenizer.json \
  --vocab_path checkpoints_v2_1b_grammar_v1/vocabularies.json \
  --metadata src/sekiro_ai/config/raga_metadata.json \
  --mood calm \
  --raga yaman \
  --taal trital \
  --tempo 84 \
  --duration 180 \
  --max_new_tokens 8192 \
  --min_new_tokens 4096 \
  --temperature 1.05 \
  --top_k 80 \
  --top_p 0.92 \
  --repetition_penalty 1.08 \
  --no_repeat_ngram_size 6 \
  --enforce_grammar \
  --use_chalan_prefix \
  --output outputs/v2_long_yaman_3min.mid
```

## 8) Batch Generation for Demo Sets

Use this exact script from the repo root. It creates a batch with presentation-friendly labels (`rage`, `very_fast`, `active`, `sad`, `happy`, `scary`) while mapping each to a mood token that your V2 checkpoint actually supports (`devotional`, `energetic`, `peaceful`, `romantic`, `serious`).

```bash
mkdir -p outputs/v2_demo_batch

CHECKPOINT="checkpoints_v2_1b_grammar_v1/best.pt"
BPE="models/tokenizer_v2/bpe_tokenizer.json"
VOCAB="checkpoints_v2_1b_grammar_v1/vocabularies.json"
META="src/sekiro_ai/config/raga_metadata.json"

while IFS='|' read -r label mood raga tempo duration; do
  out="outputs/v2_demo_batch/${label}_${raga}.mid"
  echo "Generating ${label} -> ${out}"

  python scripts/generate_v2_sample.py \
    --checkpoint "$CHECKPOINT" \
    --bpe_path "$BPE" \
    --vocab_path "$VOCAB" \
    --metadata "$META" \
    --mood "$mood" \
    --raga "$raga" \
    --taal trital \
    --tempo "$tempo" \
    --duration "$duration" \
    --max_new_tokens 8192 \
    --min_new_tokens 4096 \
    --temperature 1.12 \
    --top_k 64 \
    --top_p 0.94 \
    --repetition_penalty 1.04 \
    --no_repeat_ngram_size 4 \
    --enforce_grammar \
    --use_chalan_prefix \
    --output "$out"
done << 'EOF'
rage|serious|darbari_kanada|122|60
very_fast|energetic|sarang|146|50
active|energetic|hamsadhwani|136|55
sad|devotional|bhairavi|76|60
happy|romantic|bihag|114|55
scary|serious|malkauns|82|60
EOF
```

For demo batches, it helps to cover a wider emotional range instead of repeating only one calm setting. The current mood vocabulary is still limited in the model, so for broader labels like rage, very fast, active, sad, happy, and scary, use the closest supported mood targets:

- rage -> intense, serious, energetic
- very fast -> energetic, spirited, dynamic
- active -> energetic, rhythmic, structured
- sad -> sad, melancholic, contemplative
- happy -> joyful, festive, energetic
- scary -> mysterious, serious, profound

Suggested batch preset layout:

```python
combinations = [
    # High-energy / aggressive ideas
    {"raga": "hamsadhvani", "mood": "energetic", "tempo": 140, "duration": 50},
    {"raga": "shanmukhapriya", "mood": "intense", "tempo": 135, "duration": 55},
    {"raga": "sarang", "mood": "joyful", "tempo": 132, "duration": 55},

    # Sad / reflective ideas
    {"raga": "bhairavi", "mood": "sad", "tempo": 72, "duration": 60},
    {"raga": "asavari", "mood": "contemplative", "tempo": 68, "duration": 60},
    {"raga": "darbari", "mood": "serious", "tempo": 65, "duration": 60},

    # Happy / bright ideas
    {"raga": "bhoopali", "mood": "happy", "tempo": 108, "duration": 55},
    {"raga": "bihag", "mood": "joyful", "tempo": 112, "duration": 55},
    {"raga": "yaman", "mood": "peaceful", "tempo": 100, "duration": 60},

    # Scary / mysterious ideas
    {"raga": "malkauns", "mood": "mysterious", "tempo": 78, "duration": 60},
    {"raga": "puriya", "mood": "serious", "tempo": 70, "duration": 60},
    {"raga": "bhairav", "mood": "devotional", "tempo": 80, "duration": 60},
]
```

If you want the batch to feel less slow, raise tempo first and keep durations around 45 to 60 seconds per file. For a presentation, that usually produces a stronger mix of moods than a single long slow render.

## Notes

- Use `data/v2_tokenized_cache_curated_expr.pt`; it preserves expression data.
- Checkpoint saving now prunes older numbered checkpoints before new writes when disk headroom gets tight.
- `latest.pt` is overwritten periodically during training, so the run is recoverable even if you stop it mid-epoch.
- If epoch time is too slow, reduce `--micro_batch_size` first.
- Resume with `latest.pt` if Lightning interrupts the run.