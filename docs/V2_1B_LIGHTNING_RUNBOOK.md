# V2 1B Transformer-Flow Lightning.ai Runbook (RTX PRO 6000)

This runbook is for `1x RTX PRO 6000` on Lightning.ai.
Flow: environment setup -> curated data -> cache -> smoke -> 10-epoch calibration -> full run.

## 0) Machine

Use:

- `Quantity: 1`
- `GPU: RTX PRO 6000`
- `Interruptible`: optional (cheaper, but can stop unexpectedly)

If interruptible is enabled, always resume from `latest.pt`.

## 1) Upload Checklist

Upload:

- `src/`
- `scripts/`
- `configs/`
- `docs/`
- `requirements.txt`
- `data/midi_v2/`
- `models/tokenizer_v2/bpe_tokenizer.json` (if already trained)

Do not upload:

- `venv/`
- `checkpoints*/`
- temporary archives

## 2) Environment Setup

```bash
cd /teamspace/studios/this_studio
pip install --upgrade pip
pip install -r requirements.txt
```

Quick GPU check:

```bash
nvidia-smi
```

## 3) Build Curated Dataset (Recommended)

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

Quick validation:

```bash
cat data/midi_v2_curated_report.json | head -n 80
find data/midi_v2_curated -name '*.mid' | wc -l
find data/midi_v2_curated -name '*.expr.npy' | wc -l
```

## 4) Train BPE Tokenizer (Skip If Present)

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

Notes:

- Start with `--workers 32`.
- If RAM pressure is high, reduce to `--workers 24` or `--workers 16`.

## 5) Pre-tokenize Cache

```bash
python scripts/pre_tokenize_v2.py \
  --midi_dir data/midi_v2_curated \
  --bpe_path models/tokenizer_v2/bpe_tokenizer.json \
  --metadata src/sekiro_ai/config/raga_metadata.json \
  --output data/v2_tokenized_cache_curated_expr.pt \
  --workers 32
```

If CPU contention is high, retry with `--workers 24` or `--workers 16`.

Notes:

- This cache now preserves `expression`, `has_expression`, `tempo`, and `duration`.
- If you update `*.expr.npy` files, rebuild this cache.

## 6) Exact Commands To Run Now (Smoke Already Completed)

You already finished smoke (`✅ Epoch 1/1`), so start from here.

1. Quick check before main training:

```bash
ls -lh checkpoints_v2_1b_smoke/latest.pt
nvidia-smi
```

2. Run the required 10-epoch calibration (do not skip):

```bash
python scripts/train_v2_flow.py \
  --midi_dir data/midi_v2_curated \
  --metadata src/sekiro_ai/config/raga_metadata.json \
  --bpe_path models/tokenizer_v2/bpe_tokenizer.json \
  --config configs/1.0b_mamba_flow_budget.json \
  --cache_pt data/v2_tokenized_cache_curated_expr.pt \
  --checkpoint_dir checkpoints_v2_1b_calib10_fast \
  --epochs 10 \
  --batch_size 8 \
  --micro_batch_size 8 \
  --grad_accum 1 \
  --lr 2e-5 \
  --warmup_epochs 1 \
  --seq_length 256 \
  --workers 12 \
  --max_disk_gb 200 \
  --device cuda
```

3. If interrupted, resume 10-epoch calibration:

```bash
python scripts/train_v2_flow.py \
  --midi_dir data/midi_v2_curated \
  --metadata src/sekiro_ai/config/raga_metadata.json \
  --bpe_path models/tokenizer_v2/bpe_tokenizer.json \
  --config configs/1.0b_mamba_flow_budget.json \
  --cache_pt data/v2_tokenized_cache_curated_expr.pt \
  --checkpoint_dir checkpoints_v2_1b_calib10_fast \
  --resume checkpoints_v2_1b_calib10_fast/latest.pt \
  --epochs 10 \
  --batch_size 8 \
  --micro_batch_size 8 \
  --grad_accum 1 \
  --lr 2e-5 \
  --warmup_epochs 1 \
  --seq_length 256 \
  --workers 12 \
  --max_disk_gb 200 \
  --device cuda
```

4. Calibration pass rule:

- If epoch time is `<= 12 minutes`, proceed to Section 7.
- If slower, retry with `--micro_batch_size 4`.
- If still slow/OOM, use `--batch_size 6 --grad_accum 2`.

5. Optional speed calibration at longer context (2 epochs):

```bash
python scripts/train_v2_flow.py \
  --midi_dir data/midi_v2_curated \
  --metadata src/sekiro_ai/config/raga_metadata.json \
  --bpe_path models/tokenizer_v2/bpe_tokenizer.json \
  --config configs/1.0b_mamba_flow_budget.json \
  --cache_pt data/v2_tokenized_cache_curated_expr.pt \
  --checkpoint_dir checkpoints_v2_1b_13h_longctx \
  --epochs 2 \
  --batch_size 4 \
  --micro_batch_size 1 \
  --grad_accum 12 \
  --lr 1.5e-5 \
  --warmup_epochs 1 \
  --seq_length 1024 \
  --workers 8 \
  --max_train_steps_per_epoch 120 \
  --max_val_steps 20 \
  --max_disk_gb 200 \
  --device cuda
```

## 6.5) Grammar-Aware V2 (Recommended New Default)

Once cache with expression is built, start the grammar-aware run:

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

Generate with grammar-aware decoding:

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

## 7) 13-Hour Plan (80-100 Epochs With Longer Context)

This is mini-epoch mode. It is the only realistic path to 80-100 epochs in 13 hours on one RTX PRO 6000.

Main command:

```bash
python scripts/train_v2_flow.py \
  --midi_dir data/midi_v2_curated \
  --metadata src/sekiro_ai/config/raga_metadata.json \
  --bpe_path models/tokenizer_v2/bpe_tokenizer.json \
  --config configs/1.0b_mamba_flow_budget.json \
  --cache_pt data/v2_tokenized_cache_curated_expr.pt \
  --checkpoint_dir checkpoints_v2_1b_13h_longctx \
  --epochs 100 \
  --batch_size 4 \
  --micro_batch_size 1 \
  --grad_accum 12 \
  --lr 1.5e-5 \
  --warmup_epochs 2 \
  --seq_length 1024 \
  --workers 8 \
  --max_train_steps_per_epoch 120 \
  --max_val_steps 20 \
  --max_disk_gb 200 \
  --device cuda
```

Resume after interruption:

```bash
python scripts/train_v2_flow.py \
  --midi_dir data/midi_v2_curated \
  --metadata src/sekiro_ai/config/raga_metadata.json \
  --bpe_path models/tokenizer_v2/bpe_tokenizer.json \
  --config configs/1.0b_mamba_flow_budget.json \
  --cache_pt data/v2_tokenized_cache_curated.pt \
  --checkpoint_dir checkpoints_v2_1b_13h_longctx \
  --resume checkpoints_v2_1b_13h_longctx/latest.pt \
  --epochs 100 \
  --batch_size 4 \
  --micro_batch_size 1 \
  --grad_accum 12 \
  --lr 1.5e-5 \
  --warmup_epochs 2 \
  --seq_length 1024 \
  --workers 8 \
  --max_train_steps_per_epoch 120 \
  --max_val_steps 20 \
  --max_disk_gb 220 \
  --device cuda
```

Important:

- `--epochs` is a total target, not "extra epochs".
- Example: if you stop at epoch 37 and resume with `--epochs 100`, training continues to epoch 100.

## 8) How To Hit 13 Hours Reliably

Target epoch time for `100 epochs in 13h` is about `468s` per epoch.

After first 2-3 epochs:

- If epoch time is `> 520s`, reduce `--max_train_steps_per_epoch` to `100`.
- If epoch time is `420-520s`, keep `120`.
- If epoch time is `< 420s`, increase `--max_train_steps_per_epoch` to `140`.

## 9) Disk and Health Checks

```bash
du -sh checkpoints_v2_1b_13h_longctx
du -sh .
nvidia-smi
```

Notes:

- Trainer line `Disk used: ...` is partition-wide usage, not just this project folder.
- Lightning UI "size" can be much smaller because it tracks project/workspace usage differently.

## 10) One-Command Pipeline (Alternative)

```bash
WORKERS=32 TRAIN_WORKERS=12 BATCH_SIZE=8 MICRO_BATCH_SIZE=8 GRAD_ACCUM=1 EPOCHS=1 \
bash scripts/setup_v2_curated_pipeline.sh
```

Then use Section 7 for the 13-hour long-context run.

## 11) Generate Outputs (Long-Form, 3 Minutes)

After you have a checkpoint + vocab, generate a long-form sample:

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

Notes:

- `--min_new_tokens` prevents early EOS so you get long-form structure.
- The MIDI writer scales time-shifts so output length matches `--duration`.
