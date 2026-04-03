# Hybrid (Older Model) Retrain Runbook on Curated V2 Data

This runbook is for retraining the older HybridCVAE model on:

- MIDI data: `data/midi_v2_curated`
- Metadata: `src/sekiro_ai/config/raga_metadata.json`

It is written for your current Lightning setup and prioritizes stability + reproducibility.

## 0) Quick Answer: `features.npy` vs `expr.npy`

- For the current legacy script `scripts/train_hybrid.py`, neither is used yet (it trains token-only + CVAE losses).
- For expression-aware hybrid training, `features.npy` (dict format with `f0`, `amplitude`, `voiced`, `spectral_centroid`) is the native format expected by `src/sekiro_ai/hybrid/data/hybrid_dataset.py`.
- `expr.npy` is still useful and can be converted/mapped, but out-of-the-box the grammar-aware hybrid pipeline expects `*_features.npy`.

## 1) Finish/Stop Current V2 Run Cleanly

After your current 2-epoch long-context calibration completes, stop V2 jobs before starting hybrid retrain.

Optional check:

```bash
nvidia-smi
```

## 2) Free Disk Before Hybrid Training

Your logs show disk pressure (`228.3GB / 220GB`). Free space first.

Inspect disk usage:

```bash
du -sh checkpoints_* | sort -h
du -sh outputs data | sort -h
```

Keep only the checkpoints you still need, then remove stale ones.
Example cleanup (edit paths before running):

```bash
rm -rf checkpoints_v2_1b_quick11 checkpoints_v2_1b_quick12
```

Re-check:

```bash
df -h .
du -sh checkpoints_* | sort -h
```

## 3) Environment Setup

```bash
cd /teamspace/studios/this_studio
pip install --upgrade pip
pip install -r requirements.txt
```

## 4) Data Sanity Check

```bash
find data/midi_v2_curated -name '*.mid' | wc -l
cat data/midi_v2_curated_report.json | head -n 80
```

## 5) Smoke Test (10 Epochs)

Run this first to validate memory + throughput over multiple passes (time-capped):

```bash
python scripts/train_hybrid.py \
  --midi_dir data/midi_v2_curated \
  --metadata src/sekiro_ai/config/raga_metadata.json \
  --checkpoint_dir checkpoints_hybrid_curated_smoke \
  --epochs 10 \
  --batch_size 1 \
  --grad_accum 16 \
  --lr 3e-5 \
  --seq_length 512 \
  --workers 8 \
  --max_train_batches 600 \
  --max_val_batches 80 \
  --device cuda
```

Runtime note:

- This caps each smoke epoch to ~600 train + ~80 val batches.
- To run full-epoch smoke (slower), use `--max_train_batches 0 --max_val_batches 0`.

If OOM:

- reduce `--seq_length` to `384`
- reduce `--workers` to `4`

## 6) Main Training (Older Hybrid on New Curated Data)

Train in 2 phases:

- Phase A: time-capped to reach epoch 80.
- Phase B: uncapped for the final 20 epochs (to epoch 100).

Phase A (capped, target epoch 80):

```bash
python scripts/train_hybrid.py \
  --midi_dir data/midi_v2_curated \
  --metadata src/sekiro_ai/config/raga_metadata.json \
  --checkpoint_dir checkpoints_hybrid_curated_1024 \
  --epochs 80 \
  --batch_size 1 \
  --grad_accum 16 \
  --lr 2e-5 \
  --seq_length 512 \
  --workers 8 \
  --max_train_batches 1500 \
  --max_val_batches 200 \
  --device cuda
```

Phase B (uncapped, continue from epoch 80 to epoch 100):

```bash
python scripts/train_hybrid.py \
  --midi_dir data/midi_v2_curated \
  --metadata src/sekiro_ai/config/raga_metadata.json \
  --checkpoint_dir checkpoints_hybrid_curated_1024 \
  --resume checkpoints_hybrid_curated_1024/latest_checkpoint.pt \
  --epochs 100 \
  --batch_size 1 \
  --grad_accum 16 \
  --lr 2e-5 \
  --seq_length 512 \
  --workers 8 \
  --max_train_batches 0 \
  --max_val_batches 0 \
  --device cuda
```

Notes:

- `batch_size=1 + grad_accum` is intentional for stability with this large model.
- Phase A caps runtime per epoch.
- Phase B removes caps for full-data finishing passes.

## 6.5) Quick Test After Epoch 80 (Generate Output)

Once Phase A hits epoch 80, you can do a fast sanity check by generating a MIDI output.
If you haven't yet built the conditioning vocab, run step 8 first, then:

```bash
PYTHONPATH=src python src/sekiro_ai/hybrid/inference/generate.py \
  --checkpoint checkpoints_hybrid_curated_1024/best_model.pt \
  --vocab-path checkpoints_hybrid_curated_1024/conditioning_vocabs.json \
  --raga yaman \
  --mood peaceful \
  --taal trital \
  --tempo 84 \
  --duration 180 \
  --output outputs/hybrid_curated_yaman.mid \
  --device cuda
```

Note: generation now scales time-shift tokens so the saved MIDI duration matches the requested `--duration` (e.g., 180s ≈ 3 minutes).

## 7) Resume After Interruption

Resume commands must keep the same target `--epochs` for the phase you are in:

- If interrupted during Phase A, resume with `--epochs 80` and keep caps.
- If interrupted during Phase B, resume with `--epochs 100` and keep uncapped (`0/0`).

Resume during Phase A:

```bash
python scripts/train_hybrid.py \
  --midi_dir data/midi_v2_curated \
  --metadata src/sekiro_ai/config/raga_metadata.json \
  --checkpoint_dir checkpoints_hybrid_curated_1024 \
  --resume checkpoints_hybrid_curated_1024/latest_checkpoint.pt \
  --epochs 80 \
  --batch_size 1 \
  --grad_accum 16 \
  --lr 2e-5 \
  --seq_length 512 \
  --workers 8 \
  --max_train_batches 1500 \
  --max_val_batches 200 \
  --device cuda
```

Resume during Phase B:

```bash
python scripts/train_hybrid.py \
  --midi_dir data/midi_v2_curated \
  --metadata src/sekiro_ai/config/raga_metadata.json \
  --checkpoint_dir checkpoints_hybrid_curated_1024 \
  --resume checkpoints_hybrid_curated_1024/latest_checkpoint.pt \
  --epochs 100 \
  --batch_size 1 \
  --grad_accum 16 \
  --lr 2e-5 \
  --seq_length 512 \
  --workers 8 \
  --max_train_batches 0 \
  --max_val_batches 0 \
  --device cuda
```

`--epochs` is total target (not additional).  
Example: if interruption happens at epoch 67 in Phase A, resume with `--epochs 80`; if at epoch 92 in Phase B, resume with `--epochs 100`.

## 8) Build Conditioning Vocab (With Grammar Rules)

`scripts/train_hybrid.py` writes `vocabularies.json`, but inference grammar masking expects `raga_rules_by_idx`.
Create `conditioning_vocabs.json` once:

```bash
PYTHONPATH=src python -c "import json; from sekiro_ai.hybrid.musicology import get_raga_grammar; v='checkpoints_hybrid_curated_1024/vocabularies.json'; m='src/sekiro_ai/config/raga_metadata.json'; o='checkpoints_hybrid_curated_1024/conditioning_vocabs.json'; voc=json.load(open(v)); meta=json.load(open(m)); rules={}; r2i=voc.get('raga_to_idx',{}); [rules.setdefault(int(idx), {'vivadi_pitch_classes':sorted(get_raga_grammar(r, meta.get(r, {})).vivadi_pitch_classes), 'vadi_pitch_classes':sorted(get_raga_grammar(r, meta.get(r, {})).vadi_pitch_classes), 'samvadi_pitch_classes':sorted(get_raga_grammar(r, meta.get(r, {})).samvadi_pitch_classes), 'chalan_degrees':get_raga_grammar(r, meta.get(r, {})).chalan_degrees}) for r, idx in r2i.items()]; out={'raga_to_idx':voc.get('raga_to_idx',{}),'mood_to_idx':voc.get('mood_to_idx',{}),'taal_to_idx':voc.get('taal_to_idx',{}),'raga_rules_by_idx':rules}; json.dump(out, open(o,'w'), indent=2); print('saved', o)"
```

## 9) Generate Validation Samples (Grammar Mask Enabled)

Use the hybrid inference module with `conditioning_vocabs.json`:

```bash
PYTHONPATH=src python src/sekiro_ai/hybrid/inference/generate.py \
  --checkpoint checkpoints_hybrid_curated_1024/best_model.pt \
  --vocab-path checkpoints_hybrid_curated_1024/conditioning_vocabs.json \
  --raga yaman \
  --mood peaceful \
  --taal trital \
  --tempo 84 \
  --duration 180 \
  --output outputs/hybrid_curated_yaman.mid \
  --device cuda
```

Generate 3-5 variants by changing:

- `--tempo` (72, 84, 96)
- `--mood` (`peaceful`, `devotional`, `introspective`)

## 9) Quality Gate (Before Returning to V2)

Use these checks before you call hybrid retrain successful:

- Raga grammar: low vivadi usage for target raga.
- Phrase quality: recognizable pakad/chalan-like movement appears.
- Variation: alaap-like and taan-like contrast across samples.
- Rhythmic shape: taal feel remains coherent over longer durations.

If quality is still bland after 20+ epochs, do not tune V2 yet. First extend hybrid training to 80-100 epochs and compare again.

## 11) Next Phase (After Hybrid Baseline)

Once hybrid baseline is in place, we improve V2 with:

- expression-aware cache (no zeroed expression),
- grammar-aware losses or decoding constraints,
- stronger long-context curriculum.
