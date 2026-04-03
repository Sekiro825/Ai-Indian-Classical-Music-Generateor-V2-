# Lightning AI Training Guide - 400M Raga CVAE Model

## Quick Start

This guide will walk you through training your 400M parameter Raga CVAE model on Lightning.ai.

---

## Step 1: Create Lightning AI Account & Workspace

1. Go to [lightning.ai](https://lightning.ai) and sign up
2. Create a new **Studio**
3. Select GPU instance:
   - **Recommended**: A100 (40GB) - fastest, ~8-12 hours
   - **Alternative**: L4 (24GB) - may need smaller batch size
   - **Budget**: T4 (16GB) - requires batch_size=2, grad_accum=32

---

## Step 2: Upload These Files

Upload the following files/folders to your Lightning AI workspace:

```
📁 Your Lightning AI Workspace
├── 📁 all_midi/               # All your MIDI training files
│   ├── yaman_track1.mid
│   ├── bhairavi_track2.mid
│   └── ... (all 684+ MIDI files)
├── 📁 config/
│   └── raga_metadata.json     # Raga metadata
├── 📁 models/
│   ├── __init__.py
│   ├── transformer_cvae.py    # 400M model architecture
│   ├── dataset.py             # Dataset with augmentation
│   └── tokenizer.py           # MIDI tokenizer
├── train.py                   # Training script
├── export_model.py            # Model export script
└── requirements.txt           # Dependencies
```

### Quick Upload Method
1. Zip your local `ALLDATA` folder
2. Upload to Lightning AI
3. Unzip: `unzip ALLDATA.zip`

---

## Step 3: Install Dependencies

Open a terminal in Lightning AI and run:

```bash
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118
pip install mido tqdm numpy
```

---

## Step 4: Verify Setup

Test that the model builds correctly:

```bash
cd ~/your-workspace
python models/transformer_cvae.py
```

Expected output:
```
Model parameters: ~400,000,000 (400.0M)
Estimated size: 1.52 GB (float32)
Testing forward pass...
Output shape: torch.Size([2, 128, 491])
Testing generation...
Generated shape: torch.Size([2, 64])
```

---

## Step 5: Start Training

### A100 (40GB) - Recommended
```bash
python train.py \
    --midi_dir all_midi \
    --config_dir config \
    --checkpoint_dir checkpoints \
    --epochs 300 \
    --batch_size 4 \
    --grad_accum 16 \
    --lr 1e-4 \
    --warmup_epochs 30 \
    --use_amp \
    --device cuda
```

### L4 (24GB)
```bash
python train.py \
    --midi_dir all_midi \
    --config_dir config \
    --checkpoint_dir checkpoints \
    --epochs 300 \
    --batch_size 2 \
    --grad_accum 32 \
    --lr 1e-4 \
    --warmup_epochs 30 \
    --use_amp \
    --device cuda
```

### T4 (16GB) - Budget Option
```bash
python train.py \
    --midi_dir all_midi \
    --config_dir config \
    --checkpoint_dir checkpoints \
    --epochs 300 \
    --batch_size 1 \
    --grad_accum 64 \
    --lr 5e-5 \
    --warmup_epochs 50 \
    --use_amp \
    --seq_length 512 \
    --device cuda
```

---

## Step 6: Monitor Training

Training will show progress like:

```
🎵 Starting training for 300 epochs
   Device: cuda
   GPU: NVIDIA A100-SXM4-40GB
   Mixed Precision: True
   Effective Batch Size: 64

Epoch 1/300: 100%|██████████| 42/42 [02:15<00:00]
   loss: 5.2341  recon: 5.2100  kl: 0.0241  lr: 2.50e-05

✅ Epoch 1/300 | Time: 135.2s
   Train Loss: 5.2341 (recon: 5.2100, kl: 0.0241)
   Val Loss: 4.8932 (recon: 4.8700, kl: 0.0232)
💾 Saved best model with val_loss: 4.8932
```

### What to Watch For:
- **Reconstruction loss** should decrease steadily
- **KL loss** should stay between 5-50 (not collapse to 0)
- If KL → 0, the model is experiencing mode collapse

---

## Step 7: Resume Training (if interrupted)

```bash
python train.py \
    --resume checkpoints/latest_checkpoint.pt \
    --midi_dir all_midi \
    --config_dir config \
    --checkpoint_dir checkpoints \
    --epochs 300 \
    --batch_size 4 \
    --grad_accum 16 \
    --use_amp \
    --device cuda
```

---

## Step 8: Export Trained Model

After training completes:

```bash
python export_model.py \
    --checkpoint checkpoints/best_model.pt \
    --output exported_model
```

---

## Step 9: Download to Your PC

Download these files from Lightning AI:

1. `exported_model/` folder (contains model + tokenizer)
2. `checkpoints/best_model.pt` (full checkpoint for fine-tuning)
3. `checkpoints/training_history.json` (loss curves)

---

## Troubleshooting

### Out of Memory (OOM)
- Reduce `--batch_size` (try 2 or 1)
- Increase `--grad_accum` proportionally
- Reduce `--seq_length` to 512

### Training Too Slow
- Make sure `--use_amp` is enabled
- Check GPU is being used: `nvidia-smi`

### KL Loss Collapses to 0
- This means mode collapse - stop training
- Increase KL weight in config (change 0.005 to 0.01)
- Restart training with new checkpoint

---

## Estimated Training Times

| GPU | Batch Size | Time per Epoch | Total (300 epochs) |
|-----|------------|----------------|-------------------|
| A100 40GB | 4 | ~2-3 min | 10-15 hours |
| L4 24GB | 2 | ~4-5 min | 20-25 hours |
| T4 16GB | 1 | ~8-10 min | 40-50 hours |

---

## Files You'll Get After Training

```
📁 checkpoints/
├── best_model.pt           # Best model (use this!)
├── latest_checkpoint.pt    # Latest checkpoint
├── vocabularies.json       # Mood/Raga mappings
├── tokenizer.json          # Tokenizer config
└── training_history.json   # Loss history

📁 exported_model/
├── model.pt                # Exported model for inference
├── config.json             # Model config
└── tokenizer.json          # Tokenizer for inference
```
