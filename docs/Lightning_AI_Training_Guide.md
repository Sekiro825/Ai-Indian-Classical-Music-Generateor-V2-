# Lightning.ai Hybrid Model Training Guide

This guide covers exactly what to upload and the terminal commands required to run the highly-optimized **1.7 Billion Parameter** Hybrid MIDI-Audio model on a Lightning.ai GPU instance.

---

## 🚀 Step 1: Pre-Upload Cleanup (What to Upload)
To save upload time and Lightning.ai disk space, **ONLY** upload the required folders. 

### ✅ DO Upload These Folders & Files:
1. `hybrid/` — The complete hybrid codebase and preprocessed features.
2. `models/` — Contains tokenizer infrastructure.
3. `config/` — Contains Raga metadata maps.
4. `all_midi/` — The core 684 MIDI training files.
5. `DData/` — The 887 original audio files (needed for extraction).
6. `requirements.txt` — Base dependencies.

### ❌ Do NOT Upload These:
- `backend/` or `frontend/` or `services/` (not used during model training)
- `checkpoints/` (contains massive legacy weights)
- `venv/` (Python environments don't transfer)
- `__pycache__/`

---

## 🛠️ Step 2: Environment Setup
To save costs during setup and preprocessing, start your Lightning.ai instance on the **Default (CPU)** machine (Interruptible, 4 CPUs, 16GB). Once your files are in the workspace and the CPU instance is active, open a terminal and run:

```bash
# Update tools and install dependencies
pip install --upgrade pip
pip install -r requirements.txt

# Depending on the instance, explicitly install librosa & soundfile if missing
pip install librosa soundfile
```

---

## ⚡ Step 3: Audio Preprocessing
Before the model can train, it requires the expression maps (Pitch contours, Timbre) from the raw audio in the `DData` folder. Continue using the **Default (CPU)** machine for this step. Since we've enabled the **piptrack fast-processing** feature, it runs efficiently on CPUs.

Run the preprocessor:

```bash
python -m hybrid.data.preprocess \
    --audio-dir DData \
    --output-dir hybrid/features \
    --midi-dir all_midi \
    --workers 4
```

> **Note:** The script will automatically skip any files that are already cached inside `hybrid/features/`. Once you see `=== Preprocessing Complete ===`, you are ready for training.

---

## 🧠 Step 4: Train the Hybrid Model

**⚠️ CRITICAL: Switch Machine Type**
Before running the training script, switch your Lightning.ai machine from the CPU instance to the **RTX PRO 6000 (Interruptible)** instance.

The architecture is set to ~1.7B parameters. The training script has been heavily updated to cap storage limit blowouts and heavily penalize "robotic" outputs in favor of rich Indian Classical micro-expressions (Meend/Gamaka).

Run the core trainer:

```bash
python -m hybrid.training.train_hybrid \
    --midi-dir all_midi \
    --feature-dir hybrid/features \
    --raga-metadata config/raga_metadata.json \
    --epochs 100 \
    --batch-size 4 \
    --device cuda
```

### What to expect during training:
- **Storage:** Checkpoints will *overwrite* each other into `checkpoint_latest.pt` in the `hybrid/checkpoints/` folder. It will never blow past 15GB total.
- **Speed:** Mixed precision (`use_amp=True`) is enabled natively, accelerating training by up to 2x on modern GPUs (like the RTX PRO 6000).
- **Rhythmic Nuance:** The configuration aggressively forces the model to learn pitch bends and timber variations via bumped expression weights.

---

## 💾 Step 5: Exporting & Inference
When the model registers its best validation loss, the training script will immediately output a `best_model.pt`. As an added bonus, it will automatically compress and export:

```text
hybrid/checkpoints/cpu_best_model_int8.pt
```

This specific file is dynamically quantized down to INT8 format and stripped of 13+ GB of Adam optimizer states. 
- You can right-click this `cpu_best_model_int8.pt` file inside Lightning.ai to **Download** it locally. 
- It will load beautifully on cheap cloud CPU instances or local 16GB laptops without throwing memory limits!
