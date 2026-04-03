# Hybrid MIDI-Audio Indian Classical Music Generator — Technical Documentation

> **Version:** 0.2.0  
> **Architecture:** Hybrid Conditional VAE with Expression Conditioning  
> **Estimated Parameters:** ~1.7B (CVAE) + ~20M (Neural Synthesizer)

---

## Table of Contents

1. [Project Overview](#1-project-overview)
2. [Architecture](#2-architecture)
3. [Component Reference](#3-component-reference)
   - 3.1 [Config](#31-config)
   - 3.2 [Audio Feature Extraction](#32-audio-feature-extraction)
   - 3.3 [Expression Encoder](#33-expression-encoder)
   - 3.4 [Hybrid CVAE Model](#34-hybrid-cvae-model)
   - 3.5 [Neural Synthesizer](#35-neural-synthesizer)
   - 3.6 [Dataset & Preprocessing](#36-dataset--preprocessing)
   - 3.7 [Loss Functions](#37-loss-functions)
   - 3.8 [Training](#38-training)
   - 3.9 [Inference / Generation](#39-inference--generation)
4. [Data Pipeline](#4-data-pipeline)
5. [MIDI Tokenization](#5-midi-tokenization)
6. [Backend API Integration](#6-backend-api-integration)
7. [Current Progress](#7-current-progress)
8. [Quick Start Commands](#8-quick-start-commands)
9. [Configuration Reference](#9-configuration-reference)

---

## 1. Project Overview

This project generates **expressive Indian Classical Music** by combining two complementary representations:

| Representation | Strength | Source |
|---|---|---|
| **MIDI tokens** | Precise note structure (pitch, duration, velocity) | `all_midi/` — 684 MIDI files |
| **Audio expression** | Continuous dynamics (pitch bends, amplitude, timbre) | `DData/` — 887 audio files (WAV + MP3) |

The **Hybrid model** fuses both into a single **Conditional Variational Autoencoder (CVAE)** that is conditioned on:
- **Raga** (e.g., Yaman, Bhairavi, Malkauns)
- **Mood** (e.g., peaceful, devotional, mysterious)
- **Tempo** (BPM)
- **Duration** (seconds)

The system is backward-compatible and can operate in **MIDI-only mode** (without expression features) for deployments where audio data is unavailable.

---

## 2. Architecture

```
┌─────────────────────────────────────────────────────────────────┐
│                    TEXT PROMPT (user input)                      │
└────────────────┬────────────────────────────────────────────────┘
                 │  LLM (Gemini / OpenRouter)
                 ▼
         ┌───────────────┐
         │ MusicParameters│  (mood, raga, tempo, duration)
         └───────┬───────┘
                 │
    ┌────────────┼────────────────────────────┐
    │            │                            │
    ▼            ▼                            ▼
┌────────┐ ┌──────────┐              ┌─────────────────┐
│ Mood   │ │  Raga    │              │ Expression      │
│ Embed  │ │  Embed   │              │ Features (4D)   │
│ (128)  │ │  (128)   │              │ f0, amp, voiced │
└───┬────┘ └────┬─────┘              │ spectral_cent.  │
    │           │                    └────────┬────────┘
    │    ┌──────┘                             │
    ▼    ▼                                   ▼
┌────────────────────┐          ┌─────────────────────────┐
│  ConditioningModule │          │  GlobalExpressionEncoder │
│  + Tempo/Duration   │          │  (attention pooling)     │
│  Embeddings         │          └────────────┬────────────┘
└─────────┬──────────┘                       │
          │                                  │
          └────────────┬─────────────────────┘
                       ▼  (fused conditioning)
              ┌────────────────┐
              │   ENCODER      │  20 × PreNorm Transformer
              │   (RoPE attn)  │  with gradient checkpointing
              └───────┬────────┘
                      │
                ┌─────┴─────┐
                │  μ, log σ² │  → Latent z (512-dim)
                └─────┬─────┘
                      │ reparameterize
                      ▼
              ┌────────────────┐
              │   DECODER      │  20 × PreNorm Transformer
              │   (causal +    │  + cross-attention to z
              │    RoPE attn)  │
              └──────┬─────────┘
                     │
          ┌──────────┼──────────┐
          ▼                     ▼
  ┌───────────────┐    ┌────────────────┐
  │  Token Logits │    │ ExpressionHead │
  │  (vocab_size) │    │  (4-dim pred)  │
  └───────────────┘    └────────────────┘
          │                     │
          ▼                     ▼
     MIDI tokens        Expression features
          │                     │
          └──────────┬──────────┘
                     ▼
          ┌─────────────────────┐
          │  NeuralSynthesizer  │  (~20M params)
          │  dilated convs +    │
          │  upsampling (512×)  │
          └─────────┬───────────┘
                    ▼
               Audio waveform
```

---

## 3. Component Reference

### 3.1 Config

**File:** `hybrid/config/hybrid_config.py`

Five dataclass-based configurations govern the entire system:

| Config | Purpose | Key Defaults |
|---|---|---|
| `AudioFeatureConfig` | Audio extraction params | sr=22050, hop=512, fmin=50Hz, fmax=2000Hz |
| `ExpressionEncoderConfig` | Expression encoder arch | hidden=256, embed=128, 2 layers, 4 heads |
| `HybridCVAEConfig` | Main model architecture | embed=1280, 20 enc/dec layers, latent=512 |
| `TrainingConfig` | Training hyperparameters | batch=4, grad_accum=8, lr=1e-4, 100 epochs |
| `InferenceConfig` | Generation settings | temp=1.0, top_k=50, top_p=0.9 |

All configs support JSON serialization via `.save()` / `.load()` methods.

---

### 3.2 Audio Feature Extraction

**File:** `hybrid/models/audio_features.py`  
**Class:** `AudioFeatureExtractor`

Extracts 4 frame-level features from raw audio using **librosa**:

| Feature | Method | Range | Description |
|---|---|---|---|
| `f0` | YIN algorithm | [-1, 1] (cents/1200) | Fundamental frequency contour |
| `amplitude` | RMS energy | [0, 1] | Volume envelope |
| `voiced` | f0 range check | {0, 1} | Whether frame is voiced |
| `spectral_centroid` | librosa | [0, 1] (normalized) | Brightness / timbre indicator |

**Key methods:**
- `extract_all_features(path)` → `Dict[str, ndarray]` of shape `(T,)` each
- `chunk_features(features)` → splits into 8-second chunks with 2-second minimum
- `features_to_tensor(features)` → stacked `(T, 4)` numpy array
- `compute_dataset_statistics(feature_dir)` → mean/std per feature for z-normalization

---

### 3.3 Expression Encoder

**File:** `hybrid/models/expression_encoder.py`

Three modules handle expression at different granularities:

#### ExpressionEncoder
- **Input:** `(batch, seq_len, 4)` audio features
- **Output:** `(batch, seq_len, 128)` per-frame embeddings
- Uses a 2-layer Transformer encoder (or bidirectional LSTM fallback)
- Includes sinusoidal positional encoding

#### ExpressionHead
- **Input:** `(batch, seq_len, 1280)` decoder hidden states
- **Output:** `(batch, seq_len, 4)` predicted expression
- Per-feature activation functions: `tanh(f0)`, `sigmoid(amp)`, `sigmoid(voiced)`, `sigmoid(centroid)`
- Used during **inference** to predict expression from generated MIDI

#### GlobalExpressionEncoder
- **Input:** `(batch, seq_len, 4)` audio features
- **Output:** `(batch, 128)` single global vector
- Uses **attention pooling** over frames
- Conditions the latent space during encoding

---

### 3.4 Hybrid CVAE Model

**File:** `hybrid/models/hybrid_cvae.py`  
**Class:** `HybridCVAE` (~1.7B parameters)

Core architecture built on top of a Transformer encoder-decoder with:

- **Rotary Positional Embeddings (RoPE)** for better length generalization
- **Pre-LayerNorm** Transformer blocks (more stable training)
- **Gradient checkpointing** for memory efficiency on large models
- **Cross-attention** from decoder to latent variable `z`

#### Forward Pass (Training)
```
tokens + conditioning + expression
    → Encoder (20 layers) → mean pool → μ, logσ²
    → Reparameterize → z (512-dim)
    → Decoder (20 layers, causal, cross-attn to z)
    → Token logits + Predicted expression
```

#### Generation (Inference)
```
conditioning → Sample z from N(0,I)
    → Autoregressive decoding with top-k/top-p sampling
    → Returns tokens + expression predictions per step
    → Stops at EOS token or max_length
```

**Key classes:**
- `ConditioningModule`: fuses mood, raga, tempo, duration embeddings → `(batch, embed_dim)`
- `PreNormTransformerEncoderLayer` / `PreNormTransformerDecoderLayer`: with RoPE attention
- `RoPEMultiHeadAttention`: uses PyTorch `scaled_dot_product_attention` with rotary embeddings

---

### 3.5 Neural Synthesizer

**File:** `hybrid/models/neural_synth.py`

Two synthesis options for converting MIDI + expression → audio:

#### NeuralSynthesizer (~20M params)
- Direct waveform generation
- **Architecture:** MIDI embedding + expression conditioning → fusion → dilated residual blocks → upsampling (8×8×4×2 = 512× hop) → output convolution → Tanh [-1, 1]
- Output: `(batch, seq_len × 512)` audio samples at 22050 Hz

#### SpectrogramSynthesizer (lighter)
- Generates mel spectrograms instead (128 mels)
- Designed to pair with a vocoder (HiFi-GAN, WaveGlow)
- Uses Transformer encoder with MIDI/expression fusion

---

### 3.6 Dataset & Preprocessing

#### Preprocessing

**File:** `hybrid/data/preprocess.py`  
**Class:** `AudioPreprocessor`

- Scans `DData/` for `.wav` and `.mp3` files (filters macOS `._` files)
- Uses **multiprocessing** (`ProcessPoolExecutor`) with per-file timeout
- Extracts features → saves as `{stem}_features.npy`
- Builds `audio_midi_pairs.json` mapping audio stems to MIDI filenames
- Computes `feature_stats.json` (mean/std per feature for normalization)

**Current state:** 35 feature files cached in `hybrid/features/`

#### Dataset

**File:** `hybrid/data/hybrid_dataset.py`  
**Class:** `HybridRagaDataset`

- Loads paired MIDI tokens + cached `.npy` audio features
- Extracts raga from MIDI filenames using `extract_raga_from_filename()`
- **Sequence alignment:** interpolates features to match token length via `np.interp`
- Returns per-sample: `tokens`, `expression`, `raga`, `mood`, `tempo`, `duration`
- Custom `hybrid_collate_fn` pads batches to max length
- `create_hybrid_dataloaders()` creates 90/10 train/val split

---

### 3.7 Loss Functions

**File:** `hybrid/training/losses.py`

#### HybridLoss (main model)
```
Total = recon_weight × CE(logits, tokens)
      + kl_weight × kl_multiplier × KL(q(z|x) ‖ p(z))
      + expr_weight × [MSE(f0,amp,centroid) + 0.5·BCE(voiced)]
```

- Cross-entropy with label smoothing (0.1) and padding ignore (idx=0)
- KL annealing via `kl_weight_multiplier` (0→1 over warmup)
- Expression loss: MSE for continuous features, BCE for binary voiced flag

#### SynthesizerLoss (neural synth)
- **Multi-resolution STFT loss** at FFT sizes 512, 1024, 2048
- Combines spectral convergence + log-magnitude L1 + waveform L1

---

### 3.8 Training

**File:** `hybrid/training/train_hybrid.py`  
**Class:** `HybridTrainer`

| Feature | Detail |
|---|---|
| Optimizer | AdamW (lr=1e-4, weight_decay=0.01) |
| Scheduler | Cosine with linear warmup |
| Mixed Precision | AMP with GradScaler (enabled by default) |
| Gradient Accumulation | 8 steps (effective batch = 4×8 = 32) |
| KL Annealing | Linear over 50 epochs |
| Gradient Clipping | max_norm = 1.0 |
| Checkpointing | Overwrites `checkpoint_latest.pt` daily + keeps `best_model.pt` + `cpu_best_model_int8.pt` |

**CLI usage:**
```bash
python -m hybrid.training.train_hybrid \
    --midi-dir all_midi \
    --feature-dir hybrid/features \
    --raga-metadata config/raga_metadata.json \
    --epochs 100 \
    --batch-size 4 \
    --lr 1e-4 \
    --device cuda
```

---

### 3.9 Inference / Generation

**File:** `hybrid/inference/generate.py`  
**Class:** `HybridGenerator`

High-level API for music generation:

```python
generator = HybridGenerator.from_checkpoint("hybrid/checkpoints/best_model.pt")
result = generator.generate(
    raga="yaman",
    mood="peaceful",
    tempo=90,
    duration=60,
    temperature=1.0,
    top_k=50,
    top_p=0.9
)
# result contains: tokens, expression, audio (if synth available)
```

**Methods:**
- `generate()` → dict with tokens, expression, optional audio
- `generate_midi_file(output_path, ...)` → saves `.mid` file
- `generate_audio_file(output_path, ...)` → saves `.wav` file (requires synthesizer)

**CLI:**
```bash
python -m hybrid.inference.generate \
    --checkpoint hybrid/checkpoints/best_model.pt \
    --raga yaman --mood peaceful \
    --tempo 90 --duration 60 \
    --output generated.mid
```

---

## 4. Data Pipeline

```
DData/                         all_midi/
(887 audio files)              (684 MIDI files)
       │                              │
       ▼                              │
  preprocess.py                       │
  (AudioFeatureExtractor)             │
       │                              │
       ▼                              │
  hybrid/features/                    │
  (35 .npy files)                     │
  + audio_midi_pairs.json             │
  + feature_stats.json                │
       │                              │
       └──────────┬───────────────────┘
                  ▼
          HybridRagaDataset
          (pairs MIDI tokens + expression features)
                  │
                  ▼
          DataLoader (train: 90%, val: 10%)
```

### Dataset Composition

| Source | Type | Count | Description |
|---|---|---|---|
| `DData/` WAV files | Raga recordings | ~120 | Named by raga (yaman01-05, bhairavi01-05, etc.) |
| `DData/` WAV files | Taal patterns | ~700+ | Named by taal (trital, ektal, jhaptal, etc.) |
| `DData/` MP3 files | Hindustani ragas | ~80 | Full performances (Raag Yaman, Bhairavi, etc.) |
| `DData/` WAV files | Carnatic compositions | ~120 | Numbered catalog items |
| `all_midi/` | MIDI transcriptions | 684 | Corresponding MIDI files |

---

## 5. MIDI Tokenization

**File:** `models/tokenizer.py`  
**Class:** `MIDITokenizer`

Vocabulary structure:
- Token 0: `PAD`
- Token 1: `BOS` (beginning of sequence)
- Token 2: `EOS` (end of sequence)
- Tokens 3+: `NOTE_ON`, `NOTE_OFF`, `TIME_SHIFT`, `VELOCITY` events

**Key params:**
- 128 pitches (0-127), 32 velocity bins, 64 duration bins
- Max sequence length: 512 (configurable to 1024)
- Default vocabulary size: **491 tokens**

Helper function `extract_raga_from_filename()` maps filenames like `yaman01.mid` → `"yaman"`.

---

## 6. Backend API Integration

**File:** `backend/main.py` (FastAPI)

The backend loads either the **standard CVAE** or the **Hybrid CVAE** at startup. The API remains identical regardless of which model is active:

| Endpoint | Method | Description |
|---|---|---|
| `/generate` | POST | Submit text prompt → returns job_id |
| `/status/{job_id}` | GET | Poll job progress (0.0 → 1.0) |
| `/download/midi/{job_id}` | GET | Download generated MIDI |
| `/download/audio/{job_id}` | GET | Download generated audio (if available) |
| `/instruments` | GET | List available instruments |
| `/ragas` | GET | List available ragas with metadata |
| `/health` | GET | Health check (model loaded, parser, synth) |

The backend supports:
- **LLM text parsing** via Gemini or OpenRouter API
- **Legacy model loading** fallback for older checkpoints
- **Background task processing** with progress tracking
- **FluidSynth audio** conversion (MIDI → WAV via soundfonts)

---

## 7. Current Progress

### ✅ Completed
- [x] Full hybrid architecture implemented (config, models, data, training, inference)
- [x] Audio feature extraction pipeline with multiprocessing
- [x] 35 audio files preprocessed and cached as `.npy` features
- [x] MIDI tokenizer with 491-token vocabulary
- [x] Paired MIDI-audio dataset with alignment and collation
- [x] Training script with AMP, gradient accumulation, KL annealing
- [x] Inference pipeline with autoregressive generation
- [x] Neural synthesizer (~20M params) for direct waveform generation
- [x] Spectrogram synthesizer alternative for vocoder integration
- [x] Loss functions: HybridLoss + SynthesizerLoss
- [x] FastAPI backend with legacy model fallback
- [x] `__init__.py` module exports for clean imports

### 🔲 Remaining
- [ ] **Preprocess remaining audio files** — only 35/887 audio files have cached features
- [ ] **Train the hybrid model** — no hybrid-specific checkpoint exists yet
- [ ] **Train the neural synthesizer** — separate training required
- [ ] **Integrate hybrid model into backend** — backend currently loads the standard CVAE only
- [ ] **Data augmentation** — pitch-shift stub in `_augment_tokens` is a no-op
- [ ] **Audio-MIDI pair mapping** — needs validation (currently uses filename-based heuristic matching)
- [ ] **End-to-end integration testing** — generate → synthesize → serve via API

---

## 8. Quick Start Commands

```bash
# 1. Preprocess audio features (one-time)
python -m hybrid.data.preprocess \
    --audio-dir DData \
    --output-dir hybrid/features \
    --midi-dir all_midi \
    --workers 4

# 2. Train the hybrid model
python -m hybrid.training.train_hybrid \
    --midi-dir all_midi \
    --feature-dir hybrid/features \
    --raga-metadata config/raga_metadata.json \
    --epochs 100 \
    --batch-size 4

# 3. Generate music
python -m hybrid.inference.generate \
    --checkpoint hybrid/checkpoints/best_model.pt \
    --raga yaman \
    --mood peaceful \
    --tempo 90 \
    --duration 60 \
    --output outputs/generated.mid

# 4. Run the API server
cd backend && uvicorn main:app --host 0.0.0.0 --port 8000
```

---

## 9. Configuration Reference

### HybridCVAEConfig (complete)

| Parameter | Type | Default | Description |
|---|---|---|---|
| `vocab_size` | int | 491 | Token vocabulary size |
| `max_seq_length` | int | 1024 | Maximum sequence length |
| `embed_dim` | int | 2048 | Main embedding dimension |
| `num_heads` | int | 32 | Attention heads |
| `num_encoder_layers` | int | 20 | Encoder transformer layers |
| `num_decoder_layers` | int | 20 | Decoder transformer layers |
| `ff_dim` | int | 8192 | Feed-forward dimension (4×embed) |
| `latent_dim` | int | 1024 | VAE latent dimension |
| `dropout` | float | 0.1 | Dropout rate |
| `num_moods` | int | 36 | Number of mood categories |
| `num_ragas` | int | 19 | Number of raga categories |
| `mood_embed_dim` | int | 128 | Mood embedding size |
| `raga_embed_dim` | int | 128 | Raga embedding size |
| `tempo_embed_dim` | int | 64 | Tempo embedding size |
| `duration_embed_dim` | int | 64 | Duration embedding size |
| `kl_weight` | float | 0.005 | KL divergence weight |
| `use_expression` | bool | True | Enable expression conditioning |
| `expression_dim` | int | 128 | Expression embedding size |
| `expression_loss_weight` | float | 0.1 | Expression loss weight |
| `use_gradient_checkpointing` | bool | True | Memory-efficient training |

### TrainingConfig (complete)

| Parameter | Type | Default | Description |
|---|---|---|---|
| `midi_dir` | str | `"all_midi"` | MIDI files directory |
| `audio_dir` | str | `"DData"` | Audio files directory |
| `feature_cache_dir` | str | `"hybrid/features"` | Cached features dir |
| `batch_size` | int | 4 | Per-GPU batch size |
| `gradient_accumulation_steps` | int | 8 | Effective batch = 32 |
| `learning_rate` | float | 1e-4 | Peak learning rate |
| `weight_decay` | float | 0.01 | AdamW weight decay |
| `warmup_steps` | int | 1000 | LR warmup steps |
| `num_epochs` | int | 100 | Total training epochs |
| `reconstruction_weight` | float | 1.0 | CE loss weight |
| `kl_weight` | float | 0.005 | KL loss weight |
| `expression_weight` | float | 0.1 | Expression loss weight |
| `kl_annealing_epochs` | int | 50 | KL annealing duration |
| `use_amp` | bool | True | Mixed precision |
