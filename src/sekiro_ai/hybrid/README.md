# Hybrid MIDI-Audio Indian Classical Music Generator

This folder contains the hybrid architecture that combines MIDI structure with audio expression features for generating expressive Indian Classical Music.

## Architecture Overview

```
hybrid/
├── models/
│   ├── audio_features.py     # Extract f0, amplitude, spectral features
│   ├── expression_encoder.py # Encode expression into latent space
│   ├── hybrid_cvae.py        # Modified CVAE with expression conditioning
│   ├── neural_synth.py       # Convert MIDI+expression → audio
│   └── __init__.py
├── data/
│   ├── hybrid_dataset.py     # Dataset with paired MIDI-audio loading
│   └── preprocess.py         # Cache audio features
├── training/
│   ├── train_hybrid.py       # Training script for hybrid model
│   └── losses.py             # Expression + reconstruction losses
├── inference/
│   └── generate.py           # Generate expressive music
├── config/
│   └── hybrid_config.py      # Configuration dataclasses
└── README.md
```

## Key Features

- **Audio Feature Extraction**: f0 (pitch), amplitude, spectral centroid
- **Expression Conditioning**: Model learns to predict expression from MIDI context
- **Neural Synthesizer**: Lightweight audio generation from MIDI + expression
- **Backward Compatible**: Can still use original MIDI-only mode

## Quick Start

```bash
# 1. Preprocess audio features (one-time)
python -m hybrid.data.preprocess --audio-dir ../DData --output-dir ./features

# 2. Train hybrid model
python -m hybrid.training.train_hybrid --epochs 100

# 3. Generate music
python -m hybrid.inference.generate --raga yaman --mood calm
```
