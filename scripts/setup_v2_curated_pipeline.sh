#!/usr/bin/env bash
set -euo pipefail

# One-command setup:
# 1) curate dataset
# 2) pre-tokenize curated MIDI
# 3) run 1-epoch smoke training
#
# Usage:
#   bash scripts/setup_v2_curated_pipeline.sh
#   MIN_TRACKS=3 WORKERS=16 bash scripts/setup_v2_curated_pipeline.sh

MANIFEST="${MANIFEST:-data/midi_v2/manifest_processed.json}"
CURATED_DIR="${CURATED_DIR:-data/midi_v2_curated}"
REPORT_PATH="${REPORT_PATH:-data/midi_v2_curated_report.json}"
CACHE_PT="${CACHE_PT:-data/v2_tokenized_cache_curated.pt}"
BPE_PATH="${BPE_PATH:-models/tokenizer_v2/bpe_tokenizer.json}"
METADATA="${METADATA:-src/sekiro_ai/config/raga_metadata.json}"
CHECKPOINT_DIR="${CHECKPOINT_DIR:-checkpoints_v2_curated_smoke}"

MIN_CONFIDENCE="${MIN_CONFIDENCE:-0.8}"
MIN_TRACKS="${MIN_TRACKS:-5}"
MIN_DURATION_S="${MIN_DURATION_S:-8.0}"
WORKERS="${WORKERS:-16}"

EPOCHS="${EPOCHS:-1}"
BATCH_SIZE="${BATCH_SIZE:-2}"
GRAD_ACCUM="${GRAD_ACCUM:-2}"
MICRO_BATCH_SIZE="${MICRO_BATCH_SIZE:-1}"
SEQ_LENGTH="${SEQ_LENGTH:-256}"
TRAIN_WORKERS="${TRAIN_WORKERS:-4}"

echo "==> Step 1/3: Curating dataset into ${CURATED_DIR}"
python scripts/prepare_curated_v2_dataset.py \
  --manifest "${MANIFEST}" \
  --output_dir "${CURATED_DIR}" \
  --report_path "${REPORT_PATH}" \
  --min_confidence "${MIN_CONFIDENCE}" \
  --min_tracks_per_raga "${MIN_TRACKS}" \
  --min_duration_s "${MIN_DURATION_S}" \
  --link_mode hardlink \
  --overwrite

echo "==> Step 2/3: Pre-tokenizing curated MIDI into ${CACHE_PT}"
python scripts/pre_tokenize_v2.py \
  --midi_dir "${CURATED_DIR}" \
  --bpe_path "${BPE_PATH}" \
  --metadata "${METADATA}" \
  --output "${CACHE_PT}" \
  --workers "${WORKERS}"

echo "==> Step 3/3: Running smoke train (${EPOCHS} epoch)"
python scripts/train_v2_flow.py \
  --midi_dir "${CURATED_DIR}" \
  --metadata "${METADATA}" \
  --bpe_path "${BPE_PATH}" \
  --cache_pt "${CACHE_PT}" \
  --checkpoint_dir "${CHECKPOINT_DIR}" \
  --epochs "${EPOCHS}" \
  --batch_size "${BATCH_SIZE}" \
  --grad_accum "${GRAD_ACCUM}" \
  --micro_batch_size "${MICRO_BATCH_SIZE}" \
  --seq_length "${SEQ_LENGTH}" \
  --workers "${TRAIN_WORKERS}"

echo "Pipeline complete."
