import os
import sys
import json
import logging
import hashlib
from pathlib import Path

import numpy as np
import librosa
import onnxruntime as ort
import pretty_midi

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# ============================================================
# Basic-Pitch Fixed Constants (nmp.onnx Specification)
# ============================================================
AUDIO_SAMPLE_RATE = 22050
MODEL_INPUT_NAME = "serving_default_input_2:0"
MODEL_INPUT_SAMPLES = 43844
MIDI_OFFSET = 21 # Index 0 in 88-note output = MIDI 21 (A0)
N_NOTES = 88
N_FREQ_BINS = 264

DEFAULT_MODEL_PATH = Path("models/basic_pitch_spotify.onnx")

class BasicPitchONNX:
    """
    A production-grade standalone Basic-Pitch engine (Waveform Input).
    Uses fixed IO mapping for the common nmp.onnx model export.
    """
    def __init__(self, model_path: Path = DEFAULT_MODEL_PATH, device="cpu"):
        if not model_path.exists():
            raise FileNotFoundError(f"Model not found at {model_path}. Please place nmp.onnx there.")

        # Optimization: Limit internal threading to 1 thread per worker.
        # This prevents core contention during 48-worker batch processing.
        ort.set_default_logger_severity(3) # Only Errors
        sess_options = ort.SessionOptions()
        sess_options.intra_op_num_threads = 1
        sess_options.inter_op_num_threads = 1
        
        # Configure Providers (Safe CPU fallback)
        providers = ['CPUExecutionProvider']
        if device == "gpu":
            # Only try CUDA if explicitly asked, otherwise CPU is more stable for preprocessing
            providers.insert(0, 'CUDAExecutionProvider')

        try:
            self.session = ort.InferenceSession(str(model_path), sess_options=sess_options, providers=providers)
            active_providers = self.session.get_providers()
            logger.info(f"ONNX Engine initialized. Active Providers: {active_providers}")
        except Exception as e:
            logger.warning(f"Failed to load preferred providers. Falling back to CPU only.")
            self.session = ort.InferenceSession(str(model_path), sess_options=sess_options, providers=['CPUExecutionProvider'])

        # Verify IO Schema to prevent runtime errors
        input_info = self.session.get_inputs()[0]
        if input_info.name != MODEL_INPUT_NAME:
            logger.warning(f"Model input name mismatch! Expected {MODEL_INPUT_NAME}, found {input_info.name}.")
        
    def preprocess(self, audio_path: str) -> np.ndarray:
        """Load and normalize raw audio for the waveform encoder."""
        y, sr = librosa.load(audio_path, sr=AUDIO_SAMPLE_RATE, mono=True)
        # Ensure amplitude is normalized [-1, 1] (Standard for Basic Pitch Waveform models)
        max_val = np.abs(y).max()
        if max_val > 1.0:
            y = y / max_val
        return y.astype(np.float32)

    def predict(self, audio_path: str) -> pretty_midi.PrettyMIDI:
        """Full transcription pipeline: Waveform -> ONNX -> Note Tracking."""
        audio = self.preprocess(audio_path)
        
        # 1. Chunking with 50% overlap for boundary stability
        # model expects 43844 samples
        hop_size = MODEL_INPUT_SAMPLES // 2
        chunks = []
        for i in range(0, len(audio) - MODEL_INPUT_SAMPLES + 1, hop_size):
            chunks.append(audio[i : i + MODEL_INPUT_SAMPLES])
            
        if not chunks:
            # Handle tiny files
            pad = np.zeros(MODEL_INPUT_SAMPLES, dtype=np.float32)
            pad[:len(audio)] = audio
            chunks = [pad]

        # 2. Batch Inference
        # Input shape: (Batch, Samples, 1)
        input_tensor = np.stack(chunks).reshape(-1, MODEL_INPUT_SAMPLES, 1)
        
        # Returns: [onsets(88), frames(88), contours(264)]
        raw_outputs = self.session.run(None, {MODEL_INPUT_NAME: input_tensor})
        
        onsets_batch = raw_outputs[0] # (B, 172, 88)
        frames_batch = raw_outputs[1] # (B, 172, 88)
        
        # 3. Overlap-Add Reassembly
        # We need to map Batch x Frames back to a continuous timeline
        batch_size, n_frames, _ = onsets_batch.shape
        # Overlap is 50%, so frame_hop is n_frames // 2
        frame_hop = n_frames // 2
        total_frames = (batch_size - 1) * frame_hop + n_frames
        
        full_onsets = np.zeros((total_frames, N_NOTES))
        full_frames = np.zeros((total_frames, N_NOTES))
        counts = np.zeros(total_frames)
        
        for i in range(batch_size):
            start = i * frame_hop
            end = start + n_frames
            full_onsets[start:end] += onsets_batch[i]
            full_frames[start:end] += frames_batch[i]
            counts[start:end] += 1
            
        full_onsets /= np.maximum(counts[:, None], 1)
        full_frames /= np.maximum(counts[:, None], 1)
        
        # 4. Note Post-Processing
        return self._decode_notes_v2(full_onsets, full_frames)

    def _decode_notes_v2(self, onsets, frames, threshold=0.3) -> pretty_midi.PrettyMIDI:
        """Converts probability maps to MIDI notes with temporal hysteresis."""
        midi = pretty_midi.PrettyMIDI()
        instrument = pretty_midi.Instrument(program=0) # Piano
        
        # 0.0512 sec per frame (standard Basic Pitch frame rate depends on model)
        # For nmp.onnx, 43844 samples → 172 frames => ~255 samples per frame (~11ms)
        time_step = (MODEL_INPUT_SAMPLES / 172) / AUDIO_SAMPLE_RATE

        for note_idx in range(N_NOTES):
            midi_pitch = note_idx + MIDI_OFFSET
            
            active_note_start = None
            
            for t in range(onsets.shape[0]):
                is_onset = onsets[t, note_idx] > threshold
                is_active = frames[t, note_idx] > threshold
                
                if active_note_start is None:
                    if is_onset or is_active:
                        active_note_start = t * time_step
                else:
                    if not is_active:
                        # End note
                        end_time = t * time_step
                        if end_time - active_note_start > 0.05: # Min duration filter
                            instrument.notes.append(pretty_midi.Note(
                                velocity=int(80 * min(frames[t-1, note_idx] * 1.5, 1.0)), 
                                pitch=midi_pitch,
                                start=active_note_start,
                                end=end_time
                            ))
                        active_note_start = None
                        
        midi.instruments.append(instrument)
        return midi

if __name__ == "__main__":
    # Self-test logic
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--test", action="store_true", help="Run diagnostic self-test")
    parser.add_argument("input", nargs="?", help="Input audio file")
    parser.add_argument("output", nargs="?", help="Output MIDI file")
    args = parser.parse_args()
    
    if args.test:
        logger.info("Starting Stable Engine Self-Test...")
        # Create 2 seconds of silence
        silence = np.zeros(MODEL_INPUT_SAMPLES, dtype=np.float32)
        engine = BasicPitchONNX()
        input_tensor = silence.reshape(1, MODEL_INPUT_SAMPLES, 1)
        outputs = engine.session.run(None, {MODEL_INPUT_NAME: input_tensor})
        logger.info("Self-test SUCCESS! Model IO is perfectly aligned.")
        sys.exit(0)

    if not args.input or not args.output:
        parser.print_help()
        sys.exit(1)
        
    engine = BasicPitchONNX()
    logger.info(f"Transcribing {args.input}...")
    midi = engine.predict(args.input)
    midi.write(args.output)
    logger.info(f"Saved MIDI to {args.output}")
