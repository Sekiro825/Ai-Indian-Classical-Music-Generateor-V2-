"""
MIDI Tokenizer for Indian Classical Music
Converts MIDI files to token sequences for model training
"""

import os
import json
import numpy as np
from pathlib import Path
from typing import List, Dict, Tuple, Optional
from dataclasses import dataclass
import mido


@dataclass
class TokenizerConfig:
    """Configuration for MIDI tokenizer"""
    max_pitch: int = 127
    min_pitch: int = 0
    num_velocity_bins: int = 32
    num_duration_bins: int = 64
    max_sequence_length: int = 512
    time_step: float = 0.01  # 10ms resolution


class MIDITokenizer:
    """
    Tokenizes MIDI files into sequences for Transformer training.
    Uses a vocabulary of: NOTE_ON, NOTE_OFF, TIME_SHIFT, VELOCITY
    """
    
    # Special tokens
    PAD_TOKEN = 0
    BOS_TOKEN = 1
    EOS_TOKEN = 2
    
    def __init__(self, config: TokenizerConfig = None):
        self.config = config or TokenizerConfig()
        self._build_vocabulary()
        
    def _build_vocabulary(self):
        """Build the token vocabulary"""
        self.vocab = {}
        idx = 3  # Start after special tokens
        
        # Note-on tokens (pitch 0-127)
        self.note_on_offset = idx
        for pitch in range(128):
            self.vocab[f"NOTE_ON_{pitch}"] = idx
            idx += 1
        
        # Note-off tokens (pitch 0-127)
        self.note_off_offset = idx
        for pitch in range(128):
            self.vocab[f"NOTE_OFF_{pitch}"] = idx
            idx += 1
        
        # Time shift tokens (in 10ms increments, up to 2 seconds)
        self.time_shift_offset = idx
        for t in range(200):  # 0-2 seconds in 10ms steps
            self.vocab[f"TIME_SHIFT_{t}"] = idx
            idx += 1
        
        # Velocity tokens (binned)
        self.velocity_offset = idx
        for v in range(self.config.num_velocity_bins):
            self.vocab[f"VELOCITY_{v}"] = idx
            idx += 1
        
        self.vocab_size = idx
        self.idx_to_token = {v: k for k, v in self.vocab.items()}
        
        # Add special tokens to reverse mapping
        self.idx_to_token[self.PAD_TOKEN] = "PAD"
        self.idx_to_token[self.BOS_TOKEN] = "BOS"
        self.idx_to_token[self.EOS_TOKEN] = "EOS"
        
    def tokenize_midi(self, midi_path: str) -> List[int]:
        """
        Convert a MIDI file to a sequence of tokens
        """
        try:
            midi = mido.MidiFile(midi_path)
        except Exception as e:
            print(f"Error loading MIDI {midi_path}: {e}")
            return []
        
        tokens = [self.BOS_TOKEN]
        events = []
        
        # Extract all note events with absolute timing
        for track in midi.tracks:
            abs_time = 0
            for msg in track:
                abs_time += msg.time
                if msg.type == 'note_on':
                    events.append({
                        'time': abs_time,
                        'type': 'note_on' if msg.velocity > 0 else 'note_off',
                        'pitch': msg.note,
                        'velocity': msg.velocity
                    })
                elif msg.type == 'note_off':
                    events.append({
                        'time': abs_time,
                        'type': 'note_off',
                        'pitch': msg.note,
                        'velocity': 0
                    })
        
        # Sort events by time
        events.sort(key=lambda x: x['time'])
        
        # Convert to tokens
        prev_time = 0
        for event in events:
            # Time shift
            time_delta = event['time'] - prev_time
            if time_delta > 0:
                # Convert to 10ms steps
                ticks_per_beat = midi.ticks_per_beat
                # Approximate conversion (assuming 120 BPM if not specified)
                time_seconds = time_delta / ticks_per_beat * 0.5
                time_steps = min(int(time_seconds / self.config.time_step), 199)
                if time_steps > 0:
                    tokens.append(self.time_shift_offset + time_steps)
            
            # Velocity (only for note_on)
            if event['type'] == 'note_on' and event['velocity'] > 0:
                velocity_bin = min(
                    event['velocity'] * self.config.num_velocity_bins // 128,
                    self.config.num_velocity_bins - 1
                )
                tokens.append(self.velocity_offset + velocity_bin)
            
            # Note event
            if event['type'] == 'note_on' and event['velocity'] > 0:
                tokens.append(self.note_on_offset + event['pitch'])
            else:
                tokens.append(self.note_off_offset + event['pitch'])
            
            prev_time = event['time']
            
            # Limit sequence length
            if len(tokens) >= self.config.max_sequence_length - 1:
                break
        
        tokens.append(self.EOS_TOKEN)
        return tokens
    
    def detokenize(
        self,
        tokens: List[int],
        output_path: str,
        ticks_per_beat: int = 480,
        time_scale: float = 1.0
    ):
        """
        Convert tokens back to a MIDI file
        """
        midi = mido.MidiFile(ticks_per_beat=ticks_per_beat)
        track = mido.MidiTrack()
        midi.tracks.append(track)
        
        current_velocity = 64
        accumulated_time = 0
        
        for token in tokens:
            if token in [self.PAD_TOKEN, self.BOS_TOKEN, self.EOS_TOKEN]:
                continue
            
            token_str = self.idx_to_token.get(token, "")
            
            if token_str.startswith("TIME_SHIFT_"):
                time_steps = int(token_str.split("_")[-1])
                # Convert time steps to ticks - SCALED UP for longer duration
                time_seconds = time_steps * self.config.time_step * time_scale
                # Multiplier increased from 2 to 4 for longer playback
                accumulated_time += int(time_seconds * 4 * ticks_per_beat)  # 60 BPM feel
                
            elif token_str.startswith("VELOCITY_"):
                velocity_bin = int(token_str.split("_")[-1])
                current_velocity = (velocity_bin * 128) // self.config.num_velocity_bins
                
            elif token_str.startswith("NOTE_ON_"):
                pitch = int(token_str.split("_")[-1])
                track.append(mido.Message(
                    'note_on',
                    note=pitch,
                    velocity=current_velocity,
                    time=accumulated_time
                ))
                accumulated_time = 0
                
            elif token_str.startswith("NOTE_OFF_"):
                pitch = int(token_str.split("_")[-1])
                track.append(mido.Message(
                    'note_off',
                    note=pitch,
                    velocity=0,
                    time=accumulated_time
                ))
                accumulated_time = 0
        
        midi.save(output_path)
        return output_path

    def estimate_duration_seconds(self, tokens: List[int]) -> float:
        """
        Estimate total duration in seconds from TIME_SHIFT tokens.
        """
        total_seconds = 0.0
        for token in tokens:
            token_str = self.idx_to_token.get(token, "")
            if token_str.startswith("TIME_SHIFT_"):
                time_steps = int(token_str.split("_")[-1])
                total_seconds += time_steps * self.config.time_step
        return total_seconds
    
    def pad_sequence(self, tokens: List[int], max_length: int = None) -> np.ndarray:
        """Pad or truncate sequence to fixed length"""
        max_length = max_length or self.config.max_sequence_length
        
        if len(tokens) >= max_length:
            return np.array(tokens[:max_length])
        
        padded = tokens + [self.PAD_TOKEN] * (max_length - len(tokens))
        return np.array(padded)
    
    def save(self, path: str):
        """Save tokenizer configuration"""
        config_dict = {
            'vocab': self.vocab,
            'vocab_size': self.vocab_size,
            'config': {
                'max_pitch': self.config.max_pitch,
                'min_pitch': self.config.min_pitch,
                'num_velocity_bins': self.config.num_velocity_bins,
                'num_duration_bins': self.config.num_duration_bins,
                'max_sequence_length': self.config.max_sequence_length,
                'time_step': self.config.time_step
            }
        }
        with open(path, 'w') as f:
            json.dump(config_dict, f, indent=2)
    
    @classmethod
    def load(cls, path: str) -> 'MIDITokenizer':
        """Load tokenizer from file"""
        with open(path, 'r') as f:
            config_dict = json.load(f)
        
        config = TokenizerConfig(**config_dict['config'])
        tokenizer = cls(config)
        tokenizer.vocab = config_dict['vocab']
        tokenizer.vocab_size = config_dict['vocab_size']
        tokenizer.idx_to_token = {int(v): k for k, v in tokenizer.vocab.items()}
        tokenizer.idx_to_token[cls.PAD_TOKEN] = "PAD"
        tokenizer.idx_to_token[cls.BOS_TOKEN] = "BOS"
        tokenizer.idx_to_token[cls.EOS_TOKEN] = "EOS"
        return tokenizer


def extract_raga_from_filename(filename: str) -> str:
    """Extract raga name from MIDI filename"""
    name = Path(filename).stem.lower()
    
    # Remove common suffixes
    name = name.replace('_basic_pitch', '')
    name = name.replace('.mp3', '')
    
    # Extract raga name
    raga_patterns = [
        'yaman', 'bhairavi', 'malkauns', 'bageshree', 'bhoopali', 'bhoop',
        'asavari', 'sarang', 'darbari', 'dkanada', 'addhatrital', 'trital',
        'dadra', 'deepchandi', 'ektal', 'jhaptal', 'rupak', 'bhajani'
    ]
    
    for pattern in raga_patterns:
        if pattern in name:
            return pattern
    
    # Try to extract from "Raag X" pattern
    if 'raag' in name or 'raga' in name:
        parts = name.replace('raag', ' ').replace('raga', ' ').split()
        for part in parts:
            clean_part = ''.join(c for c in part if c.isalpha())
            if len(clean_part) > 2:
                return clean_part
    
    return 'unknown'


if __name__ == "__main__":
    # Test tokenizer
    tokenizer = MIDITokenizer()
    print(f"Vocabulary size: {tokenizer.vocab_size}")
    
    # Test on a sample file
    midi_dir = Path("d:/MUSIC_MP/EXTRACTED/ALLDATA/all_midi")
    sample_files = list(midi_dir.glob("*.mid"))[:3]
    
    for midi_file in sample_files:
        tokens = tokenizer.tokenize_midi(str(midi_file))
        raga = extract_raga_from_filename(midi_file.name)
        print(f"{midi_file.name}: {len(tokens)} tokens, Raga: {raga}")
