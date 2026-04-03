"""
Music Generation Service
Handles token generation and MIDI conversion with robust fallback for model issues
"""

import os
import random
from pathlib import Path
from typing import List, Optional
import torch

# Import from models
import sys
sys.path.insert(0, str(Path(__file__).parent.parent / "models"))

from tokenizer import MIDITokenizer


class MusicGenerator:
    """
    Wrapper for music generation with robust fallbacks
    """
    
    def __init__(self, model, tokenizer: MIDITokenizer, vocabularies: dict):
        self.model = model
        self.tokenizer = tokenizer
        self.vocabularies = vocabularies
        
        # Pre-compute some useful token ranges
        self.note_on_start = tokenizer.note_on_offset
        self.note_off_start = tokenizer.note_off_offset
        self.time_shift_start = tokenizer.time_shift_offset
        self.velocity_start = tokenizer.velocity_offset
    
    def _is_valid_generation(self, tokens: List[int]) -> bool:
        """
        Check if generated tokens represent valid music.
        Detects mode collapse and other generation failures.
        """
        if len(tokens) < 10:
            return False
        
        # Check for repeated special tokens (mode collapse symptom)
        special_tokens = [0, 1, 2]  # PAD, BOS, EOS
        special_count = sum(1 for t in tokens if t in special_tokens)
        if special_count > len(tokens) * 0.5:  # More than 50% special tokens
            return False
        
        # Check for excessive repetition of same token
        for i in range(len(tokens) - 5):
            if len(set(tokens[i:i+6])) <= 2:  # 6 consecutive tokens with only 2 unique values
                return False
        
        # Must have at least some note tokens
        note_on_count = sum(1 for t in tokens if self.note_on_start <= t < self.note_off_start)
        if note_on_count < 5:
            return False
        
        return True
    
    def generate(
        self,
        mood: str,
        raga: str, 
        tempo: int,
        duration: int,
        temperature: float = 1.5,  # Increased for more variety
        top_k: int = 80
    ) -> List[int]:
        """
        Generate music tokens with fallback to rule-based generation
        """
        # Get conditioning indices
        mood_idx = self.vocabularies['mood_to_idx'].get(mood, 0)
        raga_idx = self.vocabularies['raga_to_idx'].get(raga, 0)
        tempo_bin = min(int((tempo - 40) * 32 / 160), 31)
        duration_bin = min(int(duration * 16 / 300), 15)
        
        # Estimate max_length from duration (more tokens for longer output)
        max_length = int(duration * 15)  # Increased from 10
        max_length = min(max(100, max_length), 512)
        
        # Try model generation with higher temperature
        try:
            with torch.no_grad():
                mood_tensor = torch.tensor([mood_idx])
                raga_tensor = torch.tensor([raga_idx])
                tempo_tensor = torch.tensor([tempo_bin])
                duration_tensor = torch.tensor([duration_bin])
                
                generated = self.model.generate(
                    mood_tensor, raga_tensor, tempo_tensor, duration_tensor,
                    max_length=max_length,
                    temperature=temperature,
                    top_k=top_k
                )
            
            tokens = generated[0].tolist()
            
            # Enhanced validation
            if self._is_valid_generation(tokens):
                print(f"Model generation successful: {len(tokens)} tokens")
                return tokens
            else:
                print("Model generation invalid (mode collapse detected), using fallback...")
        except Exception as e:
            print(f"Model generation error: {e}, using fallback...")
        
        # Always use fallback if model fails
        return self._generate_fallback(raga, tempo, duration, mood)
    
    def _generate_fallback(self, raga: str, tempo: int, duration: int, mood: str = "calm") -> List[int]:
        """
        Enhanced rule-based fallback generation using raga scales.
        Creates authentic-sounding raga patterns with proper musical structure.
        """
        # Extended raga scales with characteristic phrases
        RAGA_SCALES = {
            'yaman': [0, 2, 4, 6, 7, 9, 11],  # Kalyan thaat - evening raga
            'bhairavi': [0, 1, 3, 5, 7, 8, 10],  # Bhairavi thaat - morning
            'malkauns': [0, 3, 5, 8, 10],  # Pentatonic - late night
            'bageshree': [0, 2, 3, 5, 7, 9, 10],  # Night raga
            'bhoopali': [0, 2, 4, 7, 9],  # Pentatonic - evening
            'bhoop': [0, 2, 4, 7, 9],
            'darbari': [0, 2, 3, 5, 7, 8, 9],  # Serious, night
            'asavari': [0, 2, 3, 5, 7, 8, 10],  # Morning
            'sarang': [0, 2, 4, 5, 7, 9, 11],  # Afternoon
            'dkanada': [0, 2, 3, 5, 7, 8, 10],  # Night
            'trital': [0, 2, 4, 5, 7, 9, 11],  # Default major
            'ektal': [0, 2, 4, 5, 7, 9, 11],
            'jhaptal': [0, 2, 4, 5, 7, 9, 11],
            'rupak': [0, 2, 4, 5, 7, 9, 11],
            'dadra': [0, 2, 4, 5, 7, 9, 11],
            'deepchandi': [0, 2, 4, 5, 7, 9, 11],
            'addhatrital': [0, 2, 4, 5, 7, 9, 11],
            'bhajani': [0, 2, 4, 5, 7, 9, 11],
        }
        
        # Mood affects dynamics and tempo feel
        MOOD_DYNAMICS = {
            'calm': (12, 20, 0.6),      # velocity_min, velocity_max, note_density
            'energetic': (20, 28, 1.2),
            'sad': (10, 18, 0.4),
            'happy': (18, 26, 1.0),
            'meditative': (8, 16, 0.3),
            'devotional': (14, 22, 0.5),
            'romantic': (12, 20, 0.5),
            'peaceful': (10, 18, 0.4),
        }
        
        # Get scale or use major scale
        scale = RAGA_SCALES.get(raga.lower(), [0, 2, 4, 5, 7, 9, 11])
        
        # Get mood dynamics
        vel_min, vel_max, density = MOOD_DYNAMICS.get(mood.lower(), (14, 22, 0.7))
        
        # Base pitch (around middle octave)
        base_pitch = 60
        
        tokens = [self.tokenizer.BOS_TOKEN]
        
        # Calculate note count based on duration and tempo
        notes_per_second = tempo / 60 * density
        num_notes = int(duration * notes_per_second)
        num_notes = max(30, min(num_notes, 300))  # More notes for longer output
        
        # Base time between notes (in 10ms units)
        base_time = max(8, int(60 / tempo * 100))
        
        # Generate with musical patterns
        prev_pitch = base_pitch
        phrase_length = random.randint(4, 8)
        phrase_counter = 0
        ascending = random.choice([True, False])
        
        for i in range(num_notes):
            # Add time shift with variation
            time_var = random.randint(-3, 5)
            time_shift = max(3, min(base_time + time_var, 150))
            tokens.append(self.time_shift_start + time_shift)
            
            # Add velocity (with dynamics)
            velocity_bin = random.randint(vel_min, vel_max)
            # Add crescendo/decrescendo within phrases
            phrase_pos = phrase_counter / phrase_length
            if ascending:
                velocity_bin = int(vel_min + (vel_max - vel_min) * phrase_pos)
            tokens.append(self.velocity_start + velocity_bin)
            
            # Choose note with melodic movement
            if random.random() < 0.7:  # 70% stepwise motion
                scale_idx = scale.index(min(scale, key=lambda x: abs((prev_pitch - base_pitch) % 12 - x)))
                if ascending:
                    scale_idx = min(scale_idx + random.randint(0, 2), len(scale) - 1)
                else:
                    scale_idx = max(scale_idx - random.randint(0, 2), 0)
                scale_degree = scale[scale_idx]
            else:  # 30% skip or same note
                scale_degree = random.choice(scale)
            
            # Determine octave (mostly stay in middle, occasional jumps)
            octave_offset = 0
            if random.random() < 0.1:
                octave_offset = random.choice([-12, 12])
            
            pitch = base_pitch + scale_degree + octave_offset
            pitch = max(48, min(pitch, 84))  # Keep in reasonable range
            prev_pitch = pitch
            
            # Note on
            tokens.append(self.note_on_start + pitch)
            
            # Note duration (longer notes for slow tempos)
            note_duration = random.randint(5, max(10, base_time // 2))
            tokens.append(self.time_shift_start + note_duration)
            
            # Note off
            tokens.append(self.note_off_start + pitch)
            
            # Phrase management
            phrase_counter += 1
            if phrase_counter >= phrase_length:
                phrase_counter = 0
                phrase_length = random.randint(4, 8)
                ascending = not ascending
                # Add a longer pause between phrases
                tokens.append(self.time_shift_start + random.randint(20, 50))
        
        tokens.append(self.tokenizer.EOS_TOKEN)
        
        print(f"Fallback generation: {len(tokens)} tokens, {num_notes} notes, raga={raga}")
        return tokens
    
    def tokens_to_midi(self, tokens: List[int], output_path: str) -> str:
        """Convert tokens to MIDI file"""
        return self.tokenizer.detokenize(tokens, output_path)


# Singleton instance
_generator = None


def get_generator(model, tokenizer, vocabularies) -> MusicGenerator:
    """Get or create music generator instance"""
    global _generator
    if _generator is None:
        _generator = MusicGenerator(model, tokenizer, vocabularies)
    return _generator
