"""
Dataset and DataLoader for Raga Music Training
With data augmentation for better generalization
"""

import os
import json
import random
from pathlib import Path
from typing import Dict, List, Tuple, Optional
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader

# Import tokenizer
import sys
sys.path.append(str(Path(__file__).parent))
from tokenizer import MIDITokenizer, extract_raga_from_filename


class DataAugmentation:
    """Data augmentation for MIDI token sequences"""
    
    def __init__(self, 
                 pitch_shift_range: int = 3,
                 tempo_scale_range: Tuple[float, float] = (0.9, 1.1),
                 random_crop: bool = True):
        self.pitch_shift_range = pitch_shift_range
        self.tempo_scale_range = tempo_scale_range
        self.random_crop = random_crop
    
    def pitch_shift(self, tokens: np.ndarray, note_on_offset: int, note_off_offset: int) -> np.ndarray:
        """Shift all pitches by a random amount (preserving raga character)"""
        shift = random.randint(-self.pitch_shift_range, self.pitch_shift_range)
        if shift == 0:
            return tokens
        
        new_tokens = tokens.copy()
        for i in range(len(new_tokens)):
            token = new_tokens[i]
            # Check if it's a NOTE_ON token (offset 3 to 130)
            if note_on_offset <= token < note_on_offset + 128:
                pitch = token - note_on_offset
                new_pitch = max(0, min(127, pitch + shift))
                new_tokens[i] = note_on_offset + new_pitch
            # Check if it's a NOTE_OFF token (offset 131 to 258)
            elif note_off_offset <= token < note_off_offset + 128:
                pitch = token - note_off_offset
                new_pitch = max(0, min(127, pitch + shift))
                new_tokens[i] = note_off_offset + new_pitch
        
        return new_tokens
    
    def tempo_variation(self, tokens: np.ndarray, time_shift_offset: int) -> np.ndarray:
        """Scale time shifts to vary tempo"""
        scale = random.uniform(*self.tempo_scale_range)
        if scale == 1.0:
            return tokens
        
        new_tokens = tokens.copy()
        for i in range(len(new_tokens)):
            token = new_tokens[i]
            # Check if it's a TIME_SHIFT token (offset 259 to 458)
            if time_shift_offset <= token < time_shift_offset + 200:
                time_steps = token - time_shift_offset
                new_time_steps = int(time_steps * scale)
                new_time_steps = max(0, min(199, new_time_steps))
                new_tokens[i] = time_shift_offset + new_time_steps
        
        return new_tokens
    
    def __call__(self, tokens: np.ndarray, tokenizer: MIDITokenizer) -> np.ndarray:
        """Apply random augmentations"""
        if random.random() < 0.5:
            tokens = self.pitch_shift(tokens, tokenizer.note_on_offset, tokenizer.note_off_offset)
        
        if random.random() < 0.3:
            tokens = self.tempo_variation(tokens, tokenizer.time_shift_offset)
        
        return tokens


class RagaDataset(Dataset):
    """
    Dataset for loading tokenized MIDI files with conditioning labels
    Supports data augmentation for training
    """
    
    def __init__(
        self,
        midi_dir: str,
        raga_metadata_path: str,
        tokenizer: MIDITokenizer,
        max_seq_length: int = 1024,
        cache_tokens: bool = True,
        augment: bool = False
    ):
        self.midi_dir = Path(midi_dir)
        self.tokenizer = tokenizer
        self.max_seq_length = max_seq_length
        self.cache_tokens = cache_tokens
        self.token_cache = {}
        self.augment = augment
        
        if augment:
            self.augmentation = DataAugmentation()
        
        # Load raga metadata
        with open(raga_metadata_path, 'r') as f:
            self.raga_metadata = json.load(f)
        
        # Build mood and raga vocabularies
        self._build_vocabularies()
        
        # Load MIDI files
        self.midi_files = list(self.midi_dir.glob("*.mid"))
        print(f"Found {len(self.midi_files)} MIDI files")
        
        # Create labels for each file
        self._create_labels()
    
    def _build_vocabularies(self):
        """Build mood and raga index mappings"""
        # Collect all moods
        all_moods = set()
        for raga_info in self.raga_metadata.values():
            all_moods.update(raga_info.get('moods', []))
        
        self.mood_to_idx = {mood: idx for idx, mood in enumerate(sorted(all_moods))}
        self.idx_to_mood = {idx: mood for mood, idx in self.mood_to_idx.items()}
        
        # Raga vocabulary
        self.raga_to_idx = {raga: idx for idx, raga in enumerate(sorted(self.raga_metadata.keys()))}
        self.idx_to_raga = {idx: raga for raga, idx in self.raga_to_idx.items()}
        
        # Add unknown
        self.mood_to_idx['unknown'] = len(self.mood_to_idx)
        self.raga_to_idx['unknown'] = len(self.raga_to_idx)
        
        print(f"Mood vocabulary: {len(self.mood_to_idx)} moods")
        print(f"Raga vocabulary: {len(self.raga_to_idx)} ragas")
    
    def _create_labels(self):
        """Extract labels from filenames"""
        self.labels = []
        
        for midi_file in self.midi_files:
            raga = extract_raga_from_filename(midi_file.name)
            
            # Get raga info
            raga_info = self.raga_metadata.get(raga, {})
            
            # Get mood (pick randomly from available moods for variety)
            moods = raga_info.get('moods', ['unknown'])
            mood = random.choice(moods) if moods else 'unknown'
            
            # Get tempo range
            tempo_range = raga_info.get('tempo_range', [60, 120])
            tempo = random.randint(*tempo_range)
            tempo_bin = min(int((tempo - 40) * 32 / 160), 31)
            
            # Duration bin
            duration_bin = 8
            
            self.labels.append({
                'file': str(midi_file),
                'raga': raga,
                'mood': mood,
                'moods': moods,  # Keep all moods for augmentation
                'tempo_bin': tempo_bin,
                'tempo_range': tempo_range,
                'duration_bin': duration_bin
            })
    
    def __len__(self):
        return len(self.midi_files)
    
    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        label = self.labels[idx]
        midi_file = label['file']
        
        # Get tokens (with caching)
        if self.cache_tokens and midi_file in self.token_cache:
            tokens = self.token_cache[midi_file].copy()  # Copy to avoid mutation
        else:
            tokens = self.tokenizer.tokenize_midi(midi_file)
            tokens = np.array(tokens)
            if self.cache_tokens:
                self.token_cache[midi_file] = tokens.copy()
        
        # Apply augmentation during training
        if self.augment and len(tokens) > 0:
            tokens = self.augmentation(tokens, self.tokenizer)
            
            # Random mood from available moods
            mood = random.choice(label['moods'])
            
            # Random tempo within range
            tempo = random.randint(*label['tempo_range'])
            tempo_bin = min(int((tempo - 40) * 32 / 160), 31)
        else:
            mood = label['mood']
            tempo_bin = label['tempo_bin']
        
        # Random crop for long sequences
        if self.augment and len(tokens) > self.max_seq_length:
            max_start = len(tokens) - self.max_seq_length
            start_idx = random.randint(0, max_start)
            tokens = tokens[start_idx:start_idx + self.max_seq_length]
            # Ensure BOS at start
            if tokens[0] != self.tokenizer.BOS_TOKEN:
                tokens = np.concatenate([[self.tokenizer.BOS_TOKEN], tokens[:-1]])
        
        # Pad/truncate to max length
        tokens = self.tokenizer.pad_sequence(list(tokens), self.max_seq_length)
        
        # Get indices
        mood_idx = self.mood_to_idx.get(mood, self.mood_to_idx['unknown'])
        raga_idx = self.raga_to_idx.get(label['raga'], self.raga_to_idx['unknown'])
        
        # Create padding mask
        padding_mask = tokens == 0
        
        return {
            'tokens': torch.tensor(tokens, dtype=torch.long),
            'mood': torch.tensor(mood_idx, dtype=torch.long),
            'raga': torch.tensor(raga_idx, dtype=torch.long),
            'tempo': torch.tensor(tempo_bin, dtype=torch.long),
            'duration': torch.tensor(label['duration_bin'], dtype=torch.long),
            'padding_mask': torch.tensor(padding_mask, dtype=torch.bool)
        }
    
    def save_vocabularies(self, path: str):
        """Save vocabulary mappings"""
        vocab_data = {
            'mood_to_idx': self.mood_to_idx,
            'idx_to_mood': {str(k): v for k, v in self.idx_to_mood.items()},
            'raga_to_idx': self.raga_to_idx,
            'idx_to_raga': {str(k): v for k, v in self.idx_to_raga.items()}
        }
        with open(path, 'w') as f:
            json.dump(vocab_data, f, indent=2)
    
    @classmethod
    def load_vocabularies(cls, path: str) -> Dict:
        """Load vocabulary mappings"""
        with open(path, 'r') as f:
            return json.load(f)


def create_dataloaders(
    midi_dir: str,
    raga_metadata_path: str,
    tokenizer: MIDITokenizer,
    batch_size: int = 4,
    max_seq_length: int = 1024,
    train_ratio: float = 0.9,
    num_workers: int = 0
) -> Tuple[DataLoader, DataLoader, RagaDataset]:
    """
    Create train and validation dataloaders
    """
    # Create training dataset with augmentation
    train_dataset = RagaDataset(
        midi_dir=midi_dir,
        raga_metadata_path=raga_metadata_path,
        tokenizer=tokenizer,
        max_seq_length=max_seq_length,
        augment=True  # Enable augmentation for training
    )
    
    # Create validation dataset without augmentation
    val_dataset = RagaDataset(
        midi_dir=midi_dir,
        raga_metadata_path=raga_metadata_path,
        tokenizer=tokenizer,
        max_seq_length=max_seq_length,
        augment=False
    )
    
    # Split indices
    total_size = len(train_dataset)
    indices = list(range(total_size))
    random.shuffle(indices)
    
    train_size = int(total_size * train_ratio)
    train_indices = indices[:train_size]
    val_indices = indices[train_size:]
    
    # Create subset datasets
    train_subset = torch.utils.data.Subset(train_dataset, train_indices)
    val_subset = torch.utils.data.Subset(val_dataset, val_indices)
    
    print(f"Train samples: {len(train_subset)}, Val samples: {len(val_subset)}")
    
    # Create dataloaders
    train_loader = DataLoader(
        train_subset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=True,
        drop_last=True  # Required for gradient accumulation
    )
    
    val_loader = DataLoader(
        val_subset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True
    )
    
    return train_loader, val_loader, train_dataset


if __name__ == "__main__":
    # Test dataset with augmentation
    tokenizer = MIDITokenizer()
    
    dataset = RagaDataset(
        midi_dir="all_midi",
        raga_metadata_path="config/raga_metadata.json",
        tokenizer=tokenizer,
        max_seq_length=1024,
        augment=True
    )
    
    print(f"\nDataset size: {len(dataset)}")
    
    # Test a sample
    sample = dataset[0]
    print(f"Sample tokens shape: {sample['tokens'].shape}")
    print(f"Mood: {dataset.idx_to_mood.get(sample['mood'].item(), 'unknown')}")
    print(f"Raga: {dataset.idx_to_raga.get(sample['raga'].item(), 'unknown')}")
