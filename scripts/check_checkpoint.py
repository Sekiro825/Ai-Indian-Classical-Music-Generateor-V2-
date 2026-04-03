"""Inspect all layer dimensions in checkpoint"""
import sys
sys.path.insert(0, 'd:/MUSIC_MP/EXTRACTED/ALLDATA/models')

import torch
from transformer_cvae import RagaCVAE, CVAEConfig
from tokenizer import MIDITokenizer

checkpoint_path = 'd:/MUSIC_MP/EXTRACTED/ALLDATA/checkpoints/model.pt'
checkpoint = torch.load(checkpoint_path, map_location='cpu', weights_only=False)

print("=" * 60)
print("CHECKPOINT ANALYSIS")
print("=" * 60)

# Current config
current_config = CVAEConfig()
print("\nCurrent CVAEConfig:")
print(f"  vocab_size: {current_config.vocab_size}")
print(f"  num_moods: {current_config.num_moods}")
print(f"  num_ragas: {current_config.num_ragas}")
print(f"  mood_embed_dim: {current_config.mood_embed_dim}")
print(f"  raga_embed_dim: {current_config.raga_embed_dim}")
print(f"  tempo_embed_dim: {current_config.tempo_embed_dim}")
print(f"  duration_embed_dim: {current_config.duration_embed_dim}")

# Get the saved config
if 'config' in checkpoint:
    saved_config = checkpoint['config']
    print("\nSaved Config from checkpoint:")
    print(f"  vocab_size: {saved_config.vocab_size}")
    print(f"  num_moods: {saved_config.num_moods}")
    print(f"  num_ragas: {saved_config.num_ragas}")
    print(f"  mood_embed_dim: {saved_config.mood_embed_dim}")
    print(f"  raga_embed_dim: {saved_config.raga_embed_dim}")
    print(f"  tempo_embed_dim: {saved_config.tempo_embed_dim}")
    print(f"  duration_embed_dim: {saved_config.duration_embed_dim}")

# Print conditioning layer dimensions from state dict
print("\nConditioning layers in state_dict:")
state_dict = checkpoint['model_state_dict']
for key in sorted(state_dict.keys()):
    if 'conditioning' in key:
        print(f"  {key}: {state_dict[key].shape}")

print("\nEncoder/Decoder embedding layers:")
for key in sorted(state_dict.keys()):
    if 'embedding' in key:
        print(f"  {key}: {state_dict[key].shape}")
