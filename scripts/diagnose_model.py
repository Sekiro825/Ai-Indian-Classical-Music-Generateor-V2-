"""Diagnose model output quality issues"""
import os
import sys
sys.path.insert(0, 'd:/MUSIC_MP/EXTRACTED/ALLDATA')
sys.path.insert(0, 'd:/MUSIC_MP/EXTRACTED/ALLDATA/models')

import torch
import json
from transformer_cvae import RagaCVAE, CVAEConfig
from tokenizer import MIDITokenizer

# Load checkpoint
checkpoint = torch.load('checkpoints/model.pt', map_location='cpu', weights_only=False)
config = checkpoint['config']
model = RagaCVAE(config)
model.load_state_dict(checkpoint['model_state_dict'])
model.eval()

# Load tokenizer
tokenizer = MIDITokenizer.load('checkpoints/tokenizer.json')

# Load vocabularies
with open('checkpoints/vocabularies.json', 'r') as f:
    vocabularies = json.load(f)

print("=" * 60)
print("ISSUE 1: Testing if conditioning affects output")
print("=" * 60)

# Test with different moods
results = []
for mood_name in ['calm', 'energetic', 'sad']:
    mood_idx = vocabularies['mood_to_idx'].get(mood_name, 0)
    raga_idx = vocabularies['raga_to_idx'].get('yaman', 0)
    
    with torch.no_grad():
        generated = model.generate(
            torch.tensor([mood_idx]),
            torch.tensor([raga_idx]),
            torch.tensor([15]),  # tempo
            torch.tensor([8]),   # duration
            max_length=100,
            temperature=1.2,
            top_k=50
        )
    
    tokens = generated[0].tolist()
    results.append((mood_name, tokens[:20]))
    print(f"Mood '{mood_name}': first 20 tokens = {tokens[:20]}")

# Check if results are identical
if results[0][1] == results[1][1] == results[2][1]:
    print("\n❌ PROBLEM: All outputs are IDENTICAL despite different moods!")
    print("   This indicates the model is suffering from MODE COLLAPSE")
    print("   The conditioning is not affecting the output at all.")
else:
    print("\n✅ Different moods produce different outputs")

print("\n" + "=" * 60)
print("ISSUE 2: Analyzing token distribution and timing")
print("=" * 60)

# Generate longer sequence
with torch.no_grad():
    generated = model.generate(
        torch.tensor([0]),
        torch.tensor([0]),
        torch.tensor([15]),
        torch.tensor([8]),
        max_length=200,
        temperature=1.2,
        top_k=50
    )

tokens = generated[0].tolist()
print(f"Generated {len(tokens)} tokens total")

# Analyze token types
note_ons = [t for t in tokens if tokenizer.note_on_offset <= t < tokenizer.note_off_offset]
note_offs = [t for t in tokens if tokenizer.note_off_offset <= t < tokenizer.time_shift_offset]
time_shifts = [t for t in tokens if tokenizer.time_shift_offset <= t < tokenizer.velocity_offset]
velocities = [t for t in tokens if tokenizer.velocity_offset <= t < tokenizer.vocab_size]

print(f"\nToken breakdown:")
print(f"  Note ONs: {len(note_ons)}")
print(f"  Note OFFs: {len(note_offs)}")
print(f"  Time Shifts: {len(time_shifts)}")
print(f"  Velocities: {len(velocities)}")

# Analyze time shifts
if time_shifts:
    time_values = [t - tokenizer.time_shift_offset for t in time_shifts]
    total_time_steps = sum(time_values)
    total_seconds = total_time_steps * tokenizer.config.time_step
    print(f"\nTime analysis:")
    print(f"  Time shift values: min={min(time_values)}, max={max(time_values)}, avg={sum(time_values)/len(time_values):.1f}")
    print(f"  Total time steps: {total_time_steps}")
    print(f"  Total duration: {total_seconds:.2f} seconds")
    
    if total_seconds < 5:
        print("\n❌ PROBLEM: Total duration is very short!")
        print("   Time shift tokens have low values - model isn't generating enough time between notes")

# Check for repetitive patterns
print("\n" + "=" * 60)
print("ISSUE 3: Check for repetitive patterns")
print("=" * 60)

# Check if there are repeated subsequences
token_str = ','.join(map(str, tokens[:50]))
print(f"First 50 tokens: {token_str}")

# Look for repeated patterns
for pattern_len in [3, 4, 5]:
    patterns = {}
    for i in range(len(tokens) - pattern_len):
        pattern = tuple(tokens[i:i+pattern_len])
        patterns[pattern] = patterns.get(pattern, 0) + 1
    
    repeated = {p: c for p, c in patterns.items() if c > 2}
    if repeated:
        print(f"\n  Patterns of length {pattern_len} repeated >2 times:")
        for pattern, count in sorted(repeated.items(), key=lambda x: -x[1])[:3]:
            print(f"    {pattern}: {count} times")

print("\n" + "=" * 60)
print("DIAGNOSIS SUMMARY")
print("=" * 60)
