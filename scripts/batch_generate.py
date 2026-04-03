import os
import torch
from hybrid.inference.generate import HybridGenerator

def batch_generate():
    # Adjust paths if your checkpoints are located somewhere else
    checkpoint_path = "hybrid/checkpoints/best_model.pt"
    vocab_path = "hybrid/checkpoints/conditioning_vocabs.json"
    
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Loading 1.7B Model on {device}... (this may take a minute)")
    
    # Load model once to save massive amounts of time and RAM
    generator = HybridGenerator.from_checkpoint(
        checkpoint_path=checkpoint_path,
        vocab_path=vocab_path,
        device=device
    )
    
    output_dir = "generated_outputs"
    os.makedirs(output_dir, exist_ok=True)
    
    # Define your prompts/combinations here:
    combinations = [
        # Set 1: Fast/Energetic
        {"raga": "hindol", "mood": "joyful", "tempo": 120, "duration": 60},
        {"raga": "hamsadhvani", "mood": "energetic", "tempo": 130, "duration": 60},
        {"raga": "desh", "mood": "romantic", "tempo": 110, "duration": 60},
        
        # Set 2: Devotional/Morning
        {"raga": "bhairav", "mood": "devotional", "tempo": 85, "duration": 60},
        {"raga": "ahir_bhairav", "mood": "peaceful", "tempo": 90, "duration": 60},
        {"raga": "bhairavi", "mood": "devotional", "tempo": 100, "duration": 60},
        
        # Set 3: Serious/Midnight
        {"raga": "darbari", "mood": "serious", "tempo": 70, "duration": 60},
        {"raga": "malkauns", "mood": "serious", "tempo": 80, "duration": 60},
        {"raga": "bageshri", "mood": "romantic", "tempo": 95, "duration": 60},
        
        # Set 4: Peaceful/Evening
        {"raga": "yaman", "mood": "peaceful", "tempo": 90, "duration": 60},
        {"raga": "puriya_dhanashree", "mood": "intense", "tempo": 95, "duration": 60},
        {"raga": "bihag", "mood": "joyful", "tempo": 105, "duration": 60},
    ]
    
    print("\nStarting Batch Generation...")
    for i, c in enumerate(combinations):
        out_name = f"{output_dir}/output_{i+1}_{c['raga']}_{c['mood']}.mid"
        print(f"[{i+1}/{len(combinations)}] Generating {c['raga']} ({c['mood']}) -> {out_name}")
        
        try:
            generator.generate_midi_file(
                output_path=out_name,
                raga=c['raga'],
                mood=c['mood'],
                tempo=c['tempo'],
                duration=c['duration'],
                temperature=1.2,
                top_k=200,
                top_p=0.95,
                enforce_raga_grammar=True,
                use_chalan_prefix=True
            )
        except Exception as e:
            print(f"Failed generating {c['raga']}: {e}")
            
    print(f"\n✅ All outputs saved in the '{output_dir}' folder!")
    print("You can download this folder to present the MIDI files to your teachers.")

if __name__ == "__main__":
    batch_generate()
