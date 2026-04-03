"""
Export trained model for CPU inference
Run this after training to create optimized model for deployment
"""

import os
import sys
import argparse
from pathlib import Path

import torch
import torch.nn as nn

# Add models directory to path
sys.path.insert(0, str(Path(__file__).parent / "models"))

from tokenizer import MIDITokenizer
from transformer_cvae import RagaCVAE, CVAEConfig


def export_for_cpu(
    checkpoint_path: str,
    output_dir: str = "exported_model",
    optimize: bool = True
):
    """
    Export trained model for CPU inference
    
    Args:
        checkpoint_path: Path to training checkpoint
        output_dir: Directory to save exported model
        optimize: Whether to optimize model for inference
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    print(f"📦 Loading checkpoint from: {checkpoint_path}")
    
    # Load checkpoint
    checkpoint = torch.load(checkpoint_path, map_location='cpu')
    config = checkpoint['config']
    
    # Create model
    model = RagaCVAE(config)
    model.load_state_dict(checkpoint['model_state_dict'])
    model.eval()
    
    # Count parameters
    num_params = sum(p.numel() for p in model.parameters())
    size_mb = num_params * 4 / (1024 ** 2)  # float32
    print(f"📊 Model parameters: {num_params:,}")
    print(f"📊 Model size: {size_mb:.2f} MB (float32)")
    
    if optimize:
        print("⚡ Optimizing for inference...")
        
        # Convert to half precision for smaller size (optional)
        # model = model.half()  # Uncomment for float16
        
        # Disable gradients
        for param in model.parameters():
            param.requires_grad = False
        
        # Fuse operations where possible
        model = torch.jit.optimize_for_inference(torch.jit.script(model.encoder))
        # Note: Full model scripting may need adjustments
    
    # Save model state dict (most portable)
    model_path = output_dir / "model.pt"
    torch.save({
        'config': config,
        'model_state_dict': checkpoint['model_state_dict'],
        'epoch': checkpoint.get('epoch', -1)
    }, model_path)
    print(f"💾 Saved model to: {model_path}")
    
    # Copy tokenizer
    tokenizer_src = Path("checkpoints/tokenizer.json")
    tokenizer_dst = output_dir / "tokenizer.json"
    if tokenizer_src.exists():
        import shutil
        shutil.copy(tokenizer_src, tokenizer_dst)
        print(f"💾 Copied tokenizer to: {tokenizer_dst}")
    
    # Copy vocabularies
    vocab_src = Path("checkpoints/vocabularies.json")
    vocab_dst = output_dir / "vocabularies.json"
    if vocab_src.exists():
        import shutil
        shutil.copy(vocab_src, vocab_dst)
        print(f"💾 Copied vocabularies to: {vocab_dst}")
    
    # Copy raga metadata
    raga_src = Path("config/raga_metadata.json")
    raga_dst = output_dir / "raga_metadata.json"
    if raga_src.exists():
        import shutil
        shutil.copy(raga_src, raga_dst)
        print(f"💾 Copied raga metadata to: {raga_dst}")
    
    # Create inference script
    inference_script = output_dir / "inference.py"
    with open(inference_script, 'w') as f:
        f.write('''"""
Quick inference script for CPU
"""

import torch
import json
from pathlib import Path

# Load model
checkpoint = torch.load("model.pt", map_location="cpu")
config = checkpoint["config"]

# Import model class (adjust path as needed)
import sys
sys.path.insert(0, "..")
from models.transformer_cvae import RagaCVAE

model = RagaCVAE(config)
model.load_state_dict(checkpoint["model_state_dict"])
model.eval()

# Load vocabularies
with open("vocabularies.json", "r") as f:
    vocab = json.load(f)

print("Model loaded successfully!")
print(f"Available moods: {list(vocab['mood_to_idx'].keys())}")
print(f"Available ragas: {list(vocab['raga_to_idx'].keys())}")

# Example generation
mood_idx = vocab["mood_to_idx"].get("romantic", 0)
raga_idx = vocab["raga_to_idx"].get("yaman", 0)

with torch.no_grad():
    generated = model.generate(
        torch.tensor([mood_idx]),
        torch.tensor([raga_idx]),
        torch.tensor([15]),  # tempo bin
        torch.tensor([8]),   # duration bin
        max_length=256,
        temperature=0.9
    )

print(f"Generated {generated.shape[1]} tokens")
''')
    print(f"💾 Created inference script: {inference_script}")
    
    # Create requirements for export
    req_path = output_dir / "requirements.txt"
    with open(req_path, 'w') as f:
        f.write("""# Minimal requirements for CPU inference
torch>=2.0.0
numpy>=1.24.0
mido>=1.3.0
midi2audio>=0.1.1
google-generativeai>=0.3.0
fastapi>=0.100.0
uvicorn>=0.23.0
""")
    print(f"💾 Created requirements: {req_path}")
    
    # Calculate final size
    total_size = sum(f.stat().st_size for f in output_dir.glob("*") if f.is_file())
    print(f"\n✅ Export complete!")
    print(f"📁 Output directory: {output_dir}")
    print(f"📊 Total size: {total_size / (1024**2):.2f} MB")
    
    return str(output_dir)


def main():
    parser = argparse.ArgumentParser(description='Export model for CPU inference')
    parser.add_argument('--checkpoint', type=str, default='checkpoints/best_model.pt',
                        help='Path to training checkpoint')
    parser.add_argument('--output', type=str, default='exported_model',
                        help='Output directory')
    parser.add_argument('--no-optimize', action='store_true',
                        help='Skip optimization')
    args = parser.parse_args()
    
    if not Path(args.checkpoint).exists():
        print(f"❌ Checkpoint not found: {args.checkpoint}")
        print("   Please train the model first with: python train.py")
        return
    
    export_for_cpu(
        checkpoint_path=args.checkpoint,
        output_dir=args.output,
        optimize=not args.no_optimize
    )


if __name__ == "__main__":
    main()
