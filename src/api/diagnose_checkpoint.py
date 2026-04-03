
import torch
import sys
from pathlib import Path
from dataclasses import dataclass
import types

# Add parent directory to path
BASE_DIR = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(BASE_DIR))

# Mock transformer_cvae module
mock_module = types.ModuleType("transformer_cvae")
@dataclass
class CVAEConfig:
    pass
mock_module.CVAEConfig = CVAEConfig
sys.modules["transformer_cvae"] = mock_module
sys.modules["models.transformer_cvae"] = mock_module

checkpoint_path = BASE_DIR / "checkpoints/model.pt"
if not checkpoint_path.exists():
    checkpoint_path = BASE_DIR / "checkpoints/best_model.pt"

try:
    checkpoint = torch.load(checkpoint_path, map_location='cpu', weights_only=False)
    
    cfg = checkpoint['config']
    print("\n--- CLEAN CONFIG ---")
    if isinstance(cfg, dict):
        d = cfg
    else:
        d = cfg.__dict__
    
    # Print individual fields cleanly
    for k, v in d.items():
        print(f"{k}: {v}")

except Exception as e:
    print(f"Error: {e}")
