import os
import json
import torch
import numpy as np
import argparse
from tqdm import tqdm
from pathlib import Path
from concurrent.futures import ProcessPoolExecutor
import importlib.util
from functools import partial

# Load the BPE tokenizer class from its script
def get_tokenizer_class():
    bpe_script_path = os.path.join(os.path.dirname(__file__), "train_bpe_tokenizer.py")
    spec = importlib.util.spec_from_file_location("train_bpe_tokenizer", bpe_script_path)
    bpe_module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(bpe_module)
    return bpe_module.BPEMIDITokenizer

# Global tokenizer instance for workers to reuse
_tokenizer = None
_metadata = None

TAAL_NAMES = [
    "addhatrital", "trital", "dadra", "deepchandi", "ektal", "jhaptal",
    "rupak", "bhajani", "keherwa", "adi", "misra_chapu", "khanda_chapu",
    "roopak", "unknown",
]

def init_worker(bpe_path, metadata_path):
    global _tokenizer, _metadata
    BPEMIDITokenizer = get_tokenizer_class()
    _tokenizer = BPEMIDITokenizer.load(bpe_path)
    if os.path.exists(metadata_path):
        with open(metadata_path, 'r') as f:
            _metadata = json.load(f)
    else:
        _metadata = {}

def process_file_worker(file_info):
    """
    Worker function to tokenize a single file.
    file_info: (full_path, relative_path)
    """
    global _tokenizer, _metadata
    file_path, midi_dir_path = file_info
    file_path = Path(file_path)
    midi_dir_path = Path(midi_dir_path)
    
    try:
        # Encode with BPE
        tokens = _tokenizer.encode(str(file_path))
        
        # Metadata Parsing (Mimic V2RagaDataset._scan)
        parts = file_path.relative_to(midi_dir_path).parts
        if len(parts) >= 3:
            tradition = parts[0]
            raga = parts[1]
        else:
            raga = "unknown"
            tradition = "unknown"
            name_lower = file_path.name.lower()
            for r_name in _metadata.keys():
                if r_name.lower() in name_lower:
                    raga = r_name
                    break
        
        # Mood lookup
        raga_info = _metadata.get(raga, {})
        moods = raga_info.get("moods", ["unknown"])
        mood = moods[0]

        # Infer taal from raga name or directory
        taal = "unknown"
        raga_lower = raga.lower()
        for t_name in TAAL_NAMES:
            if t_name in raga_lower:
                taal = t_name
                break

        # Tempo: use midpoint of metadata tempo_range if available
        tempo_range = raga_info.get("tempo_range", [60, 120])
        try:
            tempo = int(round((tempo_range[0] + tempo_range[1]) / 2))
        except Exception:
            tempo = 90

        # Duration (seconds): estimate from tokenized sequence
        est_seconds = _tokenizer.estimate_duration_seconds(tokens)
        duration = int(round(est_seconds)) if est_seconds and est_seconds > 0 else 60

        # Expression features (if available)
        expr_path = file_path.with_suffix('.expr.npy')
        has_expression = False
        expression = None
        if expr_path.exists():
            try:
                expression = np.load(expr_path)
                has_expression = True
            except Exception:
                expression = None
        
        return {
            'tokens': torch.tensor(tokens, dtype=torch.int16),
            'mood': mood,
            'raga': raga,
            'taal': taal,
            'tempo': tempo,
            'duration': duration,
            'expression': torch.tensor(expression, dtype=torch.float32) if expression is not None else None,
            'has_expression': has_expression,
            'file_name': file_path.name
        }
    except Exception as e:
        return None

def main():
    parser = argparse.ArgumentParser(description='TRUE Parallel Pre-tokenization for Infinite Speed')
    parser.add_argument('--midi_dir', type=str, default='data/midi_v2')
    parser.add_argument('--bpe_path', type=str, default='models/tokenizer_v2/bpe_tokenizer.json')
    parser.add_argument('--metadata', type=str, default='src/sekiro_ai/config/raga_metadata.json')
    parser.add_argument('--output', type=str, default='data/v2_tokenized_cache.pt')
    parser.add_argument('--workers', type=int, default=42)
    args = parser.parse_args()

    print(f"Scanning {args.midi_dir} for MIDI files...")
    all_files = list(Path(args.midi_dir).rglob("*.mid"))
    print(f"Found {len(all_files)} files. Preparing for 42-worker blast...")

    # Prepare worker arguments
    worker_inputs = [(str(f), args.midi_dir) for f in all_files]

    data_cache = []
    
    # Run with true parallelism
    print(f"🚀 Launching Parallel Pre-Tokenization (Workers: {args.workers})...")
    with ProcessPoolExecutor(
        max_workers=args.workers,
        initializer=init_worker,
        initargs=(args.bpe_path, args.metadata)
    ) as executor:
        results = list(tqdm(executor.map(process_file_worker, worker_inputs), total=len(all_files)))

    # Filter out failed runs
    data_cache = [r for r in results if r is not None]
    
    print(f"🏁 Successfully tokenized {len(data_cache)} / {len(all_files)} files.")
    
    print(f"💾 Saving high-speed cache to {args.output}...")
    torch.save(data_cache, args.output)
    print("✨ DONE! You are now ready for the Infinite Speed training run.")

if __name__ == '__main__':
    main()
