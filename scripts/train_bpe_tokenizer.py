"""
BPE Tokenizer Trainer for MIDI Sequences

Trains a Byte-Pair Encoding tokenizer on the raw MIDI event vocabulary.
This compresses common melodic phrases (taans, alankars, gamaka patterns)
into single tokens, allowing the model to "see" much larger musical structures
within its context window.

Usage:
    python scripts/train_bpe_tokenizer.py \
        --midi_dir data/midi \
        --extra_midi_dir data/midi_v2 \
        --output_path src/sekiro_ai/models/bpe_tokenizer.json \
        --vocab_size 10000

The output tokenizer is used by the V2 Mamba-Flow model.
"""

import os
import sys
import json
import argparse
import importlib.util
from pathlib import Path
from collections import Counter
from typing import List, Dict, Tuple
import numpy as np
from tqdm import tqdm
from concurrent.futures import ProcessPoolExecutor
import math

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))
try:
    from sekiro_ai.models.tokenizer import MIDITokenizer, TokenizerConfig
except (OSError, ModuleNotFoundError):
    tokenizer_path = Path(__file__).parent.parent / "src" / "sekiro_ai" / "models" / "tokenizer.py"
    spec = importlib.util.spec_from_file_location("sekiro_ai_models_tokenizer", tokenizer_path)
    if spec is None or spec.loader is None:
        raise
    tokenizer_module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(tokenizer_module)
    MIDITokenizer = tokenizer_module.MIDITokenizer
    TokenizerConfig = tokenizer_module.TokenizerConfig


class BPEMIDITokenizer:
    """
    Byte-Pair Encoding tokenizer for MIDI event streams.

    Phase 1: Use the base MIDITokenizer to convert MIDI files to raw event streams
    Phase 2: Learn BPE merges from the corpus to compress frequent patterns
    Phase 3: Encode new sequences using the learned merges

    The vocabulary starts with the base 491 raw tokens and expands to ~10K
    by merging common pairs into new compound tokens.
    """

    PAD_TOKEN = 0
    BOS_TOKEN = 1
    EOS_TOKEN = 2

    def __init__(self, base_tokenizer: MIDITokenizer = None):
        self.base_tokenizer = base_tokenizer or MIDITokenizer(
            TokenizerConfig(max_sequence_length=65536)  # No truncation during BPE training
        )
        self.base_vocab_size = self.base_tokenizer.vocab_size
        self.merges: List[Tuple[int, int]] = []
        self.merge_map: Dict[Tuple[int, int], int] = {}
        self.vocab_size = self.base_vocab_size
        # Reverse map: new_token -> (token_a, token_b)
        self.decompose_map: Dict[int, Tuple[int, int]] = {}

    def _get_pair_counts(self, sequences: List[List[int]]) -> Counter:
        """Count adjacent token pairs across all sequences"""
        pair_counts = Counter()
        for seq in sequences:
            for i in range(len(seq) - 1):
                pair = (seq[i], seq[i + 1])
                # Don't merge across special tokens
                if pair[0] <= 2 or pair[1] <= 2:
                    continue
                pair_counts[pair] += 1
        return pair_counts

    def _apply_merge(self, sequences: List[List[int]], pair: Tuple[int, int], new_token: int) -> List[List[int]]:
        """Replace all occurrences of pair with new_token in all sequences"""
        result = []
        for seq in sequences:
            new_seq = []
            i = 0
            while i < len(seq):
                if i < len(seq) - 1 and seq[i] == pair[0] and seq[i + 1] == pair[1]:
                    new_seq.append(new_token)
                    i += 2
                else:
                    new_seq.append(seq[i])
                    i += 1
            result.append(new_seq)
        return result

    def _tokenize_chunk(self, paths: List[str]) -> Tuple[List[List[int]], int]:
        """Tokenize a list of MIDI files (for parallel processing)"""
        seqs = []
        failed = 0
        for p in paths:
            try:
                tokens = self.base_tokenizer.tokenize_midi(str(p))
                if len(tokens) > 5:
                    seqs.append(tokens)
                else:
                    failed += 1
            except Exception:
                failed += 1
        return seqs, failed

    @staticmethod
    def _get_pair_counts_chunk(sequences: List[List[int]]) -> Counter:
        """Count pairs in a chunk of sequences (for parallel processing)"""
        counts = Counter()
        for seq in sequences:
            for i in range(len(seq) - 1):
                pair = (seq[i], seq[i + 1])
                if pair[0] <= 2 or pair[1] <= 2: continue
                counts[pair] += 1
        return counts

    @staticmethod
    def _apply_merge_chunk(sequences: List[List[int]], pair: Tuple[int, int], new_token: int) -> List[List[int]]:
        """Apply merge to a chunk of sequences (for parallel processing)"""
        result = []
        for seq in sequences:
            new_seq = []
            i = 0
            while i < len(seq):
                if i < len(seq) - 1 and seq[i] == pair[0] and seq[i + 1] == pair[1]:
                    new_seq.append(new_token)
                    i += 2
                else:
                    new_seq.append(seq[i])
                    i += 1
            result.append(new_seq)
        return result

    def train(self, midi_paths: List[str], target_vocab_size: int = 10000, verbose: bool = True):
        """
        Train BPE merges from a corpus of MIDI files.

        Args:
            midi_paths: List of paths to MIDI files
            target_vocab_size: Target vocabulary size after merges
            verbose: Print progress
        """
        if verbose:
            print(f"Tokenizing {len(midi_paths)} MIDI files with base tokenizer (vocab={self.base_vocab_size})...")

        # Phase 1: Tokenize all files (Parallelized)
        sequences = []
        failed = 0
        workers = getattr(self, 'workers', 1)

        print(f"Tokenizing MIDI files using {workers} workers...")
        chunk_size = math.ceil(len(midi_paths) / workers)
        
        with ProcessPoolExecutor(max_workers=workers) as executor:
            futures = []
            for i in range(0, len(midi_paths), chunk_size):
                chunk = midi_paths[i:i + chunk_size]
                futures.append(executor.submit(self._tokenize_chunk, chunk))
            
            for future in tqdm(futures, desc="Tokenizing Chunks"):
                chunk_seqs, chunk_failed = future.result()
                sequences.extend(chunk_seqs)
                failed += chunk_failed


        if verbose:
            total_tokens = sum(len(s) for s in sequences)
            print(f"  {len(sequences)} valid sequences, {failed} failed")
            print(f"  Total tokens before BPE: {total_tokens:,}")

        # --- Phase 2: Learning BPE Merges (TURBO MODE + SMART SAMPLING) ---
        sample_ratio = getattr(self, 'sample_ratio', 1.0)
        if sample_ratio < 1.0:
            import random
            random.shuffle(sequences)
            num_samples = int(len(sequences) * sample_ratio)
            print(f"📊 Smart Sampling: Learning patterns from {num_samples}/{len(sequences)} sequences ({sample_ratio*100:.0f}%)...")
            sequences = sequences[:num_samples]

        next_token = self.base_vocab_size
        num_merges = target_vocab_size - self.base_vocab_size
        pbar = tqdm(range(num_merges), desc="Learning BPE Merges (Flash)")
        
        # Determine number of workers
        workers = getattr(self, 'workers', 1)
        
        # Pre-split sequences into chunks once
        seq_chunks = []
        c_size = math.ceil(len(sequences) / workers) if sequences else 1
        for i in range(0, len(sequences), c_size):
            seq_chunks.append(sequences[i:i + c_size])

        # KEEP WORKERS ALIVE FOR ALL MERGES (Massive Speedup)
        with ProcessPoolExecutor(max_workers=workers) as executor:
            for merge_idx in pbar:
                # Step A: Parallel pair counting
                futures = [executor.submit(self._get_pair_counts_chunk, chunk) for chunk in seq_chunks]
                pair_counts = Counter()
                for f in futures:
                    pair_counts.update(f.result())

                if not pair_counts:
                    break

                best_pair = pair_counts.most_common(1)[0]
                pair, count = best_pair

                if count < 2:
                    break

                # Record the merge
                self.merges.append(pair)
                self.merge_map[pair] = next_token
                self.decompose_map[next_token] = pair

                # Step B: Parallel merge application (using the same living workers)
                futures = [executor.submit(self._apply_merge_chunk, chunk, pair, next_token) for chunk in seq_chunks]
                seq_chunks = [f.result() for f in futures]
                
                # Update progress bar
                if (merge_idx + 1) % 10 == 0 or merge_idx < 100:
                    pbar.set_postfix({
                        'pair': f'({pair[0]},{pair[1]})',
                        'count': count
                    })
                
                next_token += 1

        # Re-flatten sequences
        sequences = [s for chunk in seq_chunks for s in chunk]

        self.vocab_size = next_token
        if verbose:
            total_tokens = sum(len(s) for s in sequences)
            print(f"\nBPE training complete!")
            print(f"  Final vocab size: {self.vocab_size}")
            print(f"  Total tokens after BPE: {total_tokens:,}")
            compression = sum(len(self.base_tokenizer.tokenize_midi(str(p))) for p in midi_paths[:10])
            compressed = sum(len(s) for s in sequences[:10])
            if compression > 0:
                print(f"  Compression ratio (sample): {compression/max(compressed,1):.2f}x")

    def encode(self, midi_path: str) -> List[int]:
        """Encode a MIDI file using BPE merges"""
        tokens = self.base_tokenizer.tokenize_midi(midi_path)
        return self.apply_merges(tokens)

    def apply_merges(self, tokens: List[int]) -> List[int]:
        """Apply learned BPE merges to a raw token sequence"""
        for pair, new_token in self.merge_map.items():
            new_tokens = []
            i = 0
            while i < len(tokens):
                if i < len(tokens) - 1 and tokens[i] == pair[0] and tokens[i + 1] == pair[1]:
                    new_tokens.append(new_token)
                    i += 2
                else:
                    new_tokens.append(tokens[i])
                    i += 1
            tokens = new_tokens
        return tokens

    def decode(self, bpe_tokens: List[int]) -> List[int]:
        """Decompose BPE tokens back to raw tokens"""
        raw = []
        for token in bpe_tokens:
            if token in self.decompose_map:
                # Recursively decompose
                stack = [token]
                while stack:
                    t = stack.pop()
                    if t in self.decompose_map:
                        a, b = self.decompose_map[t]
                        stack.append(b)  # Push second first (LIFO)
                        stack.append(a)
                    else:
                        raw.append(t)
            else:
                raw.append(token)
        return raw

    def decode_to_midi(self, bpe_tokens: List[int], output_path: str, time_scale: float = 1.0):
        """Decode BPE tokens back to a MIDI file"""
        raw_tokens = self.decode(bpe_tokens)
        return self.base_tokenizer.detokenize(raw_tokens, output_path, time_scale=time_scale)

    def estimate_duration_seconds(self, bpe_tokens: List[int]) -> float:
        """Estimate duration in seconds from TIME_SHIFT tokens"""
        raw_tokens = self.decode(bpe_tokens)
        return self.base_tokenizer.estimate_duration_seconds(raw_tokens)

    def pad_sequence(self, tokens: List[int], max_length: int) -> np.ndarray:
        """Pad or truncate to fixed length"""
        if len(tokens) >= max_length:
            return np.array(tokens[:max_length])
        padded = tokens + [self.PAD_TOKEN] * (max_length - len(tokens))
        return np.array(padded)

    def save(self, path: str):
        """Save the BPE tokenizer"""
        data = {
            "base_vocab_size": self.base_vocab_size,
            "vocab_size": self.vocab_size,
            "merges": self.merges,
            "merge_map": {f"{k[0]},{k[1]}": v for k, v in self.merge_map.items()},
            "decompose_map": {str(k): list(v) for k, v in self.decompose_map.items()},
        }
        os.makedirs(os.path.dirname(path), exist_ok=True)
        with open(path, 'w') as f:
            json.dump(data, f, indent=2)
        print(f"Saved BPE tokenizer to {path} (vocab_size={self.vocab_size})")

    @classmethod
    def load(cls, path: str) -> 'BPEMIDITokenizer':
        """Load a trained BPE tokenizer"""
        with open(path, 'r') as f:
            data = json.load(f)

        tokenizer = cls()
        tokenizer.base_vocab_size = data["base_vocab_size"]
        tokenizer.vocab_size = data["vocab_size"]
        tokenizer.merges = [tuple(m) for m in data["merges"]]
        tokenizer.merge_map = {
            tuple(int(x) for x in k.split(",")): v
            for k, v in data["merge_map"].items()
        }
        tokenizer.decompose_map = {
            int(k): tuple(v)
            for k, v in data["decompose_map"].items()
        }
        return tokenizer


def collect_midi_files(*dirs) -> List[Path]:
    """Collect all .mid files from one or more directories"""
    files = []
    for d in dirs:
        p = Path(d)
        if p.exists():
            files.extend(p.rglob("*.mid"))
    return sorted(set(files))


def main():
    parser = argparse.ArgumentParser(description="Train BPE tokenizer on MIDI corpus")
    parser.add_argument("--midi_dir", type=str, default="data/midi",
                        help="Primary MIDI directory (existing dataset)")
    parser.add_argument("--extra_midi_dir", type=str, default="data/midi_v2",
                        help="Extra MIDI directory (new processed dataset)")
    parser.add_argument("--output_path", type=str,
                        default="src/sekiro_ai/models/bpe_tokenizer.json")
    parser.add_argument("--vocab_size", type=int, default=8000,
                        help="Target BPE vocabulary size (optimized for credits)")
    parser.add_argument("--sample_ratio", type=float, default=0.2,
                        help="Percentage of sequences to use for BPE discovery (default 0.2)")
    parser.add_argument("--workers", type=int, default=42,
                        help="Number of CPU workers for parallel BPE")
    args = parser.parse_args()

    # Collect all MIDI files
    all_midis = collect_midi_files(args.midi_dir, args.extra_midi_dir)
    print(f"Found {len(all_midis)} MIDI files across all directories")

    if len(all_midis) == 0:
        print("ERROR: No MIDI files found. Check your --midi_dir paths.")
        return

    # Train BPE
    bpe = BPEMIDITokenizer()
    bpe.workers = args.workers
    bpe.sample_ratio = args.sample_ratio
    bpe.train(
        midi_paths=[str(f) for f in all_midis],
        target_vocab_size=args.vocab_size,
        verbose=True
    )

    # Save
    bpe.save(args.output_path)

    # Quick sanity check
    print("\n--- Sanity Check ---")
    sample = all_midis[0]
    raw_tokens = bpe.base_tokenizer.tokenize_midi(str(sample))
    bpe_tokens = bpe.encode(str(sample))
    decoded_raw = bpe.decode(bpe_tokens)
    print(f"File: {sample.name}")
    print(f"Raw tokens: {len(raw_tokens)}")
    print(f"BPE tokens: {len(bpe_tokens)} ({len(raw_tokens)/max(len(bpe_tokens),1):.2f}x compression)")
    print(f"Roundtrip check: {'PASS ✅' if decoded_raw == raw_tokens else 'FAIL ❌'}")


if __name__ == "__main__":
    main()
