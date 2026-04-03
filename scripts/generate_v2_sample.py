"""
Generate a MIDI sample from a trained V2 Transformer-Flow checkpoint.

Example:
    python scripts/generate_v2_sample.py \
      --checkpoint checkpoints_v2_1b_calib10_fast/best.pt \
      --bpe_path models/tokenizer_v2/bpe_tokenizer.json \
      --vocab_path checkpoints_v2_1b_calib10_fast/vocabularies.json \
      --mood calm \
      --raga yaman \
      --taal trital \
      --tempo 90 \
      --duration 45 \
      --max_new_tokens 512 \
      --temperature 1.0 \
      --top_k 40 \
      --output outputs/v2_calib10_yaman.mid
"""

import argparse
import importlib.util
import json
import random
from pathlib import Path

import numpy as np
import torch

import sys
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from sekiro_ai.v2.transformer_flow_model import TransformerFlowModel
from sekiro_ai.hybrid.musicology import get_raga_grammar


def _load_bpe_tokenizer(bpe_path: str):
    bpe_script_path = Path(__file__).parent / "train_bpe_tokenizer.py"
    spec = importlib.util.spec_from_file_location("train_bpe_tokenizer", bpe_script_path)
    bpe_module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(bpe_module)
    BPEMIDITokenizer = bpe_module.BPEMIDITokenizer
    return BPEMIDITokenizer.load(bpe_path)


def _sample_next(
    logits: torch.Tensor,
    temperature: float,
    top_k: int,
    top_p: float,
    repetition_penalty: float,
    no_repeat_ngram_size: int,
    recent_tokens: list,
    ban_eos: bool = False,
    eos_token: int = 2
) -> int:
    logits = logits / max(temperature, 1e-6)

    if repetition_penalty and repetition_penalty != 1.0 and recent_tokens:
        for token_id in set(recent_tokens):
            if logits[..., token_id] > 0:
                logits[..., token_id] = logits[..., token_id] / repetition_penalty
            else:
                logits[..., token_id] = logits[..., token_id] * repetition_penalty

    if ban_eos:
        logits = logits.clone()
        logits[..., eos_token] = float("-inf")
    if no_repeat_ngram_size and no_repeat_ngram_size > 0 and len(recent_tokens) >= no_repeat_ngram_size - 1:
        n = no_repeat_ngram_size
        prefix = recent_tokens[-(n - 1):]
        banned = set()
        for i in range(len(recent_tokens) - n + 1):
            if recent_tokens[i:i + n - 1] == prefix:
                banned.add(recent_tokens[i + n - 1])
        if banned:
            logits = logits.clone()
            logits[..., list(banned)] = float("-inf")
    if top_k > 0:
        vals, idx = torch.topk(logits, k=min(top_k, logits.shape[-1]), dim=-1)
        probs = torch.softmax(vals, dim=-1)
        next_rel = torch.multinomial(probs, num_samples=1)
        next_tok = idx.gather(-1, next_rel)
    else:
        probs = torch.softmax(logits, dim=-1)
        if top_p and top_p < 1.0:
            sorted_probs, sorted_idx = torch.sort(probs, descending=True, dim=-1)
            cumulative = torch.cumsum(sorted_probs, dim=-1)
            mask = cumulative > top_p
            mask[..., 0] = False
            sorted_probs = sorted_probs.masked_fill(mask, 0.0)
            sorted_probs = sorted_probs / sorted_probs.sum(dim=-1, keepdim=True).clamp_min(1e-8)
            next_rel = torch.multinomial(sorted_probs, num_samples=1)
            next_tok = sorted_idx.gather(-1, next_rel)
        else:
            next_tok = torch.multinomial(probs, num_samples=1)
    return int(next_tok.item())


def _build_vivadi_token_ids(bpe, vivadi_pitch_classes):
    if not vivadi_pitch_classes:
        return []
    note_on_offset = bpe.base_tokenizer.note_on_offset

    def decompose_token(token_id):
        if token_id in bpe.decompose_map:
            stack = [token_id]
            raw = []
            while stack:
                t = stack.pop()
                if t in bpe.decompose_map:
                    a, b = bpe.decompose_map[t]
                    stack.append(b)
                    stack.append(a)
                else:
                    raw.append(t)
            return raw
        return [token_id]

    vivadi_ids = set()
    for tid in range(bpe.vocab_size):
        raw_tokens = decompose_token(tid)
        for rt in raw_tokens:
            if note_on_offset <= rt < note_on_offset + 128:
                pc = (rt - note_on_offset) % 12
                if pc in vivadi_pitch_classes:
                    vivadi_ids.add(tid)
                    break
    return sorted(list(vivadi_ids))


def _build_chalan_prefix(bpe, chalan_degrees):
    if not chalan_degrees:
        return []
    base = bpe.base_tokenizer
    base_pitch = 60
    velocity_token = base.velocity_offset + (base.config.num_velocity_bins // 2)
    time_shift_token = base.time_shift_offset + 10  # ~100ms
    raw = []
    for pc in chalan_degrees:
        pitch = max(0, min(127, base_pitch + int(pc)))
        raw.extend([
            velocity_token,
            base.note_on_offset + pitch,
            time_shift_token,
            base.note_off_offset + pitch,
            time_shift_token
        ])
    return bpe.apply_merges(raw)


def main():
    parser = argparse.ArgumentParser(description="Generate one sample from a V2 checkpoint")
    parser.add_argument("--checkpoint", required=True, help="Path to checkpoint (.pt)")
    parser.add_argument("--bpe_path", required=True, help="Path to BPE tokenizer json")
    parser.add_argument("--vocab_path", required=True, help="Path to vocabularies.json")
    parser.add_argument("--output", default="outputs/v2_sample.mid", help="Output MIDI path")
    parser.add_argument("--mood", default="calm")
    parser.add_argument("--raga", default="yaman")
    parser.add_argument("--taal", default="trital")
    parser.add_argument("--metadata", default="src/sekiro_ai/config/raga_metadata.json")
    parser.add_argument("--tempo", type=float, default=90.0)
    parser.add_argument("--duration", type=float, default=45.0)
    parser.add_argument("--max_new_tokens", type=int, default=512)
    parser.add_argument("--min_new_tokens", type=int, default=0)
    parser.add_argument("--temperature", type=float, default=1.0)
    parser.add_argument("--top_k", type=int, default=40)
    parser.add_argument("--top_p", type=float, default=1.0)
    parser.add_argument("--repetition_penalty", type=float, default=1.0)
    parser.add_argument("--no_repeat_ngram_size", type=int, default=0)
    parser.add_argument("--enforce_grammar", action="store_true")
    parser.add_argument("--use_chalan_prefix", action="store_true")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", default="auto", choices=["auto", "cuda", "cpu"])
    args = parser.parse_args()

    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)

    device = "cuda" if (args.device == "auto" and torch.cuda.is_available()) else args.device
    if device == "auto":
        device = "cpu"

    bpe = _load_bpe_tokenizer(args.bpe_path)
    with open(args.vocab_path, "r") as f:
        vocab = json.load(f)

    raga_meta = {}
    if Path(args.metadata).exists():
        with open(args.metadata, "r") as f:
            raga_meta = json.load(f)

    mood_idx = vocab.get("mood_to_idx", {}).get(args.mood, 0)
    raga_idx = vocab.get("raga_to_idx", {}).get(args.raga, 0)
    taal_to_idx = vocab.get("taal_to_idx", {})
    taal_idx = taal_to_idx.get(args.taal, taal_to_idx.get("unknown", 0))

    ckpt = torch.load(args.checkpoint, map_location=device, weights_only=False)
    config = ckpt["config"]

    model = TransformerFlowModel(config).to(device)
    model.load_state_dict(ckpt["model_state_dict"])
    model.eval()

    if args.max_new_tokens <= 0:
        raise ValueError("--max_new_tokens must be > 0")

    seq_cap = int(getattr(config, "max_seq_length", args.max_new_tokens + 1))
    tokens = [1]  # BOS

    grammar = get_raga_grammar(args.raga, raga_meta.get(args.raga, {}))
    vivadi_token_ids = _build_vivadi_token_ids(bpe, grammar.vivadi_pitch_classes) if args.enforce_grammar else []
    if args.use_chalan_prefix:
        tokens.extend(_build_chalan_prefix(bpe, grammar.chalan_degrees))

    mood_t = torch.tensor([mood_idx], device=device)
    raga_t = torch.tensor([raga_idx], device=device)
    taal_t = torch.tensor([taal_idx], device=device)
    tempo_t = torch.tensor([args.tempo], dtype=torch.float32, device=device)
    duration_t = torch.tensor([args.duration], dtype=torch.float32, device=device)

    taal_cycle_map = {
        "trital": 16, "ektal": 12, "jhaptal": 10, "rupak": 7, "roopak": 7,
        "dadra": 6, "keherwa": 8, "deepchandi": 14, "addhatrital": 8,
        "bhajani": 8, "adi": 8, "misra_chapu": 7, "khanda_chapu": 5, "unknown": 8,
    }
    taal_cycle = taal_cycle_map.get(args.taal, 8)

    with torch.no_grad():
        for _ in range(args.max_new_tokens):
            inp = torch.tensor([tokens], dtype=torch.long, device=device)
            out = model(
                inp,
                mood_t,
                raga_t,
                taal_t,
                tempo_t,
                duration_t,
                expression=None,
                taal_cycle_len=taal_cycle,
                flow_timestep=None,
                noisy_expression=None,
                padding_mask=(inp == 0),
            )
            ban_eos = args.min_new_tokens > 0 and len(tokens) < args.min_new_tokens
            logits = out["logits"][:, -1, :]
            if vivadi_token_ids:
                logits = logits.clone()
                logits[..., vivadi_token_ids] = float("-inf")
            next_token = _sample_next(
                logits,
                args.temperature,
                args.top_k,
                args.top_p,
                args.repetition_penalty,
                args.no_repeat_ngram_size,
                tokens,
                ban_eos=ban_eos
            )
            tokens.append(next_token)

            if next_token == 2:  # EOS
                break
            if len(tokens) >= seq_cap:
                break

    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    est_seconds = bpe.estimate_duration_seconds(tokens)
    if est_seconds > 0:
        time_scale = max(1.0, float(args.duration) / est_seconds)
    else:
        time_scale = 1.0
    bpe.decode_to_midi(tokens, str(output_path), time_scale=time_scale)

    print(f"Generated {len(tokens)} tokens")
    print(f"MIDI saved: {output_path}")


if __name__ == "__main__":
    main()
