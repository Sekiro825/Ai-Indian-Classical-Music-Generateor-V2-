"""
V2 Training Script: Transformer-Flow Indian Classical Music Generator

Joint training of:
1. Autoregressive BPE token prediction (cross-entropy)
2. Flow-matching for expression contours (MSE on velocity field)
3. Direct expression prediction (auxiliary, MSE)

Designed for Lightning.ai RTX PRO 6000 (96GB VRAM).

Usage:
    # Full training
    python scripts/train_v2_flow.py --midi_dir data/midi_v2 --epochs 150

    # Resume
    python scripts/train_v2_flow.py --midi_dir data/midi_v2 --epochs 150 --resume checkpoints_v2/latest.pt

    # Smoke test
    python scripts/train_v2_flow.py --midi_dir data/midi_v2 --epochs 1 --batch_size 2 --seq_length 256
"""

import os
import sys
import json
import time
import math
import argparse
from pathlib import Path
from datetime import datetime

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset, random_split
from torch.amp import autocast, GradScaler
from tqdm import tqdm
import numpy as np

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from sekiro_ai.v2.config import MambaFlowConfig
from sekiro_ai.v2.transformer_flow_model import TransformerFlowModel
from sekiro_ai.hybrid.musicology import get_raga_grammar


def get_cosine_schedule_with_warmup(optimizer, num_warmup_steps, num_training_steps, num_cycles=0.5):
    """
    Standard cosine schedule with warmup.
    """
    def lr_lambda(current_step):
        if current_step < num_warmup_steps:
            return float(current_step) / float(max(1, num_warmup_steps))
        progress = float(current_step - num_warmup_steps) / float(max(1, num_training_steps - num_warmup_steps))
        return max(0.0, 0.5 * (1.0 + math.cos(math.pi * float(num_cycles) * 2.0 * progress)))

    return optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)


def build_note_on_token_maps(bpe_tokenizer):
    """
    Build BPE token-id maps for NOTE_ON pitch classes.
    Returns:
      - note_on_token_ids: list of BPE token ids that contain any NOTE_ON
      - note_on_token_ids_by_pc: dict pc -> list of BPE token ids that include NOTE_ON of that pitch class
    """
    note_on_offset = bpe_tokenizer.base_tokenizer.note_on_offset
    vocab_size = bpe_tokenizer.vocab_size
    note_on_token_ids = set()
    note_on_token_ids_by_pc = {pc: set() for pc in range(12)}

    def decompose_token(token_id):
        if token_id in bpe_tokenizer.decompose_map:
            stack = [token_id]
            raw = []
            while stack:
                t = stack.pop()
                if t in bpe_tokenizer.decompose_map:
                    a, b = bpe_tokenizer.decompose_map[t]
                    stack.append(b)
                    stack.append(a)
                else:
                    raw.append(t)
            return raw
        return [token_id]

    for tid in range(vocab_size):
        raw_tokens = decompose_token(tid)
        for rt in raw_tokens:
            if note_on_offset <= rt < note_on_offset + 128:
                note_on_token_ids.add(tid)
                pc = (rt - note_on_offset) % 12
                note_on_token_ids_by_pc[pc].add(tid)

    return (
        sorted(note_on_token_ids),
        {pc: sorted(list(s)) for pc, s in note_on_token_ids_by_pc.items()}
    )


# ============================================================
# Dataset
# ============================================================

class V2RagaDataset(Dataset):
    """
    Dataset for V2 model: BPE tokens + aligned audio expression features.
    Reads from raga-organized directory: data/midi_v2/{tradition}/{raga}/

    Each MIDI file should have a companion .expr.npy file containing
    the aligned [f0, amplitude, voiced, spectral_centroid] features.
    """

    TAAL_MAP = {
        "addhatrital": 0, "trital": 1, "dadra": 2, "deepchandi": 3,
        "ektal": 4, "jhaptal": 5, "rupak": 6, "bhajani": 7,
        "keherwa": 8, "adi": 9, "misra_chapu": 10, "khanda_chapu": 11,
        "roopak": 12, "unknown": 13,
    }

    TAAL_CYCLES = {
        "trital": 16, "ektal": 12, "jhaptal": 10, "rupak": 7,
        "dadra": 6, "keherwa": 8, "deepchandi": 14, "addhatrital": 8,
        "bhajani": 8, "adi": 8, "misra_chapu": 7, "khanda_chapu": 5,
        "roopak": 7, "unknown": 8,
    }

    def __init__(self, midi_dir: str, raga_metadata_path: str,
                 bpe_tokenizer, max_seq_length: int = 4096,
                 augment: bool = False):
        self.midi_dir = Path(midi_dir)
        self.bpe = bpe_tokenizer
        self.max_seq_length = max_seq_length
        self.augment = augment

        with open(raga_metadata_path, 'r') as f:
            self.raga_metadata = json.load(f)

        self._scan()
        self._build_vocabs()

    def _scan(self):
        self.items = []  # (midi_path, expr_path_or_None, raga, tradition)

        for tradition_dir in self.midi_dir.iterdir():
            if not tradition_dir.is_dir():
                continue
            tradition = tradition_dir.name

            for raga_dir in tradition_dir.iterdir():
                if not raga_dir.is_dir():
                    continue
                raga = raga_dir.name

                for midi_file in raga_dir.glob("*.mid"):
                    # Check for companion expression file
                    expr_path = midi_file.with_suffix('.expr.npy')
                    self.items.append((
                        str(midi_file),
                        str(expr_path) if expr_path.exists() else None,
                        raga,
                        tradition
                    ))

        # Also scan flat directory (for existing data/midi/)
        flat_midis = list(self.midi_dir.glob("*.mid"))
        for midi_file in flat_midis:
            expr_path = midi_file.with_suffix('.expr.npy')
            raga = self._extract_raga(midi_file.name)
            self.items.append((
                str(midi_file), str(expr_path) if expr_path.exists() else None,
                raga, "unknown"
            ))

        print(f"V2Dataset: {len(self.items)} files, "
              f"{len(set(i[2] for i in self.items))} ragas")

    def _extract_raga(self, filename: str) -> str:
        name = filename.lower().replace('_basic_pitch', '').replace('.mid', '').replace('.mp3', '')
        patterns = [
            'yaman', 'bhairavi', 'malkauns', 'bageshree', 'bhoopali', 'bhoop',
            'asavari', 'sarang', 'darbari', 'dkanada', 'todi', 'khamaj',
            'marwa', 'puriya', 'kedar', 'bihag', 'jog', 'desh',
        ]
        for p in patterns:
            if p in name:
                return p
        if 'raag' in name or 'raga' in name:
            parts = name.replace('raag', ' ').replace('raga', ' ').split()
            for part in parts:
                clean = ''.join(c for c in part if c.isalpha())
                if len(clean) > 2:
                    return clean
        return 'unknown'

    def _build_vocabs(self):
        unique_ragas = sorted(set(i[2] for i in self.items))
        all_moods = set()
        for r in unique_ragas:
            info = self.raga_metadata.get(r, {})
            all_moods.update(info.get("moods", ["unknown"]))
        all_moods.add("unknown")

        self.mood_to_idx = {m: i for i, m in enumerate(sorted(all_moods))}
        self.raga_to_idx = {r: i for i, r in enumerate(unique_ragas)}
        self.raga_to_idx.setdefault("unknown", len(self.raga_to_idx))
        self.taal_to_idx = dict(self.TAAL_MAP)

        print(f"  Vocabs: {len(self.mood_to_idx)} moods, "
              f"{len(self.raga_to_idx)} ragas, {len(self.taal_to_idx)} taals")

    def __len__(self):
        return len(self.items)

    def __getitem__(self, idx):
        midi_path, expr_path, raga, tradition = self.items[idx]

        # Tokenize with BPE
        tokens = self.bpe.encode(midi_path)
        tokens = np.array(tokens)

        # Load expression features if available
        has_expression = False
        if expr_path is not None:
            try:
                expression = np.load(expr_path)  # (frames, 4)
                has_expression = True
            except Exception:
                expression = np.zeros((len(tokens), 4), dtype=np.float32)
        else:
            expression = np.zeros((len(tokens), 4), dtype=np.float32)

        # Augmentation: random crop for long sequences
        if self.augment and len(tokens) > self.max_seq_length:
            start = np.random.randint(0, len(tokens) - self.max_seq_length)
            tokens = tokens[start:start + self.max_seq_length]
            # Proportionally crop expression
            expr_len = len(expression)
            ratio = expr_len / max(len(self.bpe.encode(midi_path)), 1)
            expr_start = int(start * ratio)
            expr_end = expr_start + int(self.max_seq_length * ratio)
            expression = expression[expr_start:min(expr_end, expr_len)]

        # Pad/truncate tokens
        if len(tokens) >= self.max_seq_length:
            tokens = tokens[:self.max_seq_length]
        else:
            tokens = np.pad(tokens, (0, self.max_seq_length - len(tokens)))

        # Pad/truncate expression to match token length
        if len(expression) >= self.max_seq_length:
            expression = expression[:self.max_seq_length]
        else:
            pad_len = self.max_seq_length - len(expression)
            expression = np.pad(expression, ((0, pad_len), (0, 0)))

        # Metadata
        raga_info = self.raga_metadata.get(raga, {})
        moods = raga_info.get("moods", ["unknown"])
        mood = np.random.choice(moods) if self.augment and moods else moods[0]
        tempo_range = raga_info.get("tempo_range", [60, 120])
        tempo = np.random.randint(*tempo_range) if self.augment else sum(tempo_range) // 2

        # Determine taal and cycle length
        taal = "unknown"
        taal_cycle = 8
        for t_name, t_idx in self.TAAL_MAP.items():
            if t_name in raga.lower():
                taal = t_name
                taal_cycle = self.TAAL_CYCLES.get(t_name, 8)
                break

        padding_mask = tokens == 0

        return {
            "tokens": torch.tensor(tokens, dtype=torch.long),
            "expression": torch.tensor(expression, dtype=torch.float32),
            "has_expression": torch.tensor(has_expression, dtype=torch.bool),
            "mood": torch.tensor(self.mood_to_idx.get(mood, 0), dtype=torch.long),
            "raga": torch.tensor(self.raga_to_idx.get(raga, 0), dtype=torch.long),
            "taal": torch.tensor(self.taal_to_idx.get(taal, 13), dtype=torch.long),
            "taal_cycle": torch.tensor(taal_cycle, dtype=torch.long),
            "tempo": torch.tensor(tempo, dtype=torch.long),
            "duration": torch.tensor(300, dtype=torch.long),  # 5 min default
            "padding_mask": torch.tensor(padding_mask, dtype=torch.bool),
        }


class V2CachedDataset(Dataset):
    """
    High-speed dataset that loads pre-tokenized tensors from a cache file.
    Eliminates all MIDI parsing and BPE tokenization during the training loop.
    """
    TAAL_CYCLES = {
        "trital": 16, "ektal": 12, "jhaptal": 10, "rupak": 7,
        "dadra": 6, "keherwa": 8, "deepchandi": 14, "addhatrital": 8,
        "bhajani": 8, "adi": 8, "misra_chapu": 7, "khanda_chapu": 5,
        "roopak": 7, "unknown": 8,
    }

    def __init__(self, cache_path, max_seq_length=512):
        print(f"🚀 Loading high-speed dataset cache: {cache_path}...")
        self.data = torch.load(cache_path)
        self.max_seq_length = max_seq_length

        # Build reverse-engineered vocabs from cache
        self.moods = sorted(list(set(d['mood'] for d in self.data)))
        self.ragas = sorted(list(set(d['raga'] for d in self.data)))
        self.taals = sorted(list(set(d['taal'] for d in self.data)))

        self.mood_to_idx = {m: i for i, m in enumerate(self.moods)}
        self.raga_to_idx = {r: i for i, r in enumerate(self.ragas)}
        self.taal_to_idx = {t: i for i, t in enumerate(self.taals)}

        print(f"🚀 Cache Ready: {len(self.data)} sequences | "
              f"{len(self.moods)} moods, {len(self.ragas)} ragas")

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        d = self.data[idx]
        tokens = d['tokens'].long()
        expression = d.get('expression', None)
        has_expression = d.get('has_expression', False)

        # Truncate/Pad
        if len(tokens) > self.max_seq_length:
            tokens = tokens[:self.max_seq_length]
        else:
            tokens = F.pad(tokens, (0, self.max_seq_length - len(tokens)))

        # Metadata
        mood_idx = self.mood_to_idx.get(d['mood'], 0)
        raga_idx = self.raga_to_idx.get(d['raga'], 0)
        taal_idx = self.taal_to_idx.get(d['taal'], 0)
        taal_cycle = self.TAAL_CYCLES.get(d['taal'], 8)

        # Expression handling
        if expression is None:
            expression = torch.zeros((self.max_seq_length, 4), dtype=torch.float32)
            has_expression = False
        else:
            if not torch.is_tensor(expression):
                expression = torch.tensor(expression, dtype=torch.float32)
            # Align expression length with token length
            if expression.size(0) > self.max_seq_length:
                expression = expression[:self.max_seq_length]
            else:
                pad_len = self.max_seq_length - expression.size(0)
                if pad_len > 0:
                    expression = F.pad(expression, (0, 0, 0, pad_len))
            has_expression = bool(has_expression)

        return {
            "tokens": tokens,
            "expression": expression,
            "has_expression": torch.tensor(has_expression, dtype=torch.bool),
            "mood": torch.tensor(mood_idx, dtype=torch.long),
            "raga": torch.tensor(raga_idx, dtype=torch.long),
            "taal": torch.tensor(taal_idx, dtype=torch.long),
            "taal_cycle": torch.tensor(taal_cycle, dtype=torch.long),
            "tempo": torch.tensor(int(d.get('tempo', 90)), dtype=torch.long),
            "duration": torch.tensor(int(d.get('duration', 300)), dtype=torch.long),
            "padding_mask": tokens == 0,
        }


# ============================================================
# Cosine Schedule
# ============================================================

def get_cosine_schedule_with_warmup(optimizer, warmup_steps, total_steps, min_lr_ratio=0.1):
    def lr_lambda(step):
        if step < warmup_steps:
            return float(step) / float(max(1, warmup_steps))
        progress = float(step - warmup_steps) / float(max(1, total_steps - warmup_steps))
        return max(min_lr_ratio, 0.5 * (1.0 + math.cos(math.pi * progress)))
    return optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)


# ============================================================
# V2 Trainer
# ============================================================

class V2Trainer:
    """
    Trainer for V2 Transformer-Flow model.

    Three-loss training:
    1. Token loss (cross-entropy on BPE tokens — autoregressive)
    2. Flow loss (MSE on velocity field — teaches sharp expression)
    3. Expression loss (direct MSE — auxiliary / warm-start)
    """

    def __init__(self, model, train_loader, val_loader, optimizer,
                 scheduler=None, device='cuda', checkpoint_dir='checkpoints_v2',
                 grad_accum_steps=4, use_amp=True, config=None,
                 max_disk_gb=200.0, micro_batch_size=None,
                 checkpoint_every_optimizer_steps=20,
                 max_train_steps_per_epoch=None, max_val_steps=None,
                 grammar_rules_by_idx=None,
                 note_on_token_ids=None,
                 note_on_token_ids_by_pc=None):
        self.model = model.to(device)
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.optimizer = optimizer
        self.scheduler = scheduler
        self.device = device
        self.checkpoint_dir = Path(checkpoint_dir)
        self.checkpoint_dir.mkdir(parents=True, exist_ok=True)
        self.grad_accum_steps = grad_accum_steps
        self.config = config or model.config
        self.max_disk_gb = max_disk_gb
        self.micro_batch_size = micro_batch_size
        self.checkpoint_every_optimizer_steps = checkpoint_every_optimizer_steps
        self.max_train_steps_per_epoch = max_train_steps_per_epoch
        self.max_val_steps = max_val_steps
        self.grammar_rules_by_idx = grammar_rules_by_idx or {}
        self.note_on_token_ids = note_on_token_ids or []
        self.note_on_token_ids_by_pc = note_on_token_ids_by_pc or {pc: [] for pc in range(12)}
        self._note_on_token_ids_tensor = None

        self.amp_dtype = torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16
        self.scaler = GradScaler('cuda') if (use_amp and self.amp_dtype == torch.float16) else None
        self.use_amp = use_amp

        self.best_val_loss = float('inf')
        self.history = {
            'train_loss': [], 'train_token': [], 'train_flow': [], 'train_expr': [],
            'train_grammar': [],
            'val_loss': [], 'val_token': [], 'val_flow': [], 'val_grammar': [],
            'learning_rate': [], 'epoch_time': [],
        }
        self._resume_state = {'epoch': 0, 'next_batch_idx': 0, 'optimizer_step': 0}

    def _note_on_mask(self, targets: torch.Tensor) -> torch.Tensor:
        if not self.note_on_token_ids:
            return torch.zeros_like(targets, dtype=torch.bool)
        if self._note_on_token_ids_tensor is None or self._note_on_token_ids_tensor.device != targets.device:
            self._note_on_token_ids_tensor = torch.tensor(
                self.note_on_token_ids, dtype=targets.dtype, device=targets.device
            )
        return torch.isin(targets, self._note_on_token_ids_tensor)

    def _pitch_class_logprob(self, logits: torch.Tensor, pitch_classes):
        token_ids = []
        for pc in pitch_classes:
            token_ids.extend(self.note_on_token_ids_by_pc.get(int(pc) % 12, []))
        token_ids = [idx for idx in set(token_ids) if idx < logits.size(-1)]
        if not token_ids:
            return torch.zeros(logits.size(0), device=logits.device)
        probs = F.softmax(logits, dim=-1)[:, token_ids]
        summed = probs.sum(dim=-1).clamp(min=1e-9)
        return torch.log(summed)

    def _grammar_loss(self, logits: torch.Tensor, targets: torch.Tensor, ragas: torch.Tensor) -> torch.Tensor:
        if not self.grammar_rules_by_idx or not self.note_on_token_ids:
            return torch.tensor(0.0, device=logits.device)

        B, T, V = logits.shape
        log_probs = F.log_softmax(logits, dim=-1)
        total = torch.tensor(0.0, device=logits.device)
        count = 0

        for b in range(B):
            rule = self.grammar_rules_by_idx.get(int(ragas[b].item()), {})
            vivadi = rule.get("vivadi_pitch_classes", [])
            vadi = rule.get("vadi_pitch_classes", [])
            samvadi = rule.get("samvadi_pitch_classes", [])
            if not vivadi and not vadi and not samvadi:
                continue

            vivadi_ids = []
            for pc in vivadi:
                vivadi_ids.extend(self.note_on_token_ids_by_pc.get(int(pc) % 12, []))
            vivadi_ids = [idx for idx in set(vivadi_ids) if idx < V]

            note_on_mask = self._note_on_mask(targets[b])
            sample_loss = torch.tensor(0.0, device=logits.device)
            if vivadi_ids and note_on_mask.any():
                vivadi_logprob = torch.logsumexp(log_probs[b, :, vivadi_ids], dim=-1)
                sample_loss = sample_loss + self.config.vivadi_penalty_multiplier * vivadi_logprob[note_on_mask].exp().mean()

            if vadi:
                vadi_bonus = self._pitch_class_logprob(logits[b], vadi)
                if note_on_mask.any():
                    vadi_bonus = vadi_bonus[note_on_mask].mean()
                else:
                    vadi_bonus = vadi_bonus.mean()
                sample_loss = sample_loss - self.config.vadi_reward_weight * vadi_bonus

            if samvadi:
                samvadi_bonus = self._pitch_class_logprob(logits[b], samvadi)
                if note_on_mask.any():
                    samvadi_bonus = samvadi_bonus[note_on_mask].mean()
                else:
                    samvadi_bonus = samvadi_bonus.mean()
                sample_loss = sample_loss - self.config.samvadi_reward_weight * samvadi_bonus

            total = total + sample_loss
            count += 1

        if count == 0:
            return torch.tensor(0.0, device=logits.device)
        return total / count

    def _split_micro_batches(self, batch):
        """Yield micro-batches sliced along batch dimension for lower VRAM usage."""
        if not self.micro_batch_size or self.micro_batch_size <= 0:
            yield batch
            return

        batch_size = batch['tokens'].size(0)
        if self.micro_batch_size >= batch_size:
            yield batch
            return

        for start in range(0, batch_size, self.micro_batch_size):
            end = min(start + self.micro_batch_size, batch_size)
            micro = {}
            for key, value in batch.items():
                if torch.is_tensor(value) and value.dim() > 0 and value.size(0) == batch_size:
                    micro[key] = value[start:end]
                else:
                    micro[key] = value
            yield micro

    def compute_loss(self, batch):
        tokens = batch['tokens'].to(self.device)
        expression = batch['expression'].to(self.device)
        has_expression = batch['has_expression'].to(self.device)
        mood = batch['mood'].to(self.device)
        raga = batch['raga'].to(self.device)
        taal = batch['taal'].to(self.device)
        taal_cycle = batch['taal_cycle'][0].item()  # Assume same in batch
        tempo = batch['tempo'].to(self.device)
        duration = batch['duration'].to(self.device)
        padding_mask = batch['padding_mask'].to(self.device)

        B, L = tokens.shape

        # --- Flow-matching setup ---
        # Sample random timestep t ~ U(0, 1)
        t = torch.rand(B, device=self.device)

        # Create noisy expression: x_t = (1-t)*noise + t*data
        noise = torch.randn_like(expression)
        t_expand = t.view(B, 1, 1)
        noisy_expression = (1 - t_expand) * noise + t_expand * expression

        # Target velocity: v = data - noise
        target_velocity = expression - noise

        has_any_expression = bool(has_expression.any().item())

        # Forward pass
        outputs = self.model(
            tokens, mood, raga, taal, tempo, duration,
            expression=expression if has_any_expression else None,
            taal_cycle_len=taal_cycle,
            flow_timestep=t if has_any_expression else None,
            noisy_expression=noisy_expression if has_any_expression else None,
            padding_mask=padding_mask,
        )

        # --- Token loss (autoregressive, shifted) ---
        logits = outputs['logits'][:, :-1, :]        # Predict next token
        targets = tokens[:, 1:]                        # Shifted targets

        token_loss = F.cross_entropy(
            logits.contiguous().view(-1, logits.size(-1)),
            targets.contiguous().view(-1),
            ignore_index=0,  # PAD
            reduction='mean',
            label_smoothing=0.1,
        )

        # --- Flow loss (MSE on velocity field) ---
        flow_loss = torch.tensor(0.0, device=self.device)
        if 'flow_velocity' in outputs:
            pred_velocity = outputs['flow_velocity']
            # Only compute on samples that have real expression data
            if has_any_expression:
                mask = has_expression.view(B, 1, 1).float()
                flow_loss = F.mse_loss(pred_velocity * mask, target_velocity * mask)

        # --- Direct expression loss (auxiliary) ---
        expr_loss = torch.tensor(0.0, device=self.device)
        if 'predicted_expression' in outputs:
            pred_expr = outputs['predicted_expression']
            if has_any_expression:
                mask = has_expression.view(B, 1, 1).float()
                expr_loss = F.mse_loss(pred_expr * mask, expression * mask)

        # --- Grammar loss (vivadi penalty / vadi reward) ---
        grammar_loss = torch.tensor(0.0, device=self.device)
        if self.config.grammar_loss_weight > 0:
            grammar_loss = self._grammar_loss(logits, targets, raga)

        # Combined loss
        total_loss = (
            token_loss +
            self.config.flow_loss_weight * flow_loss +
            self.config.expression_loss_weight * expr_loss +
            self.config.grammar_loss_weight * grammar_loss
        )

        return total_loss, token_loss, flow_loss, expr_loss, grammar_loss

    def train_epoch(self, epoch, total_epochs, start_batch_idx=0):
        self.model.train()
        total_loss = total_token = total_flow = total_expr = total_grammar = 0.0
        num_batches = 0

        full_steps = max(1, len(self.train_loader) // self.grad_accum_steps)
        num_steps = full_steps
        if self.max_train_steps_per_epoch and self.max_train_steps_per_epoch > 0:
            num_steps = min(full_steps, self.max_train_steps_per_epoch)
        if start_batch_idx > 0:
            num_steps = max(1, num_steps - (start_batch_idx // self.grad_accum_steps))
        pbar = tqdm(total=num_steps, desc=f"Epoch {epoch+1}/{total_epochs}")
        self.optimizer.zero_grad()
        opt_steps = 0

        for batch_idx, batch in enumerate(self.train_loader):
            if batch_idx < start_batch_idx:
                continue
            full_batch_size = batch['tokens'].size(0)
            agg_loss = agg_token = agg_flow = agg_expr = agg_grammar = 0.0

            for micro in self._split_micro_batches(batch):
                micro_size = micro['tokens'].size(0)
                weight = micro_size / max(full_batch_size, 1)

                with autocast('cuda', dtype=self.amp_dtype, enabled=self.use_amp):
                    loss, token_loss, flow_loss, expr_loss, grammar_loss = self.compute_loss(micro)
                    scaled_loss = (loss * weight) / self.grad_accum_steps

                if self.scaler:
                    self.scaler.scale(scaled_loss).backward()
                else:
                    scaled_loss.backward()

                agg_loss += loss.item() * weight
                agg_token += token_loss.item() * weight
                agg_flow += flow_loss.item() * weight
                agg_expr += expr_loss.item() * weight
                agg_grammar += grammar_loss.item() * weight if isinstance(grammar_loss, torch.Tensor) else float(grammar_loss) * weight

            if (batch_idx + 1) % self.grad_accum_steps == 0:
                if self.scaler:
                    self.scaler.unscale_(self.optimizer)
                    torch.nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)
                    self.scaler.step(self.optimizer)
                    self.scaler.update()
                else:
                    torch.nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)
                    self.optimizer.step()
                self.optimizer.zero_grad()
                if self.scheduler:
                    self.scheduler.step()
                pbar.update(1)
                opt_steps += 1

                if (
                    self.checkpoint_every_optimizer_steps
                    and self.checkpoint_every_optimizer_steps > 0
                    and opt_steps % self.checkpoint_every_optimizer_steps == 0
                ):
                    self.save_checkpoint(
                        epoch,
                        is_best=False,
                        next_batch_idx=batch_idx + 1,
                        optimizer_step=opt_steps,
                    )

                pbar.set_postfix({
                    'loss': agg_loss,
                    'tok': agg_token,
                    'flow': agg_flow,
                    'lr': f"{self.optimizer.param_groups[0]['lr']:.2e}"
                })
                if opt_steps >= num_steps:
                    break

            total_loss += agg_loss
            total_token += agg_token
            total_flow += agg_flow
            total_expr += agg_expr
            total_grammar += agg_grammar
            num_batches += 1

            if batch_idx % 10 == 0:
                lr = self.optimizer.param_groups[0]['lr']
                pbar.set_postfix({
                    'loss': f'{agg_loss:.4f}',
                    'tok': f'{agg_token:.4f}',
                    'flow': f'{agg_flow:.4f}',
                    'lr': f'{lr:.2e}'
                })

        n = max(num_batches, 1)
        return {
            'loss': total_loss / n, 'token': total_token / n,
            'flow': total_flow / n, 'expr': total_expr / n,
            'grammar': total_grammar / n,
        }

    @torch.no_grad()
    def validate(self):
        self.model.eval()
        total_loss = total_token = total_flow = total_grammar = 0.0
        num_batches = 0

        val_total = len(self.val_loader)
        if self.max_val_steps and self.max_val_steps > 0:
            val_total = min(val_total, self.max_val_steps)

        for batch_idx, batch in enumerate(tqdm(self.val_loader, desc="Validation", total=val_total)):
            if self.max_val_steps and self.max_val_steps > 0 and batch_idx >= self.max_val_steps:
                break
            with autocast('cuda', dtype=self.amp_dtype, enabled=self.use_amp):
                loss, token_loss, flow_loss, _, grammar_loss = self.compute_loss(batch)
            total_loss += loss.item()
            total_token += token_loss.item()
            total_flow += flow_loss.item()
            total_grammar += grammar_loss.item() if isinstance(grammar_loss, torch.Tensor) else float(grammar_loss)
            num_batches += 1

        n = max(num_batches, 1)
        return {
            'loss': total_loss / n, 'token': total_token / n,
            'flow': total_flow / n, 'grammar': total_grammar / n
        }

    def _get_disk_usage_gb(self) -> float:
        """Get total disk usage of the project workspace in GB"""
        import shutil
        # Check the disk usage of the partition where the project lives
        workspace = Path('.').resolve()
        try:
            usage = shutil.disk_usage(workspace)
            used_gb = (usage.total - usage.free) / (1024**3)
            return used_gb
        except Exception:
            return 0.0

    def _get_checkpoint_size_gb(self) -> float:
        """Get total size of checkpoints directory in GB"""
        total = 0
        for f in self.checkpoint_dir.rglob('*'):
            if f.is_file():
                total += f.stat().st_size
        return total / (1024**3)

    def _prune_numbered_checkpoints(self, keep: int = 1):
        """Keep only the newest numbered checkpoints."""
        numbered = sorted(self.checkpoint_dir.glob('epoch_*.pt'))
        if keep < 0:
            keep = 0
        for old in numbered[:-keep] if keep else numbered:
            old.unlink()
            print(f"🗑️  Removed old checkpoint: {old.name}")

    def _enforce_checkpoint_budget(self):
        """Prune numbered checkpoints first so saves stay within the project budget."""
        safety_buffer_gb = 10.0
        if self._get_disk_usage_gb() >= max(0.0, self.max_disk_gb - safety_buffer_gb):
            self._prune_numbered_checkpoints(keep=0)

    def save_checkpoint(self, epoch, is_best=False, next_batch_idx=0, optimizer_step=0):
        # If disk is getting tight, remove numbered checkpoints before writing new files.
        self._enforce_checkpoint_budget()

        # ---- FULL checkpoint (latest.pt) — includes optimizer for resuming ----
        full_ckpt = {
            'epoch': epoch,
            'next_batch_idx': next_batch_idx,
            'optimizer_step': optimizer_step,
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'config': self.config,
            'history': self.history,
            'best_val_loss': self.best_val_loss,
        }
        if self.scheduler:
            full_ckpt['scheduler_state_dict'] = self.scheduler.state_dict()
        if self.scaler:
            full_ckpt['scaler_state_dict'] = self.scaler.state_dict()

        torch.save(full_ckpt, self.checkpoint_dir / 'latest.pt')

        # ---- LIGHTWEIGHT checkpoint (numbered) — weights only, no optimizer ----
        # Saves ~3x less disk (no Adam momentum/variance buffers)
        if (epoch + 1) % 25 == 0:  # Save every 25 epochs instead of 10
            light_ckpt = {
                'epoch': epoch,
                'next_batch_idx': next_batch_idx,
                'optimizer_step': optimizer_step,
                'model_state_dict': self.model.state_dict(),
                'config': self.config,
                'best_val_loss': self.best_val_loss,
            }
            torch.save(light_ckpt, self.checkpoint_dir / f'epoch_{epoch+1}.pt')

        if is_best:
            torch.save(full_ckpt, self.checkpoint_dir / 'best.pt')
            print(f"💾 Best model saved (val_loss={self.best_val_loss:.4f})")

        # ---- Aggressive cleanup: keep only 2 numbered checkpoints ----
        self._prune_numbered_checkpoints(keep=1)

        # ---- Disk usage monitoring ----
        ckpt_gb = self._get_checkpoint_size_gb()
        disk_gb = self._get_disk_usage_gb()
        print(f"💿 Checkpoints: {ckpt_gb:.1f}GB | Disk used: {disk_gb:.1f}GB / {self.max_disk_gb:.0f}GB limit")

        # ---- EMERGENCY CLEANUP: Trigger only when usage exceeds the configured limit ----
        if disk_gb > self.max_disk_gb:
            print(f"\n☢️  EMERGENCY DISK CLEANUP (Used: {disk_gb:.1f}GB / Limit: {self.max_disk_gb}GB)")
            self._prune_numbered_checkpoints(keep=0)
            
            # If still over limit, warn about latest.pt
            if self._get_disk_usage_gb() > self.max_disk_gb:
                print("⚠️  WARNING: Disk usage still exceeds limit even after cleanup!")
                print("   The model weights + optimizer states (latest.pt) may be too large for this partition.")

    def load_checkpoint(self, path):
        ckpt = torch.load(path, map_location=self.device, weights_only=False)
        self.model.load_state_dict(ckpt['model_state_dict'])
        self.optimizer.load_state_dict(ckpt['optimizer_state_dict'])
        if self.scheduler and 'scheduler_state_dict' in ckpt:
            self.scheduler.load_state_dict(ckpt['scheduler_state_dict'])
        if self.scaler and 'scaler_state_dict' in ckpt:
            self.scaler.load_state_dict(ckpt['scaler_state_dict'])
        self.history = ckpt.get('history', self.history)
        self.best_val_loss = ckpt.get('best_val_loss', float('inf'))
        self._resume_state = {
            'epoch': int(ckpt.get('epoch', 0)),
            'next_batch_idx': int(ckpt.get('next_batch_idx', 0)),
            'optimizer_step': int(ckpt.get('optimizer_step', 0)),
        }
        return self._resume_state

    def train(self, num_epochs, resume_from=None):
        start_epoch = 0
        start_batch_idx = 0
        if resume_from:
            resume_state = self.load_checkpoint(resume_from)
            start_epoch = resume_state['epoch']
            start_batch_idx = resume_state['next_batch_idx']
            print(f"▶ Resumed from epoch {start_epoch}, batch {start_batch_idx}")

        params = self.model.get_num_params()
        print(f"\n🎵 V2 Transformer-Flow Trainer")
        print(f"   Parameters: {params:,} ({params/1e9:.2f}B)")
        print(f"   Device: {self.device}")
        if self.device == 'cuda':
            print(f"   GPU: {torch.cuda.get_device_name(0)}")
            print(f"   VRAM: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f}GB")
        print(f"   AMP dtype: {self.amp_dtype}")
        print(f"   Grad Accum: {self.grad_accum_steps}")
        print(f"   Effective batch: {self.train_loader.batch_size * self.grad_accum_steps}")
        if self.micro_batch_size and self.micro_batch_size > 0:
            print(f"   Micro-batch size: {self.micro_batch_size}")
        if self.max_train_steps_per_epoch and self.max_train_steps_per_epoch > 0:
            print(f"   Max train steps/epoch: {self.max_train_steps_per_epoch}")
        if self.max_val_steps and self.max_val_steps > 0:
            print(f"   Max val steps/epoch: {self.max_val_steps}")
        print(f"   Seq length: {self.model.config.max_seq_length}")
        print(f"   Vocab size: {self.model.config.vocab_size}")
        print()

        for epoch in range(start_epoch, num_epochs):
            t0 = time.time()
            epoch_start_batch_idx = start_batch_idx if epoch == start_epoch else 0
            train_m = self.train_epoch(epoch, num_epochs, start_batch_idx=epoch_start_batch_idx)
            val_m = self.validate()
            elapsed = time.time() - t0

            self.history['train_loss'].append(train_m['loss'])
            self.history['train_token'].append(train_m['token'])
            self.history['train_flow'].append(train_m['flow'])
            self.history['train_expr'].append(train_m['expr'])
            self.history['train_grammar'].append(train_m.get('grammar', 0.0))
            self.history['val_loss'].append(val_m['loss'])
            self.history['val_token'].append(val_m['token'])
            self.history['val_flow'].append(val_m['flow'])
            self.history['val_grammar'].append(val_m.get('grammar', 0.0))
            self.history['learning_rate'].append(self.optimizer.param_groups[0]['lr'])
            self.history['epoch_time'].append(elapsed)

            is_best = val_m['loss'] < self.best_val_loss
            if is_best:
                self.best_val_loss = val_m['loss']
            self.save_checkpoint(epoch + 1, is_best, next_batch_idx=0, optimizer_step=0)

            print(f"\n✅ Epoch {epoch+1}/{num_epochs} ({elapsed:.0f}s)")
            print(f"   Train: loss={train_m['loss']:.4f} tok={train_m['token']:.4f} "
                  f"flow={train_m['flow']:.4f} expr={train_m['expr']:.4f} "
                  f"grammar={train_m.get('grammar', 0.0):.4f}")
            print(f"   Val:   loss={val_m['loss']:.4f} tok={val_m['token']:.4f} "
                  f"flow={val_m['flow']:.4f} grammar={val_m.get('grammar', 0.0):.4f}")
            start_batch_idx = 0

        with open(self.checkpoint_dir / 'history.json', 'w') as f:
            json.dump(self.history, f, indent=2)
        print("\n🎵 Training complete!")


# ============================================================
# Main
# ============================================================

def main():
    parser = argparse.ArgumentParser(description='Train V2 Transformer-Flow Model')
    parser.add_argument('--midi_dir', type=str, default='data/midi_v2')
    parser.add_argument('--metadata', type=str, default='src/sekiro_ai/config/raga_metadata.json')
    parser.add_argument('--bpe_path', type=str, default='models/tokenizer_v2/bpe_tokenizer.json')
    parser.add_argument('--checkpoint_dir', type=str, default='checkpoints_v2')
    parser.add_argument('--epochs', type=int, default=150)
    parser.add_argument('--batch_size', type=int, default=8)
    parser.add_argument('--grad_accum', type=int, default=8)
    parser.add_argument('--micro_batch_size', type=int, default=1,
                        help='If >0, split each loader batch into smaller micro-batches for lower VRAM.')
    parser.add_argument('--lr', type=float, default=3e-5)
    parser.add_argument('--warmup_epochs', type=int, default=10)
    parser.add_argument('--seq_length', type=int, default=None,
                        help='Override sequence length. If omitted, uses config max_seq_length.')
    parser.add_argument('--device', type=str, default='auto')
    parser.add_argument('--resume', type=str, default=None)
    parser.add_argument('--workers', type=int, default=4)
    parser.add_argument('--config', type=str, default=None,
                        help='Path to V2 config JSON (e.g., configs/1.0b_mamba_flow_budget.json)')
    parser.add_argument('--cache_pt', type=str, default=None,
                        help='Path to pre-tokenized cache .pt file')
    parser.add_argument('--max_disk_gb', type=float, default=200.0,
                        help='Max disk usage in GB before aggressive cleanup')
    parser.add_argument('--checkpoint_every_optimizer_steps', type=int, default=20,
                        help='If >0, save latest.pt every N optimizer steps for recoverability.')
    parser.add_argument('--max_train_steps_per_epoch', type=int, default=0,
                        help='If >0, cap optimizer steps per epoch (mini-epoch mode).')
    parser.add_argument('--max_val_steps', type=int, default=0,
                        help='If >0, cap validation batches per epoch.')
    args = parser.parse_args()

    device = 'cuda' if (args.device == 'auto' and torch.cuda.is_available()) else args.device
    if device == 'auto':
        device = 'cpu'
    print(f"Device: {device}")
    if device == 'cuda':
        # Throughput knobs (safe on RTX PRO 6000 / Ampere+)
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True
        torch.backends.cudnn.benchmark = True

        # Stability: avoid sporadic torch.compile + CUDA graph replay crashes.
        # Set SEKIRO_DISABLE_CUDAGRAPHS=0 to re-enable if you want to test later.
        if os.environ.get("SEKIRO_DISABLE_CUDAGRAPHS", "1") == "1":
            try:
                import torch._inductor.config as inductor_config
                inductor_config.triton.cudagraphs = False
                print("⚙️  Disabled torch inductor cudagraphs for stability.")
            except Exception:
                pass

        torch.set_float32_matmul_precision('high')
        torch.cuda.empty_cache()

    # Load BPE tokenizer
    # Load BPE tokenizer (Robust import)
    import importlib.util
    bpe_script_path = Path(__file__).parent / "train_bpe_tokenizer.py"
    spec = importlib.util.spec_from_file_location("train_bpe_tokenizer", bpe_script_path)
    bpe_module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(bpe_module)
    BPEMIDITokenizer = bpe_module.BPEMIDITokenizer

    if Path(args.bpe_path).exists():
        bpe = BPEMIDITokenizer.load(args.bpe_path)
        print(f"Loaded BPE tokenizer: vocab_size={bpe.vocab_size}")
    else:
        print(f"⚠ BPE tokenizer not found at {args.bpe_path}")
        print("  Run: python scripts/train_bpe_tokenizer.py first")
        print("  Falling back to base tokenizer (vocab=491)")
        bpe = BPEMIDITokenizer()

    # Load raga metadata (for grammar rules)
    raga_metadata = {}
    if Path(args.metadata).exists():
        with open(args.metadata, 'r') as f:
            raga_metadata = json.load(f)

    # Resolve sequence length before dataset init.
    config_from_file = None
    if args.config and Path(args.config).exists():
        print(f"Loading model config from {args.config}...")
        config_from_file = MambaFlowConfig.load(args.config)

    seq_length = args.seq_length
    if seq_length is None:
        seq_length = config_from_file.max_seq_length if config_from_file is not None else 512
    print(f"Sequence length: {seq_length}")

    # Dataset Choice
    if args.cache_pt and os.path.exists(args.cache_pt):
        dataset = V2CachedDataset(
            cache_path=args.cache_pt,
            max_seq_length=seq_length
        )
    else:
        # Fallback to standard MIDI dataset
        dataset = V2RagaDataset(
            midi_dir=args.midi_dir,
            raga_metadata_path=args.metadata,
            bpe_tokenizer=bpe,
            max_seq_length=seq_length,
            augment=True,
        )

    # Grammar rules + NOTE_ON maps (for grammar loss)
    grammar_rules_by_idx = {}
    for raga_name, raga_idx in dataset.raga_to_idx.items():
        grammar = get_raga_grammar(raga_name, raga_metadata.get(raga_name, {}))
        grammar_rules_by_idx[int(raga_idx)] = {
            "vivadi_pitch_classes": sorted(grammar.vivadi_pitch_classes),
            "vadi_pitch_classes": sorted(grammar.vadi_pitch_classes),
            "samvadi_pitch_classes": sorted(grammar.samvadi_pitch_classes),
        }
    note_on_token_ids, note_on_token_ids_by_pc = build_note_on_token_maps(bpe)

    train_size = int(0.9 * len(dataset))
    val_size = len(dataset) - train_size
    train_ds, val_ds = random_split(dataset, [train_size, val_size], generator=torch.Generator().manual_seed(42))

    loader_kwargs = {}
    if args.workers > 0:
        loader_kwargs = {
            "persistent_workers": True,
            "prefetch_factor": 4,
        }

    train_loader = DataLoader(
        train_ds,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.workers,
        pin_memory=True,
        drop_last=True,
        **loader_kwargs
    )
    val_loader = DataLoader(
        val_ds,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.workers,
        pin_memory=True,
        **loader_kwargs
    )

    # Model config
    if config_from_file is not None:
        config = config_from_file
        # Always override with runtime-determined values
        config.backbone = "transformer"
        config.vocab_size = bpe.vocab_size
        config.max_seq_length = seq_length
        config.num_moods = len(dataset.mood_to_idx)
        config.num_ragas = len(dataset.raga_to_idx)
    else:
        print("Using default model architecture (2.7B scale)...")
        config = MambaFlowConfig(
            vocab_size=bpe.vocab_size,
            max_seq_length=seq_length,
            d_model=2560,
            n_layers=64,
            num_moods=len(dataset.mood_to_idx),
            num_ragas=len(dataset.raga_to_idx),
        )
        config.backbone = "transformer"

    model = TransformerFlowModel(config)
    
    # Stability Optimization: High precision but NO torch.compile on Windows
    if device == 'cuda':
        print("⚡ Enabling Blackwell TensorCore & Memory-Efficient allocation...")
        # Prevent fragmentation OOMs
        os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"
        torch.set_float32_matmul_precision('high')
        torch.cuda.empty_cache()
    
    params = model.get_num_params()
    print(f"\n🎵 Model: {params:,} parameters ({params/1e9:.2f}B)")
    print(f"   Backbone: {config.backbone}")
    print(f"   BF16 size: {params*2/1e9:.1f}GB")

    # Save vocabs
    os.makedirs(args.checkpoint_dir, exist_ok=True)
    vocab_data = {
        'mood_to_idx': dataset.mood_to_idx,
        'raga_to_idx': dataset.raga_to_idx,
        'taal_to_idx': dataset.taal_to_idx,
    }
    with open(os.path.join(args.checkpoint_dir, 'vocabularies.json'), 'w') as f:
        json.dump(vocab_data, f, indent=2)

    # Optimizer (prefer 8-bit states on CUDA to reduce memory)
    optimizer = None
    if device == 'cuda':
        try:
            import bitsandbytes as bnb
            optimizer = bnb.optim.AdamW8bit(
                model.parameters(),
                lr=args.lr,
                weight_decay=0.01,
                betas=(0.9, 0.95),
            )
            print("Using optimizer: bitsandbytes AdamW8bit")
        except Exception as exc:
            print(f"⚠️  bitsandbytes unavailable ({exc}); falling back to torch.optim.AdamW")
    if optimizer is None:
        optimizer = optim.AdamW(
            model.parameters(),
            lr=args.lr,
            weight_decay=0.01,
            betas=(0.9, 0.95),
        )

    # Scheduler
    steps_per_epoch = max(1, len(train_loader) // args.grad_accum)
    if args.max_train_steps_per_epoch and args.max_train_steps_per_epoch > 0:
        steps_per_epoch = min(steps_per_epoch, args.max_train_steps_per_epoch)
    total_steps = steps_per_epoch * args.epochs
    warmup_steps = steps_per_epoch * args.warmup_epochs
    scheduler = get_cosine_schedule_with_warmup(optimizer, warmup_steps, total_steps)

    # Trainer
    trainer = V2Trainer(
        model=model, train_loader=train_loader, val_loader=val_loader,
        optimizer=optimizer, scheduler=scheduler, device=device,
        checkpoint_dir=args.checkpoint_dir, grad_accum_steps=args.grad_accum,
        use_amp=True, config=config, max_disk_gb=args.max_disk_gb,
        micro_batch_size=args.micro_batch_size,
        checkpoint_every_optimizer_steps=args.checkpoint_every_optimizer_steps,
        max_train_steps_per_epoch=(args.max_train_steps_per_epoch if args.max_train_steps_per_epoch > 0 else None),
        max_val_steps=(args.max_val_steps if args.max_val_steps > 0 else None),
        grammar_rules_by_idx=grammar_rules_by_idx,
        note_on_token_ids=note_on_token_ids,
        note_on_token_ids_by_pc=note_on_token_ids_by_pc,
    )

    trainer.train(num_epochs=args.epochs, resume_from=args.resume)


if __name__ == "__main__":
    main()
