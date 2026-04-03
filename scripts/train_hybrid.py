"""
Training Script for 2.5B HybridCVAE Model
Designed 1) for Lightning.ai with RTX PRO 6000 (96GB VRAM)
Supports: BF16 mixed precision, gradient checkpointing, gradient accumulation,
          cosine scheduler with warmup, checkpoint resume (for spot instance interrupts)

Usage:
    # Full training
    python scripts/train_hybrid.py --midi_dir data/midi_v2 --epochs 150 --batch_size 8

    # Resume after Lightning.ai interruption
    python scripts/train_hybrid.py --midi_dir data/midi_v2 --epochs 150 --resume checkpoints/latest_checkpoint.pt

    # Quick smoke test
    python scripts/train_hybrid.py --midi_dir data/midi_v2 --epochs 1 --batch_size 2
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

# Add project src directory to path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from sekiro_ai.models.tokenizer import MIDITokenizer, TokenizerConfig


# ============================================================
# Dataset for Raga-Organized MIDI Directory
# ============================================================

class HybridRagaDataset(Dataset):
    """
    Loads MIDIs from data/midi_v2/{tradition}/{raga}/*.mid organized structure.
    Supports taal labels and expression features.
    """

    # Known taal patterns
    TAAL_MAP = {
        "addhatrital": 0, "trital": 1, "dadra": 2, "deepchandi": 3,
        "ektal": 4, "jhaptal": 5, "rupak": 6, "bhajani": 7,
        "keherwa": 8, "adi": 9, "misra_chapu": 10, "khanda_chapu": 11,
        "roopak": 12, "unknown": 13,
    }

    def __init__(
        self,
        midi_dir: str,
        raga_metadata_path: str,
        tokenizer: MIDITokenizer,
        max_seq_length: int = 1024,
        augment: bool = False,
    ):
        self.midi_dir = Path(midi_dir)
        self.tokenizer = tokenizer
        self.max_seq_length = max_seq_length
        self.augment = augment
        self.token_cache = {}

        # Load raga metadata
        with open(raga_metadata_path, 'r') as f:
            self.raga_metadata = json.load(f)

        # Build vocabularies from actual directory structure
        self._scan_directory()
        self._build_vocabularies()

    def _scan_directory(self):
        """Scan the raga-organized directory to find all MIDI files"""
        self.midi_files = []
        self.file_ragas = []
        self.file_traditions = []

        for tradition_dir in self.midi_dir.iterdir():
            if not tradition_dir.is_dir():
                continue
            tradition = tradition_dir.name  # "carnatic" or "hindustani"

            for raga_dir in tradition_dir.iterdir():
                if not raga_dir.is_dir():
                    continue
                raga = raga_dir.name

                for midi_file in raga_dir.glob("*.mid"):
                    self.midi_files.append(str(midi_file))
                    self.file_ragas.append(raga)
                    self.file_traditions.append(tradition)

        print(f"Found {len(self.midi_files)} MIDI files across "
              f"{len(set(self.file_ragas))} ragas")

    def _build_vocabularies(self):
        """Build mood, raga, taal index mappings"""
        # Collect ragas from directory (ground truth)
        unique_ragas = sorted(set(self.file_ragas))

        # Collect moods from metadata
        all_moods = set()
        for raga in unique_ragas:
            raga_info = self.raga_metadata.get(raga, {})
            all_moods.update(raga_info.get("moods", []))
        all_moods.add("unknown")

        self.mood_to_idx = {m: i for i, m in enumerate(sorted(all_moods))}
        self.idx_to_mood = {i: m for m, i in self.mood_to_idx.items()}

        self.raga_to_idx = {r: i for i, r in enumerate(unique_ragas)}
        self.raga_to_idx["unknown"] = len(self.raga_to_idx)
        self.idx_to_raga = {i: r for r, i in self.raga_to_idx.items()}

        self.taal_to_idx = dict(self.TAAL_MAP)
        self.idx_to_taal = {i: t for t, i in self.taal_to_idx.items()}

        print(f"Vocabularies: {len(self.mood_to_idx)} moods, "
              f"{len(self.raga_to_idx)} ragas, {len(self.taal_to_idx)} taals")

    def __len__(self):
        return len(self.midi_files)

    def __getitem__(self, idx):
        midi_file = self.midi_files[idx]
        raga = self.file_ragas[idx]

        # Tokenize (with caching)
        if midi_file in self.token_cache:
            tokens = self.token_cache[midi_file].copy()
        else:
            tokens = self.tokenizer.tokenize_midi(midi_file)
            tokens = np.array(tokens)
            self.token_cache[midi_file] = tokens.copy()

        # Augmentation: random crop for long sequences
        if self.augment and len(tokens) > self.max_seq_length:
            max_start = len(tokens) - self.max_seq_length
            start = np.random.randint(0, max_start)
            tokens = tokens[start:start + self.max_seq_length]

        # Pad/truncate
        tokens = self.tokenizer.pad_sequence(list(tokens), self.max_seq_length)

        # Get metadata
        raga_info = self.raga_metadata.get(raga, {})
        moods = raga_info.get("moods", ["unknown"])
        mood = np.random.choice(moods) if self.augment and moods else moods[0] if moods else "unknown"

        tempo_range = raga_info.get("tempo_range", [60, 120])
        tempo = np.random.randint(*tempo_range) if self.augment else sum(tempo_range) // 2

        # Duration (arbitrary bin for now)
        duration = 60

        # Taal: check if raga is actually a taal pattern
        taal = "unknown"
        if raga in self.TAAL_MAP:
            taal = raga

        mood_idx = self.mood_to_idx.get(mood, self.mood_to_idx.get("unknown", 0))
        raga_idx = self.raga_to_idx.get(raga, self.raga_to_idx.get("unknown", 0))
        taal_idx = self.taal_to_idx.get(taal, self.taal_to_idx.get("unknown", 13))

        padding_mask = tokens == 0

        return {
            "tokens": torch.tensor(tokens, dtype=torch.long),
            "mood": torch.tensor(mood_idx, dtype=torch.long),
            "raga": torch.tensor(raga_idx, dtype=torch.long),
            "taal": torch.tensor(taal_idx, dtype=torch.long),
            "tempo": torch.tensor(tempo, dtype=torch.long),
            "duration": torch.tensor(duration, dtype=torch.long),
            "padding_mask": torch.tensor(padding_mask, dtype=torch.bool),
        }

    def save_vocabularies(self, path: str):
        vocab = {
            "mood_to_idx": self.mood_to_idx,
            "raga_to_idx": self.raga_to_idx,
            "taal_to_idx": self.taal_to_idx,
            "idx_to_mood": {str(k): v for k, v in self.idx_to_mood.items()},
            "idx_to_raga": {str(k): v for k, v in self.idx_to_raga.items()},
            "idx_to_taal": {str(k): v for k, v in self.idx_to_taal.items()},
        }
        with open(path, 'w') as f:
            json.dump(vocab, f, indent=2)


# ============================================================
# Cosine Schedule with Warmup
# ============================================================

def get_cosine_schedule_with_warmup(optimizer, warmup_steps, total_steps, min_lr_ratio=0.1):
    def lr_lambda(step):
        if step < warmup_steps:
            return float(step) / float(max(1, warmup_steps))
        progress = float(step - warmup_steps) / float(max(1, total_steps - warmup_steps))
        return max(min_lr_ratio, 0.5 * (1.0 + math.cos(math.pi * progress)))
    return optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)


# ============================================================
# Trainer
# ============================================================

class HybridTrainer:
    """
    Trainer for 2.5B HybridCVAE — optimized for RTX PRO 6000 (96GB).
    Supports BF16 mixed precision, gradient accumulation, checkpoint resume.
    """

    def __init__(
        self,
        model,
        train_loader,
        val_loader,
        optimizer,
        scheduler=None,
        device='cuda',
        checkpoint_dir='checkpoints',
        grad_accum_steps=4,
        use_amp=True,
        kl_annealing_epochs=50,
        expression_weight=0.5,
        grammar_weight=0.25,
        max_train_batches=0,
        max_val_batches=0,
    ):
        self.model = model.to(device)
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.optimizer = optimizer
        self.scheduler = scheduler
        self.device = device
        self.checkpoint_dir = Path(checkpoint_dir)
        self.checkpoint_dir.mkdir(parents=True, exist_ok=True)
        self.grad_accum_steps = grad_accum_steps
        self.use_amp = use_amp
        self.kl_annealing_epochs = kl_annealing_epochs
        self.expression_weight = expression_weight
        self.grammar_weight = grammar_weight
        self.max_train_batches = max(0, int(max_train_batches))
        self.max_val_batches = max(0, int(max_val_batches))

        # Use BF16 if available (Ampere+), else FP16
        self.amp_dtype = torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16
        self.scaler = GradScaler('cuda') if (use_amp and self.amp_dtype == torch.float16) else None

        self.best_val_loss = float('inf')
        self.history = {
            'train_loss': [], 'train_recon': [], 'train_kl': [],
            'val_loss': [], 'val_recon': [], 'val_kl': [],
            'learning_rate': [], 'epoch_time': [],
        }

    def compute_loss(self, batch, kl_weight):
        """Compute combined loss for HybridCVAE"""
        tokens = batch['tokens'].to(self.device)
        mood = batch['mood'].to(self.device)
        raga = batch['raga'].to(self.device)
        taal = batch['taal'].to(self.device)
        tempo = batch['tempo'].to(self.device)
        duration = batch['duration'].to(self.device)
        padding_mask = batch['padding_mask'].to(self.device)

        # Forward pass (no expression features for now — will add when extracted)
        outputs = self.model(tokens, mood, raga, taal, tempo, duration,
                             expression=None, padding_mask=padding_mask)

        logits = outputs['logits']
        mu = outputs['mu']
        logvar = outputs['logvar']

        # Reconstruction loss with label smoothing
        recon_loss = F.cross_entropy(
            logits.view(-1, logits.size(-1)),
            tokens.view(-1),
            ignore_index=0,
            reduction='mean',
            label_smoothing=0.1
        )

        # KL divergence with free bits
        kl_per_dim = -0.5 * (1 + logvar - mu.pow(2) - logvar.exp())
        free_bits = 0.5
        kl_loss = torch.mean(torch.clamp(kl_per_dim, min=free_bits))

        total_loss = recon_loss + kl_weight * kl_loss
        return total_loss, recon_loss, kl_loss

    def train_epoch(self, epoch, total_epochs):
        self.model.train()
        total_loss = 0.0
        total_recon = 0.0
        total_kl = 0.0
        num_batches = 0

        # Cyclical KL annealing
        cycle_len = self.kl_annealing_epochs
        cycle_pos = epoch % cycle_len
        kl_weight = min(1.0, cycle_pos / (cycle_len * 0.5)) * self.model.config.kl_weight
        kl_weight = max(kl_weight, self.model.config.kl_weight * 0.2)

        train_batches = len(self.train_loader)
        if self.max_train_batches > 0:
            train_batches = min(train_batches, self.max_train_batches)
        pbar = tqdm(enumerate(self.train_loader), total=train_batches, desc=f"Epoch {epoch+1}/{total_epochs}")
        self.optimizer.zero_grad()

        for batch_idx, batch in pbar:
            if batch_idx >= train_batches:
                break
            with autocast('cuda', dtype=self.amp_dtype, enabled=self.use_amp):
                loss, recon_loss, kl_loss = self.compute_loss(batch, kl_weight)
                loss = loss / self.grad_accum_steps

            if self.scaler:
                self.scaler.scale(loss).backward()
            else:
                loss.backward()

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

            total_loss += loss.item() * self.grad_accum_steps
            total_recon += recon_loss.item()
            total_kl += kl_loss.item()
            num_batches += 1

            if batch_idx % 10 == 0:
                lr = self.optimizer.param_groups[0]['lr']
                pbar.set_postfix({
                    'loss': f'{loss.item() * self.grad_accum_steps:.4f}',
                    'recon': f'{recon_loss.item():.4f}',
                    'kl': f'{kl_loss.item():.4f}',
                    'lr': f'{lr:.2e}'
                })

        return {
            'loss': total_loss / max(num_batches, 1),
            'recon': total_recon / max(num_batches, 1),
            'kl': total_kl / max(num_batches, 1),
        }

    @torch.no_grad()
    def validate(self):
        self.model.eval()
        total_loss = 0.0
        total_recon = 0.0
        total_kl = 0.0
        num_batches = 0
        kl_weight = self.model.config.kl_weight

        val_batches = len(self.val_loader)
        if self.max_val_batches > 0:
            val_batches = min(val_batches, self.max_val_batches)

        pbar = tqdm(enumerate(self.val_loader), total=val_batches, desc="Validation")
        for batch_idx, batch in pbar:
            if batch_idx >= val_batches:
                break
            with autocast('cuda', dtype=self.amp_dtype, enabled=self.use_amp):
                loss, recon_loss, kl_loss = self.compute_loss(batch, kl_weight)

            total_loss += loss.item()
            total_recon += recon_loss.item()
            total_kl += kl_loss.item()
            num_batches += 1

        return {
            'loss': total_loss / max(num_batches, 1),
            'recon': total_recon / max(num_batches, 1),
            'kl': total_kl / max(num_batches, 1),
        }

    def save_checkpoint(self, epoch, is_best=False):
        checkpoint = {
            'epoch': epoch,
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'config': self.model.config,
            'history': self.history,
            'best_val_loss': self.best_val_loss,
        }
        if self.scheduler:
            checkpoint['scheduler_state_dict'] = self.scheduler.state_dict()
        if self.scaler:
            checkpoint['scaler_state_dict'] = self.scaler.state_dict()

        # Always save latest (for interruption recovery)
        torch.save(checkpoint, self.checkpoint_dir / 'latest_checkpoint.pt')

        # Save numbered checkpoint every 10 epochs
        if (epoch + 1) % 10 == 0:
            torch.save(checkpoint, self.checkpoint_dir / f'checkpoint_epoch_{epoch+1}.pt')

        if is_best:
            torch.save(checkpoint, self.checkpoint_dir / 'best_model.pt')
            print(f"💾 Saved best model (val_loss: {self.best_val_loss:.4f})")

        # Cleanup old checkpoints (keep last 3)
        numbered = sorted(self.checkpoint_dir.glob('checkpoint_epoch_*.pt'))
        for old in numbered[:-3]:
            old.unlink()

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
        return ckpt['epoch']

    def train(self, num_epochs, resume_from=None):
        start_epoch = 0
        if resume_from:
            start_epoch = self.load_checkpoint(resume_from) + 1
            print(f"▶ Resumed from epoch {start_epoch}")

        print(f"\n🎵 Training 2.5B HybridCVAE")
        print(f"   Device: {self.device}")
        if self.device == 'cuda':
            print(f"   GPU: {torch.cuda.get_device_name(0)}")
            print(f"   VRAM: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB")
        print(f"   AMP: {self.amp_dtype}")
        print(f"   Grad Accum: {self.grad_accum_steps}")
        print(f"   Effective Batch: {self.train_loader.batch_size * self.grad_accum_steps}")
        print(f"   Train batches: {len(self.train_loader)}")
        print(f"   Val batches: {len(self.val_loader)}\n")
        if self.max_train_batches > 0 or self.max_val_batches > 0:
            train_cap = self.max_train_batches if self.max_train_batches > 0 else len(self.train_loader)
            val_cap = self.max_val_batches if self.max_val_batches > 0 else len(self.val_loader)
            print(f"   Batch caps: train={train_cap}, val={val_cap}\n")

        for epoch in range(start_epoch, num_epochs):
            t0 = time.time()
            train_metrics = self.train_epoch(epoch, num_epochs)
            val_metrics = self.validate()
            elapsed = time.time() - t0

            self.history['train_loss'].append(train_metrics['loss'])
            self.history['train_recon'].append(train_metrics['recon'])
            self.history['train_kl'].append(train_metrics['kl'])
            self.history['val_loss'].append(val_metrics['loss'])
            self.history['val_recon'].append(val_metrics['recon'])
            self.history['val_kl'].append(val_metrics['kl'])
            self.history['learning_rate'].append(self.optimizer.param_groups[0]['lr'])
            self.history['epoch_time'].append(elapsed)

            is_best = val_metrics['loss'] < self.best_val_loss
            if is_best:
                self.best_val_loss = val_metrics['loss']

            self.save_checkpoint(epoch, is_best)

            print(f"\n✅ Epoch {epoch+1}/{num_epochs} | {elapsed:.0f}s")
            print(f"   Train: loss={train_metrics['loss']:.4f} recon={train_metrics['recon']:.4f} kl={train_metrics['kl']:.4f}")
            print(f"   Val:   loss={val_metrics['loss']:.4f} recon={val_metrics['recon']:.4f} kl={val_metrics['kl']:.4f}")
            print(f"   LR: {self.optimizer.param_groups[0]['lr']:.2e}")

        # Save history
        with open(self.checkpoint_dir / 'training_history.json', 'w') as f:
            json.dump(self.history, f, indent=2)

        print("\n🎵 Training complete!")


# ============================================================
# Main
# ============================================================

def main():
    parser = argparse.ArgumentParser(description='Train 2.5B HybridCVAE Model')
    parser.add_argument('--midi_dir', type=str, default='data/midi_v2',
                        help='Raga-organized MIDI directory')
    parser.add_argument('--metadata', type=str, default='src/sekiro_ai/config/raga_metadata.json',
                        help='Raga metadata JSON')
    parser.add_argument('--checkpoint_dir', type=str, default='checkpoints')
    parser.add_argument('--epochs', type=int, default=150)
    parser.add_argument('--batch_size', type=int, default=8)
    parser.add_argument('--grad_accum', type=int, default=4)
    parser.add_argument('--lr', type=float, default=6e-5,
                        help='Peak learning rate (lower for 2.5B model)')
    parser.add_argument('--warmup_epochs', type=int, default=0)
    parser.add_argument('--seq_length', type=int, default=1024)
    parser.add_argument('--device', type=str, default='auto')
    parser.add_argument('--resume', type=str, default=None)
    parser.add_argument('--workers', type=int, default=4)
    parser.add_argument('--max_train_batches', type=int, default=0,
                        help='Limit train batches per epoch (0 = full epoch)')
    parser.add_argument('--max_val_batches', type=int, default=0,
                        help='Limit val batches per epoch (0 = full epoch)')
    args = parser.parse_args()

    # Device
    if args.device == 'auto':
        device = 'cuda' if torch.cuda.is_available() else 'cpu'
    else:
        device = args.device
    print(f"Using device: {device}")

    # Tokenizer
    tokenizer = MIDITokenizer(TokenizerConfig(max_sequence_length=args.seq_length))

    # Dataset
    full_dataset = HybridRagaDataset(
        midi_dir=args.midi_dir,
        raga_metadata_path=args.metadata,
        tokenizer=tokenizer,
        max_seq_length=args.seq_length,
        augment=True,
    )

    # Split 90/10
    train_size = int(0.9 * len(full_dataset))
    val_size = len(full_dataset) - train_size
    train_dataset, val_dataset = random_split(full_dataset, [train_size, val_size])

    train_loader = DataLoader(
        train_dataset, batch_size=args.batch_size, shuffle=True,
        num_workers=args.workers, pin_memory=True, drop_last=True
    )
    val_loader = DataLoader(
        val_dataset, batch_size=args.batch_size, shuffle=False,
        num_workers=args.workers, pin_memory=True
    )

    # Save vocabularies
    os.makedirs(args.checkpoint_dir, exist_ok=True)
    full_dataset.save_vocabularies(os.path.join(args.checkpoint_dir, 'vocabularies.json'))
    tokenizer.save(os.path.join(args.checkpoint_dir, 'tokenizer.json'))

    # Build model with 2.5B config
    from sekiro_ai.hybrid.config.hybrid_config import HybridCVAEConfig, ExpressionEncoderConfig
    from sekiro_ai.hybrid.models.hybrid_cvae import HybridCVAE

    config = HybridCVAEConfig(
        vocab_size=tokenizer.vocab_size,
        max_seq_length=args.seq_length,
        # 2.6B parameter config for RTX PRO 6000 (96GB)
        embed_dim=2048,
        num_heads=32,
        num_encoder_layers=24,
        num_decoder_layers=24,
        ff_dim=8192,
        latent_dim=1024,
        dropout=0.1,
        # Conditioning
        num_moods=len(full_dataset.mood_to_idx),
        num_ragas=len(full_dataset.raga_to_idx),
        num_taals=len(full_dataset.taal_to_idx),
        mood_embed_dim=128,
        raga_embed_dim=128,
        taal_embed_dim=64,
        tempo_embed_dim=64,
        duration_embed_dim=64,
        # VAE
        kl_weight=0.005,
        # Expression
        use_expression=True,
        expression_dim=128,
        expression_encoder_config=ExpressionEncoderConfig(
            input_dim=4,
            hidden_dim=256,
            embed_dim=128,
            num_layers=2,
            num_heads=4,
            dropout=0.1,
        ),
        expression_loss_weight=0.5,
        # Training
        use_gradient_checkpointing=True,
    )

    model = HybridCVAE(config)
    params = sum(p.numel() for p in model.parameters())
    print(f"\n🎵 Model parameters: {params:,} ({params/1e9:.2f}B)")
    print(f"   Model size (BF16): {params * 2 / 1e9:.2f} GB")
    print(f"   Model size (FP32): {params * 4 / 1e9:.2f} GB")

    # Optimizer
    optimizer = optim.AdamW(
        model.parameters(),
        lr=args.lr,
        weight_decay=0.01,
        betas=(0.9, 0.95),
        eps=1e-8,
    )

    # Scheduler
    effective_train_batches = len(train_loader)
    if args.max_train_batches > 0:
        effective_train_batches = min(effective_train_batches, args.max_train_batches)
    steps_per_epoch = max(1, effective_train_batches // args.grad_accum)
    total_steps = steps_per_epoch * args.epochs
    warmup_steps = steps_per_epoch * args.warmup_epochs
    scheduler = get_cosine_schedule_with_warmup(optimizer, warmup_steps, total_steps)

    # Trainer
    trainer = HybridTrainer(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        optimizer=optimizer,
        scheduler=scheduler,
        device=device,
        checkpoint_dir=args.checkpoint_dir,
        grad_accum_steps=args.grad_accum,
        use_amp=True,
        max_train_batches=args.max_train_batches,
        max_val_batches=args.max_val_batches,
    )

    trainer.train(num_epochs=args.epochs, resume_from=args.resume)


if __name__ == "__main__":
    main()
