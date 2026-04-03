"""
Training Script for Hybrid MIDI-Audio Model
Supports gradient accumulation, mixed precision, and KL annealing.
"""

import os
import sys
import json
import time
import argparse
from pathlib import Path
from typing import Dict, Optional

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torch.amp import autocast, GradScaler

# Add parent directories to path
sys.path.insert(0, str(Path(__file__).parent.parent))
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from sekiro_ai.hybrid.models.hybrid_cvae import HybridCVAE
from sekiro_ai.hybrid.config.hybrid_config import HybridCVAEConfig, TrainingConfig
from sekiro_ai.hybrid.data.hybrid_dataset import create_hybrid_dataloaders
from sekiro_ai.hybrid.training.losses import HybridLoss
from sekiro_ai.hybrid.musicology import get_raga_grammar

# Import tokenizer from main project
from sekiro_ai.models.tokenizer import MIDITokenizer


def get_cosine_schedule_with_warmup(
    optimizer: optim.Optimizer,
    num_warmup_steps: int,
    num_training_steps: int,
    min_lr_ratio: float = 0.1
):
    """Cosine learning rate schedule with linear warmup"""
    def lr_lambda(current_step: int) -> float:
        if current_step < num_warmup_steps:
            return float(current_step) / float(max(1, num_warmup_steps))
        progress = float(current_step - num_warmup_steps) / float(max(1, num_training_steps - num_warmup_steps))
        return max(min_lr_ratio, 0.5 * (1.0 + torch.cos(torch.tensor(progress * 3.14159)).item()))
    
    return optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)


class HybridTrainer:
    """
    Trainer for Hybrid CVAE model with expression conditioning.
    """
    
    def __init__(
        self,
        model: HybridCVAE,
        train_loader: DataLoader,
        val_loader: DataLoader,
        config: TrainingConfig,
        device: str = 'cuda'
    ):
        self.model = model.to(device)
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.config = config
        self.device = device
        
        # Optimizer
        self.optimizer = optim.AdamW(
            model.parameters(),
            lr=config.learning_rate,
            weight_decay=config.weight_decay
        )
        
        # Scheduler
        num_training_steps = len(train_loader) * config.num_epochs // config.gradient_accumulation_steps
        self.scheduler = get_cosine_schedule_with_warmup(
            self.optimizer,
            num_warmup_steps=config.warmup_steps,
            num_training_steps=num_training_steps
        )
        
        # Loss function
        self.loss_fn = HybridLoss(
            reconstruction_weight=config.reconstruction_weight,
            kl_weight=config.kl_weight,
            expression_weight=config.expression_weight,
            grammar_weight=config.grammar_weight,
            vivadi_penalty_multiplier=config.vivadi_penalty_multiplier,
            vadi_reward_weight=config.vadi_reward_weight,
            samvadi_reward_weight=config.samvadi_reward_weight
        )
        
        # Mixed precision
        self.scaler = GradScaler() if config.use_amp else None
        self.use_amp = config.use_amp
        
        # Checkpointing
        self.checkpoint_dir = Path(config.checkpoint_dir)
        self.checkpoint_dir.mkdir(parents=True, exist_ok=True)
        
        # Tracking
        self.global_step = 0
        self.best_val_loss = float('inf')
    
    def get_kl_weight(self, epoch: int) -> float:
        """KL annealing: gradually increase KL weight"""
        if epoch >= self.config.kl_annealing_epochs:
            return 1.0
        return epoch / self.config.kl_annealing_epochs
    
    def train_epoch(self, epoch: int) -> Dict[str, float]:
        """Train for one epoch"""
        self.model.train()
        
        total_losses = {
            'total': 0.0, 'reconstruction': 0.0, 
            'kl': 0.0, 'expression': 0.0, 'grammar': 0.0
        }
        num_batches = 0
        
        kl_weight = self.get_kl_weight(epoch)
        
        self.optimizer.zero_grad()
        
        for batch_idx, batch in enumerate(self.train_loader):
            # Move to device
            tokens = batch['tokens'].to(self.device)
            expression = batch['expression'].to(self.device)
            mood = batch['mood'].to(self.device)
            raga = batch['raga'].to(self.device)
            tempo = batch['tempo'].to(self.device)
            duration = batch['duration'].to(self.device)
            taal = batch['taal'].to(self.device)
            
            # Forward pass
            with autocast('cuda', enabled=self.use_amp):
                outputs = self.model(
                    tokens=tokens,
                    mood=mood,
                    raga=raga,
                    taal=taal,
                    tempo=tempo,
                    duration=duration,
                    expression=expression
                )
                
                # Compute loss (shift tokens for next-token prediction)
                target_tokens = tokens[:, 1:]
                output_logits = outputs['logits'][:, :-1]
                outputs['logits'] = output_logits
                
                losses = self.loss_fn(
                    outputs,
                    target_tokens,
                    expression[:, :-1],
                    ragas=raga,
                    kl_weight_multiplier=kl_weight
                )
                
                loss = losses['total'] / self.config.gradient_accumulation_steps
            
            # Backward pass
            if self.use_amp:
                self.scaler.scale(loss).backward()
            else:
                loss.backward()
            
            # Accumulate
            for k, v in losses.items():
                if k in total_losses:
                    total_losses[k] += v.item()
            num_batches += 1
            
            # Optimizer step
            if (batch_idx + 1) % self.config.gradient_accumulation_steps == 0:
                if self.use_amp:
                    self.scaler.unscale_(self.optimizer)
                    torch.nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)
                    self.scaler.step(self.optimizer)
                    self.scaler.update()
                else:
                    torch.nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)
                    self.optimizer.step()
                
                self.scheduler.step()
                self.optimizer.zero_grad()
                self.global_step += 1
            
            # Logging
            if (batch_idx + 1) % self.config.log_every_steps == 0:
                avg_loss = total_losses['total'] / num_batches
                print(f"  Step {batch_idx+1}/{len(self.train_loader)}, Loss: {avg_loss:.4f}")
        
        # Average losses
        for k in total_losses:
            total_losses[k] /= max(num_batches, 1)
        
        return total_losses
    
    @torch.no_grad()
    def validate(self) -> Dict[str, float]:
        """Validate the model"""
        self.model.eval()
        
        total_losses = {
            'total': 0.0, 'reconstruction': 0.0, 
            'kl': 0.0, 'expression': 0.0, 'grammar': 0.0
        }
        num_batches = 0
        
        for batch in self.val_loader:
            tokens = batch['tokens'].to(self.device)
            expression = batch['expression'].to(self.device)
            mood = batch['mood'].to(self.device)
            raga = batch['raga'].to(self.device)
            tempo = batch['tempo'].to(self.device)
            duration = batch['duration'].to(self.device)
            taal = batch['taal'].to(self.device)
            
            with autocast('cuda', enabled=self.use_amp):
                outputs = self.model(
                    tokens=tokens,
                    mood=mood,
                    raga=raga,
                    taal=taal,
                    tempo=tempo,
                    duration=duration,
                    expression=expression
                )
                
                target_tokens = tokens[:, 1:]
                outputs['logits'] = outputs['logits'][:, :-1]
                
                losses = self.loss_fn(
                    outputs,
                    target_tokens,
                    expression[:, :-1],
                    ragas=raga,
                    kl_weight_multiplier=1.0
                )
            
            for k, v in losses.items():
                if k in total_losses:
                    total_losses[k] += v.item()
            num_batches += 1
        
        for k in total_losses:
            total_losses[k] /= max(num_batches, 1)
        
        return total_losses
    
    def save_checkpoint(self, epoch: int, is_best: bool = False):
        """Save model checkpoint"""
        checkpoint = {
            'epoch': epoch,
            'global_step': self.global_step,
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'scheduler_state_dict': self.scheduler.state_dict(),
            'best_val_loss': self.best_val_loss,
            'config': self.model.config.__dict__
        }
        
        # Overwrite latest to prevent 300GB+ storage blowouts
        path = self.checkpoint_dir / 'checkpoint_latest.pt'
        torch.save(checkpoint, path)
        print(f"Saved checkpoint: {path}")
        
        if is_best:
            best_path = self.checkpoint_dir / 'best_model.pt'
            torch.save(checkpoint, best_path)
            print(f"Saved best model: {best_path}")

    def export_cpu_model(self):
        """
        Exports a stripped-down, INT8 dynamically quantized model
        for CPU inference on 16GB/32GB RAM systems.
        """
        print("Exporting INT8 Quantized CPU model...")
        import copy
        
        # Move model to CPU temporarily
        cpu_model = copy.deepcopy(self.model).cpu()
        cpu_model.eval()
        
        # Apply INT8 dynamic quantization to Linear layers
        quantized_model = torch.quantization.quantize_dynamic(
            cpu_model, 
            {torch.nn.Linear}, 
            dtype=torch.qint8
        )
        
        # Save ONLY the quantized weights and config, NO optimizer bloat
        cpu_checkpoint = {
            'model_state_dict': quantized_model.state_dict(),
            'config': self.model.config.__dict__
        }
        
        cpu_path = self.checkpoint_dir / 'cpu_best_model_int8.pt'
        torch.save(cpu_checkpoint, cpu_path)
        print(f"Saved highly compressed CPU inference model: {cpu_path}")
    
    def load_checkpoint(self, path: str):
        """Load model checkpoint"""
        checkpoint = torch.load(path, map_location=self.device, weights_only=False)
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        self.scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
        self.global_step = checkpoint['global_step']
        self.best_val_loss = checkpoint.get('best_val_loss', float('inf'))
        return checkpoint['epoch']
    
    def train(self, num_epochs: int, resume_from: Optional[str] = None):
        """Full training loop"""
        start_epoch = 0
        
        if resume_from:
            start_epoch = self.load_checkpoint(resume_from) + 1
            print(f"Resumed from epoch {start_epoch}")
        
        print(f"\n{'='*60}")
        print(f"Starting training for {num_epochs} epochs")
        print(f"Device: {self.device}")
        print(f"Model parameters: {self.model.get_num_params():,}")
        print(f"{'='*60}\n")
        
        for epoch in range(start_epoch, num_epochs):
            epoch_start = time.time()
            
            print(f"Epoch {epoch+1}/{num_epochs}")
            print("-" * 40)
            
            # Train
            train_losses = self.train_epoch(epoch)
            
            # Validate
            val_losses = self.validate()
            
            epoch_time = time.time() - epoch_start
            
            # Log
            print(f"\nEpoch {epoch+1} Summary (took {epoch_time:.1f}s):")
            print(f"  Train Loss: {train_losses['total']:.4f}")
            print(f"    Recon: {train_losses['reconstruction']:.4f}")
            print(f"    KL: {train_losses['kl']:.4f}")
            print(f"    Expr: {train_losses['expression']:.4f}")
            print(f"  Val Loss: {val_losses['total']:.4f}")
            
            # Checkpointing
            is_best = val_losses['total'] < self.best_val_loss
            if is_best:
                self.best_val_loss = val_losses['total']
            
            if (epoch + 1) % self.config.save_every_epochs == 0 or is_best:
                self.save_checkpoint(epoch, is_best)
            
            print()
        
        print("Training complete!")


def main():
    parser = argparse.ArgumentParser(description="Train Hybrid MIDI-Audio Model")
    parser.add_argument("--midi-dir", type=str, default="all_midi")
    parser.add_argument("--feature-dir", type=str, default="hybrid/features")
    parser.add_argument("--raga-metadata", type=str, default="config/raga_metadata.json")
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--resume", type=str, default=None)
    parser.add_argument("--device", type=str, default="cuda")
    
    args = parser.parse_args()
    
    # Check device
    device = args.device
    if device == "cuda" and not torch.cuda.is_available():
        print("CUDA not available, using CPU")
        device = "cpu"
    
    # Create tokenizer
    tokenizer = MIDITokenizer()
    
    # Create config
    model_config = HybridCVAEConfig()
    training_config = TrainingConfig(
        midi_dir=args.midi_dir,
        feature_cache_dir=args.feature_dir,
        raga_metadata_path=args.raga_metadata,
        batch_size=args.batch_size,
        learning_rate=args.lr,
        num_epochs=args.epochs
    )
    
    # Create dataloaders
    train_loader, val_loader, dataset = create_hybrid_dataloaders(
        midi_dir=args.midi_dir,
        feature_dir=args.feature_dir,
        raga_metadata_path=args.raga_metadata,
        tokenizer=tokenizer,
        batch_size=args.batch_size,
        max_seq_length=model_config.max_seq_length
    )
    
    # Update config with dataset info
    model_config.num_moods = len(dataset.mood_to_idx)
    model_config.num_ragas = len(dataset.raga_to_idx)
    model_config.num_taals = len(dataset.taal_to_idx)
    
    # Create model
    model = HybridCVAE(model_config)

    # Build grammar rules indexed by raga id for grammar-aware loss.
    grammar_rules_by_idx = {}
    for raga_name, raga_idx in dataset.raga_to_idx.items():
        grammar = get_raga_grammar(raga_name, dataset.raga_metadata.get(raga_name, {}))
        grammar_rules_by_idx[raga_idx] = {
            "vivadi_pitch_classes": sorted(grammar.vivadi_pitch_classes),
            "vadi_pitch_classes": sorted(grammar.vadi_pitch_classes),
            "samvadi_pitch_classes": sorted(grammar.samvadi_pitch_classes),
            "chalan_degrees": grammar.chalan_degrees,
        }
    
    # Create trainer
    trainer = HybridTrainer(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        config=training_config,
        device=device
    )
    trainer.loss_fn.raga_rules_by_idx = grammar_rules_by_idx
    trainer.loss_fn.note_on_offset = tokenizer.note_on_offset

    # Save vocab/conditioning mappings for inference convenience.
    vocabs_path = Path(training_config.checkpoint_dir) / "conditioning_vocabs.json"
    vocabs_path.parent.mkdir(parents=True, exist_ok=True)
    with open(vocabs_path, "w") as f:
        json.dump(
            {
                "raga_to_idx": dataset.raga_to_idx,
                "mood_to_idx": dataset.mood_to_idx,
                "taal_to_idx": dataset.taal_to_idx,
                "raga_rules_by_idx": grammar_rules_by_idx,
            },
            f,
            indent=2
        )
    print(f"Saved conditioning mappings: {vocabs_path}")
    
    # Train
    trainer.train(num_epochs=args.epochs, resume_from=args.resume)


if __name__ == "__main__":
    main()
