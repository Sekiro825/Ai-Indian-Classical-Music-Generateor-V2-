"""
Training Script for 400M Raga CVAE Model
Designed to run on Lightning.ai with A100/L40 GPU
Supports gradient accumulation, mixed precision, and warmup scheduler
"""

import os
import sys
import json
import time
import argparse
from pathlib import Path
from datetime import datetime

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torch.cuda.amp import autocast, GradScaler
from tqdm import tqdm

# Add project src directory to path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from sekiro_ai.models.tokenizer import MIDITokenizer, TokenizerConfig
from sekiro_ai.models.transformer_cvae import RagaCVAE, CVAEConfig
from sekiro_ai.models.dataset import RagaDataset, create_dataloaders


def get_cosine_schedule_with_warmup(optimizer, num_warmup_steps, num_training_steps, min_lr_ratio=0.1):
    """Cosine learning rate schedule with linear warmup"""
    import math
    
    def lr_lambda(current_step):
        if current_step < num_warmup_steps:
            return float(current_step) / float(max(1, num_warmup_steps))
        progress = float(current_step - num_warmup_steps) / float(max(1, num_training_steps - num_warmup_steps))
        return max(min_lr_ratio, 0.5 * (1.0 + math.cos(math.pi * progress)))
    
    return optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)


class Trainer:
    """
    Trainer class for 400M Raga CVAE model with gradient accumulation and mixed precision
    """
    
    def __init__(
        self,
        model: RagaCVAE,
        train_loader: DataLoader,
        val_loader: DataLoader,
        optimizer: optim.Optimizer,
        scheduler: optim.lr_scheduler._LRScheduler = None,
        device: str = 'cuda',
        checkpoint_dir: str = 'checkpoints',
        log_interval: int = 10,
        kl_annealing_epochs: int = 50,
        grad_accum_steps: int = 1,
        use_amp: bool = False
    ):
        self.model = model.to(device)
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.optimizer = optimizer
        self.scheduler = scheduler
        self.device = device
        self.checkpoint_dir = Path(checkpoint_dir)
        self.checkpoint_dir.mkdir(parents=True, exist_ok=True)
        self.log_interval = log_interval
        self.kl_annealing_epochs = kl_annealing_epochs
        self.grad_accum_steps = grad_accum_steps
        self.use_amp = use_amp
        
        # Mixed precision scaler
        self.scaler = GradScaler() if use_amp else None
        
        self.best_val_loss = float('inf')
        self.history = {
            'train_loss': [],
            'train_recon': [],
            'train_kl': [],
            'val_loss': [],
            'val_recon': [],
            'val_kl': [],
            'learning_rate': []
        }
    
    def train_epoch(self, epoch: int, total_epochs: int) -> dict:
        """Train for one epoch with gradient accumulation and mixed precision"""
        self.model.train()
        
        total_loss = 0
        total_recon = 0
        total_kl = 0
        num_batches = 0
        
        # Cyclical KL annealing with longer warmup for large models
        cycle_length = self.kl_annealing_epochs
        cycle_position = epoch % cycle_length
        kl_weight = min(1.0, cycle_position / (cycle_length * 0.5)) * self.model.config.kl_weight
        kl_weight = max(kl_weight, self.model.config.kl_weight * 0.2)  # 20% minimum
        
        pbar = tqdm(self.train_loader, desc=f"Epoch {epoch+1}/{total_epochs}")
        
        self.optimizer.zero_grad()
        
        for batch_idx, batch in enumerate(pbar):
            # Move to device
            tokens = batch['tokens'].to(self.device)
            mood = batch['mood'].to(self.device)
            raga = batch['raga'].to(self.device)
            tempo = batch['tempo'].to(self.device)
            duration = batch['duration'].to(self.device)
            padding_mask = batch['padding_mask'].to(self.device)
            
            # Forward pass with mixed precision
            with autocast(enabled=self.use_amp):
                logits, mu, logvar = self.model(tokens, mood, raga, tempo, duration, padding_mask)
                
                # Compute loss with label smoothing
                recon_loss = nn.functional.cross_entropy(
                    logits.view(-1, self.model.config.vocab_size),
                    tokens.view(-1),
                    ignore_index=0,
                    reduction='mean',
                    label_smoothing=0.1
                )
                
                # KL divergence with free bits
                kl_per_dim = -0.5 * (1 + logvar - mu.pow(2) - logvar.exp())
                free_bits = 0.5
                kl_loss = torch.mean(torch.clamp(kl_per_dim, min=free_bits))
                
                loss = recon_loss + kl_weight * kl_loss
                loss = loss / self.grad_accum_steps  # Scale loss for accumulation
            
            # Backward pass with gradient scaling
            if self.use_amp:
                self.scaler.scale(loss).backward()
            else:
                loss.backward()
            
            # Gradient accumulation step
            if (batch_idx + 1) % self.grad_accum_steps == 0:
                if self.use_amp:
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
            
            # Accumulate metrics (multiply back for correct logging)
            total_loss += loss.item() * self.grad_accum_steps
            total_recon += recon_loss.item()
            total_kl += kl_loss.item()
            num_batches += 1
            
            # Update progress
            if batch_idx % self.log_interval == 0:
                current_lr = self.optimizer.param_groups[0]['lr']
                pbar.set_postfix({
                    'loss': f'{loss.item() * self.grad_accum_steps:.4f}',
                    'recon': f'{recon_loss.item():.4f}',
                    'kl': f'{kl_loss.item():.4f}',
                    'kl_w': f'{kl_weight:.4f}',
                    'lr': f'{current_lr:.2e}'
                })
        
        return {
            'loss': total_loss / num_batches,
            'recon': total_recon / num_batches,
            'kl': total_kl / num_batches
        }
    
    @torch.no_grad()
    def validate(self) -> dict:
        """Validate the model"""
        self.model.eval()
        
        total_loss = 0
        total_recon = 0
        total_kl = 0
        num_batches = 0
        
        for batch in tqdm(self.val_loader, desc="Validation"):
            tokens = batch['tokens'].to(self.device)
            mood = batch['mood'].to(self.device)
            raga = batch['raga'].to(self.device)
            tempo = batch['tempo'].to(self.device)
            duration = batch['duration'].to(self.device)
            padding_mask = batch['padding_mask'].to(self.device)
            
            with autocast(enabled=self.use_amp):
                logits, mu, logvar = self.model(tokens, mood, raga, tempo, duration, padding_mask)
                loss, recon_loss, kl_loss = self.model.loss(logits, tokens, mu, logvar, padding_mask)
            
            total_loss += loss.item()
            total_recon += recon_loss.item()
            total_kl += kl_loss.item()
            num_batches += 1
        
        return {
            'loss': total_loss / num_batches,
            'recon': total_recon / num_batches,
            'kl': total_kl / num_batches
        }
    
    def save_checkpoint(self, epoch: int, is_best: bool = False):
        """Save model checkpoint"""
        checkpoint = {
            'epoch': epoch,
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'config': self.model.config,
            'history': self.history,
            'best_val_loss': self.best_val_loss
        }
        
        if self.scheduler:
            checkpoint['scheduler_state_dict'] = self.scheduler.state_dict()
        
        if self.scaler:
            checkpoint['scaler_state_dict'] = self.scaler.state_dict()
        
        # Save regular checkpoint every 10 epochs
        if epoch % 10 == 0:
            path = self.checkpoint_dir / f'checkpoint_epoch_{epoch}.pt'
            torch.save(checkpoint, path)
        
        # Always save latest
        latest_path = self.checkpoint_dir / 'latest_checkpoint.pt'
        torch.save(checkpoint, latest_path)
        
        # Save best model
        if is_best:
            best_path = self.checkpoint_dir / 'best_model.pt'
            torch.save(checkpoint, best_path)
            print(f"💾 Saved best model with val_loss: {self.best_val_loss:.4f}")
        
        # Keep only last 3 numbered checkpoints
        checkpoints = sorted(self.checkpoint_dir.glob('checkpoint_epoch_*.pt'))
        for old_ckpt in checkpoints[:-3]:
            old_ckpt.unlink()
    
    def load_checkpoint(self, path: str):
        """Load model checkpoint"""
        checkpoint = torch.load(path, map_location=self.device, weights_only=False)
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        if self.scheduler and 'scheduler_state_dict' in checkpoint:
            self.scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
        if self.scaler and 'scaler_state_dict' in checkpoint:
            self.scaler.load_state_dict(checkpoint['scaler_state_dict'])
        self.history = checkpoint.get('history', self.history)
        self.best_val_loss = checkpoint.get('best_val_loss', float('inf'))
        return checkpoint['epoch']
    
    def train(self, num_epochs: int, resume_from: str = None):
        """Full training loop"""
        start_epoch = 0
        
        if resume_from:
            start_epoch = self.load_checkpoint(resume_from) + 1
            print(f"Resuming from epoch {start_epoch}")
        
        print(f"\n🎵 Starting training for {num_epochs} epochs")
        print(f"   Device: {self.device}")
        print(f"   Mixed Precision: {self.use_amp}")
        print(f"   Gradient Accumulation: {self.grad_accum_steps}")
        print(f"   Effective Batch Size: {self.train_loader.batch_size * self.grad_accum_steps}")
        print(f"   Train batches: {len(self.train_loader)}")
        print(f"   Val batches: {len(self.val_loader)}\n")
        
        for epoch in range(start_epoch, num_epochs):
            start_time = time.time()
            
            # Train
            train_metrics = self.train_epoch(epoch, num_epochs)
            self.history['train_loss'].append(train_metrics['loss'])
            self.history['train_recon'].append(train_metrics['recon'])
            self.history['train_kl'].append(train_metrics['kl'])
            self.history['learning_rate'].append(self.optimizer.param_groups[0]['lr'])
            
            # Validate
            val_metrics = self.validate()
            self.history['val_loss'].append(val_metrics['loss'])
            self.history['val_recon'].append(val_metrics['recon'])
            self.history['val_kl'].append(val_metrics['kl'])
            
            # Check for best model
            is_best = val_metrics['loss'] < self.best_val_loss
            if is_best:
                self.best_val_loss = val_metrics['loss']
            
            # Save checkpoint
            self.save_checkpoint(epoch, is_best)
            
            # Log
            elapsed = time.time() - start_time
            print(f"\n✅ Epoch {epoch+1}/{num_epochs} | Time: {elapsed:.1f}s")
            print(f"   Train Loss: {train_metrics['loss']:.4f} (recon: {train_metrics['recon']:.4f}, kl: {train_metrics['kl']:.4f})")
            print(f"   Val Loss: {val_metrics['loss']:.4f} (recon: {val_metrics['recon']:.4f}, kl: {val_metrics['kl']:.4f})")
            print(f"   Learning Rate: {self.optimizer.param_groups[0]['lr']:.2e}")
        
        # Save training history
        with open(self.checkpoint_dir / 'training_history.json', 'w') as f:
            json.dump(self.history, f, indent=2)
        
        print("\n🎵 Training complete!")
        return self.history


def main():
    parser = argparse.ArgumentParser(description='Train 400M Raga CVAE Model')
    parser.add_argument('--midi_dir', type=str, default='data/midi',
                        help='Directory containing MIDI files')
    parser.add_argument('--config_dir', type=str, default='src/sekiro_ai/config',
                        help='Directory containing config files')
    parser.add_argument('--checkpoint_dir', type=str, default='checkpoints',
                        help='Directory for saving checkpoints')
    parser.add_argument('--epochs', type=int, default=300,
                        help='Number of training epochs')
    parser.add_argument('--batch_size', type=int, default=4,
                        help='Batch size (small for large model)')
    parser.add_argument('--grad_accum', type=int, default=16,
                        help='Gradient accumulation steps (effective batch = batch_size * grad_accum)')
    parser.add_argument('--lr', type=float, default=1e-4,
                        help='Peak learning rate')
    parser.add_argument('--warmup_epochs', type=int, default=30,
                        help='Number of warmup epochs')
    parser.add_argument('--seq_length', type=int, default=1024,
                        help='Maximum sequence length')
    parser.add_argument('--device', type=str, default='auto',
                        help='Device (cuda/cpu/auto)')
    parser.add_argument('--resume', type=str, default=None,
                        help='Path to checkpoint to resume from')
    parser.add_argument('--use_amp', action='store_true',
                        help='Use automatic mixed precision (FP16)')
    parser.add_argument('--export', action='store_true',
                        help='Export model for CPU after training')
    parser.add_argument('--export_dir', type=str, default='exported_model',
                        help='Directory for exported model')
    args = parser.parse_args()
    
    # Set device
    if args.device == 'auto':
        device = 'cuda' if torch.cuda.is_available() else 'cpu'
    else:
        device = args.device
    print(f"Using device: {device}")
    
    if device == 'cuda':
        print(f"GPU: {torch.cuda.get_device_name(0)}")
        print(f"GPU Memory: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB")
    
    # Create tokenizer
    tokenizer = MIDITokenizer(TokenizerConfig(max_sequence_length=args.seq_length))
    
    # Create dataloaders
    train_loader, val_loader, dataset = create_dataloaders(
        midi_dir=args.midi_dir,
        raga_metadata_path=os.path.join(args.config_dir, 'raga_metadata.json'),
        tokenizer=tokenizer,
        batch_size=args.batch_size,
        max_seq_length=args.seq_length
    )
    
    # Save vocabulary mappings
    os.makedirs(args.checkpoint_dir, exist_ok=True)
    dataset.save_vocabularies(os.path.join(args.checkpoint_dir, 'vocabularies.json'))
    tokenizer.save(os.path.join(args.checkpoint_dir, 'tokenizer.json'))
    
    # Create 400M model
    config = CVAEConfig(
        vocab_size=tokenizer.vocab_size,
        max_seq_length=args.seq_length,
        num_moods=len(dataset.mood_to_idx),
        num_ragas=len(dataset.raga_to_idx),
        use_gradient_checkpointing=True  # Required for 400M model
    )
    model = RagaCVAE(config)
    
    params = sum(p.numel() for p in model.parameters())
    print(f"\n🎵 Model parameters: {params:,} ({params/1e6:.1f}M)")
    print(f"   Model size: {params * 4 / 1e9:.2f} GB (FP32)")
    
    # Calculate total training steps
    steps_per_epoch = len(train_loader) // args.grad_accum
    total_steps = steps_per_epoch * args.epochs
    warmup_steps = steps_per_epoch * args.warmup_epochs
    
    # Create optimizer with weight decay
    optimizer = optim.AdamW(
        model.parameters(), 
        lr=args.lr, 
        weight_decay=0.01,
        betas=(0.9, 0.98),
        eps=1e-8
    )
    
    # Create cosine scheduler with warmup
    scheduler = get_cosine_schedule_with_warmup(optimizer, warmup_steps, total_steps)
    
    # Create trainer
    trainer = Trainer(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        optimizer=optimizer,
        scheduler=scheduler,
        device=device,
        checkpoint_dir=args.checkpoint_dir,
        grad_accum_steps=args.grad_accum,
        use_amp=args.use_amp,
        kl_annealing_epochs=50
    )
    
    # Train
    trainer.train(num_epochs=args.epochs, resume_from=args.resume)
    
    # Auto-export for CPU after training
    if args.export:
        print("\n📦 Exporting model for CPU inference...")
        from export_model import export_for_cpu
        export_for_cpu(
            checkpoint_path=os.path.join(args.checkpoint_dir, 'best_model.pt'),
            output_dir=args.export_dir,
            optimize=False
        )
        print(f"✅ Model exported to: {args.export_dir}")


if __name__ == "__main__":
    main()
