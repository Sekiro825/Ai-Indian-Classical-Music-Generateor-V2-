"""
Loss Functions for Hybrid MIDI-Audio Training
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, Optional, List


def compute_kl_divergence(mu: torch.Tensor, logvar: torch.Tensor) -> torch.Tensor:
    """
    Compute KL divergence between learned distribution and standard normal.
    
    KL(q(z|x) || p(z)) = -0.5 * sum(1 + logvar - mu^2 - exp(logvar))
    """
    kl = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp(), dim=-1)
    return kl.mean()


class HybridLoss(nn.Module):
    """
    Combined loss for hybrid MIDI-Audio model.
    
    Loss = reconstruction_loss + kl_weight * kl_loss + expression_weight * expression_loss
    """
    
    def __init__(
        self,
        reconstruction_weight: float = 1.0,
        kl_weight: float = 0.005,
        expression_weight: float = 0.1,
        label_smoothing: float = 0.1,
        grammar_weight: float = 0.25,
        vivadi_penalty_multiplier: float = 10.0,
        vadi_reward_weight: float = 0.15,
        samvadi_reward_weight: float = 0.08,
        raga_rules_by_idx: Optional[Dict[int, Dict[str, List[int]]]] = None,
        note_on_offset: int = 3
    ):
        super().__init__()
        
        self.reconstruction_weight = reconstruction_weight
        self.kl_weight = kl_weight
        self.expression_weight = expression_weight
        self.grammar_weight = grammar_weight
        self.vivadi_penalty_multiplier = vivadi_penalty_multiplier
        self.vadi_reward_weight = vadi_reward_weight
        self.samvadi_reward_weight = samvadi_reward_weight
        self.raga_rules_by_idx = raga_rules_by_idx or {}
        self.note_on_offset = note_on_offset
        
        # Cross-entropy with label smoothing for token reconstruction
        self.ce_loss = nn.CrossEntropyLoss(
            ignore_index=0,  # Ignore padding token
            label_smoothing=label_smoothing
        )
        
        # MSE for expression features
        self.mse_loss = nn.MSELoss()
        
    def _pitch_class_logprob(
        self,
        logits: torch.Tensor,
        pitch_classes: List[int]
    ) -> torch.Tensor:
        if not pitch_classes:
            return torch.zeros(logits.size(0), device=logits.device)
        token_indices = [self.note_on_offset + pc + 12 * k for pc in pitch_classes for k in range(11)]
        token_indices = [idx for idx in token_indices if idx < logits.size(-1)]
        if not token_indices:
            return torch.zeros(logits.size(0), device=logits.device)
        probs = F.softmax(logits, dim=-1)[:, token_indices]
        summed = probs.sum(dim=-1).clamp(min=1e-9)
        return torch.log(summed)

    def _grammar_loss(
        self,
        logits: torch.Tensor,
        targets: torch.Tensor,
        ragas: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        if ragas is None or not self.raga_rules_by_idx:
            return torch.tensor(0.0, device=logits.device)

        B, T, V = logits.shape
        log_probs = F.log_softmax(logits, dim=-1)
        total = torch.tensor(0.0, device=logits.device)
        count = 0

        for b in range(B):
            rule = self.raga_rules_by_idx.get(int(ragas[b].item()), {})
            vivadi = rule.get("vivadi_pitch_classes", [])
            vadi = rule.get("vadi_pitch_classes", [])
            samvadi = rule.get("samvadi_pitch_classes", [])
            if not vivadi and not vadi and not samvadi:
                continue

            vivadi_tokens = [self.note_on_offset + pc + 12 * k for pc in vivadi for k in range(11)]
            vivadi_tokens = [idx for idx in vivadi_tokens if idx < V]

            sample_loss = torch.tensor(0.0, device=logits.device)
            if vivadi_tokens:
                vivadi_logprob = torch.logsumexp(log_probs[b, :, vivadi_tokens], dim=-1)
                note_on_mask = (targets[b] >= self.note_on_offset) & (targets[b] < self.note_on_offset + 128)
                if note_on_mask.any():
                    sample_loss = sample_loss + self.vivadi_penalty_multiplier * vivadi_logprob[note_on_mask].exp().mean()

            if vadi:
                vadi_bonus = self._pitch_class_logprob(logits[b], vadi).mean()
                sample_loss = sample_loss - self.vadi_reward_weight * vadi_bonus

            if samvadi:
                samvadi_bonus = self._pitch_class_logprob(logits[b], samvadi).mean()
                sample_loss = sample_loss - self.samvadi_reward_weight * samvadi_bonus

            total = total + sample_loss
            count += 1

        if count == 0:
            return torch.tensor(0.0, device=logits.device)
        return total / count
    
    def forward(
        self,
        outputs: Dict[str, torch.Tensor],
        targets: torch.Tensor,
        expression_targets: Optional[torch.Tensor] = None,
        ragas: Optional[torch.Tensor] = None,
        kl_weight_multiplier: float = 1.0
    ) -> Dict[str, torch.Tensor]:
        """
        Compute all losses.
        
        Args:
            outputs: Model outputs dict with 'logits', 'mu', 'logvar', 'predicted_expression'
            targets: (batch, seq_len) target token indices
            expression_targets: (batch, seq_len, 4) target expression features
            kl_weight_multiplier: For KL annealing (0 to 1)
            
        Returns:
            Dictionary with individual losses and total loss
        """
        losses = {}
        
        # 1. Token reconstruction loss
        logits = outputs['logits']
        # Flatten for cross-entropy
        B, T, V = logits.shape
        logits_flat = logits.contiguous().view(B * T, V)
        targets_flat = targets.contiguous().view(B * T)
        
        recon_loss = self.ce_loss(logits_flat, targets_flat)
        losses['reconstruction'] = recon_loss
        
        # 2. KL divergence loss
        mu = outputs['mu']
        logvar = outputs['logvar']
        kl_loss = compute_kl_divergence(mu, logvar)
        losses['kl'] = kl_loss
        
        # 3. Expression loss (if available)
        expr_loss = torch.tensor(0.0, device=logits.device)
        if expression_targets is not None and outputs.get('predicted_expression') is not None:
            pred_expr = outputs['predicted_expression']
            
            # Align lengths if necessary
            min_len = min(pred_expr.size(1), expression_targets.size(1))
            pred_expr = pred_expr[:, :min_len]
            expression_targets = expression_targets[:, :min_len]
            
            # MSE for continuous features (f0, amplitude, spectral_centroid)
            continuous_mask = [0, 1, 3]  # f0, amplitude, centroid
            mse = F.mse_loss(
                pred_expr[:, :, continuous_mask],
                expression_targets[:, :, continuous_mask]
            )
            
            # BCE for voiced flag (using with_logits for autocast safety)
            voiced_logits = pred_expr[:, :, 2]
            voiced_target = expression_targets[:, :, 2]
            bce = F.binary_cross_entropy_with_logits(voiced_logits, voiced_target)
            
            expr_loss = mse + 0.5 * bce
            
        losses['expression'] = expr_loss

        grammar_loss = self._grammar_loss(logits, targets, ragas)
        losses['grammar'] = grammar_loss
        
        # Total loss with annealing
        effective_kl_weight = self.kl_weight * kl_weight_multiplier
        
        total_loss = (
            self.reconstruction_weight * recon_loss +
            effective_kl_weight * kl_loss +
            self.expression_weight * expr_loss +
            self.grammar_weight * grammar_loss
        )
        
        losses['total'] = total_loss
        losses['effective_kl_weight'] = torch.tensor(effective_kl_weight)
        
        return losses


class SynthesizerLoss(nn.Module):
    """
    Loss for training the neural synthesizer.
    
    Uses multi-resolution STFT loss for waveform generation.
    """
    
    def __init__(
        self,
        fft_sizes: tuple = (512, 1024, 2048),
        hop_sizes: tuple = (128, 256, 512),
        win_sizes: tuple = (512, 1024, 2048)
    ):
        super().__init__()
        
        self.fft_sizes = fft_sizes
        self.hop_sizes = hop_sizes
        self.win_sizes = win_sizes
    
    def stft(self, x: torch.Tensor, fft_size: int, hop_size: int, win_size: int):
        """Compute STFT magnitude"""
        window = torch.hann_window(win_size, device=x.device)
        stft_out = torch.stft(
            x, fft_size, hop_size, win_size, window,
            return_complex=True
        )
        return stft_out.abs()
    
    def forward(
        self,
        pred_audio: torch.Tensor,
        target_audio: torch.Tensor
    ) -> Dict[str, torch.Tensor]:
        """
        Compute multi-resolution STFT loss.
        
        Args:
            pred_audio: (batch, samples) predicted waveform
            target_audio: (batch, samples) target waveform
            
        Returns:
            Dictionary with spectral loss, L1 loss, and total
        """
        losses = {}
        
        # L1 loss on waveform
        l1_loss = F.l1_loss(pred_audio, target_audio)
        losses['l1'] = l1_loss
        
        # Multi-resolution STFT loss
        spectral_loss = 0.0
        for fft_size, hop_size, win_size in zip(self.fft_sizes, self.hop_sizes, self.win_sizes):
            pred_stft = self.stft(pred_audio, fft_size, hop_size, win_size)
            target_stft = self.stft(target_audio, fft_size, hop_size, win_size)
            
            # Spectral convergence
            sc = torch.norm(target_stft - pred_stft, p='fro') / (torch.norm(target_stft, p='fro') + 1e-8)
            
            # Log magnitude L1
            log_pred = torch.log(pred_stft + 1e-8)
            log_target = torch.log(target_stft + 1e-8)
            lm = F.l1_loss(log_pred, log_target)
            
            spectral_loss += sc + lm
        
        spectral_loss /= len(self.fft_sizes)
        losses['spectral'] = spectral_loss
        
        # Total
        losses['total'] = l1_loss + spectral_loss
        
        return losses


if __name__ == "__main__":
    # Test losses
    batch_size = 4
    seq_len = 128
    vocab_size = 491
    
    # Test HybridLoss
    loss_fn = HybridLoss()
    
    outputs = {
        'logits': torch.randn(batch_size, seq_len, vocab_size),
        'mu': torch.randn(batch_size, 512),
        'logvar': torch.randn(batch_size, 512),
        'predicted_expression': torch.randn(batch_size, seq_len, 4)
    }
    
    targets = torch.randint(0, vocab_size, (batch_size, seq_len))
    expression_targets = torch.randn(batch_size, seq_len, 4)
    
    losses = loss_fn(outputs, targets, expression_targets, kl_weight_multiplier=0.5)
    
    print("HybridLoss:")
    for k, v in losses.items():
        print(f"  {k}: {v.item():.4f}")
    
    # Test SynthesizerLoss
    synth_loss_fn = SynthesizerLoss()
    
    pred_audio = torch.randn(batch_size, 22050)
    target_audio = torch.randn(batch_size, 22050)
    
    synth_losses = synth_loss_fn(pred_audio, target_audio)
    
    print("\nSynthesizerLoss:")
    for k, v in synth_losses.items():
        print(f"  {k}: {v.item():.4f}")
