"""
Neural Synthesizer for Hybrid MIDI-Audio Architecture
Converts MIDI tokens + expression features → audio waveform.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, Tuple
import numpy as np


class ResidualBlock(nn.Module):
    """Residual block with dilated convolutions for audio generation"""
    
    def __init__(
        self, 
        channels: int, 
        kernel_size: int = 3, 
        dilation: int = 1,
        dropout: float = 0.1
    ):
        super().__init__()
        padding = (kernel_size - 1) * dilation // 2
        
        self.conv1 = nn.Conv1d(channels, channels, kernel_size, padding=padding, dilation=dilation)
        self.conv2 = nn.Conv1d(channels, channels, kernel_size, padding=padding, dilation=dilation)
        self.norm1 = nn.GroupNorm(8, channels)
        self.norm2 = nn.GroupNorm(8, channels)
        self.dropout = nn.Dropout(dropout)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        residual = x
        x = self.norm1(x)
        x = F.gelu(x)
        x = self.conv1(x)
        x = self.norm2(x)
        x = F.gelu(x)
        x = self.dropout(x)
        x = self.conv2(x)
        return x + residual


class UpsampleBlock(nn.Module):
    """Upsampling block for audio generation"""
    
    def __init__(self, in_channels: int, out_channels: int, upsample_factor: int):
        super().__init__()
        self.upsample = nn.ConvTranspose1d(
            in_channels, out_channels, 
            kernel_size=upsample_factor * 2,
            stride=upsample_factor,
            padding=upsample_factor // 2
        )
        self.norm = nn.GroupNorm(8, out_channels)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.upsample(x)
        x = self.norm(x)
        return F.gelu(x)


class ExpressionConditioner(nn.Module):
    """Conditions the synthesis on expression features"""
    
    def __init__(self, expression_dim: int, hidden_dim: int):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(expression_dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, hidden_dim)
        )
    
    def forward(self, expression: torch.Tensor) -> torch.Tensor:
        """
        Args:
            expression: (batch, seq_len, expression_dim)
        Returns:
            (batch, hidden_dim, seq_len) conditioning signal
        """
        x = self.net(expression)  # (batch, seq_len, hidden_dim)
        return x.transpose(1, 2)  # (batch, hidden_dim, seq_len)


class NeuralSynthesizer(nn.Module):
    """
    Lightweight neural synthesizer for generating audio from MIDI + expression.
    
    Architecture:
    1. Encode MIDI events and expression features
    2. Upsample from frame rate to audio sample rate
    3. Generate waveform with dilated convolutions
    
    ~20M parameters, designed to run on consumer GPUs
    """
    
    def __init__(
        self,
        midi_vocab_size: int = 491,
        expression_dim: int = 4,
        hidden_dim: int = 256,
        num_upsample_blocks: int = 4,
        upsample_factors: Tuple[int, ...] = (8, 8, 4, 2),  # Total: 512x (hop_length)
        num_residual_blocks: int = 4,
        dropout: float = 0.1
    ):
        super().__init__()
        
        self.hidden_dim = hidden_dim
        self.hop_length = np.prod(upsample_factors)
        
        # MIDI embedding
        self.midi_embedding = nn.Embedding(midi_vocab_size, hidden_dim)
        
        # Expression conditioning
        self.expression_conditioner = ExpressionConditioner(expression_dim, hidden_dim)
        
        # Fusion layer
        self.fusion = nn.Sequential(
            nn.Conv1d(hidden_dim * 2, hidden_dim, 1),
            nn.GroupNorm(8, hidden_dim),
            nn.GELU()
        )
        
        # Pre-processing residual blocks
        self.pre_blocks = nn.ModuleList([
            ResidualBlock(hidden_dim, dilation=2**i, dropout=dropout)
            for i in range(num_residual_blocks)
        ])
        
        # Upsampling path
        channels = [hidden_dim]
        for i, factor in enumerate(upsample_factors):
            out_ch = hidden_dim // (2 ** (i + 1))
            out_ch = max(out_ch, 32)  # Minimum channels
            channels.append(out_ch)
        
        self.upsample_blocks = nn.ModuleList()
        self.upsample_residuals = nn.ModuleList()
        
        for i, factor in enumerate(upsample_factors):
            in_ch = channels[i]
            out_ch = channels[i + 1]
            self.upsample_blocks.append(UpsampleBlock(in_ch, out_ch, factor))
            self.upsample_residuals.append(
                ResidualBlock(out_ch, dropout=dropout)
            )
        
        # Output projection
        self.output_conv = nn.Sequential(
            nn.Conv1d(channels[-1], channels[-1], 7, padding=3),
            nn.GELU(),
            nn.Conv1d(channels[-1], 1, 1),
            nn.Tanh()  # Audio samples in [-1, 1]
        )
    
    def forward(
        self, 
        midi_tokens: torch.Tensor,
        expression: torch.Tensor,
        token_mask: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """
        Generate audio waveform from MIDI tokens and expression.
        
        Args:
            midi_tokens: (batch, seq_len) MIDI token indices
            expression: (batch, seq_len, 4) expression features
            token_mask: (batch, seq_len) mask for valid tokens
            
        Returns:
            audio: (batch, seq_len * hop_length) audio waveform
        """
        batch_size, seq_len = midi_tokens.shape
        
        # Embed MIDI tokens
        midi_emb = self.midi_embedding(midi_tokens)  # (batch, seq_len, hidden_dim)
        midi_emb = midi_emb.transpose(1, 2)  # (batch, hidden_dim, seq_len)
        
        # Condition on expression
        expr_cond = self.expression_conditioner(expression)  # (batch, hidden_dim, seq_len)
        
        # Fuse MIDI and expression
        x = torch.cat([midi_emb, expr_cond], dim=1)  # (batch, hidden_dim*2, seq_len)
        x = self.fusion(x)  # (batch, hidden_dim, seq_len)
        
        # Pre-processing
        for block in self.pre_blocks:
            x = block(x)
        
        # Upsample to audio rate
        for upsample, residual in zip(self.upsample_blocks, self.upsample_residuals):
            x = upsample(x)
            x = residual(x)
        
        # Generate audio
        audio = self.output_conv(x).squeeze(1)  # (batch, audio_length)
        
        return audio
    
    def get_num_params(self) -> int:
        return sum(p.numel() for p in self.parameters())


class SpectrogramSynthesizer(nn.Module):
    """
    Alternative synthesizer that generates mel spectrograms.
    Use with a vocoder (HiFi-GAN, WaveGlow) for final audio.
    
    Smaller and faster than waveform synthesis.
    """
    
    def __init__(
        self,
        midi_vocab_size: int = 491,
        expression_dim: int = 4,
        hidden_dim: int = 256,
        n_mels: int = 128,
        num_blocks: int = 4,
        dropout: float = 0.1
    ):
        super().__init__()
        
        self.n_mels = n_mels
        
        # MIDI encoder
        self.midi_embedding = nn.Embedding(midi_vocab_size, hidden_dim)
        
        # Expression encoder
        self.expression_encoder = nn.Sequential(
            nn.Linear(expression_dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, hidden_dim)
        )
        
        # Fusion
        self.fusion = nn.Linear(hidden_dim * 2, hidden_dim)
        
        # Transformer decoder
        decoder_layer = nn.TransformerEncoderLayer(
            d_model=hidden_dim,
            nhead=4,
            dim_feedforward=hidden_dim * 4,
            dropout=dropout,
            activation='gelu',
            batch_first=True
        )
        self.transformer = nn.TransformerEncoder(decoder_layer, num_layers=num_blocks)
        
        # Mel projection
        self.mel_proj = nn.Linear(hidden_dim, n_mels)
    
    def forward(
        self, 
        midi_tokens: torch.Tensor,
        expression: torch.Tensor
    ) -> torch.Tensor:
        """
        Generate mel spectrogram from MIDI + expression.
        
        Args:
            midi_tokens: (batch, seq_len)
            expression: (batch, seq_len, 4)
            
        Returns:
            mel: (batch, seq_len, n_mels) mel spectrogram
        """
        # Encode inputs
        midi_emb = self.midi_embedding(midi_tokens)
        expr_emb = self.expression_encoder(expression)
        
        # Fuse
        x = torch.cat([midi_emb, expr_emb], dim=-1)
        x = self.fusion(x)
        
        # Transform
        x = self.transformer(x)
        
        # Project to mel
        mel = self.mel_proj(x)
        
        return mel


if __name__ == "__main__":
    # Test NeuralSynthesizer
    print("Testing NeuralSynthesizer...")
    synth = NeuralSynthesizer()
    print(f"Parameters: {synth.get_num_params():,}")
    
    batch_size = 2
    seq_len = 128
    
    midi = torch.randint(0, 491, (batch_size, seq_len))
    expr = torch.randn(batch_size, seq_len, 4)
    
    audio = synth(midi, expr)
    print(f"Input tokens: {midi.shape}")
    print(f"Input expression: {expr.shape}")
    print(f"Output audio: {audio.shape}")
    print(f"Audio duration: {audio.shape[1] / 22050:.2f}s @ 22050Hz")
    
    # Test SpectrogramSynthesizer
    print("\nTesting SpectrogramSynthesizer...")
    spec_synth = SpectrogramSynthesizer()
    print(f"Parameters: {sum(p.numel() for p in spec_synth.parameters()):,}")
    
    mel = spec_synth(midi, expr)
    print(f"Output mel: {mel.shape}")
