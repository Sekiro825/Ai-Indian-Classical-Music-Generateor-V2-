"""
400M Parameter Transformer-based Conditional Variational Autoencoder for Raga Music Generation
Designed for training on Lightning.ai with A100/L40 GPU
"""

import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.checkpoint import checkpoint
from dataclasses import dataclass
from typing import Optional, Tuple


@dataclass
class CVAEConfig:
    """Configuration for 400M Parameter Conditional VAE model"""
    vocab_size: int = 491  # Must match tokenizer.vocab_size
    max_seq_length: int = 1024  # Longer sequences
    embed_dim: int = 1280  # Large embedding dimension
    num_heads: int = 20  # More attention heads
    num_encoder_layers: int = 20  # Deep encoder
    num_decoder_layers: int = 20  # Deep decoder
    ff_dim: int = 5120  # Large feedforward dimension
    latent_dim: int = 512  # Larger latent space
    dropout: float = 0.1
    
    # Conditioning dimensions (larger for better control)
    num_moods: int = 36
    num_ragas: int = 19
    mood_embed_dim: int = 128
    raga_embed_dim: int = 128
    tempo_embed_dim: int = 64
    duration_embed_dim: int = 64
    
    # Training settings
    kl_weight: float = 0.005  # Higher for larger model
    use_gradient_checkpointing: bool = True  # Required for 400M on 24GB


class RotaryPositionalEmbedding(nn.Module):
    """Rotary Positional Embedding (RoPE) for better length generalization"""
    
    def __init__(self, dim: int, max_seq_len: int = 2048, base: float = 10000.0):
        super().__init__()
        self.dim = dim
        self.max_seq_len = max_seq_len
        self.base = base
        
        # Compute inverse frequencies
        inv_freq = 1.0 / (base ** (torch.arange(0, dim, 2).float() / dim))
        self.register_buffer('inv_freq', inv_freq)
        
        # Precompute rotary embeddings
        self._build_cache(max_seq_len)
    
    def _build_cache(self, seq_len: int):
        t = torch.arange(seq_len, device=self.inv_freq.device)
        freqs = torch.einsum('i,j->ij', t, self.inv_freq)
        emb = torch.cat((freqs, freqs), dim=-1)
        self.register_buffer('cos_cached', emb.cos().unsqueeze(0))
        self.register_buffer('sin_cached', emb.sin().unsqueeze(0))
    
    def forward(self, x: torch.Tensor, seq_len: int = None) -> Tuple[torch.Tensor, torch.Tensor]:
        if seq_len is None:
            seq_len = x.size(1)
        
        if seq_len > self.cos_cached.size(1):
            self._build_cache(seq_len)
        
        return (
            self.cos_cached[:, :seq_len, :],
            self.sin_cached[:, :seq_len, :]
        )


def rotate_half(x: torch.Tensor) -> torch.Tensor:
    """Rotates half the hidden dims of the input."""
    x1 = x[..., :x.shape[-1] // 2]
    x2 = x[..., x.shape[-1] // 2:]
    return torch.cat((-x2, x1), dim=-1)


def apply_rotary_pos_emb(q: torch.Tensor, k: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
    """Apply rotary positional embedding to queries and keys."""
    q_embed = (q * cos) + (rotate_half(q) * sin)
    k_embed = (k * cos) + (rotate_half(k) * sin)
    return q_embed, k_embed


class RoPEMultiHeadAttention(nn.Module):
    """Multi-Head Attention with Rotary Positional Embeddings"""
    
    def __init__(self, embed_dim: int, num_heads: int, dropout: float = 0.1):
        super().__init__()
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.head_dim = embed_dim // num_heads
        
        self.q_proj = nn.Linear(embed_dim, embed_dim)
        self.k_proj = nn.Linear(embed_dim, embed_dim)
        self.v_proj = nn.Linear(embed_dim, embed_dim)
        self.out_proj = nn.Linear(embed_dim, embed_dim)
        
        self.dropout = nn.Dropout(dropout)
        self.rotary = RotaryPositionalEmbedding(self.head_dim)
    
    def forward(self, query: torch.Tensor, key: torch.Tensor, value: torch.Tensor,
                attn_mask: Optional[torch.Tensor] = None,
                key_padding_mask: Optional[torch.Tensor] = None,
                is_causal: bool = False) -> torch.Tensor:
        batch_size, seq_len, _ = query.shape
        
        # Project
        q = self.q_proj(query).view(batch_size, seq_len, self.num_heads, self.head_dim).transpose(1, 2)
        k = self.k_proj(key).view(batch_size, -1, self.num_heads, self.head_dim).transpose(1, 2)
        v = self.v_proj(value).view(batch_size, -1, self.num_heads, self.head_dim).transpose(1, 2)
        
        # Apply RoPE
        cos, sin = self.rotary(q, seq_len)
        cos = cos.unsqueeze(1)  # Add head dimension
        sin = sin.unsqueeze(1)
        q, k = apply_rotary_pos_emb(q, k, cos, sin)
        
        # Attention
        scale = math.sqrt(self.head_dim)
        attn_weights = torch.matmul(q, k.transpose(-2, -1)) / scale
        
        if is_causal:
            causal_mask = torch.triu(torch.ones(seq_len, seq_len, device=query.device), diagonal=1).bool()
            attn_weights = attn_weights.masked_fill(causal_mask, float('-inf'))
        
        if attn_mask is not None:
            attn_weights = attn_weights + attn_mask
        
        if key_padding_mask is not None:
            attn_weights = attn_weights.masked_fill(
                key_padding_mask.unsqueeze(1).unsqueeze(2), float('-inf')
            )
        
        attn_weights = F.softmax(attn_weights, dim=-1)
        attn_weights = self.dropout(attn_weights)
        
        output = torch.matmul(attn_weights, v)
        output = output.transpose(1, 2).contiguous().view(batch_size, seq_len, self.embed_dim)
        
        return self.out_proj(output)


class PreNormTransformerEncoderLayer(nn.Module):
    """Pre-LayerNorm Transformer Encoder Layer (more stable for deep models)"""
    
    def __init__(self, embed_dim: int, num_heads: int, ff_dim: int, dropout: float = 0.1):
        super().__init__()
        self.norm1 = nn.LayerNorm(embed_dim)
        self.attn = RoPEMultiHeadAttention(embed_dim, num_heads, dropout)
        self.norm2 = nn.LayerNorm(embed_dim)
        
        self.ffn = nn.Sequential(
            nn.Linear(embed_dim, ff_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(ff_dim, embed_dim),
            nn.Dropout(dropout)
        )
    
    def forward(self, x: torch.Tensor, padding_mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        # Pre-norm attention
        normed = self.norm1(x)
        x = x + self.attn(normed, normed, normed, key_padding_mask=padding_mask)
        
        # Pre-norm FFN
        x = x + self.ffn(self.norm2(x))
        return x


class PreNormTransformerDecoderLayer(nn.Module):
    """Pre-LayerNorm Transformer Decoder Layer with cross-attention"""
    
    def __init__(self, embed_dim: int, num_heads: int, ff_dim: int, dropout: float = 0.1):
        super().__init__()
        self.norm1 = nn.LayerNorm(embed_dim)
        self.self_attn = RoPEMultiHeadAttention(embed_dim, num_heads, dropout)
        
        self.norm2 = nn.LayerNorm(embed_dim)
        self.cross_attn = nn.MultiheadAttention(embed_dim, num_heads, dropout=dropout, batch_first=True)
        
        self.norm3 = nn.LayerNorm(embed_dim)
        self.ffn = nn.Sequential(
            nn.Linear(embed_dim, ff_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(ff_dim, embed_dim),
            nn.Dropout(dropout)
        )
    
    def forward(self, x: torch.Tensor, memory: torch.Tensor,
                tgt_padding_mask: Optional[torch.Tensor] = None,
                is_causal: bool = True) -> torch.Tensor:
        # Pre-norm self-attention (causal)
        normed = self.norm1(x)
        x = x + self.self_attn(normed, normed, normed, key_padding_mask=tgt_padding_mask, is_causal=is_causal)
        
        # Pre-norm cross-attention
        normed = self.norm2(x)
        cross_out, _ = self.cross_attn(normed, memory, memory)
        x = x + cross_out
        
        # Pre-norm FFN
        x = x + self.ffn(self.norm3(x))
        return x


class ConditioningModule(nn.Module):
    """Encodes conditioning inputs with larger embeddings"""
    
    def __init__(self, config: CVAEConfig):
        super().__init__()
        self.config = config
        
        self.mood_embedding = nn.Embedding(config.num_moods, config.mood_embed_dim)
        self.raga_embedding = nn.Embedding(config.num_ragas, config.raga_embed_dim)
        self.tempo_bins = nn.Embedding(32, config.tempo_embed_dim)
        self.duration_bins = nn.Embedding(16, config.duration_embed_dim)
        
        total_cond_dim = (config.mood_embed_dim + config.raga_embed_dim + 
                         config.tempo_embed_dim + config.duration_embed_dim)
        
        # Project to embedding dimension
        self.projection = nn.Sequential(
            nn.Linear(total_cond_dim, config.embed_dim),
            nn.GELU(),
            nn.LayerNorm(config.embed_dim)
        )
    
    def forward(self, mood: torch.Tensor, raga: torch.Tensor, 
                tempo: torch.Tensor, duration: torch.Tensor) -> torch.Tensor:
        mood_emb = self.mood_embedding(mood)
        raga_emb = self.raga_embedding(raga)
        tempo_emb = self.tempo_bins(tempo)
        duration_emb = self.duration_bins(duration)
        
        combined = torch.cat([mood_emb, raga_emb, tempo_emb, duration_emb], dim=-1)
        return self.projection(combined)


class TransformerEncoder(nn.Module):
    """Deep Transformer encoder with gradient checkpointing"""
    
    def __init__(self, config: CVAEConfig):
        super().__init__()
        self.config = config
        
        self.token_embedding = nn.Embedding(config.vocab_size, config.embed_dim)
        self.embed_scale = math.sqrt(config.embed_dim)
        
        self.layers = nn.ModuleList([
            PreNormTransformerEncoderLayer(
                config.embed_dim, config.num_heads, config.ff_dim, config.dropout
            ) for _ in range(config.num_encoder_layers)
        ])
        
        self.final_norm = nn.LayerNorm(config.embed_dim)
        
        # Latent space projection
        self.to_mu = nn.Linear(config.embed_dim, config.latent_dim)
        self.to_logvar = nn.Linear(config.embed_dim, config.latent_dim)
    
    def _forward_layer(self, layer, x, padding_mask):
        return layer(x, padding_mask)
    
    def forward(self, x: torch.Tensor, conditioning: torch.Tensor, 
                padding_mask: Optional[torch.Tensor] = None) -> Tuple[torch.Tensor, torch.Tensor]:
        # Embed tokens
        x = self.token_embedding(x) * self.embed_scale
        
        # Add conditioning as first token
        conditioning = conditioning.unsqueeze(1)
        x = torch.cat([conditioning, x[:, :-1]], dim=1)
        
        # Apply layers with gradient checkpointing
        for layer in self.layers:
            if self.config.use_gradient_checkpointing and self.training:
                x = checkpoint(self._forward_layer, layer, x, padding_mask, use_reentrant=False)
            else:
                x = layer(x, padding_mask)
        
        x = self.final_norm(x)
        
        # Pool (use mean of non-padded positions)
        if padding_mask is not None:
            mask = ~padding_mask.unsqueeze(-1)
            x = (x * mask).sum(dim=1) / mask.sum(dim=1).clamp(min=1)
        else:
            x = x.mean(dim=1)
        
        return self.to_mu(x), self.to_logvar(x)


class TransformerDecoder(nn.Module):
    """Deep Transformer decoder with cross-attention and gradient checkpointing"""
    
    def __init__(self, config: CVAEConfig):
        super().__init__()
        self.config = config
        
        self.token_embedding = nn.Embedding(config.vocab_size, config.embed_dim)
        self.embed_scale = math.sqrt(config.embed_dim)
        
        # Latent to embedding
        self.latent_projection = nn.Sequential(
            nn.Linear(config.latent_dim, config.embed_dim),
            nn.GELU(),
            nn.LayerNorm(config.embed_dim)
        )
        
        self.layers = nn.ModuleList([
            PreNormTransformerDecoderLayer(
                config.embed_dim, config.num_heads, config.ff_dim, config.dropout
            ) for _ in range(config.num_decoder_layers)
        ])
        
        self.final_norm = nn.LayerNorm(config.embed_dim)
        self.output_projection = nn.Linear(config.embed_dim, config.vocab_size)
    
    def _forward_layer(self, layer, x, memory, tgt_padding_mask, is_causal):
        return layer(x, memory, tgt_padding_mask, is_causal)
    
    def forward(self, z: torch.Tensor, conditioning: torch.Tensor,
                target: Optional[torch.Tensor] = None,
                target_padding_mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        # Project latent
        z_emb = self.latent_projection(z).unsqueeze(1)
        
        # Combine with conditioning for memory
        memory = torch.cat([z_emb, conditioning.unsqueeze(1)], dim=1)
        
        # Embed target tokens
        target_emb = self.token_embedding(target) * self.embed_scale
        
        # Apply layers
        x = target_emb
        for layer in self.layers:
            if self.config.use_gradient_checkpointing and self.training:
                x = checkpoint(self._forward_layer, layer, x, memory, target_padding_mask, True, use_reentrant=False)
            else:
                x = layer(x, memory, target_padding_mask, is_causal=True)
        
        x = self.final_norm(x)
        return self.output_projection(x)
    
    @torch.no_grad()
    def generate(self, z: torch.Tensor, conditioning: torch.Tensor,
                 max_length: int = 512, temperature: float = 1.0,
                 top_k: int = 50, top_p: float = 0.95,
                 bos_token: int = 1, eos_token: int = 2) -> torch.Tensor:
        """Autoregressive generation with nucleus sampling"""
        batch_size = z.size(0)
        device = z.device
        
        # Project latent and conditioning to memory
        z_emb = self.latent_projection(z).unsqueeze(1)
        memory = torch.cat([z_emb, conditioning.unsqueeze(1)], dim=1)
        
        # Start with BOS token
        generated = torch.full((batch_size, 1), bos_token, dtype=torch.long, device=device)
        finished = torch.zeros(batch_size, dtype=torch.bool, device=device)
        
        for _ in range(max_length - 1):
            # Embed current sequence
            target_emb = self.token_embedding(generated) * self.embed_scale
            
            # Apply decoder layers
            x = target_emb
            for layer in self.layers:
                x = layer(x, memory, is_causal=True)
            
            x = self.final_norm(x)
            logits = self.output_projection(x[:, -1, :])
            
            # Apply temperature
            logits = logits / temperature
            
            # Top-k filtering
            if top_k > 0:
                indices_to_remove = logits < torch.topk(logits, top_k)[0][..., -1, None]
                logits = logits.masked_fill(indices_to_remove, float('-inf'))
            
            # Top-p (nucleus) filtering
            if top_p < 1.0:
                sorted_logits, sorted_indices = torch.sort(logits, descending=True)
                cumulative_probs = torch.cumsum(F.softmax(sorted_logits, dim=-1), dim=-1)
                sorted_indices_to_remove = cumulative_probs > top_p
                sorted_indices_to_remove[..., 1:] = sorted_indices_to_remove[..., :-1].clone()
                sorted_indices_to_remove[..., 0] = 0
                indices_to_remove = sorted_indices_to_remove.scatter(1, sorted_indices, sorted_indices_to_remove)
                logits = logits.masked_fill(indices_to_remove, float('-inf'))
            
            # Sample
            probs = F.softmax(logits, dim=-1)
            next_token = torch.multinomial(probs, 1)
            
            # Update finished sequences
            finished = finished | (next_token.squeeze(-1) == eos_token)
            
            # Append token
            generated = torch.cat([generated, next_token], dim=1)
            
            if finished.all():
                break
        
        return generated


class RagaCVAE(nn.Module):
    """400M Parameter Conditional Variational Autoencoder for Raga Music Generation"""
    
    def __init__(self, config: CVAEConfig = None):
        super().__init__()
        self.config = config or CVAEConfig()
        
        self.conditioning = ConditioningModule(self.config)
        self.encoder = TransformerEncoder(self.config)
        self.decoder = TransformerDecoder(self.config)
    
    def reparameterize(self, mu: torch.Tensor, logvar: torch.Tensor) -> torch.Tensor:
        if self.training:
            std = torch.exp(0.5 * logvar)
            eps = torch.randn_like(std)
            return mu + eps * std
        return mu
    
    def forward(self, x: torch.Tensor, mood: torch.Tensor, raga: torch.Tensor,
                tempo: torch.Tensor, duration: torch.Tensor,
                padding_mask: Optional[torch.Tensor] = None) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        cond = self.conditioning(mood, raga, tempo, duration)
        mu, logvar = self.encoder(x, cond, padding_mask)
        z = self.reparameterize(mu, logvar)
        logits = self.decoder(z, cond, x, padding_mask)
        return logits, mu, logvar
    
    def generate(self, mood: torch.Tensor, raga: torch.Tensor,
                 tempo: torch.Tensor, duration: torch.Tensor,
                 max_length: int = 512, temperature: float = 1.0,
                 top_k: int = 50, top_p: float = 0.95) -> torch.Tensor:
        batch_size = mood.size(0)
        device = mood.device
        
        cond = self.conditioning(mood, raga, tempo, duration)
        z = torch.randn(batch_size, self.config.latent_dim, device=device)
        
        return self.decoder.generate(
            z, cond, max_length, temperature, top_k, top_p,
            bos_token=1, eos_token=2
        )
    
    def loss(self, logits: torch.Tensor, targets: torch.Tensor,
             mu: torch.Tensor, logvar: torch.Tensor,
             padding_mask: Optional[torch.Tensor] = None) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        # Reconstruction loss with label smoothing
        recon_loss = F.cross_entropy(
            logits.view(-1, self.config.vocab_size),
            targets.view(-1),
            ignore_index=0,
            reduction='mean',
            label_smoothing=0.1
        )
        
        # KL divergence with free bits
        kl_per_dim = -0.5 * (1 + logvar - mu.pow(2) - logvar.exp())
        free_bits = 0.5
        kl_loss = torch.mean(torch.clamp(kl_per_dim, min=free_bits))
        
        total_loss = recon_loss + self.config.kl_weight * kl_loss
        return total_loss, recon_loss, kl_loss


def count_parameters(model: nn.Module) -> int:
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


if __name__ == "__main__":
    config = CVAEConfig()
    model = RagaCVAE(config)
    
    params = count_parameters(model)
    print(f"Model parameters: {params:,} ({params/1e6:.1f}M)")
    print(f"Estimated size: {params * 4 / (1024**3):.2f} GB (float32)")
    
    # Test forward pass (small batch for testing)
    batch_size = 2
    seq_len = 128
    
    x = torch.randint(0, config.vocab_size, (batch_size, seq_len))
    mood = torch.randint(0, config.num_moods, (batch_size,))
    raga = torch.randint(0, config.num_ragas, (batch_size,))
    tempo = torch.randint(0, 32, (batch_size,))
    duration = torch.randint(0, 16, (batch_size,))
    
    print("\nTesting forward pass...")
    logits, mu, logvar = model(x, mood, raga, tempo, duration)
    print(f"Output shape: {logits.shape}")
    
    print("\nTesting generation...")
    model.eval()
    with torch.no_grad():
        generated = model.generate(mood, raga, tempo, duration, max_length=64)
    print(f"Generated shape: {generated.shape}")
