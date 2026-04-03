"""
Hybrid CVAE Model for Indian Classical Music Generation
Combines MIDI structure with audio expression conditioning.
"""

import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.checkpoint import checkpoint
from typing import Optional, Tuple, Dict
from dataclasses import dataclass

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

from sekiro_ai.hybrid.config.hybrid_config import HybridCVAEConfig, ExpressionEncoderConfig
from .expression_encoder import ExpressionEncoder, ExpressionHead, GlobalExpressionEncoder


class RotaryPositionalEmbedding(nn.Module):
    """Rotary Positional Embedding (RoPE) for better length generalization"""
    
    def __init__(self, dim: int, max_seq_len: int = 2048, base: float = 10000.0):
        super().__init__()
        self.dim = dim
        self.max_seq_len = max_seq_len
        
        inv_freq = 1.0 / (base ** (torch.arange(0, dim, 2).float() / dim))
        self.register_buffer('inv_freq', inv_freq)
        
        self._build_cache(max_seq_len)
    
    def _build_cache(self, seq_len: int):
        t = torch.arange(seq_len, device=self.inv_freq.device)
        freqs = torch.einsum('i,j->ij', t, self.inv_freq)
        emb = torch.cat((freqs, freqs), dim=-1)
        self.register_buffer('cos_cached', emb.cos())
        self.register_buffer('sin_cached', emb.sin())
    
    def forward(self, x: torch.Tensor, seq_len: int = None) -> Tuple[torch.Tensor, torch.Tensor]:
        if seq_len is None:
            seq_len = x.shape[1]
        if seq_len > self.max_seq_len:
            # Expand RoPE cache on demand for longer sequences.
            self.max_seq_len = seq_len
            self._build_cache(seq_len)
        return self.cos_cached[:seq_len], self.sin_cached[:seq_len]


def rotate_half(x: torch.Tensor) -> torch.Tensor:
    x1, x2 = x[..., :x.shape[-1]//2], x[..., x.shape[-1]//2:]
    return torch.cat((-x2, x1), dim=-1)


def apply_rotary_pos_emb(q, k, cos, sin):
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
    
    def forward(
        self, 
        query: torch.Tensor, 
        key: torch.Tensor, 
        value: torch.Tensor,
        attn_mask: Optional[torch.Tensor] = None,
        key_padding_mask: Optional[torch.Tensor] = None,
        is_causal: bool = False,
        use_rope: bool = True
    ) -> torch.Tensor:
        batch_size, seq_len, _ = query.shape
        
        q = self.q_proj(query).view(batch_size, seq_len, self.num_heads, self.head_dim).transpose(1, 2)
        k = self.k_proj(key).view(batch_size, -1, self.num_heads, self.head_dim).transpose(1, 2)
        v = self.v_proj(value).view(batch_size, -1, self.num_heads, self.head_dim).transpose(1, 2)
        
        if use_rope:
            cos, sin = self.rotary(q, seq_len)
            cos = cos.unsqueeze(0).unsqueeze(0)
            sin = sin.unsqueeze(0).unsqueeze(0)
            q, k = apply_rotary_pos_emb(q, k, cos, sin)
        
        attn_output = F.scaled_dot_product_attention(
            q, k, v, attn_mask=attn_mask, is_causal=is_causal, dropout_p=self.dropout.p if self.training else 0.0
        )
        
        attn_output = attn_output.transpose(1, 2).contiguous().view(batch_size, seq_len, self.embed_dim)
        return self.out_proj(attn_output)


class PreNormTransformerEncoderLayer(nn.Module):
    """Pre-LayerNorm Transformer Encoder Layer"""
    
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
        x = x + self.attn(self.norm1(x), self.norm1(x), self.norm1(x), key_padding_mask=padding_mask, use_rope=True)
        x = x + self.ffn(self.norm2(x))
        return x


class PreNormTransformerDecoderLayer(nn.Module):
    """Pre-LayerNorm Transformer Decoder Layer with cross-attention"""
    
    def __init__(self, embed_dim: int, num_heads: int, ff_dim: int, dropout: float = 0.1):
        super().__init__()
        self.norm1 = nn.LayerNorm(embed_dim)
        self.self_attn = RoPEMultiHeadAttention(embed_dim, num_heads, dropout)
        self.norm2 = nn.LayerNorm(embed_dim)
        self.cross_attn = RoPEMultiHeadAttention(embed_dim, num_heads, dropout)
        self.norm3 = nn.LayerNorm(embed_dim)
        self.ffn = nn.Sequential(
            nn.Linear(embed_dim, ff_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(ff_dim, embed_dim),
            nn.Dropout(dropout)
        )
    
    def forward(
        self, 
        x: torch.Tensor, 
        memory: torch.Tensor,
        tgt_padding_mask: Optional[torch.Tensor] = None,
        is_causal: bool = True
    ) -> torch.Tensor:
        x = x + self.self_attn(self.norm1(x), self.norm1(x), self.norm1(x), is_causal=is_causal, use_rope=True)
        x = x + self.cross_attn(self.norm2(x), memory, memory, use_rope=False)
        x = x + self.ffn(self.norm3(x))
        return x


class ConditioningModule(nn.Module):
    """Encodes conditioning inputs (mood, raga, taal, tempo, duration)"""
    
    def __init__(self, config: HybridCVAEConfig):
        super().__init__()
        self.config = config
        
        self.mood_embedding = nn.Embedding(config.num_moods, config.mood_embed_dim)
        self.raga_embedding = nn.Embedding(config.num_ragas, config.raga_embed_dim)
        self.taal_embedding = nn.Embedding(config.num_taals, config.taal_embed_dim)
        self.tempo_embedding = nn.Linear(1, config.tempo_embed_dim)
        self.duration_embedding = nn.Linear(1, config.duration_embed_dim)
        
        cond_dim = (
            config.mood_embed_dim +
            config.raga_embed_dim +
            config.taal_embed_dim +
            config.tempo_embed_dim +
            config.duration_embed_dim
        )
        
        self.projection = nn.Sequential(
            nn.Linear(cond_dim, config.embed_dim),
            nn.GELU(),
            nn.LayerNorm(config.embed_dim)
        )
    
    def forward(
        self, 
        mood: torch.Tensor, 
        raga: torch.Tensor,
        taal: torch.Tensor,
        tempo: torch.Tensor, 
        duration: torch.Tensor
    ) -> torch.Tensor:
        mood_emb = self.mood_embedding(mood)
        raga_emb = self.raga_embedding(raga)
        taal_emb = self.taal_embedding(taal)
        tempo_emb = self.tempo_embedding(tempo.unsqueeze(-1).float())
        duration_emb = self.duration_embedding(duration.unsqueeze(-1).float())
        
        combined = torch.cat([mood_emb, raga_emb, taal_emb, tempo_emb, duration_emb], dim=-1)
        return self.projection(combined)


class HybridCVAE(nn.Module):
    """
    Hybrid Conditional VAE for Indian Classical Music Generation.
    Combines MIDI token modeling with audio expression conditioning.
    """
    
    def __init__(self, config: HybridCVAEConfig):
        super().__init__()
        self.config = config
        
        # Token embedding
        self.token_embedding = nn.Embedding(config.vocab_size, config.embed_dim)
        
        # Conditioning module
        self.conditioning = ConditioningModule(config)
        
        # **NEW: Expression modules**
        if config.use_expression:
            self.expression_encoder = ExpressionEncoder(
                input_dim=config.expression_encoder_config.input_dim,
                hidden_dim=config.expression_encoder_config.hidden_dim,
                embed_dim=config.expression_dim,
                num_layers=config.expression_encoder_config.num_layers,
                num_heads=config.expression_encoder_config.num_heads,
                dropout=config.expression_encoder_config.dropout
            )
            
            self.global_expression_encoder = GlobalExpressionEncoder(
                input_dim=config.expression_encoder_config.input_dim,
                hidden_dim=config.expression_encoder_config.hidden_dim,
                embed_dim=config.expression_dim
            )
            
            self.expression_head = ExpressionHead(
                hidden_dim=config.embed_dim,
                output_dim=config.expression_encoder_config.input_dim
            )
            
            # Project expression to embed_dim for fusion
            self.expression_projection = nn.Linear(config.expression_dim, config.embed_dim)
        
        # Encoder
        self.encoder_layers = nn.ModuleList([
            PreNormTransformerEncoderLayer(config.embed_dim, config.num_heads, config.ff_dim, config.dropout)
            for _ in range(config.num_encoder_layers)
        ])
        self.encoder_norm = nn.LayerNorm(config.embed_dim)
        
        # Latent space
        self.to_mu = nn.Linear(config.embed_dim, config.latent_dim)
        self.to_logvar = nn.Linear(config.embed_dim, config.latent_dim)
        self.from_latent = nn.Linear(config.latent_dim, config.embed_dim)
        
        # Decoder
        self.decoder_layers = nn.ModuleList([
            PreNormTransformerDecoderLayer(config.embed_dim, config.num_heads, config.ff_dim, config.dropout)
            for _ in range(config.num_decoder_layers)
        ])
        self.decoder_norm = nn.LayerNorm(config.embed_dim)
        
        # Output head
        self.output_head = nn.Linear(config.embed_dim, config.vocab_size)
        
        # Initialize weights
        self.apply(self._init_weights)
    
    def _init_weights(self, module):
        if isinstance(module, nn.Linear):
            nn.init.normal_(module.weight, std=0.02)
            if module.bias is not None:
                nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            nn.init.normal_(module.weight, std=0.02)
    
    def encode(
        self, 
        tokens: torch.Tensor, 
        conditioning: torch.Tensor,
        expression: Optional[torch.Tensor] = None,
        padding_mask: Optional[torch.Tensor] = None
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Encode tokens to latent distribution parameters"""
        x = self.token_embedding(tokens)
        x = x + conditioning.unsqueeze(1)
        
        # Add expression conditioning if provided
        if expression is not None and self.config.use_expression:
            expr_encoded = self.expression_encoder(expression)
            expr_projected = self.expression_projection(expr_encoded)
            # Align lengths if necessary
            min_len = min(x.size(1), expr_projected.size(1))
            x = x[:, :min_len] + expr_projected[:, :min_len]
        
        # Encode
        for layer in self.encoder_layers:
            if self.config.use_gradient_checkpointing and self.training:
                x = checkpoint(layer, x, padding_mask, use_reentrant=False)
            else:
                x = layer(x, padding_mask)
        
        x = self.encoder_norm(x)
        
        # Pool to single vector (mean pooling)
        if padding_mask is not None:
            mask = ~padding_mask.unsqueeze(-1)
            x = (x * mask).sum(dim=1) / mask.sum(dim=1)
        else:
            x = x.mean(dim=1)
        
        mu = self.to_mu(x)
        logvar = self.to_logvar(x)
        
        return mu, logvar
    
    def reparameterize(self, mu: torch.Tensor, logvar: torch.Tensor) -> torch.Tensor:
        """Reparameterization trick"""
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mu + eps * std
    
    def decode(
        self, 
        z: torch.Tensor, 
        tokens: torch.Tensor,
        conditioning: torch.Tensor,
        padding_mask: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """Decode from latent to token logits"""
        # Prepare decoder input (shifted tokens)
        x = self.token_embedding(tokens)
        x = x + conditioning.unsqueeze(1)
        
        # Expand latent for cross-attention
        memory = self.from_latent(z).unsqueeze(1)
        
        # Decode
        for layer in self.decoder_layers:
            if self.config.use_gradient_checkpointing and self.training:
                x = checkpoint(layer, x, memory, padding_mask, True, use_reentrant=False)
            else:
                x = layer(x, memory, padding_mask, is_causal=True)
        
        x = self.decoder_norm(x)
        
        return x
    
    def forward(
        self,
        tokens: torch.Tensor,
        mood: torch.Tensor,
        raga: torch.Tensor,
        taal: torch.Tensor,
        tempo: torch.Tensor,
        duration: torch.Tensor,
        expression: Optional[torch.Tensor] = None,
        padding_mask: Optional[torch.Tensor] = None
    ) -> Dict[str, torch.Tensor]:
        """
        Forward pass for training.
        
        Args:
            tokens: (batch, seq_len) token indices
            mood: (batch,) mood indices
            raga: (batch,) raga indices
            taal: (batch,) taal indices
            tempo: (batch,) tempo values
            duration: (batch,) duration values
            expression: (batch, seq_len, 4) audio features [f0, amp, voiced, centroid]
            padding_mask: (batch, seq_len) True for padding positions
            
        Returns:
            Dictionary with logits, mu, logvar, and predicted_expression
        """
        # Get conditioning vector
        conditioning = self.conditioning(mood, raga, taal, tempo, duration)
        
        # Add global expression to conditioning if available
        if expression is not None and self.config.use_expression:
            global_expr = self.global_expression_encoder(expression)
            conditioning = conditioning + self.expression_projection(global_expr)
        
        # Encode to latent
        mu, logvar = self.encode(tokens, conditioning, expression, padding_mask)
        
        # Sample latent
        z = self.reparameterize(mu, logvar)
        
        # Decode
        decoder_output = self.decode(z, tokens, conditioning, padding_mask)
        
        # Token logits
        logits = self.output_head(decoder_output)
        
        # Predict expression if using hybrid mode
        predicted_expression = None
        if self.config.use_expression:
            predicted_expression = self.expression_head(decoder_output)
        
        return {
            'logits': logits,
            'mu': mu,
            'logvar': logvar,
            'z': z,
            'predicted_expression': predicted_expression
        }
    
    @torch.no_grad()
    def generate(
        self,
        mood: torch.Tensor,
        raga: torch.Tensor,
        taal: torch.Tensor,
        tempo: torch.Tensor,
        duration: torch.Tensor,
        max_length: int = 512,
        temperature: float = 1.0,
        top_k: int = 50,
        top_p: float = 0.9,
        bos_token: int = 1,
        prefix_tokens: Optional[torch.Tensor] = None,
        token_mask: Optional[torch.Tensor] = None,
        min_length: int = 0
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        """
        Generate tokens autoregressively.
        
        Returns:
            tokens: (batch, seq_len) generated token indices
            expression: (batch, seq_len, 4) predicted expression features
        """
        batch_size = mood.shape[0]
        device = mood.device
        
        # Get conditioning
        conditioning = self.conditioning(mood, raga, taal, tempo, duration)
        
        # Sample latent from prior
        z = torch.randn(batch_size, self.config.latent_dim, device=device)
        
        if prefix_tokens is not None and prefix_tokens.numel() > 0:
            tokens = prefix_tokens.to(device=device, dtype=torch.long)
        else:
            tokens = torch.full((batch_size, 1), bos_token, dtype=torch.long, device=device)
        
        all_expression = []
        
        for _ in range(max_length - 1):
            # Decode current sequence
            decoder_output = self.decode(z, tokens, conditioning)
            
            # Get next token logits
            next_logits = self.output_head(decoder_output[:, -1:, :]) / temperature
            if token_mask is not None:
                mask = token_mask.to(device=device).view(1, 1, -1).bool()
                next_logits = next_logits.masked_fill(~mask, float('-inf'))
            
            # Prevent EOS token before min_length is reached
            if tokens.size(1) < min_length:
                next_logits[..., 2] = float('-inf')  # 2 is EOS token
                
            # Apply top-k filtering
            if top_k > 0:
                indices_to_remove = next_logits < torch.topk(next_logits, top_k)[0][..., -1, None]
                next_logits[indices_to_remove] = float('-inf')
            
            # Apply top-p (nucleus) filtering
            if top_p < 1.0:
                sorted_logits, sorted_indices = torch.sort(next_logits, descending=True)
                cumulative_probs = torch.cumsum(F.softmax(sorted_logits, dim=-1), dim=-1)
                sorted_indices_to_remove = cumulative_probs > top_p
                sorted_indices_to_remove[..., 1:] = sorted_indices_to_remove[..., :-1].clone()
                sorted_indices_to_remove[..., 0] = 0
                indices_to_remove = sorted_indices_to_remove.scatter(-1, sorted_indices, sorted_indices_to_remove)
                next_logits[indices_to_remove] = float('-inf')
            
            # Sample
            probs = F.softmax(next_logits, dim=-1)
            next_token = torch.multinomial(probs.squeeze(1), num_samples=1)
            
            tokens = torch.cat([tokens, next_token], dim=1)
            
            # Predict expression for this step
            if self.config.use_expression:
                expr = self.expression_head(decoder_output[:, -1:, :])
                all_expression.append(expr)
            
            # Stop if EOS (token 2)
            if (next_token == 2).all():
                break
        
        expression = None
        if all_expression:
            expression = torch.cat(all_expression, dim=1)
        
        return tokens, expression
    
    def get_num_params(self) -> int:
        """Count total parameters"""
        return sum(p.numel() for p in self.parameters())


if __name__ == "__main__":
    # Test the model
    config = HybridCVAEConfig()
    model = HybridCVAE(config)
    
    print(f"Model parameters: {model.get_num_params():,}")
    
    # Test forward pass
    batch_size = 2
    seq_len = 128
    
    tokens = torch.randint(0, config.vocab_size, (batch_size, seq_len))
    mood = torch.randint(0, config.num_moods, (batch_size,))
    raga = torch.randint(0, config.num_ragas, (batch_size,))
    taal = torch.randint(0, config.num_taals, (batch_size,))
    tempo = torch.randint(60, 180, (batch_size,))
    duration = torch.randint(30, 120, (batch_size,))
    expression = torch.randn(batch_size, seq_len, 4)
    
    outputs = model(tokens, mood, raga, taal, tempo, duration, expression)
    
    print(f"Logits shape: {outputs['logits'].shape}")
    print(f"Mu shape: {outputs['mu'].shape}")
    print(f"Predicted expression shape: {outputs['predicted_expression'].shape}")
    
    # Test generation
    gen_tokens, gen_expr = model.generate(
        mood[:1], raga[:1], taal[:1], tempo[:1], duration[:1],
        max_length=64
    )
    print(f"Generated tokens shape: {gen_tokens.shape}")
    print(f"Generated expression shape: {gen_expr.shape}")
