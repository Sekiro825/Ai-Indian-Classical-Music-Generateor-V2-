"""
V2 Transformer-Flow Model for Indian Classical Music Generation
Decoder-only Transformer backbone with RoPE and optional sliding-window attention.
"""

import math
from typing import Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.checkpoint import checkpoint

from .config import MambaFlowConfig, ExpressionEncoderConfig


class RMSNorm(nn.Module):
    def __init__(self, dim: int, eps: float = 1e-5):
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(dim))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        var = x.pow(2).mean(dim=-1, keepdim=True)
        x = x * torch.rsqrt(var + self.eps)
        return self.weight * x


def _rotate_half(x: torch.Tensor) -> torch.Tensor:
    x1 = x[..., ::2]
    x2 = x[..., 1::2]
    return torch.stack((-x2, x1), dim=-1).flatten(-2)


class RotaryEmbedding(nn.Module):
    def __init__(self, dim: int, max_position: int, theta: float = 10000.0, scaling: float = 1.0):
        super().__init__()
        inv_freq = 1.0 / (theta ** (torch.arange(0, dim, 2).float() / dim))
        t = torch.arange(max_position, dtype=torch.float32) / max(scaling, 1e-6)
        freqs = torch.outer(t, inv_freq)
        emb = torch.cat((freqs, freqs), dim=-1)
        self.register_buffer("cos_cached", emb.cos(), persistent=False)
        self.register_buffer("sin_cached", emb.sin(), persistent=False)

    def forward(self, q: torch.Tensor, k: torch.Tensor, seq_len: int) -> Tuple[torch.Tensor, torch.Tensor]:
        cos = self.cos_cached[:seq_len].unsqueeze(0).unsqueeze(0).to(dtype=q.dtype, device=q.device)
        sin = self.sin_cached[:seq_len].unsqueeze(0).unsqueeze(0).to(dtype=q.dtype, device=q.device)
        q = (q * cos) + (_rotate_half(q) * sin)
        k = (k * cos) + (_rotate_half(k) * sin)
        return q, k


class CausalSelfAttention(nn.Module):
    def __init__(self, config: MambaFlowConfig):
        super().__init__()
        self.d_model = config.d_model
        self.n_heads = config.transformer.n_heads
        self.head_dim = self.d_model // self.n_heads
        if self.head_dim * self.n_heads != self.d_model:
            raise ValueError("d_model must be divisible by transformer.n_heads")

        self.attn_window = int(config.transformer.attention_window)
        self.q_proj = nn.Linear(self.d_model, self.d_model, bias=False)
        self.k_proj = nn.Linear(self.d_model, self.d_model, bias=False)
        self.v_proj = nn.Linear(self.d_model, self.d_model, bias=False)
        self.o_proj = nn.Linear(self.d_model, self.d_model, bias=False)
        self.dropout = nn.Dropout(config.dropout)

        self.rope = RotaryEmbedding(
            dim=self.head_dim,
            max_position=config.max_seq_length,
            theta=config.transformer.rope_theta,
            scaling=config.transformer.rope_scaling,
        )

    def _build_attn_mask(self, seq_len: int, device: torch.device) -> Optional[torch.Tensor]:
        if self.attn_window <= 0:
            return None
        idx = torch.arange(seq_len, device=device)
        causal = idx[:, None] >= idx[None, :]
        local = (idx[:, None] - idx[None, :]) < self.attn_window
        allow = causal & local
        return ~allow

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        bsz, seq_len, _ = x.shape
        q = self.q_proj(x).view(bsz, seq_len, self.n_heads, self.head_dim).transpose(1, 2)
        k = self.k_proj(x).view(bsz, seq_len, self.n_heads, self.head_dim).transpose(1, 2)
        v = self.v_proj(x).view(bsz, seq_len, self.n_heads, self.head_dim).transpose(1, 2)

        q, k = self.rope(q, k, seq_len)
        attn_mask = self._build_attn_mask(seq_len, x.device)

        y = F.scaled_dot_product_attention(
            q, k, v,
            attn_mask=attn_mask,
            dropout_p=self.dropout.p if self.training else 0.0,
            is_causal=attn_mask is None,
        )
        y = y.transpose(1, 2).contiguous().view(bsz, seq_len, self.d_model)
        return self.dropout(self.o_proj(y))


class FeedForward(nn.Module):
    def __init__(self, config: MambaFlowConfig):
        super().__init__()
        hidden_dim = int(config.d_model * config.transformer.ff_mult)
        self.w1 = nn.Linear(config.d_model, hidden_dim, bias=False)
        self.w2 = nn.Linear(hidden_dim, config.d_model, bias=False)
        self.w3 = nn.Linear(config.d_model, hidden_dim, bias=False)
        self.dropout = nn.Dropout(config.dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = F.silu(self.w1(x)) * self.w3(x)
        x = self.w2(x)
        return self.dropout(x)


class TransformerBlock(nn.Module):
    def __init__(self, config: MambaFlowConfig):
        super().__init__()
        self.attn_norm = RMSNorm(config.d_model, eps=config.transformer.norm_eps)
        self.ffn_norm = RMSNorm(config.d_model, eps=config.transformer.norm_eps)
        self.attn = CausalSelfAttention(config)
        self.ffn = FeedForward(config)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x + self.attn(self.attn_norm(x))
        x = x + self.ffn(self.ffn_norm(x))
        return x


class ExpressionEncoder(nn.Module):
    def __init__(self, config: ExpressionEncoderConfig):
        super().__init__()
        self.projection = nn.Sequential(
            nn.Linear(config.input_dim, config.hidden_dim),
            nn.GELU(),
            nn.LayerNorm(config.hidden_dim),
            nn.Dropout(config.dropout),
            nn.Linear(config.hidden_dim, config.embed_dim),
            nn.LayerNorm(config.embed_dim),
        )
        self.attention_weight = nn.Linear(config.embed_dim, 1)

    def forward(self, expression: torch.Tensor, padding_mask: Optional[torch.Tensor] = None) -> Tuple[torch.Tensor, torch.Tensor]:
        frame_emb = self.projection(expression)
        attn = self.attention_weight(frame_emb)
        if padding_mask is not None:
            attn = attn.masked_fill(padding_mask.unsqueeze(-1), float('-inf'))
        attn = F.softmax(attn, dim=1)
        global_emb = (frame_emb * attn).sum(dim=1)
        return frame_emb, global_emb


class ConditioningModule(nn.Module):
    def __init__(self, config: MambaFlowConfig):
        super().__init__()
        self.mood_emb = nn.Embedding(config.num_moods, config.mood_embed_dim)
        self.raga_emb = nn.Embedding(config.num_ragas, config.raga_embed_dim)
        self.taal_emb = nn.Embedding(config.taal.num_taals, config.taal.embed_dim)
        self.tempo_emb = nn.Linear(1, config.tempo_embed_dim)
        self.duration_emb = nn.Linear(1, config.duration_embed_dim)
        self.projection = nn.Sequential(
            nn.Linear(config.total_cond_dim(), config.d_model),
            nn.GELU(),
            nn.LayerNorm(config.d_model),
        )

    def forward(self, mood, raga, taal, tempo, duration, global_expression: Optional[torch.Tensor] = None) -> torch.Tensor:
        m = self.mood_emb(mood)
        r = self.raga_emb(raga)
        t = self.taal_emb(taal)
        tp = self.tempo_emb(tempo.unsqueeze(-1).float())
        d = self.duration_emb(duration.unsqueeze(-1).float())
        cond = self.projection(torch.cat([m, r, t, tp, d], dim=-1))
        return cond


class TaalPositionEncoder(nn.Module):
    def __init__(self, d_model: int, max_cycle: int = 16):
        super().__init__()
        self.max_cycle = max_cycle
        self.cycle_emb = nn.Embedding(max_cycle, d_model)
        self.strength_emb = nn.Embedding(4, d_model)

    def forward(self, seq_len: int, taal_cycle_len: int, device: torch.device) -> torch.Tensor:
        positions = torch.arange(seq_len, device=device) % min(taal_cycle_len, self.max_cycle)
        pos_emb = self.cycle_emb(positions)
        strengths = torch.ones(seq_len, dtype=torch.long, device=device) * 3
        strengths[0::taal_cycle_len] = 0
        str_emb = self.strength_emb(strengths)
        return (pos_emb + str_emb).unsqueeze(0)


class FlowMatchingHead(nn.Module):
    def __init__(self, d_model: int, expression_dim: int = 4, hidden_dim: int = 512):
        super().__init__()
        self.time_emb = nn.Sequential(nn.Linear(1, hidden_dim), nn.SiLU(), nn.Linear(hidden_dim, d_model))
        self.velocity_net = nn.Sequential(
            nn.Linear(d_model + expression_dim, hidden_dim),
            nn.GELU(),
            nn.LayerNorm(hidden_dim),
            nn.Linear(hidden_dim, expression_dim),
        )

    def forward(self, hidden_states: torch.Tensor, noisy_expression: torch.Tensor, timestep: torch.Tensor) -> torch.Tensor:
        t_emb = self.time_emb(timestep.unsqueeze(-1))
        h = hidden_states + t_emb.unsqueeze(1)
        return self.velocity_net(torch.cat([h, noisy_expression], dim=-1))


class TransformerFlowModel(nn.Module):
    def __init__(self, config: MambaFlowConfig):
        super().__init__()
        self.config = config
        self.token_embedding = nn.Embedding(config.vocab_size, config.d_model)
        self.token_dropout = nn.Dropout(config.dropout)
        self.conditioning = ConditioningModule(config)
        if config.use_expression:
            self.expression_encoder = ExpressionEncoder(config.expression)
            self.expression_proj = nn.Linear(config.expression.embed_dim, config.d_model)
        self.taal_position = TaalPositionEncoder(config.d_model)
        self.layers = nn.ModuleList([TransformerBlock(config) for _ in range(config.n_layers)])
        self.final_norm = RMSNorm(config.d_model, eps=config.transformer.norm_eps)
        self.token_head = nn.Linear(config.d_model, config.vocab_size, bias=False)
        if config.use_expression:
            self.flow_head = FlowMatchingHead(config.d_model, config.expression.input_dim)
            self.expression_head = nn.Linear(config.d_model, config.expression.input_dim)
        self.apply(self._init_weights)

    def _init_weights(self, module):
        if isinstance(module, (nn.Linear, nn.Embedding)):
            nn.init.normal_(module.weight, std=0.02)

    def _run_backbone(self, x: torch.Tensor) -> torch.Tensor:
        for layer in self.layers:
            if self.config.use_gradient_checkpointing and self.training:
                x = checkpoint(layer, x, use_reentrant=False)
            else:
                x = layer(x)
        return self.final_norm(x)

    def forward(self, tokens, mood, raga, taal, tempo, duration, expression=None, taal_cycle_len=8,
                flow_timestep=None, noisy_expression=None, padding_mask=None):
        _, seq_len = tokens.shape
        x = self.token_embedding(tokens)
        x = self.token_dropout(x)
        global_expr = None
        if expression is not None and self.config.use_expression:
            frame_expr, global_expr = self.expression_encoder(expression, padding_mask)
            x = x + self.expression_proj(frame_expr[:, :seq_len])
        x = x + self.conditioning(mood, raga, taal, tempo, duration, global_expr).unsqueeze(1)
        x = x + self.taal_position(seq_len, taal_cycle_len, x.device)
        hidden = self._run_backbone(x)
        logits = self.token_head(hidden)
        result = {'logits': logits}
        if self.config.use_expression:
            result['predicted_expression'] = self.expression_head(hidden)
            if flow_timestep is not None and noisy_expression is not None:
                result['flow_velocity'] = self.flow_head(hidden, noisy_expression, flow_timestep)
        return result

    def get_num_params(self) -> int:
        return sum(p.numel() for p in self.parameters())
