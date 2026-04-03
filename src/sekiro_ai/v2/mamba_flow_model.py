"""
V2 Mamba-Flow Model for Indian Classical Music Generation
Mamba Backbone (Fused Blackwell Overdrive Version)
"""

import math
import os
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.checkpoint import checkpoint
from typing import Optional, Dict, Tuple
from dataclasses import dataclass

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

from v2.config import MambaFlowConfig, ExpressionEncoderConfig

# --- FUSED SCAN ENGINE (The Blackwell Nitro Boost) ---
# Kept opt-in for stability across CUDA graph / compiler runtime combinations.
def _recurrent_scan_impl(delta_A, delta_B_x, C):
    """
    args:
        delta_A: (B, L, D, N)
        delta_B_x: (B, L, D, N)
        C: (B, L, N)
    returns:
        y: (B, L, D)
    """
    B, L, D, N = delta_A.shape
    device = delta_A.device
    dtype = delta_A.dtype
    
    # We use float32 for the accumulation to prevent precision-loss
    h = torch.zeros(B, D, N, device=device, dtype=torch.float32)
    all_y = []
    
    # Pre-transpose C for easier multiplication
    # C is (B, L, N), we want (B, L, 1, N) for multiplication with (B, D, N) or similar
    
    for t in range(L):
        # Recurrence: h_t = A_t * h_{t-1} + B_t*x_t
        # delta_A is already log-space/pre-exp in some implementations, 
        # but here we assume it's the multiplier. 
        # delta_A: (B, D, N), h: (B, D, N)
        h = (delta_A[:, t].to(torch.float32) * h) + delta_B_x[:, t].to(torch.float32)
        
        # Output: y_t = h_t * C_t
        # h: (B, D, N), C: (B, N) -> y: (B, D)
        # Use einsum for safety and clarity
        y_t = torch.einsum('bdn,bn->bd', h, C[:, t].to(torch.float32))
        all_y.append(y_t)
        
    return torch.stack(all_y, dim=1).to(dtype)


if os.environ.get("SEKIRO_COMPILE_SCAN", "0") == "1":
    _fused_recurrent_scan = torch.compile(
        _recurrent_scan_impl, mode="reduce-overhead", fullgraph=False
    )
else:
    _fused_recurrent_scan = _recurrent_scan_impl

# ============================================================
# Mamba Block (Pure PyTorch fallback if mamba-ssm not available)
# ============================================================

class SelectiveSSM(nn.Module):
    def __init__(self, d_model: int, d_state: int = 64, d_conv: int = 4,
                 expand: int = 2, dt_rank: int = None, dropout: float = 0.1):
        super().__init__()
        self.d_model = d_model
        self.d_state = d_state
        self.d_inner = d_model * expand
        self.dt_rank = dt_rank or math.ceil(d_model / 16)

        self.in_proj = nn.Linear(d_model, self.d_inner * 2, bias=False)

        self.conv1d = nn.Conv1d(
            self.d_inner, self.d_inner,
            kernel_size=d_conv, padding=d_conv - 1,
            groups=self.d_inner, bias=True
        )

        self.x_proj = nn.Linear(self.d_inner, self.dt_rank + d_state * 2, bias=False)
        self.dt_proj = nn.Linear(self.dt_rank, self.d_inner, bias=True)

        A = torch.arange(1, d_state + 1, dtype=torch.float32).unsqueeze(0).expand(self.d_inner, -1)
        self.A_log = nn.Parameter(torch.log(A))

        self.D = nn.Parameter(torch.ones(self.d_inner))
        self.out_proj = nn.Linear(self.d_inner, d_model, bias=False)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        batch, seq_len, _ = x.shape
        xz = self.in_proj(x)
        x_stream, z = xz.chunk(2, dim=-1)

        x_conv = x_stream.transpose(1, 2)
        x_conv = self.conv1d(x_conv)[:, :, :seq_len]
        x_conv = x_conv.transpose(1, 2)
        x_conv = F.silu(x_conv)

        ssm_params = self.x_proj(x_conv)
        dt, B_param, C_param = torch.split(
            ssm_params, [self.dt_rank, self.d_state, self.d_state], dim=-1
        )
        dt = F.softplus(self.dt_proj(dt))

        # Discretize A
        A = -torch.exp(self.A_log)

        # Low-memory scan: do not materialize (B, L, D, N) tensors.
        # This avoids multi-GB temporary allocations on larger models.
        h = torch.zeros(
            batch, self.d_inner, self.d_state,
            device=x.device, dtype=torch.float32
        )
        y_steps = []
        A_fp32 = A.to(torch.float32)

        for t in range(seq_len):
            dt_t = dt[:, t].to(torch.float32)                  # (B, D)
            B_t = B_param[:, t].to(torch.float32)              # (B, N)
            C_t = C_param[:, t].to(torch.float32)              # (B, N)
            x_t = x_conv[:, t].to(torch.float32)               # (B, D)

            delta_A_t = torch.exp(dt_t.unsqueeze(-1) * A_fp32.unsqueeze(0))
            delta_Bx_t = (dt_t * x_t).unsqueeze(-1) * B_t.unsqueeze(1)
            h = delta_A_t * h + delta_Bx_t
            y_t = torch.einsum('bdn,bn->bd', h, C_t)
            y_steps.append(y_t)

        y = torch.stack(y_steps, dim=1).to(x_conv.dtype)

        y = y + x_conv * self.D.unsqueeze(0).unsqueeze(0)
        y = y * F.silu(z)
        y = self.dropout(self.out_proj(y))
        return y


class MambaLayer(nn.Module):
    def __init__(self, d_model: int, d_state: int = 64, d_conv: int = 4,
                 expand: int = 2, dropout: float = 0.1):
        super().__init__()
        self.norm = nn.LayerNorm(d_model)
        self.ssm = SelectiveSSM(d_model, d_state, d_conv, expand, dropout=dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x + self.ssm(self.norm(x))


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


class MambaFlowModel(nn.Module):
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
        self.layers = nn.ModuleList([
            MambaLayer(config.d_model, config.mamba.d_state, config.mamba.d_conv, config.mamba.expand, config.dropout)
            for _ in range(config.n_layers)
        ])
        self.final_norm = nn.LayerNorm(config.d_model)
        self.token_head = nn.Linear(config.d_model, config.vocab_size)
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
        B, L = tokens.shape
        x = self.token_embedding(tokens)
        x = self.token_dropout(x)
        global_expr = None
        if expression is not None and self.config.use_expression:
            frame_expr, global_expr = self.expression_encoder(expression, padding_mask)
            x = x + self.expression_proj(frame_expr[:, :L])
        x = x + self.conditioning(mood, raga, taal, tempo, duration, global_expr).unsqueeze(1)
        x = x + self.taal_position(L, taal_cycle_len, x.device)
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
