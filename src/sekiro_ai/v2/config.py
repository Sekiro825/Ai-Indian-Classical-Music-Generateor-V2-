"""
V2 Configuration for Mamba-Flow Indian Classical Music Generator

Replaces the CVAE-based HybridCVAEConfig with a SSM + Flow-Matching architecture.
Designed for ~2.5B parameters on RTX PRO 6000 (96GB VRAM).
"""

from dataclasses import dataclass, field
from typing import Optional, List
import json


@dataclass
class ExpressionEncoderConfig:
    """Config for audio expression feature encoder"""
    input_dim: int = 4       # [f0, amplitude, voiced, spectral_centroid]
    hidden_dim: int = 512
    embed_dim: int = 256     # Expression embedding dimension
    num_layers: int = 3
    dropout: float = 0.1


@dataclass
class MambaBlockConfig:
    """Config for individual Mamba (S6) block"""
    d_state: int = 64        # SSM state expansion factor
    d_conv: int = 4          # Local convolution width
    expand: int = 2          # Block expansion factor
    dt_rank: str = "auto"    # Rank for dt projection
    dt_min: float = 0.001
    dt_max: float = 0.1
    dt_init: str = "random"
    dt_scale: float = 1.0
    conv_bias: bool = True
    bias: bool = False


@dataclass
class TransformerConfig:
    """Config for decoder-only Transformer backbone."""
    n_heads: int = 16
    n_kv_heads: int = 16
    ff_mult: float = 4.0
    rope_theta: float = 10000.0
    rope_scaling: float = 1.0
    attention_window: int = 0  # 0 means full causal attention
    norm_eps: float = 1e-5


@dataclass
class FlowMatchingConfig:
    """Config for the flow-matching (continuous normalizing flow) engine"""
    num_flow_steps: int = 100       # ODE steps during inference
    sigma_min: float = 0.001        # Minimum noise level
    solver: str = "euler"           # ODE solver: "euler" or "midpoint"
    # During training, we sample t ~ U(0,1) and compute the vector field
    # The model predicts: v(x_t, t) where x_t = (1-t)*noise + t*data


@dataclass
class TaalConfig:
    """Taal-aware rhythmic structure config"""
    num_taals: int = 14
    # Taal cycle lengths (in beats) for structural awareness
    taal_cycles: dict = field(default_factory=lambda: {
        "trital": 16, "ektal": 12, "jhaptal": 10, "rupak": 7,
        "dadra": 6, "keherwa": 8, "deepchandi": 14,
        "addhatrital": 8, "bhajani": 8,
        "adi": 8, "misra_chapu": 7, "khanda_chapu": 5,
        "roopak": 7, "unknown": 8,
    })
    embed_dim: int = 128
    # Whether to inject taal position (sam, khali, etc.) as explicit conditioning
    use_taal_position: bool = True


@dataclass
class MambaFlowConfig:
    """
    Master config for the V2 Mamba-Flow Indian Classical Music Generator.

    Architecture:
    - Mamba SSM backbone (O(N) linear scaling, infinite generation)
    - Flow-Matching generative engine (sharp pitch/expression, no VAE blur)
    - Dual-stream: BPE MIDI tokens + continuous audio expression
    - Taal-aware rhythmic conditioning

    Default values target ~2.5B parameters on RTX PRO 6000 (96GB).
    """

    # ---- Token vocabulary (set after BPE training) ----
    vocab_size: int = 10000          # BPE vocab (expanded from 491)
    max_seq_length: int = 4096       # 4x larger than V1 (Mamba handles it in O(N))

    # ---- Backbone (~1B-2B) ----
    backbone: str = "transformer"   # "transformer" (recommended) or "mamba"

    # ---- Transformer backbone ----
    transformer: TransformerConfig = field(default_factory=TransformerConfig)

    # ---- Legacy Mamba fields (kept for backward compatibility) ----
    d_model: int = 2560              # Model dimension
    n_layers: int = 64               # Total Mamba layers
    mamba: MambaBlockConfig = field(default_factory=MambaBlockConfig)
    dropout: float = 0.1

    # ---- Flow-Matching ----
    flow: FlowMatchingConfig = field(default_factory=FlowMatchingConfig)

    # ---- Conditioning ----
    num_moods: int = 36
    num_ragas: int = 50
    mood_embed_dim: int = 128
    raga_embed_dim: int = 256        # Larger: raga is the most important conditioning signal
    taal: TaalConfig = field(default_factory=TaalConfig)
    tempo_embed_dim: int = 64
    duration_embed_dim: int = 64

    # ---- Expression (audio contours) ----
    use_expression: bool = True
    expression: ExpressionEncoderConfig = field(default_factory=ExpressionEncoderConfig)

    # ---- Dual heads ----
    # The model predicts both:
    # 1) next BPE token (discrete, cross-entropy)
    # 2) expression contour (continuous, flow-matched MSE)
    expression_loss_weight: float = 0.5
    flow_loss_weight: float = 1.0
    grammar_loss_weight: float = 0.15
    vivadi_penalty_multiplier: float = 10.0
    vadi_reward_weight: float = 0.15
    samvadi_reward_weight: float = 0.08

    # ---- Training ----
    use_gradient_checkpointing: bool = True

    def total_cond_dim(self) -> int:
        """Total dimension of all conditioning signals"""
        return (
            self.mood_embed_dim +
            self.raga_embed_dim +
            self.taal.embed_dim +
            self.tempo_embed_dim +
            self.duration_embed_dim
        )

    def save(self, path: str):
        """Save config to JSON"""
        def to_dict(obj):
            if hasattr(obj, '__dict__'):
                return {k: to_dict(v) for k, v in obj.__dict__.items()}
            return obj

        with open(path, 'w') as f:
            json.dump(to_dict(self), f, indent=2)

    @classmethod
    def load(cls, path: str) -> 'MambaFlowConfig':
        """Load config from JSON"""
        with open(path, 'r') as f:
            data = json.load(f)

        # Reconstruct nested configs
        if 'mamba' in data:
            data['mamba'] = MambaBlockConfig(**data['mamba'])
        if 'transformer' in data:
            data['transformer'] = TransformerConfig(**data['transformer'])
        if 'flow' in data:
            data['flow'] = FlowMatchingConfig(**data['flow'])
        if 'taal' in data:
            data['taal'] = TaalConfig(**data['taal'])
        if 'expression' in data:
            data['expression'] = ExpressionEncoderConfig(**data['expression'])

        return cls(**data)


if __name__ == "__main__":
    config = MambaFlowConfig()
    print("=== Mamba-Flow V2 Config ===")
    print(f"d_model: {config.d_model}")
    print(f"n_layers: {config.n_layers}")
    print(f"max_seq_length: {config.max_seq_length}")
    print(f"vocab_size: {config.vocab_size}")
    print(f"expression: {config.use_expression}")
    print(f"flow steps: {config.flow.num_flow_steps}")

    # Rough param estimate for Mamba:
    # Each Mamba layer: ~8 * d_model^2 (with expand=2)
    d = config.d_model
    n = config.n_layers
    expand = config.mamba.expand
    layer_params = 2 * expand * d * d + expand * d * config.mamba.d_state * 2
    total = n * layer_params + config.vocab_size * d  # + embeddings
    print(f"\nEstimated params: ~{total/1e9:.2f}B")
    print(f"BF16 size: ~{total*2/1e9:.1f}GB")
