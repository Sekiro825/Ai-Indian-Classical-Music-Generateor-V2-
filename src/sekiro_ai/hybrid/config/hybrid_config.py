"""
Configuration for Hybrid MIDI-Audio Model
Updated for 2.5B parameter model on RTX PRO 6000 (96GB VRAM)
"""

from dataclasses import dataclass, field
from typing import Optional, List
import json
from pathlib import Path


@dataclass
class AudioFeatureConfig:
    """Configuration for audio feature extraction"""
    sample_rate: int = 22050
    hop_length: int = 512
    n_fft: int = 2048
    fmin: float = 50.0
    fmax: float = 2000.0
    chunk_duration: float = 8.0
    feature_dim: int = 4  # [f0, amplitude, voiced, spectral_centroid]
    use_fast_pitch: bool = True


@dataclass
class ExpressionEncoderConfig:
    """Configuration for expression encoder"""
    input_dim: int = 4
    hidden_dim: int = 256
    embed_dim: int = 128
    num_layers: int = 2
    num_heads: int = 4
    dropout: float = 0.1
    use_transformer: bool = True


@dataclass
class HybridCVAEConfig:
    """
    Configuration for Hybrid Conditional VAE model.
    Default values produce a ~2.5 Billion Parameter model.
    """
    # Token vocabulary
    vocab_size: int = 491
    max_seq_length: int = 1024

    # Model dimensions (~2.6B config for RTX PRO 6000 96GB)
    embed_dim: int = 2048
    num_heads: int = 32
    num_encoder_layers: int = 24
    num_decoder_layers: int = 24
    ff_dim: int = 8192  # 4 * embed_dim
    latent_dim: int = 1024
    dropout: float = 0.1

    # Conditioning dimensions
    num_moods: int = 36
    num_ragas: int = 50  # Expanded for new datasets
    num_taals: int = 14
    mood_embed_dim: int = 128
    raga_embed_dim: int = 128
    taal_embed_dim: int = 64
    tempo_embed_dim: int = 64
    duration_embed_dim: int = 64

    # VAE settings
    kl_weight: float = 0.005

    # Expression settings
    use_expression: bool = True
    expression_dim: int = 128
    expression_encoder_config: ExpressionEncoderConfig = field(
        default_factory=ExpressionEncoderConfig
    )
    expression_loss_weight: float = 0.5

    # Training settings
    use_gradient_checkpointing: bool = True

    def save(self, path: str):
        """Save config to JSON file"""
        config_dict = {
            k: v if not isinstance(v, ExpressionEncoderConfig) else v.__dict__
            for k, v in self.__dict__.items()
        }
        with open(path, 'w') as f:
            json.dump(config_dict, f, indent=2)

    @classmethod
    def load(cls, path: str) -> 'HybridCVAEConfig':
        """Load config from JSON file"""
        with open(path, 'r') as f:
            config_dict = json.load(f)

        if 'expression_encoder_config' in config_dict:
            enc_config = config_dict.pop('expression_encoder_config')
            config_dict['expression_encoder_config'] = ExpressionEncoderConfig(**enc_config)

        return cls(**config_dict)


@dataclass
class TrainingConfig:
    """Configuration for hybrid model training on RTX PRO 6000"""
    # Data
    midi_dir: str = "data/midi_v2"
    audio_dir: str = "data/audio_features"
    feature_cache_dir: str = "src/sekiro_ai/hybrid/features"
    raga_metadata_path: str = "src/sekiro_ai/config/raga_metadata.json"

    # Training (optimized for 96GB VRAM)
    batch_size: int = 8
    gradient_accumulation_steps: int = 4
    learning_rate: float = 6e-5
    weight_decay: float = 0.01
    warmup_steps: int = 1000
    num_epochs: int = 150
    max_seq_length: int = 1024

    # Loss weights
    reconstruction_weight: float = 1.0
    kl_weight: float = 0.005
    expression_weight: float = 0.5
    grammar_weight: float = 0.25
    vivadi_penalty_multiplier: float = 10.0
    vadi_reward_weight: float = 0.15
    samvadi_reward_weight: float = 0.08
    kl_annealing_epochs: int = 50

    # Checkpointing
    checkpoint_dir: str = "checkpoints"
    save_every_epochs: int = 5
    log_every_steps: int = 10

    # Hardware
    use_amp: bool = True
    num_workers: int = 4

    def save(self, path: str):
        with open(path, 'w') as f:
            json.dump(self.__dict__, f, indent=2)

    @classmethod
    def load(cls, path: str) -> 'TrainingConfig':
        with open(path, 'r') as f:
            return cls(**json.load(f))


@dataclass
class InferenceConfig:
    """Configuration for inference/generation"""
    checkpoint_path: str = "checkpoints/best_model.pt"

    # Generation settings
    max_length: int = 1024
    temperature: float = 1.0
    top_k: int = 50
    top_p: float = 0.9
    enforce_raga_grammar: bool = True
    use_chalan_prefix: bool = True
    grammar_temperature_floor: float = 0.75


DEFAULT_AUDIO_CONFIG = AudioFeatureConfig()
DEFAULT_EXPRESSION_CONFIG = ExpressionEncoderConfig()
DEFAULT_MODEL_CONFIG = HybridCVAEConfig()
DEFAULT_TRAINING_CONFIG = TrainingConfig()
DEFAULT_INFERENCE_CONFIG = InferenceConfig()


if __name__ == "__main__":
    print("=== Audio Feature Config ===")
    print(DEFAULT_AUDIO_CONFIG)

    print("\n=== Expression Encoder Config ===")
    print(DEFAULT_EXPRESSION_CONFIG)

    print("\n=== Hybrid CVAE Config (2.5B) ===")
    print(DEFAULT_MODEL_CONFIG)

    print("\n=== Training Config (RTX PRO 6000) ===")
    print(DEFAULT_TRAINING_CONFIG)

    # Calculate approximate model size
    config = DEFAULT_MODEL_CONFIG
    vocab_size = config.vocab_size
    embed_dim = config.embed_dim
    num_enc = config.num_encoder_layers
    num_dec = config.num_decoder_layers
    ff_dim = config.ff_dim

    embedding_params = vocab_size * embed_dim
    encoder_params = num_enc * (4 * embed_dim**2 + 2 * embed_dim * ff_dim)
    decoder_params = num_dec * (6 * embed_dim**2 + 2 * embed_dim * ff_dim)
    expression_params = 500_000

    total = embedding_params + encoder_params + decoder_params + expression_params
    print(f"\n=== Estimated Model Size ===")
    print(f"Total parameters: ~{total / 1e9:.2f}B")
    print(f"BF16 size: ~{total * 2 / 1e9:.2f} GB")
    print(f"FP32 size: ~{total * 4 / 1e9:.2f} GB")
