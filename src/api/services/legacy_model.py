
import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from dataclasses import dataclass
from typing import Optional, Tuple

@dataclass
class LegacyCVAEConfig:
    """
    Configuration for Legacy CVAE model (approx 47M parameters)
    Matches the architecture of the provided checkpoint.
    """
    # --- Model Dimensions ---
    vocab_size: int = 491        # Number of unique MIDI tokens (notes, velocities, time)
    max_seq_length: int = 512    # Maximum context window (how far back it looks)
    embed_dim: int = 512         # Vector size for each note (d_model)
    
    # --- Transformer Structure ---
    num_heads: int = 8           # Number of parallel attention heads
    num_encoder_layers: int = 8  # Layers in "Listening" part
    num_decoder_layers: int = 8  # Layers in "Generating" part
    ff_dim: int = 2048           # Internal size of feed-forward networks
    
    # --- Variational Components ---
    latent_dim: int = 256        # Size of the "Idea Vector" (z) - bottleneck
    dropout: float = 0.1         # Regularization to prevent overfitting
    
    # --- Conditioning (Controls) ---
    num_moods: int = 36          # Number of supported emotions
    num_ragas: int = 19          # Number of supported Ragas
    mood_embed_dim: int = 64     # Size of mood embedding
    raga_embed_dim: int = 64     # Size of raga embedding
    tempo_embed_dim: int = 32
    duration_embed_dim: int = 32
    
    kl_weight: float = 0.001


class PositionalEncoding(nn.Module):
    def __init__(self, d_model, max_len=5000):
        super(PositionalEncoding, self).__init__()
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        # [1, max_len, d_model] to match checkpoint format
        pe = pe.unsqueeze(0)
        self.register_buffer('pe', pe)

    def forward(self, x):
        # x: [seq_len, batch, dim] - transpose, add pe, transpose back
        x = x.transpose(0, 1)  # [batch, seq_len, dim]
        x = x + self.pe[:, :x.size(1), :]
        x = x.transpose(0, 1)  # [seq_len, batch, dim]
        return x


class LegacyConditioningModule(nn.Module):
    def __init__(self, config: LegacyCVAEConfig):
        super().__init__()
        self.config = config
        
        self.mood_embedding = nn.Embedding(config.num_moods, config.mood_embed_dim)
        self.raga_embedding = nn.Embedding(config.num_ragas, config.raga_embed_dim)
        self.tempo_bins = nn.Embedding(32, config.tempo_embed_dim)
        self.duration_bins = nn.Embedding(16, config.duration_embed_dim)
        
        total_cond_dim = (config.mood_embed_dim + config.raga_embed_dim + 
                         config.tempo_embed_dim + config.duration_embed_dim)
        
        # Legacy projection was a single Linear layer likely followed by activation or just Linear
        # Inspecting keys: 'conditioning.projection.weight' implies single layer or nn.Linear
        # If it was Sequential, it would be 'conditioning.projection.0.weight'
        self.projection = nn.Linear(total_cond_dim, config.embed_dim)
    
    def forward(self, mood, raga, tempo, duration):
        mood_emb = self.mood_embedding(mood)
        raga_emb = self.raga_embedding(raga)
        tempo_emb = self.tempo_bins(tempo)
        duration_emb = self.duration_bins(duration)
        
        combined = torch.cat([mood_emb, raga_emb, tempo_emb, duration_emb], dim=-1)
        return F.relu(self.projection(combined)) # Assuming ReLU based on common practice, but could be raw


class LegacyTransformerEncoder(nn.Module):
    def __init__(self, config: LegacyCVAEConfig):
        super().__init__()
        self.config = config
        
        self.token_embedding = nn.Embedding(config.vocab_size, config.embed_dim)
        self.pos_encoding = PositionalEncoding(config.embed_dim, config.max_seq_length)
        
        self.embed_scale = math.sqrt(config.embed_dim)
        
        encoder_layers = nn.TransformerEncoderLayer(
            config.embed_dim, 
            config.num_heads, 
            config.ff_dim, 
            config.dropout
        )
        self.transformer = nn.TransformerEncoder(encoder_layers, config.num_encoder_layers)
        
        self.to_mu = nn.Linear(config.embed_dim, config.latent_dim)
        self.to_logvar = nn.Linear(config.embed_dim, config.latent_dim)

    def forward(self, x, conditioning, padding_mask=None):
        # x: [batch, seq_len]
        # conditioning: [batch, embed_dim]
        
        x = self.token_embedding(x) * self.embed_scale
        x = x.transpose(0, 1) # [seq_len, batch, dim]
        
        # Add conditioning as first token
        conditioning = conditioning.unsqueeze(0) # [1, batch, dim]
        x = torch.cat([conditioning, x], dim=0) # [seq_len+1, batch, dim]
        
        x = self.pos_encoding(x)
        
        # Padding mask needs adjustment for extra token
        if padding_mask is not None:
             # Add False (not padded) for conditioning token
             cond_mask = torch.zeros((padding_mask.size(0), 1), dtype=torch.bool, device=padding_mask.device)
             padding_mask = torch.cat([cond_mask, padding_mask], dim=1)
        
        x = self.transformer(x, src_key_padding_mask=padding_mask)
        
        # Pool (mean of first token or all? Legacy usually pooled all or used first)
        # Let's assume mean pooling for CVAE
        x = x.transpose(0, 1) # [batch, seq_len, dim]
        x = x.mean(dim=1)
        
        return self.to_mu(x), self.to_logvar(x)


class LegacyTransformerDecoder(nn.Module):
    def __init__(self, config: LegacyCVAEConfig):
        super().__init__()
        self.config = config
        
        self.token_embedding = nn.Embedding(config.vocab_size, config.embed_dim)
        self.pos_encoding = PositionalEncoding(config.embed_dim, config.max_seq_length)
        self.embed_scale = math.sqrt(config.embed_dim)
        
        self.latent_projection = nn.Linear(config.latent_dim, config.embed_dim)
        
        decoder_layers = nn.TransformerDecoderLayer(
            config.embed_dim,
            config.num_heads,
            config.ff_dim,
            config.dropout
        )
        self.transformer = nn.TransformerDecoder(decoder_layers, config.num_decoder_layers)
        
        self.output_projection = nn.Linear(config.embed_dim, config.vocab_size)
    
    def forward(self, z, conditioning, target, target_padding_mask=None):
        # z: [batch, latent_dim]
        # conditioning: [batch, embed_dim]
        # target: [batch, seq_len]
        
        z_emb = self.latent_projection(z)
        
        # Memory is z + conditioning
        memory = (z_emb + conditioning).unsqueeze(0) # [1, batch, dim]
        
        # Target embedding
        tgt = self.token_embedding(target) * self.embed_scale
        tgt = tgt.transpose(0, 1) # [seq_len, batch, dim]
        tgt = self.pos_encoding(tgt)
        
        # Causal mask
        sz = tgt.size(0)
        mask = torch.triu(torch.ones(sz, sz, device=tgt.device), diagonal=1).bool()
        
        out = self.transformer(
            tgt, 
            memory, 
            tgt_mask=mask,
            tgt_key_padding_mask=target_padding_mask
        )
        
        out = out.transpose(0, 1) # [batch, seq_len, dim]
        return self.output_projection(out)

    def generate(self, z, conditioning, max_length=512, temperature=1.0, top_k=50):
        # Simple greedy/top-k generation
        batch_size = z.size(0)
        device = z.device
        
        z_emb = self.latent_projection(z)
        memory = (z_emb + conditioning).unsqueeze(0)
        
        generated = torch.full((batch_size, 1), 1, dtype=torch.long, device=device) # BOS=1
        
        for _ in range(max_length):
            tgt = self.token_embedding(generated) * self.embed_scale
            tgt = tgt.transpose(0, 1)
            tgt = self.pos_encoding(tgt)
            
            sz = tgt.size(0)
            mask = torch.triu(torch.ones(sz, sz, device=device), diagonal=1).bool()
            
            out = self.transformer(tgt, memory, tgt_mask=mask)
            logits = self.output_projection(out[-1, :, :])
            
            logits = logits / temperature
            if top_k > 0:
                indices_to_remove = logits < torch.topk(logits, top_k)[0][..., -1, None]
                logits = logits.masked_fill(indices_to_remove, float('-inf'))
            
            probs = F.softmax(logits, dim=-1)
            next_token = torch.multinomial(probs, 1)
            
            generated = torch.cat([generated, next_token], dim=1)
            
            if (next_token == 2).all(): # EOS=2
                break
                
        return generated


class LegacyRagaCVAE(nn.Module):
    """Legacy CVAE support"""
    def __init__(self, config: LegacyCVAEConfig = None):
        super().__init__()
        self.config = config or LegacyCVAEConfig()
        
        self.conditioning = LegacyConditioningModule(self.config)
        self.encoder = LegacyTransformerEncoder(self.config)
        self.decoder = LegacyTransformerDecoder(self.config)
        
    def reparameterize(self, mu, logvar):
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mu + eps * std
        
    def forward(self, x, mood, raga, tempo, duration, padding_mask=None):
        cond = self.conditioning(mood, raga, tempo, duration)
        mu, logvar = self.encoder(x, cond, padding_mask)
        z = self.reparameterize(mu, logvar)
        logits = self.decoder(z, cond, x, padding_mask)
        return logits, mu, logvar
    
    def generate(self, mood, raga, tempo, duration, max_length=512, temperature=1.0, top_k=50):
        batch_size = mood.size(0)
        device = mood.device
        
        cond = self.conditioning(mood, raga, tempo, duration)
        z = torch.randn(batch_size, self.config.latent_dim, device=device)
        
        return self.decoder.generate(z, cond, max_length, temperature, top_k)
