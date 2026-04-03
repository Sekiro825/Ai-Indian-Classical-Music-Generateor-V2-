"""
Expression Encoder for Hybrid MIDI-Audio Architecture
Encodes audio expression features (f0, amplitude, spectral) into a latent representation.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional


class PositionalEncoding(nn.Module):
    """Sinusoidal positional encoding for sequence models"""
    
    def __init__(self, d_model: int, max_len: int = 2048, dropout: float = 0.1):
        super().__init__()
        self.dropout = nn.Dropout(p=dropout)
        
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-torch.log(torch.tensor(10000.0)) / d_model))
        
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0)  # (1, max_len, d_model)
        
        self.register_buffer('pe', pe)
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: (batch, seq_len, d_model)
        """
        x = x + self.pe[:, :x.size(1), :]
        return self.dropout(x)


class ExpressionEncoder(nn.Module):
    """
    Encodes audio expression features into a latent representation.
    
    Input: (batch, seq_len, 4) - [f0, amplitude, voiced, spectral_centroid]
    Output: (batch, seq_len, embed_dim) - expression embeddings
    """
    
    def __init__(
        self,
        input_dim: int = 4,  # [f0, amplitude, voiced, spectral_centroid]
        hidden_dim: int = 256,
        embed_dim: int = 128,
        num_layers: int = 2,
        num_heads: int = 4,
        dropout: float = 0.1,
        use_transformer: bool = True
    ):
        super().__init__()
        
        self.input_dim = input_dim
        self.embed_dim = embed_dim
        self.use_transformer = use_transformer
        
        # Project input features to hidden dimension
        self.input_projection = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.GELU(),
            nn.LayerNorm(hidden_dim),
            nn.Dropout(dropout)
        )
        
        if use_transformer:
            # Transformer encoder for temporal modeling
            self.pos_encoding = PositionalEncoding(hidden_dim, dropout=dropout)
            
            encoder_layer = nn.TransformerEncoderLayer(
                d_model=hidden_dim,
                nhead=num_heads,
                dim_feedforward=hidden_dim * 4,
                dropout=dropout,
                activation='gelu',
                batch_first=True
            )
            self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        else:
            # Simpler bi-directional LSTM
            self.lstm = nn.LSTM(
                input_size=hidden_dim,
                hidden_size=hidden_dim // 2,
                num_layers=num_layers,
                batch_first=True,
                bidirectional=True,
                dropout=dropout if num_layers > 1 else 0
            )
        
        # Output projection to embedding dimension
        self.output_projection = nn.Sequential(
            nn.Linear(hidden_dim, embed_dim),
            nn.LayerNorm(embed_dim)
        )
        
    def forward(
        self, 
        expression: torch.Tensor,
        padding_mask: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """
        Args:
            expression: (batch, seq_len, 4) audio features
            padding_mask: (batch, seq_len) True for padded positions
            
        Returns:
            (batch, seq_len, embed_dim) expression embeddings
        """
        # Project to hidden dimension
        x = self.input_projection(expression)  # (batch, seq_len, hidden_dim)
        
        if self.use_transformer:
            x = self.pos_encoding(x)
            x = self.transformer(x, src_key_padding_mask=padding_mask)
        else:
            x, _ = self.lstm(x)
        
        # Project to output dimension
        x = self.output_projection(x)  # (batch, seq_len, embed_dim)
        
        return x


class ExpressionHead(nn.Module):
    """
    Predicts expression features from decoder hidden states.
    Used during inference to generate expression from MIDI.
    
    Input: (batch, seq_len, hidden_dim) - decoder output
    Output: (batch, seq_len, 4) - predicted [f0, amplitude, voiced, spectral_centroid]
    """
    
    def __init__(
        self,
        hidden_dim: int = 1280,
        intermediate_dim: int = 256,
        output_dim: int = 4,
        dropout: float = 0.1
    ):
        super().__init__()
        
        self.output_dim = output_dim
        
        self.head = nn.Sequential(
            nn.Linear(hidden_dim, intermediate_dim),
            nn.GELU(),
            nn.LayerNorm(intermediate_dim),
            nn.Dropout(dropout),
            nn.Linear(intermediate_dim, intermediate_dim // 2),
            nn.GELU(),
            nn.Linear(intermediate_dim // 2, output_dim)
        )
        
        # Separate heads for different feature types with appropriate activations
        self.f0_head = nn.Linear(output_dim, 1)  # Unbounded (normalized cents)
        self.amp_head = nn.Linear(output_dim, 1)  # Sigmoid for [0, 1]
        self.voiced_head = nn.Linear(output_dim, 1)  # Sigmoid for probability
        self.centroid_head = nn.Linear(output_dim, 1)  # Sigmoid for [0, 1]
        
    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        """
        Args:
            hidden_states: (batch, seq_len, hidden_dim)
            
        Returns:
            (batch, seq_len, 4) predicted expression features
        """
        x = self.head(hidden_states)  # (batch, seq_len, output_dim)
        
        # Apply appropriate activations for each feature
        f0 = torch.tanh(self.f0_head(x))  # [-1, 1] for normalized cents
        amp = torch.sigmoid(self.amp_head(x))  # [0, 1]
        voiced = torch.sigmoid(self.voiced_head(x))  # [0, 1] probability
        centroid = torch.sigmoid(self.centroid_head(x))  # [0, 1]
        
        return torch.cat([f0, amp, voiced, centroid], dim=-1)


class GlobalExpressionEncoder(nn.Module):
    """
    Encodes expression features into a single global vector.
    Used for conditioning the latent space.
    
    Input: (batch, seq_len, 4) audio features
    Output: (batch, embed_dim) global expression vector
    """
    
    def __init__(
        self,
        input_dim: int = 4,
        hidden_dim: int = 256,
        embed_dim: int = 128,
        dropout: float = 0.1
    ):
        super().__init__()
        
        self.frame_encoder = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.GELU(),
            nn.LayerNorm(hidden_dim),
            nn.Dropout(dropout)
        )
        
        # Attention pooling to get a single vector
        self.attention = nn.Sequential(
            nn.Linear(hidden_dim, 1),
            nn.Softmax(dim=1)
        )
        
        self.output = nn.Sequential(
            nn.Linear(hidden_dim, embed_dim),
            nn.LayerNorm(embed_dim)
        )
        
    def forward(
        self, 
        expression: torch.Tensor,
        padding_mask: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """
        Args:
            expression: (batch, seq_len, 4)
            padding_mask: (batch, seq_len) True for padded positions
            
        Returns:
            (batch, embed_dim) global expression embedding
        """
        # Encode each frame
        x = self.frame_encoder(expression)  # (batch, seq_len, hidden_dim)
        
        # Compute attention weights
        attn_weights = self.attention(x)  # (batch, seq_len, 1)
        
        # Mask out padded positions
        if padding_mask is not None:
            attn_weights = attn_weights.masked_fill(
                padding_mask.unsqueeze(-1), 
                float('-inf')
            )
            attn_weights = F.softmax(attn_weights, dim=1)
        
        # Weighted sum
        pooled = (x * attn_weights).sum(dim=1)  # (batch, hidden_dim)
        
        # Project to output dimension
        return self.output(pooled)  # (batch, embed_dim)


if __name__ == "__main__":
    # Test modules
    batch_size = 4
    seq_len = 345  # ~8s at 22050Hz with hop_length=512
    
    # Test ExpressionEncoder
    encoder = ExpressionEncoder()
    x = torch.randn(batch_size, seq_len, 4)
    out = encoder(x)
    print(f"ExpressionEncoder output: {out.shape}")  # (4, 345, 128)
    
    # Test ExpressionHead
    head = ExpressionHead()
    hidden = torch.randn(batch_size, seq_len, 1280)
    pred = head(hidden)
    print(f"ExpressionHead output: {pred.shape}")  # (4, 345, 4)
    
    # Test GlobalExpressionEncoder
    global_enc = GlobalExpressionEncoder()
    global_out = global_enc(x)
    print(f"GlobalExpressionEncoder output: {global_out.shape}")  # (4, 128)
    
    # Count parameters
    for name, model in [("ExpressionEncoder", encoder), 
                        ("ExpressionHead", head), 
                        ("GlobalExpressionEncoder", global_enc)]:
        params = sum(p.numel() for p in model.parameters())
        print(f"{name}: {params:,} parameters")
