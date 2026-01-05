"""
FrequencyAttentionFusion: Cross-attention between LL context and HF bands

Key idea from WeConvene ECCV2024:
"Low-frequency coefficients are encoded first and used as prior for high-frequency"
→ We enhance this with attention mechanism for better correlation modeling
"""
import torch
import torch.nn as nn


class FrequencyAttentionFusion(nn.Module):
    """
    Cross-attention between LL context and HF bands
    - Query: HF band features (LH/HL/HH)
    - Key/Value: LL context features
    
    This allows HF bands to selectively attend to relevant LL structures,
    improving compression efficiency by exploiting frequency correlation.
    """
    def __init__(self, dim=320, num_heads=8, dropout=0.0):
        super().__init__()
        self.dim = dim
        self.num_heads = num_heads
        
        # Cross-attention layer
        self.cross_attn = nn.MultiheadAttention(
            embed_dim=dim, 
            num_heads=num_heads,
            dropout=dropout,
            batch_first=True
        )
        
        # Layer normalization
        self.norm1 = nn.LayerNorm(dim)
        self.norm2 = nn.LayerNorm(dim)
        
        # Feed-forward network
        self.ffn = nn.Sequential(
            nn.Linear(dim, dim * 4),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(dim * 4, dim),
            nn.Dropout(dropout)
        )
    
    def forward(self, hf_band, ll_context):
        """
        Args:
            hf_band: (B, M, H, W) - HF band (LH/HL/HH)
            ll_context: (B, M, H, W) - LL context features
        Returns:
            fused: (B, M, H, W) - Attention-fused features
            attn_weights: (B, num_heads, H*W, H*W) - Attention weights for visualization
        """
        B, M, H, W = hf_band.shape
        
        # Reshape to sequence: (B, H*W, M)
        hf_seq = hf_band.flatten(2).transpose(1, 2)  # (B, HW, M)
        ll_seq = ll_context.flatten(2).transpose(1, 2)  # (B, HW, M)
        
        # Cross-attention: HF queries LL context
        attn_out, attn_weights = self.cross_attn(
            query=self.norm1(hf_seq),
            key=ll_seq,
            value=ll_seq
        )
        hf_seq = hf_seq + attn_out  # Residual connection
        
        # Feed-forward network with residual
        hf_seq = hf_seq + self.ffn(self.norm2(hf_seq))
        
        # Reshape back: (B, M, H, W)
        fused = hf_seq.transpose(1, 2).reshape(B, M, H, W)
        
        return fused, attn_weights
    
    def extra_repr(self):
        return f'dim={self.dim}, num_heads={self.num_heads}'
