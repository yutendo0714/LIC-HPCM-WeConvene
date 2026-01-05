"""
LLContextEncoder: Extract rich context from LL band

The LL band contains the most important structural information.
We extract rich contextual features from LL to guide HF band processing.

Architecture:
1. Self-attention: Capture long-range dependencies in LL
2. Convolutional layers: Extract local spatial context
3. Residual connections: Preserve original LL information
"""
import torch
import torch.nn as nn


class LLContextEncoder(nn.Module):
    """
    Extract rich context from LL band to guide HF processing
    
    Motivation:
    - LL band contains global structure and most energy
    - Better LL context → Better HF prediction
    - Self-attention captures long-range structural patterns
    - CNN captures local texture patterns
    
    Output context is used as Key/Value in cross-attention with HF bands.
    """
    def __init__(self, channels=320, num_heads=8, dropout=0.0):
        super().__init__()
        self.channels = channels
        
        # Self-attention on LL band
        # This captures global structural dependencies
        self.self_attn = nn.MultiheadAttention(
            embed_dim=channels,
            num_heads=num_heads,
            dropout=dropout,
            batch_first=True
        )
        self.norm1 = nn.LayerNorm(channels)
        
        # Spatial context extraction via CNN
        # This captures local patterns and textures
        self.conv_layers = nn.Sequential(
            nn.Conv2d(channels, channels, 3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(channels, channels, 3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(channels, channels, 3, padding=1)
        )
        self.norm2 = nn.BatchNorm2d(channels)
        
        # Optional: Additional refinement
        self.refine = nn.Sequential(
            nn.Conv2d(channels, channels, 1),
            nn.ReLU(inplace=True),
            nn.Conv2d(channels, channels, 1)
        )
    
    def forward(self, ll):
        """
        Extract rich context from LL band
        
        Args:
            ll: (B, M, H, W) - LL band features
        Returns:
            context: (B, M, H, W) - Rich LL context for cross-attention
        """
        B, M, H, W = ll.shape
        
        # Step 1: Self-attention for global context
        ll_seq = ll.flatten(2).transpose(1, 2)  # (B, H*W, M)
        attn_out, _ = self.self_attn(
            self.norm1(ll_seq),
            self.norm1(ll_seq),
            self.norm1(ll_seq)
        )
        ll_attn = (ll_seq + attn_out).transpose(1, 2).reshape(B, M, H, W)
        
        # Step 2: Spatial context extraction via CNN
        context = self.conv_layers(ll_attn)
        context = self.norm2(context) + ll  # Residual with original LL
        
        # Step 3: Final refinement
        context = context + self.refine(context)  # Residual
        
        return context
    
    def extra_repr(self):
        return f'channels={self.channels}'
