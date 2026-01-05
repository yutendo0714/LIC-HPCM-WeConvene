"""
ChannelwiseFrequencyGate: Adaptive channel-wise weights for frequency bands

Motivation:
- LH (horizontal): Contains vertical edges → Important for structure
- HL (vertical): Contains horizontal edges → Important for structure  
- HH (diagonal): Contains texture details → Can be compressed more aggressively

This module learns per-channel importance for each frequency band.
"""
import torch
import torch.nn as nn


class ChannelwiseFrequencyGate(nn.Module):
    """
    Learn adaptive weights for each frequency band per channel
    
    Key insight from traditional image compression:
    - Edge information (LH, HL) is more perceptually important
    - Texture information (HH) can tolerate more compression
    
    We make this adaptive and learnable per channel.
    """
    def __init__(self, channels=320):
        super().__init__()
        self.channels = channels
        
        # Learnable per-band, per-channel weights
        # Initialize with prior knowledge:
        # - LH/HL: Full weight (edges are important)
        # - HH: Lower weight (texture can be compressed more)
        self.weight_lh = nn.Parameter(torch.ones(1, channels, 1, 1))
        self.weight_hl = nn.Parameter(torch.ones(1, channels, 1, 1))
        self.weight_hh = nn.Parameter(torch.ones(1, channels, 1, 1) * 0.8)  # Init lower
        
        # Global context-based adaptive gating
        self.gap = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Sequential(
            nn.Linear(channels, channels // 4),
            nn.ReLU(inplace=True),
            nn.Linear(channels // 4, channels),
            nn.Sigmoid()  # Output in [0, 1]
        )
    
    def forward(self, lh, hl, hh):
        """
        Args:
            lh: (B, M, H, W) - Horizontal detail band
            hl: (B, M, H, W) - Vertical detail band
            hh: (B, M, H, W) - Diagonal detail band
        Returns:
            lh_weighted, hl_weighted, hh_weighted: Adaptively weighted bands
        """
        # Compute global context from all HF bands
        # This helps the model understand overall image characteristics
        hf_concat = torch.cat([lh, hl, hh], dim=1)  # (B, 3M, H, W)
        global_ctx = self.gap(hf_concat).squeeze(-1).squeeze(-1)  # (B, 3M)
        
        # Average across the 3 bands to get per-channel context
        # (B, 3, M) -> (B, M)
        ctx = global_ctx.view(global_ctx.size(0), 3, -1).mean(dim=1)
        
        # Generate adaptive channel weights
        adaptive_weight = self.fc(ctx).unsqueeze(-1).unsqueeze(-1)  # (B, M, 1, 1)
        
        # Apply both learnable and adaptive weights
        # Learnable weights: Capture general frequency importance
        # Adaptive weights: Adjust based on input content
        lh_weighted = lh * self.weight_lh * adaptive_weight
        hl_weighted = hl * self.weight_hl * adaptive_weight
        hh_weighted = hh * self.weight_hh * adaptive_weight
        
        return lh_weighted, hl_weighted, hh_weighted
    
    def extra_repr(self):
        return f'channels={self.channels}'
