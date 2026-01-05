import torch
import torch.nn as nn
import torch.nn.functional as F
from .wavelet import DWT2D, IDWT2D
from .conv import conv1x1


class WeConvBlock(nn.Module):
    """
    Wavelet-enhanced Convolution Block
    Processes features in wavelet domain for better frequency-aware representation
    """
    def __init__(self, channels, mlp_ratio=4, partial_ratio=4):
        super().__init__()
        self.channels = channels
        
        # Wavelet transform
        self.dwt = DWT2D("haar")
        self.idwt = IDWT2D("haar")
        
        # LL branch (low-frequency): deeper processing
        self.ll_conv1 = nn.Conv2d(channels, channels, 3, 1, 1, groups=channels)
        self.ll_conv2 = nn.Conv2d(channels, channels, 1, 1, 0)
        
        # HF branch (high-frequency): lighter processing
        self.hf_conv = nn.Conv2d(channels * 3, channels * 3, 3, 1, 1, groups=channels * 3)
        
        # Fusion
        self.fusion = nn.Sequential(
            nn.Conv2d(channels * 4, channels, 1, 1, 0),
            nn.GELU(),
            nn.Conv2d(channels, channels, 3, 1, 1, groups=channels),
        )
        
        # MLP (channel mixing)
        hidden_dim = int(channels * mlp_ratio)
        self.mlp = nn.Sequential(
            nn.Conv2d(channels, hidden_dim, 1, 1, 0),
            nn.GELU(),
            nn.Conv2d(hidden_dim, channels, 1, 1, 0),
        )
        
        self.norm1 = nn.LayerNorm(channels)
        self.norm2 = nn.LayerNorm(channels)
        
    def forward(self, x):
        B, C, H, W = x.shape
        identity = x
        
        # DWT
        x_dwt = self.dwt(x)
        x_ll, x_lh, x_hl, x_hh = torch.chunk(x_dwt, 4, dim=1)
        
        # Process LL (structural information)
        x_ll = self.ll_conv1(x_ll)
        x_ll = F.gelu(x_ll)
        x_ll = self.ll_conv2(x_ll)
        
        # Process HF (texture information)
        x_hf = torch.cat([x_lh, x_hl, x_hh], dim=1)
        x_hf = self.hf_conv(x_hf)
        x_hf = F.gelu(x_hf)
        
        # Split HF back
        x_lh, x_hl, x_hh = torch.chunk(x_hf, 3, dim=1)
        
        # IDWT
        x_dwt_processed = torch.cat([x_ll, x_lh, x_hl, x_hh], dim=1)
        x = self.idwt(x_dwt_processed)
        
        # Fusion with residual
        x = self.fusion(torch.cat([x, identity], dim=1)) if x.shape == identity.shape else x
        
        # LayerNorm + residual
        x_norm = x.permute(0, 2, 3, 1)  # B,H,W,C
        x_norm = self.norm1(x_norm)
        x_norm = x_norm.permute(0, 3, 1, 2)  # B,C,H,W
        x = x + identity
        
        # MLP
        identity2 = x
        x_mlp = self.mlp(x)
        x_mlp_norm = x_mlp.permute(0, 2, 3, 1)
        x_mlp_norm = self.norm2(x_mlp_norm)
        x_mlp_norm = x_mlp_norm.permute(0, 3, 1, 2)
        x = x_mlp_norm + identity2
        
        return x


class WeConvBlockSimple(nn.Module):
    """
    Simplified WeConv block for initial testing
    Only applies wavelet domain processing without heavy operations
    """
    def __init__(self, channels):
        super().__init__()
        self.channels = channels
        
        self.dwt = DWT2D("haar")
        self.idwt = IDWT2D("haar")
        
        # Simple 1x1 convolutions in wavelet domain
        self.ll_conv = conv1x1(channels, channels)
        self.hf_conv = conv1x1(channels * 3, channels * 3)
        
        # Residual connection
        self.alpha = nn.Parameter(torch.ones(1) * 0.1)
        
    def forward(self, x):
        identity = x
        
        # DWT
        x_dwt = self.dwt(x)
        x_ll, x_lh, x_hl, x_hh = torch.chunk(x_dwt, 4, dim=1)
        
        # Process in wavelet domain
        x_ll = self.ll_conv(x_ll)
        x_hf = torch.cat([x_lh, x_hl, x_hh], dim=1)
        x_hf = self.hf_conv(x_hf)
        x_lh, x_hl, x_hh = torch.chunk(x_hf, 3, dim=1)
        
        # IDWT
        x_dwt_processed = torch.cat([x_ll, x_lh, x_hl, x_hh], dim=1)
        x = self.idwt(x_dwt_processed)
        
        # Weighted residual
        x = identity + self.alpha * x
        
        return x
