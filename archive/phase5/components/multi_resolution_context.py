"""
MultiResolutionContextBuilder: Build context from coarse to fine resolution

Inspired by progressive decoding in ELIC (CVPR 2022) and VCT (ICLR 2022):
"Multi-scale context captures both global structure and local details"

This module extracts context at multiple resolutions and fuses them.
"""
import torch
import torch.nn as nn
import torch.nn.functional as F


class MultiResolutionContextBuilder(nn.Module):
    """
    Build context from coarse to fine resolution
    
    Architecture:
    1. Coarse path: 4× downsampled → CNN → upsample
    2. Medium path: 2× downsampled → CNN → upsample
    3. Fine path: Original resolution → CNN
    4. Fusion: Concatenate and merge all scales
    
    This provides rich multi-scale context for better prediction.
    """
    def __init__(self, M, num_scales=3):
        super().__init__()
        self.M = M
        self.num_scales = num_scales
        
        # Coarse-scale context (4× downsampled)
        self.coarse_context = nn.Sequential(
            nn.AvgPool2d(4, 4),  # Downsample
            nn.Conv2d(M, M, 3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(M, M, 3, padding=1),
            nn.ReLU(inplace=True),
            # Upsample handled in forward()
        )
        
        # Medium-scale context (2× downsampled)
        self.medium_context = nn.Sequential(
            nn.AvgPool2d(2, 2),  # Downsample
            nn.Conv2d(M, M, 3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(M, M, 3, padding=1),
            nn.ReLU(inplace=True),
            # Upsample handled in forward()
        )
        
        # Fine-scale context (original resolution)
        self.fine_context = nn.Sequential(
            nn.Conv2d(M, M, 3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(M, M, 3, padding=1),
            nn.ReLU(inplace=True)
        )
        
        # Multi-scale fusion
        self.fusion = nn.Sequential(
            nn.Conv2d(M * num_scales, M*2, 1),  # 1x1 bottleneck
            nn.ReLU(inplace=True),
            nn.Conv2d(M*2, M, 1)
        )
    
    def forward(self, y_hat_so_far):
        """
        Extract multi-resolution context
        
        Args:
            y_hat_so_far: (B, M, H, W) - Already decoded portions
                Can be partially decoded (masked) or fully decoded
        Returns:
            multi_res_context: (B, M, H, W) - Rich multi-scale context
        """
        B, M, H, W = y_hat_so_far.shape
        
        # Coarse-scale processing
        coarse = self.coarse_context(y_hat_so_far)
        coarse = F.interpolate(coarse, size=(H, W), mode='nearest')
        
        # Medium-scale processing
        medium = self.medium_context(y_hat_so_far)
        medium = F.interpolate(medium, size=(H, W), mode='nearest')
        
        # Fine-scale processing
        fine = self.fine_context(y_hat_so_far)
        
        # Concatenate all scales
        multi_scale = torch.cat([coarse, medium, fine], dim=1)
        
        # Fuse into unified context
        context = self.fusion(multi_scale)
        
        return context
    
    def extra_repr(self):
        return f'M={self.M}, num_scales={self.num_scales}'


class LightweightMultiResContext(nn.Module):
    """
    Lightweight version using depthwise separable convolutions
    """
    def __init__(self, M):
        super().__init__()
        self.M = M
        
        # Depthwise separable convolutions for efficiency
        def depthwise_separable(in_c, out_c, kernel_size=3):
            return nn.Sequential(
                nn.Conv2d(in_c, in_c, kernel_size, padding=kernel_size//2, groups=in_c),
                nn.ReLU(inplace=True),
                nn.Conv2d(in_c, out_c, 1),
                nn.ReLU(inplace=True)
            )
        
        self.coarse_context = depthwise_separable(M, M)
        self.medium_context = depthwise_separable(M, M)
        self.fine_context = depthwise_separable(M, M)
        
        # Lightweight fusion
        self.fusion = nn.Sequential(
            nn.Conv2d(M * 3, M, 1),
            nn.ReLU(inplace=True)
        )
    
    def forward(self, y_hat_so_far):
        B, M, H, W = y_hat_so_far.shape
        
        # Multi-scale processing
        coarse_in = F.avg_pool2d(y_hat_so_far, 4, 4)
        coarse = self.coarse_context(coarse_in)
        coarse = F.interpolate(coarse, size=(H, W), mode='nearest')
        
        medium_in = F.avg_pool2d(y_hat_so_far, 2, 2)
        medium = self.medium_context(medium_in)
        medium = F.interpolate(medium, size=(H, W), mode='nearest')
        
        fine = self.fine_context(y_hat_so_far)
        
        # Fusion
        multi_scale = torch.cat([coarse, medium, fine], dim=1)
        context = self.fusion(multi_scale)
        
        return context
    
    def extra_repr(self):
        return f'M={self.M} (lightweight)'


class AdaptiveMultiResContext(nn.Module):
    """
    Adaptive version that learns scale weights
    
    Some images benefit more from coarse context (structure-heavy)
    Others benefit from fine context (texture-heavy)
    """
    def __init__(self, M):
        super().__init__()
        self.M = M
        
        # Multi-scale extractors
        self.coarse_context = nn.Sequential(
            nn.Conv2d(M, M, 3, padding=1),
            nn.ReLU(inplace=True)
        )
        self.medium_context = nn.Sequential(
            nn.Conv2d(M, M, 3, padding=1),
            nn.ReLU(inplace=True)
        )
        self.fine_context = nn.Sequential(
            nn.Conv2d(M, M, 3, padding=1),
            nn.ReLU(inplace=True)
        )
        
        # Learnable scale importance
        self.scale_weights = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),  # Global context
            nn.Conv2d(M, M // 4, 1),
            nn.ReLU(inplace=True),
            nn.Conv2d(M // 4, 3, 1),  # 3 scale weights
            nn.Softmax(dim=1)  # Normalize to sum to 1
        )
        
        # Fusion
        self.fusion = nn.Sequential(
            nn.Conv2d(M * 3, M, 1),
            nn.ReLU(inplace=True)
        )
    
    def forward(self, y_hat_so_far):
        B, M, H, W = y_hat_so_far.shape
        
        # Multi-scale processing
        coarse_in = F.avg_pool2d(y_hat_so_far, 4, 4)
        coarse = self.coarse_context(coarse_in)
        coarse = F.interpolate(coarse, size=(H, W), mode='nearest')
        
        medium_in = F.avg_pool2d(y_hat_so_far, 2, 2)
        medium = self.medium_context(medium_in)
        medium = F.interpolate(medium, size=(H, W), mode='nearest')
        
        fine = self.fine_context(y_hat_so_far)
        
        # Learn adaptive scale weights
        weights = self.scale_weights(y_hat_so_far)  # (B, 3, 1, 1)
        w_coarse, w_medium, w_fine = weights.chunk(3, dim=1)
        
        # Weighted combination
        weighted_coarse = coarse * w_coarse
        weighted_medium = medium * w_medium
        weighted_fine = fine * w_fine
        
        # Fusion
        multi_scale = torch.cat([weighted_coarse, weighted_medium, weighted_fine], dim=1)
        context = self.fusion(multi_scale)
        
        return context
    
    def extra_repr(self):
        return f'M={self.M} (adaptive)'


# Factory function
def create_multi_res_context(M, variant='standard'):
    """
    Factory function for multi-resolution context
    
    Args:
        M: Number of channels
        variant: 'standard', 'lightweight', or 'adaptive'
    """
    if variant == 'lightweight':
        return LightweightMultiResContext(M)
    elif variant == 'adaptive':
        return AdaptiveMultiResContext(M)
    else:
        return MultiResolutionContextBuilder(M)


# Testing
if __name__ == "__main__":
    print("Multi-Resolution Context Builder Test")
    print("=" * 50)
    
    M = 320
    B, H, W = 2, 32, 32
    
    # Test standard version
    print(f"\nTesting MultiResolutionContextBuilder (M={M})...")
    builder = MultiResolutionContextBuilder(M)
    x = torch.randn(B, M, H, W)
    
    with torch.no_grad():
        out = builder(x)
    
    print(f"Input shape: {x.shape}")
    print(f"Output shape: {out.shape}")
    assert out.shape == (B, M, H, W), "Output shape mismatch"
    print("✓ Standard version passed")
    
    # Test lightweight version
    print(f"\nTesting LightweightMultiResContext (M={M})...")
    builder_light = LightweightMultiResContext(M)
    
    with torch.no_grad():
        out_light = builder_light(x)
    
    print(f"Output shape: {out_light.shape}")
    assert out_light.shape == (B, M, H, W), "Output shape mismatch"
    print("✓ Lightweight version passed")
    
    # Test adaptive version
    print(f"\nTesting AdaptiveMultiResContext (M={M})...")
    builder_adaptive = AdaptiveMultiResContext(M)
    
    with torch.no_grad():
        out_adaptive = builder_adaptive(x)
    
    print(f"Output shape: {out_adaptive.shape}")
    assert out_adaptive.shape == (B, M, H, W), "Output shape mismatch"
    print("✓ Adaptive version passed")
    
    # Compare parameters
    params_standard = sum(p.numel() for p in builder.parameters())
    params_light = sum(p.numel() for p in builder_light.parameters())
    params_adaptive = sum(p.numel() for p in builder_adaptive.parameters())
    
    print(f"\nParameter comparison:")
    print(f"Standard: {params_standard:,} parameters")
    print(f"Lightweight: {params_light:,} parameters ({(1-params_light/params_standard)*100:.1f}% reduction)")
    print(f"Adaptive: {params_adaptive:,} parameters")
    
    print("\n✓ All tests passed!")
