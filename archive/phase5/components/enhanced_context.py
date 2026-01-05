"""
EnhancedContextNetwork: Deeper context aggregation with spatial attention

Motivation from STF (CVPR 2023) and MLIC (CVPR 2023):
"Deeper context aggregation captures complex spatial patterns"

Standard HPCM context: 2-3 layers
Phase 5 Enhanced: 5 layers + spatial attention
"""
import torch
import torch.nn as nn


class EnhancedContextNetwork(nn.Module):
    """
    Deeper context aggregation network with spatial attention
    
    Architecture:
    1. Spatial attention on input context features
    2. Deep 5-layer CNN processing
    3. Output: scale and mean parameters
    
    This replaces HPCM's standard y_spatial_prior modules
    """
    def __init__(self, M, use_attention=True):
        super().__init__()
        self.M = M
        self.use_attention = use_attention
        
        # Spatial attention module
        if use_attention:
            self.spatial_attn = nn.Sequential(
                nn.Conv2d(M*3, M, 1),
                nn.ReLU(inplace=True),
                nn.Conv2d(M, M // 4, 1),
                nn.ReLU(inplace=True),
                nn.Conv2d(M // 4, 1, 1),
                nn.Sigmoid()  # Attention weights [0, 1]
            )
        
        # Deeper CNN stack (5 layers vs HPCM's 2-3)
        self.conv_stack = nn.Sequential(
            nn.Conv2d(M*3, M*2, 3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(M*2, M*2, 3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(M*2, M*2, 3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(M*2, M, 3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(M, 2*M, 1)  # Output: scale + mean
        )
        
        # Residual connection
        self.residual_proj = nn.Conv2d(M*3, 2*M, 1)
    
    def forward(self, context_features):
        """
        Args:
            context_features: (B, 3M, H, W) - Aggregated context
                Typically: [decoded_latents, hyper_params, additional_context]
        Returns:
            params: (B, 2M, H, W) - scale and mean parameters
        """
        # Spatial attention (content-adaptive weighting)
        if self.use_attention:
            attn_weights = self.spatial_attn(context_features)  # (B, 1, H, W)
            context_attended = context_features * attn_weights
        else:
            context_attended = context_features
        
        # Deep CNN processing
        params = self.conv_stack(context_attended)
        
        # Residual connection for stable training
        params = params + self.residual_proj(context_features)
        
        return params
    
    def extra_repr(self):
        return f'M={self.M}, use_attention={self.use_attention}'


class LightweightEnhancedContext(nn.Module):
    """
    Lightweight version for faster inference
    
    Trade-off: Slightly lower compression ratio but faster
    """
    def __init__(self, M):
        super().__init__()
        self.M = M
        
        # Lightweight attention (depthwise separable)
        self.spatial_attn = nn.Sequential(
            nn.Conv2d(M*3, M*3, 3, padding=1, groups=M*3),  # Depthwise
            nn.Conv2d(M*3, 1, 1),  # Pointwise
            nn.Sigmoid()
        )
        
        # Shallower but wider CNN (3 layers)
        self.conv_stack = nn.Sequential(
            nn.Conv2d(M*3, M*3, 3, padding=1, groups=M),  # Grouped conv
            nn.ReLU(inplace=True),
            nn.Conv2d(M*3, M*2, 1),  # 1x1 bottleneck
            nn.ReLU(inplace=True),
            nn.Conv2d(M*2, 2*M, 1)  # Output projection
        )
        
        self.residual_proj = nn.Conv2d(M*3, 2*M, 1)
    
    def forward(self, context_features):
        attn_weights = self.spatial_attn(context_features)
        context_attended = context_features * attn_weights
        params = self.conv_stack(context_attended)
        params = params + self.residual_proj(context_features)
        return params
    
    def extra_repr(self):
        return f'M={self.M} (lightweight)'


# Factory function for easy switching
def create_enhanced_context(M, lightweight=False, use_attention=True):
    """
    Factory function to create context network
    
    Args:
        M: Number of channels
        lightweight: Use lightweight version
        use_attention: Enable spatial attention
    Returns:
        EnhancedContextNetwork instance
    """
    if lightweight:
        return LightweightEnhancedContext(M)
    else:
        return EnhancedContextNetwork(M, use_attention=use_attention)


# Testing
if __name__ == "__main__":
    print("Enhanced Context Network Test")
    print("=" * 50)
    
    M = 320
    B, H, W = 2, 32, 32
    
    # Test standard version
    print(f"\nTesting standard EnhancedContextNetwork (M={M})...")
    net_standard = EnhancedContextNetwork(M)
    x = torch.randn(B, M*3, H, W)
    
    with torch.no_grad():
        out = net_standard(x)
    
    print(f"Input shape: {x.shape}")
    print(f"Output shape: {out.shape}")
    assert out.shape == (B, 2*M, H, W), "Output shape mismatch"
    print("✓ Standard version passed")
    
    # Test lightweight version
    print(f"\nTesting LightweightEnhancedContext (M={M})...")
    net_light = LightweightEnhancedContext(M)
    
    with torch.no_grad():
        out_light = net_light(x)
    
    print(f"Output shape: {out_light.shape}")
    assert out_light.shape == (B, 2*M, H, W), "Output shape mismatch"
    print("✓ Lightweight version passed")
    
    # Compare parameter counts
    params_standard = sum(p.numel() for p in net_standard.parameters())
    params_light = sum(p.numel() for p in net_light.parameters())
    
    print(f"\nParameter comparison:")
    print(f"Standard: {params_standard:,} parameters")
    print(f"Lightweight: {params_light:,} parameters")
    print(f"Reduction: {(1 - params_light/params_standard)*100:.1f}%")
    
    print("\n✓ All tests passed!")
