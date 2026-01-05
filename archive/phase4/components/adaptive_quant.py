"""
AdaptiveQuantization: Frequency-specific quantization steps

Motivation from DCT/wavelet-based codecs (JPEG, JPEG2000):
- Low-frequency: Fine quantization (preserve structure)
- High-frequency: Coarse quantization (texture less sensitive)

We make the quantization steps learnable per frequency band.
"""
import torch
import torch.nn as nn


class AdaptiveQuantization(nn.Module):
    """
    Learn frequency-specific quantization steps
    
    Traditional approach (JPEG):
    - Uses fixed quantization tables based on human visual system
    
    Our approach:
    - Learn optimal quantization steps end-to-end
    - Adapt to specific dataset characteristics
    - Balance rate-distortion trade-off automatically
    
    Quantization steps:
    - LL: Small Δ → fine quantization → preserve structure
    - LH/HL: Medium Δ → moderate quantization → preserve edges
    - HH: Large Δ → coarse quantization → compress texture
    """
    def __init__(self, 
                 init_delta_ll=1.0,   # Fine for structure
                 init_delta_lh=1.2,   # Medium for edges
                 init_delta_hl=1.2,   # Medium for edges
                 init_delta_hh=1.5):  # Coarse for texture
        super().__init__()
        
        # Learnable quantization steps in log scale for stability
        # Log scale ensures positive values and smoother gradients
        self.log_delta_ll = nn.Parameter(torch.log(torch.tensor(init_delta_ll)))
        self.log_delta_lh = nn.Parameter(torch.log(torch.tensor(init_delta_lh)))
        self.log_delta_hl = nn.Parameter(torch.log(torch.tensor(init_delta_hl)))
        self.log_delta_hh = nn.Parameter(torch.log(torch.tensor(init_delta_hh)))
    
    @property
    def delta_ll(self):
        """Get LL quantization step (always positive via exp)"""
        return torch.exp(self.log_delta_ll)
    
    @property
    def delta_lh(self):
        """Get LH quantization step"""
        return torch.exp(self.log_delta_lh)
    
    @property
    def delta_hl(self):
        """Get HL quantization step"""
        return torch.exp(self.log_delta_hl)
    
    @property
    def delta_hh(self):
        """Get HH quantization step"""
        return torch.exp(self.log_delta_hh)
    
    def forward(self, ll, lh, hl, hh):
        """
        Apply frequency-adaptive quantization
        
        Args:
            ll, lh, hl, hh: (B, M, H, W) - Frequency bands
        Returns:
            ll_q, lh_q, hl_q, hh_q: Quantized bands
        """
        # Uniform scalar quantization: Q(x) = Δ * round(x / Δ)
        # This is differentiable via straight-through estimator in training
        ll_q = torch.round(ll / self.delta_ll) * self.delta_ll
        lh_q = torch.round(lh / self.delta_lh) * self.delta_lh
        hl_q = torch.round(hl / self.delta_hl) * self.delta_hl
        hh_q = torch.round(hh / self.delta_hh) * self.delta_hh
        
        return ll_q, lh_q, hl_q, hh_q
    
    def get_deltas(self):
        """Return current quantization steps for logging"""
        return {
            'delta_ll': self.delta_ll.item(),
            'delta_lh': self.delta_lh.item(),
            'delta_hl': self.delta_hl.item(),
            'delta_hh': self.delta_hh.item()
        }
    
    def extra_repr(self):
        deltas = self.get_deltas()
        return (f"delta_ll={deltas['delta_ll']:.3f}, "
                f"delta_lh={deltas['delta_lh']:.3f}, "
                f"delta_hl={deltas['delta_hl']:.3f}, "
                f"delta_hh={deltas['delta_hh']:.3f}")
