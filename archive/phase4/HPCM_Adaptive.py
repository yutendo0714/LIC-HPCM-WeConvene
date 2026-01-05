"""
HPCM_Adaptive (Phase 4)
Attention-based Frequency Fusion + Adaptive Quantization

Key innovations:
1. FrequencyAttentionFusion: Cross-attention LL→HF
2. ChannelwiseFrequencyGate: Adaptive channel weights per band
3. AdaptiveQuantization: Learnable frequency-specific quantization
4. LLContextEncoder: Rich LL context extraction

Expected improvement: -7~-12% BD-rate over Phase 3
"""
import math
import torch
from torch import nn
import torch.nn.functional as F

from einops import rearrange

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from src.models.base import BB as basemodel
from src.layers import PConvRB, conv2x2_down, deconv2x2_up, DWConvRB, conv1x1, conv4x4_down, deconv4x4_up
from src.layers import WeConvBlock, WeConvBlockSimple
from src.layers.wavelet import DWT2D, IDWT2D

# Phase 4 components
from components import (
    FrequencyAttentionFusion,
    ChannelwiseFrequencyGate,
    AdaptiveQuantization,
    LLContextEncoder
)


class g_a_weconv(nn.Module):
    """Enhanced encoder with WeConv blocks (from Phase 3)"""
    def __init__(self, use_simple=True):
        super().__init__()
        
        mlp_ratio = 4
        partial_ratio = 4
        WeConv = WeConvBlockSimple if use_simple else WeConvBlock
        
        self.stage1 = nn.Sequential(
            conv4x4_down(3, 96),
            PConvRB(96, mlp_ratio=mlp_ratio, partial_ratio=partial_ratio),
            PConvRB(96, mlp_ratio=mlp_ratio, partial_ratio=partial_ratio),
        )
        
        self.stage2 = nn.Sequential(
            conv2x2_down(96, 192),
            PConvRB(192, mlp_ratio=mlp_ratio, partial_ratio=partial_ratio),
            PConvRB(192, mlp_ratio=mlp_ratio, partial_ratio=partial_ratio),
        )
        
        self.stage3 = nn.Sequential(
            conv2x2_down(192, 384),
            PConvRB(384, mlp_ratio=mlp_ratio, partial_ratio=partial_ratio),
            WeConv(384),
            PConvRB(384, mlp_ratio=mlp_ratio, partial_ratio=partial_ratio),
            WeConv(384),
            PConvRB(384, mlp_ratio=mlp_ratio, partial_ratio=partial_ratio),
            conv2x2_down(384, 320)
        )

    def forward(self, x):
        x = self.stage1(x)
        x = self.stage2(x)
        x = self.stage3(x)
        return x


class g_s_weconv(nn.Module):
    """Enhanced decoder with WeConv blocks (from Phase 3)"""
    def __init__(self, use_simple=True):
        super().__init__()
        
        mlp_ratio = 4
        partial_ratio = 4
        WeConv = WeConvBlockSimple if use_simple else WeConvBlock
        
        self.stage1 = nn.Sequential(
            deconv2x2_up(320, 384),
            PConvRB(384, mlp_ratio=mlp_ratio, partial_ratio=partial_ratio),
            WeConv(384),
            PConvRB(384, mlp_ratio=mlp_ratio, partial_ratio=partial_ratio),
            WeConv(384),
            PConvRB(384, mlp_ratio=mlp_ratio, partial_ratio=partial_ratio),
        )
        
        self.stage2 = nn.Sequential(
            deconv2x2_up(384, 192),
            PConvRB(192, mlp_ratio=mlp_ratio, partial_ratio=partial_ratio),
            PConvRB(192, mlp_ratio=mlp_ratio, partial_ratio=partial_ratio),
        )
        
        self.stage3 = nn.Sequential(
            deconv2x2_up(192, 96),
            PConvRB(96, mlp_ratio=mlp_ratio, partial_ratio=partial_ratio),
            PConvRB(96, mlp_ratio=mlp_ratio, partial_ratio=partial_ratio),
            deconv4x4_up(96, 3)
        )

    def forward(self, x):
        x = self.stage1(x)
        x = self.stage2(x)
        x = self.stage3(x)
        return x


class h_a_wavelet(nn.Module):
    """Wavelet-domain hyper encoder (from Phase 3)"""
    def __init__(self, M=320, N=256):
        super().__init__()
        self.M = M
        self.N = N
        
        # DWT for wavelet decomposition
        self.dwt = DWT2D(wave='haar')
        
        # Initial processing
        self.initial = nn.Sequential(
            conv1x1(M, M),
            nn.ReLU(inplace=True)
        )
        
        # 4x downsample to (M, H/8, W/8)
        self.down = nn.Sequential(
            conv4x4_down(M, M),
            nn.ReLU(inplace=True),
            conv1x1(M, M),
            nn.ReLU(inplace=True)
        )
        
        # Process LL and HF separately
        self.ll_path = nn.Sequential(
            conv1x1(M, M),
            nn.ReLU(inplace=True),
            conv1x1(M, N // 2)
        )
        
        self.hf_path = nn.Sequential(
            conv1x1(3*M, M),
            nn.ReLU(inplace=True),
            conv1x1(M, N // 2)
        )
        
    def forward(self, x):
        x = self.initial(x)
        x = self.down(x)
        
        # Wavelet decomposition
        x_dwt = self.dwt(x)
        ll, lh, hl, hh = torch.chunk(x_dwt, 4, dim=1)
        hf = torch.cat([lh, hl, hh], dim=1)
        
        # Separate processing
        ll_out = self.ll_path(ll)
        hf_out = self.hf_path(hf)
        
        # Combine
        z = torch.cat([ll_out, hf_out], dim=1)
        return z


class h_s_wavelet(nn.Module):
    """Wavelet-domain hyper decoder with frequency-adaptive params (from Phase 3)"""
    def __init__(self, M=320, N=256):
        super().__init__()
        self.M = M
        self.N = N
        
        # Initial processing
        self.initial = nn.Sequential(
            conv1x1(N, M),
            nn.ReLU(inplace=True)
        )
        
        # 4x upsample
        self.up = nn.Sequential(
            deconv4x4_up(M, M),
            nn.ReLU(inplace=True),
            conv1x1(M, M * 3 // 2),
            nn.ReLU(inplace=True)
        )
        
        # Frequency-adaptive parameter generation
        # For LL band (higher importance)
        self.ll_params = nn.Sequential(
            conv1x1(M * 3 // 2, M),
            nn.ReLU(inplace=True),
            conv1x1(M, M * 2)  # scale + mean
        )
        
        # For HF bands (3 bands: LH, HL, HH)
        self.hf_params = nn.Sequential(
            conv1x1(M * 3 // 2, M),
            nn.ReLU(inplace=True),
            conv1x1(M, M * 3 * 2)  # (scale + mean) × 3 bands
        )
        
    def forward(self, z):
        x = self.initial(z)
        x = self.up(x)
        
        # Generate frequency-specific parameters
        ll_p = self.ll_params(x)
        hf_p = self.hf_params(x)
        
        # Concatenate all parameters
        params = torch.cat([ll_p, hf_p], dim=1)
        return params


class HPCM_Adaptive(basemodel):
    """
    Phase 4: Attention-based Frequency Fusion + Adaptive Quantization
    
    Innovations over Phase 3:
    1. LLContextEncoder: Extract rich context from LL band
    2. FrequencyAttentionFusion: Cross-attention LL→HF for each band
    3. ChannelwiseFrequencyGate: Adaptive importance weighting per band
    4. AdaptiveQuantization: Learnable frequency-specific quantization steps
    
    Args:
        M: Number of latent channels (default: 320)
        N: Number of hyper-prior channels (default: 256)
        use_simple_weconv: Use simple WeConv version (default: True)
        skip_s3_for_hf: Skip s3 processing for HF bands (default: False)
        use_adaptive_quant: Use adaptive quantization (default: True)
        attention_heads: Number of attention heads (default: 8)
    """
    def __init__(self, M=320, N=256, 
                 use_simple_weconv=True, 
                 skip_s3_for_hf=False,
                 use_adaptive_quant=True,
                 attention_heads=8):
        super().__init__(M=M, N=N)
        
        self.M = M
        self.N = N
        self.skip_s3_for_hf = skip_s3_for_hf
        self.use_adaptive_quant = use_adaptive_quant
        
        # Wavelet transforms
        self.dwt = DWT2D(wave='haar')
        self.idwt = IDWT2D(wave='haar')
        
        # Main encoder/decoder with WeConv (from Phase 3)
        self.g_a_weconv = g_a_weconv(use_simple=use_simple_weconv)
        self.g_s_weconv = g_s_weconv(use_simple=use_simple_weconv)
        
        # Wavelet-domain hyper-prior (from Phase 3)
        self.h_a_wavelet = h_a_wavelet(M=M, N=N)
        self.h_s_wavelet = h_s_wavelet(M=M, N=N)
        
        # ===== Phase 4 New Components =====
        
        # 1. LL Context Encoder
        self.ll_context_encoder = LLContextEncoder(
            channels=4*M, 
            num_heads=attention_heads
        )
        
        # 2. Frequency Attention Fusion (one per HF band)
        self.freq_attention_lh = FrequencyAttentionFusion(
            dim=4*M, 
            num_heads=attention_heads
        )
        self.freq_attention_hl = FrequencyAttentionFusion(
            dim=4*M, 
            num_heads=attention_heads
        )
        self.freq_attention_hh = FrequencyAttentionFusion(
            dim=4*M, 
            num_heads=attention_heads
        )
        
        # 3. Channel-wise Frequency Gate
        self.channel_gate = ChannelwiseFrequencyGate(channels=4*M)
        
        # 4. Adaptive Quantization
        if use_adaptive_quant:
            self.adaptive_quant = AdaptiveQuantization(
                init_delta_ll=1.0,   # Fine for structure
                init_delta_lh=1.2,   # Medium for horizontal edges
                init_delta_hl=1.2,   # Medium for vertical edges
                init_delta_hh=1.5    # Coarse for diagonal texture
            )
        else:
            self.adaptive_quant = None
        
        # Parameter prediction for wavelet bands
        # LL: 2*M params (scale + mean) for M channels, downsampled 4x
        # HF: 2*M params per band × 3 bands
        self.param_aggregator_ll = nn.Sequential(
            conv1x1(M * 2, M * 2),
            nn.ReLU(inplace=True),
            conv1x1(M * 2, M * 2)
        )
        
        self.param_aggregator_hf = nn.Sequential(
            conv1x1(M * 2, M * 2),
            nn.ReLU(inplace=True),
            conv1x1(M * 2, M * 2)
        )
    
    def forward(self, x, skip_s3_for_hf=None):
        """
        Forward pass with Phase 4 adaptive processing
        
        Args:
            x: Input image (B, 3, H, W)
            skip_s3_for_hf: Override skip_s3 setting (optional)
        Returns:
            dict with x_hat and likelihoods
        """
        if skip_s3_for_hf is None:
            skip_s3_for_hf = self.skip_s3_for_hf
        
        # Encode
        y = self.g_a_weconv(x)
        
        # Hyper-prior
        z = self.h_a_wavelet(y)
        z_hat, z_likelihoods = self.entropy_bottleneck(z)
        params = self.h_s_wavelet(z_hat)
        
        # ===== Phase 4: Adaptive Frequency Processing =====
        
        # 1. DWT decomposition
        y_dwt = self.dwt(y)  # (B, 4M, H/2, W/2)
        y_ll, y_lh, y_hl, y_hh = torch.chunk(y_dwt, 4, dim=1)
        
        # Extract parameters for LL (first 2M channels)
        params_ll = params[:, :2*self.M, :, :]
        params_ll = F.interpolate(params_ll, size=y_ll.shape[2:], mode='nearest')
        params_ll = self.param_aggregator_ll(params_ll)
        
        # 2. Encode LL band first (full HPCM processing)
        y_ll_hat, y_ll_likelihood = self.forward_hpcm(
            y_ll, params_ll, skip_s3=False  # LL always uses full processing
        )
        
        # 3. Extract rich context from decoded LL
        ll_context = self.ll_context_encoder(y_ll_hat)
        
        # Extract parameters for HF bands (remaining 6M channels)
        params_hf = params[:, 2*self.M:, :, :]
        params_hf = F.interpolate(params_hf, size=y_lh.shape[2:], mode='nearest')
        params_hf = self.param_aggregator_hf(params_hf)
        
        # Split for 3 HF bands
        params_lh, params_hl, params_hh = torch.chunk(params_hf, 3, dim=1)
        
        # 4. Attention-based fusion: HF bands attend to LL context
        y_lh_fused, attn_lh = self.freq_attention_lh(y_lh, ll_context)
        y_hl_fused, attn_hl = self.freq_attention_hl(y_hl, ll_context)
        y_hh_fused, attn_hh = self.freq_attention_hh(y_hh, ll_context)
        
        # 5. Channel-wise adaptive gating
        y_lh_weighted, y_hl_weighted, y_hh_weighted = self.channel_gate(
            y_lh_fused, y_hl_fused, y_hh_fused
        )
        
        # 6. Adaptive quantization (training only)
        if self.training and self.adaptive_quant is not None:
            y_ll_hat, y_lh_weighted, y_hl_weighted, y_hh_weighted = \
                self.adaptive_quant(
                    y_ll_hat, y_lh_weighted, y_hl_weighted, y_hh_weighted
                )
        
        # 7. Encode HF bands
        y_lh_hat, y_lh_likelihood = self.forward_hpcm(
            y_lh_weighted, params_lh, skip_s3=skip_s3_for_hf
        )
        y_hl_hat, y_hl_likelihood = self.forward_hpcm(
            y_hl_weighted, params_hl, skip_s3=skip_s3_for_hf
        )
        y_hh_hat, y_hh_likelihood = self.forward_hpcm(
            y_hh_weighted, params_hh, skip_s3=skip_s3_for_hf
        )
        
        # 8. IDWT fusion
        y_hat_dwt = torch.cat([y_ll_hat, y_lh_hat, y_hl_hat, y_hh_hat], dim=1)
        y_hat = self.idwt(y_hat_dwt)
        
        # 9. Decode
        x_hat = self.g_s_weconv(y_hat).clamp(0, 1)
        
        # Combine likelihoods
        y_likelihood = y_ll_likelihood * y_lh_likelihood * y_hl_likelihood * y_hh_likelihood
        
        return {
            "x_hat": x_hat,
            "likelihoods": {"y": y_likelihood, "z": z_likelihoods}
        }
    
    def compress(self, x, skip_s3_for_hf=None):
        """Compress with Phase 4 adaptive processing"""
        if skip_s3_for_hf is None:
            skip_s3_for_hf = self.skip_s3_for_hf
            
        # Encode
        y = self.g_a_weconv(x)
        
        # Hyper-prior
        z = self.h_a_wavelet(y)
        z_strings = self.entropy_bottleneck.compress(z)
        z_hat = self.entropy_bottleneck.decompress(z_strings, z.size()[-2:])
        params = self.h_s_wavelet(z_hat)
        
        # DWT
        y_dwt = self.dwt(y)
        y_ll, y_lh, y_hl, y_hh = torch.chunk(y_dwt, 4, dim=1)
        
        # Parameters
        params_ll = params[:, :2*self.M, :, :]
        params_ll = F.interpolate(params_ll, size=y_ll.shape[2:], mode='nearest')
        params_ll = self.param_aggregator_ll(params_ll)
        
        params_hf = params[:, 2*self.M:, :, :]
        params_hf = F.interpolate(params_hf, size=y_lh.shape[2:], mode='nearest')
        params_hf = self.param_aggregator_hf(params_hf)
        params_lh, params_hl, params_hh = torch.chunk(params_hf, 3, dim=1)
        
        # Compress LL
        y_ll_strings = self.compress_hpcm(y_ll, params_ll, skip_s3=False)
        y_ll_hat = self.decompress_hpcm(y_ll_strings, params_ll, skip_s3=False)
        
        # LL context
        ll_context = self.ll_context_encoder(y_ll_hat)
        
        # Attention fusion
        y_lh_fused, _ = self.freq_attention_lh(y_lh, ll_context)
        y_hl_fused, _ = self.freq_attention_hl(y_hl, ll_context)
        y_hh_fused, _ = self.freq_attention_hh(y_hh, ll_context)
        
        # Channel gate
        y_lh_weighted, y_hl_weighted, y_hh_weighted = self.channel_gate(
            y_lh_fused, y_hl_fused, y_hh_fused
        )
        
        # Compress HF
        y_lh_strings = self.compress_hpcm(y_lh_weighted, params_lh, skip_s3=skip_s3_for_hf)
        y_hl_strings = self.compress_hpcm(y_hl_weighted, params_hl, skip_s3=skip_s3_for_hf)
        y_hh_strings = self.compress_hpcm(y_hh_weighted, params_hh, skip_s3=skip_s3_for_hf)
        
        return {
            "strings": [y_ll_strings, y_lh_strings, y_hl_strings, y_hh_strings, z_strings],
            "shape": z.size()[-2:]
        }
    
    def decompress(self, strings, shape, skip_s3_for_hf=None):
        """Decompress with Phase 4 adaptive processing"""
        if skip_s3_for_hf is None:
            skip_s3_for_hf = self.skip_s3_for_hf
            
        y_ll_strings, y_lh_strings, y_hl_strings, y_hh_strings, z_strings = strings
        
        # Hyper-prior
        z_hat = self.entropy_bottleneck.decompress(z_strings, shape)
        params = self.h_s_wavelet(z_hat)
        
        # Get latent shape (inverse of g_a downsampling: 16x)
        latent_height = shape[0] * 4
        latent_width = shape[1] * 4
        ll_shape = (latent_height // 2, latent_width // 2)
        
        # Parameters
        params_ll = params[:, :2*self.M, :, :]
        params_ll = F.interpolate(params_ll, size=ll_shape, mode='nearest')
        params_ll = self.param_aggregator_ll(params_ll)
        
        params_hf = params[:, 2*self.M:, :, :]
        params_hf = F.interpolate(params_hf, size=ll_shape, mode='nearest')
        params_hf = self.param_aggregator_hf(params_hf)
        params_lh, params_hl, params_hh = torch.chunk(params_hf, 3, dim=1)
        
        # Decompress LL
        y_ll_hat = self.decompress_hpcm(y_ll_strings, params_ll, skip_s3=False)
        
        # LL context
        ll_context = self.ll_context_encoder(y_ll_hat)
        
        # Decompress HF (we need to decode first, then apply fusion)
        # Note: In actual compression, we'd need to handle this differently
        # For now, we decompress directly
        y_lh_hat = self.decompress_hpcm(y_lh_strings, params_lh, skip_s3=skip_s3_for_hf)
        y_hl_hat = self.decompress_hpcm(y_hl_strings, params_hl, skip_s3=skip_s3_for_hf)
        y_hh_hat = self.decompress_hpcm(y_hh_strings, params_hh, skip_s3=skip_s3_for_hf)
        
        # IDWT
        y_hat_dwt = torch.cat([y_ll_hat, y_lh_hat, y_hl_hat, y_hh_hat], dim=1)
        y_hat = self.idwt(y_hat_dwt)
        
        # Decode
        x_hat = self.g_s_weconv(y_hat).clamp(0, 1)
        
        return {"x_hat": x_hat}
