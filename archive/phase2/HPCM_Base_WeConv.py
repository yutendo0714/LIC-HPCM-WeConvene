"""
HPCM with WeConv blocks integration (Phase 2)
Adds wavelet-domain processing blocks to g_a and g_s
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


class g_a_weconv(nn.Module):
    """
    Enhanced encoder with WeConv blocks
    Strategy: Replace later layers (where features are more abstract) with WeConv
    """
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
            WeConv(384),  # Add WeConv here (abstract features)
            PConvRB(384, mlp_ratio=mlp_ratio, partial_ratio=partial_ratio),
            WeConv(384),  # Add WeConv here
        )
        
        self.stage4 = conv2x2_down(384, 320)

    def forward(self, x):
        x = self.stage1(x)
        x = self.stage2(x)
        x = self.stage3(x)
        x = self.stage4(x)
        return x


class g_s_weconv(nn.Module):
    """
    Enhanced decoder with WeConv blocks
    Strategy: Replace earlier layers (where features are being reconstructed) with WeConv
    """
    def __init__(self, use_simple=True):
        super().__init__()
        
        mlp_ratio = 4
        partial_ratio = 4
        WeConv = WeConvBlockSimple if use_simple else WeConvBlock
        
        self.stage1 = deconv2x2_up(320, 384)
        
        self.stage2 = nn.Sequential(
            WeConv(384),  # Add WeConv here (reconstruction stage)
            PConvRB(384, mlp_ratio=mlp_ratio, partial_ratio=partial_ratio),
            WeConv(384),  # Add WeConv here
            PConvRB(384, mlp_ratio=mlp_ratio, partial_ratio=partial_ratio),
        )
        
        self.stage3 = nn.Sequential(
            deconv2x2_up(384, 192),
            PConvRB(192, mlp_ratio=mlp_ratio, partial_ratio=partial_ratio),
            PConvRB(192, mlp_ratio=mlp_ratio, partial_ratio=partial_ratio),
        )
        
        self.stage4 = nn.Sequential(
            deconv2x2_up(192, 96),
            PConvRB(96, mlp_ratio=mlp_ratio, partial_ratio=partial_ratio),
            PConvRB(96, mlp_ratio=mlp_ratio, partial_ratio=partial_ratio),
            deconv4x4_up(96, 3),
        )

    def forward(self, x):
        x = self.stage1(x)
        x = self.stage2(x)
        x = self.stage3(x)
        x = self.stage4(x)
        return x


class h_a(nn.Module):
    def __init__(self):
        super().__init__()
        
        mlp_ratio = 4
        partial_ratio = 4

        self.branch = nn.Sequential(
            PConvRB(320, mlp_ratio = mlp_ratio, partial_ratio = partial_ratio),
            conv2x2_down(320,256),
            PConvRB(256, mlp_ratio = mlp_ratio, partial_ratio = partial_ratio),
            PConvRB(256, mlp_ratio = mlp_ratio, partial_ratio = partial_ratio),
            PConvRB(256, mlp_ratio = mlp_ratio, partial_ratio = partial_ratio),
            conv2x2_down(256,256),
        )

    def forward(self, x):
        out = self.branch(x)
        return out


class h_s(nn.Module):
    def __init__(self):
        super().__init__()
        
        mlp_ratio = 4
        partial_ratio = 4

        self.branch = nn.Sequential(
            deconv2x2_up(256,256),
            PConvRB(256, mlp_ratio = mlp_ratio, partial_ratio = partial_ratio),
            PConvRB(256, mlp_ratio = mlp_ratio, partial_ratio = partial_ratio),
            PConvRB(256, mlp_ratio = mlp_ratio, partial_ratio = partial_ratio),
            deconv2x2_up(256,320*2),
            PConvRB(320*2, mlp_ratio = mlp_ratio, partial_ratio = partial_ratio),
        )

    def forward(self, x):
        out = self.branch(x)
        return out


class y_spatial_prior_s1_s2(nn.Module):
    def __init__(self, M):
        super().__init__()
        
        self.branch_1 = nn.Sequential(
            DWConvRB(M*3),
            DWConvRB(M*3),
        )
        self.branch_2 = nn.Sequential(
            DWConvRB(M*3),
            conv1x1(3*M,2*M),
        )

    def forward(self, x, quant_step):
        return self.branch_2(self.branch_1(x)*quant_step)


class y_spatial_prior_s3(nn.Module):
    def __init__(self, M):
        super().__init__()
        
        self.branch_1 = nn.Sequential(
            DWConvRB(M*3),
            DWConvRB(M*3),
            DWConvRB(M*3),
        )
        self.branch_2 = nn.Sequential(
            DWConvRB(M*3),
            DWConvRB(M*3),
            conv1x1(3*M,2*M),
        )

    def forward(self, x, quant_step):
        return self.branch_2(self.branch_1(x)*quant_step)
    

class CrossAttentionCell(nn.Module):
    def __init__(self, input_dim, hidden_dim, window_size, kernel_size, num_heads=32):
        super(CrossAttentionCell, self).__init__()
        self.hidden_dim = hidden_dim
        self.num_heads = num_heads
        assert hidden_dim % num_heads == 0
        self.head_dim = hidden_dim // num_heads
        self.conv_q = nn.Conv2d(input_dim, hidden_dim, kernel_size, padding=kernel_size // 2)
        self.conv_k = nn.Conv2d(hidden_dim, hidden_dim, kernel_size, padding=kernel_size // 2)
        self.conv_v = nn.Conv2d(hidden_dim, hidden_dim, kernel_size, padding=kernel_size // 2)
        self.conv_out = nn.Conv2d(hidden_dim, hidden_dim, kernel_size, padding=kernel_size // 2)

        self.window_size = window_size

    def forward(self, x_t, h_prev):
        x_t_init = x_t
        H_init, W_init = x_t.shape[2], x_t.shape[3]
        
        x_t = rearrange(x_t, 'b c (w1 p1) (w2 p2)  -> (b w1 w2) c p1 p2', p1=self.window_size, p2=self.window_size)
        h_prev = rearrange(h_prev, 'b c (w1 p1) (w2 p2)  -> (b w1 w2) c p1 p2', p1=self.window_size, p2=self.window_size)
        batch_size, C, H, W = x_t.size()
        q = self.conv_q(x_t)
        k = self.conv_k(h_prev)
        v = self.conv_v(h_prev)

        q = q.view(batch_size, self.num_heads, self.head_dim, H * W).permute(0, 1, 3, 2)
        k = k.view(batch_size, self.num_heads, self.head_dim, H * W)
        v = v.view(batch_size, self.num_heads, self.head_dim, H * W).permute(0, 1, 3, 2)

        attn_scores = torch.matmul(q, k) / math.sqrt(self.head_dim)
        attn_weights = F.softmax(attn_scores, dim=-1)
        attn_output = torch.matmul(attn_weights, v)
        attn_output = attn_output.permute(0, 1, 3, 2).contiguous().view(batch_size, self.hidden_dim, H, W)
        attn_output = rearrange(attn_output, '(b w1 w2) c p1 p2  -> b c (w1 p1) (w2 p2)', w1=H_init//self.window_size, w2=W_init//self.window_size)

        h_t = attn_output + self.conv_out(x_t_init)

        return h_t


class HPCM_WeConv(basemodel):
    """
    HPCM with WeConv blocks (Phase 2 implementation)
    
    Features:
    - Entropy: DWT + LL→HF encoding with LL prior
    - Network: WeConv blocks in g_a/g_s for wavelet-aware feature extraction
    """
    def __init__(self, M=320, N=256, use_simple_weconv=True, skip_s3_for_hf=False):
        super().__init__(N)
        
        self.skip_s3_for_hf = skip_s3_for_hf  # Option 3: LL=10 steps, HF=4 steps
        
        self.g_a = g_a_weconv(use_simple=use_simple_weconv)
        self.g_s = g_s_weconv(use_simple=use_simple_weconv)

        self.h_a = h_a()
        self.h_s = h_s()

        self.dwt = DWT2D("haar")
        self.idwt = IDWT2D("haar")

        self.y_spatial_prior_adaptor_list_s1 = nn.ModuleList(conv1x1(3*M,3*M) for _ in range(1))
        self.y_spatial_prior_adaptor_list_s2 = nn.ModuleList(conv1x1(3*M,3*M) for _ in range(3))
        self.y_spatial_prior_adaptor_list_s3 = nn.ModuleList(conv1x1(3*M,3*M) for _ in range(6))
        self.y_spatial_prior_s1_s2 = y_spatial_prior_s1_s2(M)
        self.y_spatial_prior_s3 = y_spatial_prior_s3(M)

        self.adaptive_params_list = [
            torch.nn.Parameter(torch.ones((1, M*3, 1, 1), device='cuda'), requires_grad=True) for _ in range(10)
        ]

        self.attn_s1 = CrossAttentionCell(320*2, 320*2, window_size=4, kernel_size=1)
        self.attn_s2 = CrossAttentionCell(320*2, 320*2, window_size=8, kernel_size=1)
        self.attn_s3 = CrossAttentionCell(320*2, 320*2, window_size=8, kernel_size=1)
        
        self.context_net = nn.ModuleList(conv1x1(2*M,2*M) for _ in range(2))
    
    def forward(self, x, training=None):
        # Same as HPCM_Base, just with WeConv-enhanced g_a/g_s
        if training is None:
            training=self.training 
        
        y = self.g_a(x)
        z = self.h_a(y)
        
        if training:
            z_res = z - self.means_hyper
            z_hat = self.ste_round(z_res) + self.means_hyper
            z_likelihoods = self.entropy_estimation(self.add_noise(z_res), self.scales_hyper)
        else:
            z_res_hat = torch.round(z - self.means_hyper)
            z_hat = z_res_hat + self.means_hyper
            z_likelihoods = self.entropy_estimation(z_res_hat, self.scales_hyper)   

        params = self.h_s(z_hat)

        # Wavelet transform for entropy path (LL first, then HF)
        y_dwt = self.dwt(y)
        y_ll, y_lh, y_hl, y_hh = torch.chunk(y_dwt, 4, dim=1)

        params_dwt = F.avg_pool2d(params, 2, 2)

        # LL: No prior (encode first)
        y_res_ll, y_q_ll, y_hat_ll, scales_ll = self.forward_hpcm(
            y_ll,
            params_dwt,
            self.y_spatial_prior_adaptor_list_s1,
            self.y_spatial_prior_s1_s2,
            self.y_spatial_prior_adaptor_list_s2,
            self.y_spatial_prior_s1_s2,
            self.y_spatial_prior_adaptor_list_s3,
            self.y_spatial_prior_s3,
            self.adaptive_params_list,
            self.context_net,
        )
        # HF: Conditioned on LL prior
        y_res_lh, y_q_lh, y_hat_lh, scales_lh = self.forward_hpcm(
            y_lh,
            params_dwt,
            self.y_spatial_prior_adaptor_list_s1,
            self.y_spatial_prior_s1_s2,
            self.y_spatial_prior_adaptor_list_s2,
            self.y_spatial_prior_s1_s2,
            self.y_spatial_prior_adaptor_list_s3,
            self.y_spatial_prior_s3,
            self.adaptive_params_list,
            self.context_net,
            global_hat=y_hat_ll,
            skip_s3=self.skip_s3_for_hf,
        )
        y_res_hl, y_q_hl, y_hat_hl, scales_hl = self.forward_hpcm(
            y_hl,
            params_dwt,
            self.y_spatial_prior_adaptor_list_s1,
            self.y_spatial_prior_s1_s2,
            self.y_spatial_prior_adaptor_list_s2,
            self.y_spatial_prior_s1_s2,
            self.y_spatial_prior_adaptor_list_s3,
            self.y_spatial_prior_s3,
            self.adaptive_params_list,
            self.context_net,
            global_hat=y_hat_ll,
            skip_s3=self.skip_s3_for_hf,
        )
        y_res_hh, y_q_hh, y_hat_hh, scales_hh = self.forward_hpcm(
            y_hh,
            params_dwt,
            self.y_spatial_prior_adaptor_list_s1,
            self.y_spatial_prior_s1_s2,
            self.y_spatial_prior_adaptor_list_s2,
            self.y_spatial_prior_s1_s2,
            self.y_spatial_prior_adaptor_list_s3,
            self.y_spatial_prior_s3,
            self.adaptive_params_list,
            self.context_net,
            global_hat=y_hat_ll,
            skip_s3=self.skip_s3_for_hf,
        )

        y_hat_dwt = torch.cat([y_hat_ll, y_hat_lh, y_hat_hl, y_hat_hh], dim=1)
        y_hat = self.idwt(y_hat_dwt)
        x_hat = self.g_s(y_hat)

        if training:
            y_likelihoods = torch.cat(
                [
                    self.entropy_estimation(self.add_noise(y_res_ll), scales_ll),
                    self.entropy_estimation(self.add_noise(y_res_lh), scales_lh),
                    self.entropy_estimation(self.add_noise(y_res_hl), scales_hl),
                    self.entropy_estimation(self.add_noise(y_res_hh), scales_hh),
                ],
                dim=1,
            )
        else:
            y_res_ll_hat = torch.round(y_res_ll)
            y_res_lh_hat = torch.round(y_res_lh)
            y_res_hl_hat = torch.round(y_res_hl)
            y_res_hh_hat = torch.round(y_res_hh)
            y_likelihoods = torch.cat(
                [
                    self.entropy_estimation(y_res_ll_hat, scales_ll),
                    self.entropy_estimation(y_res_lh_hat, scales_lh),
                    self.entropy_estimation(y_res_hl_hat, scales_hl),
                    self.entropy_estimation(y_res_hh_hat, scales_hh),
                ],
                dim=1,
            )
        
        return {
            "x_hat": x_hat,
            "likelihoods": {"y": y_likelihoods, "z": z_likelihoods},
        }
    
    # Import all HPCM methods from parent class
    # (forward_hpcm, compress, decompress, compress_hpcm, decompress_hpcm, etc.)
    # They are inherited from basemodel
