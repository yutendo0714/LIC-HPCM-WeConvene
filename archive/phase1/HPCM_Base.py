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
from src.layers.wavelet import DWT2D, IDWT2D

class g_a(nn.Module):
    def __init__(self):
        super().__init__()
        
        mlp_ratio = 4
        partial_ratio = 4
        
        self.branch = nn.Sequential(
            conv4x4_down(3,96),
            PConvRB(96, mlp_ratio = mlp_ratio, partial_ratio = partial_ratio),
            PConvRB(96, mlp_ratio = mlp_ratio, partial_ratio = partial_ratio),
            conv2x2_down(96,192),
            PConvRB(192, mlp_ratio = mlp_ratio, partial_ratio = partial_ratio),
            PConvRB(192, mlp_ratio = mlp_ratio, partial_ratio = partial_ratio),
            conv2x2_down(192,384),
            PConvRB(384, mlp_ratio = mlp_ratio, partial_ratio = partial_ratio),
            PConvRB(384, mlp_ratio = mlp_ratio, partial_ratio = partial_ratio),
            PConvRB(384, mlp_ratio = mlp_ratio, partial_ratio = partial_ratio),
            PConvRB(384, mlp_ratio = mlp_ratio, partial_ratio = partial_ratio),
            conv2x2_down(384,320),
        )

    def forward(self, x):
        return self.branch(x)

class g_s(nn.Module):
    def __init__(self):
        super().__init__()
        
        mlp_ratio = 4
        partial_ratio = 4

        self.branch = nn.Sequential(
            deconv2x2_up(320,384),
            PConvRB(384, mlp_ratio = mlp_ratio, partial_ratio = partial_ratio),
            PConvRB(384, mlp_ratio = mlp_ratio, partial_ratio = partial_ratio),
            PConvRB(384, mlp_ratio = mlp_ratio, partial_ratio = partial_ratio),
            PConvRB(384, mlp_ratio = mlp_ratio, partial_ratio = partial_ratio),
            deconv2x2_up(384,192),
            PConvRB(192, mlp_ratio = mlp_ratio, partial_ratio = partial_ratio),
            PConvRB(192, mlp_ratio = mlp_ratio, partial_ratio = partial_ratio),
            deconv2x2_up(192,96),
            PConvRB(96, mlp_ratio = mlp_ratio, partial_ratio = partial_ratio),
            PConvRB(96, mlp_ratio = mlp_ratio, partial_ratio = partial_ratio),
            deconv4x4_up(96,3),
        )

    def forward(self, x):
        out = self.branch(x)
        return out

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
        
        # Dynamically adjust window_size based on spatial resolution
        actual_window = min(self.window_size, H_init, W_init)
        
        # Skip attention if resolution is too small
        if H_init < 2 or W_init < 2 or actual_window < 2:
            return self.conv_out(x_t_init) + x_t_init
        
        x_t = rearrange(x_t, 'b c (w1 p1) (w2 p2)  -> (b w1 w2) c p1 p2', p1=actual_window, p2=actual_window)
        h_prev = rearrange(h_prev, 'b c (w1 p1) (w2 p2)  -> (b w1 w2) c p1 p2', p1=actual_window, p2=actual_window)
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
        attn_output = rearrange(attn_output, '(b w1 w2) c p1 p2  -> b c (w1 p1) (w2 p2)', w1=H_init//actual_window, w2=W_init//actual_window)

        h_t = attn_output + self.conv_out(x_t_init)

        return h_t

class HPCM(basemodel):
    def __init__(self, M=320, N=256, skip_s3_for_hf=False):
        super().__init__(N)
        
        self.skip_s3_for_hf = skip_s3_for_hf  # Option 3: LL=10 steps, HF=4 steps
        
        self.g_a = g_a()
        self.g_s = g_s()

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

        # LL
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
        # HF subbands conditioned on decoded LL
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
    
    def forward_hpcm(self, y, common_params, 
                              y_spatial_prior_adaptor_list_s1, y_spatial_prior_s1, 
                              y_spatial_prior_adaptor_list_s2, y_spatial_prior_s2, 
                              y_spatial_prior_adaptor_list_s3, y_spatial_prior_s3, 
                              adaptive_params_list, context_net, write=False, global_hat=None, skip_s3=False):
        B, C, H, W = y.size()
        dtype = common_params.dtype
        device = common_params.device

        global_hat_s1 = global_hat_s2 = global_hat_s3 = None
        if global_hat is not None:
            global_hat_s1 = F.avg_pool2d(global_hat, 4, 4)
            global_hat_s2 = F.avg_pool2d(global_hat, 2, 2)
            global_hat_s3 = global_hat

        ############### 2-step scale-1 (s1) (4× downsample) coding
        # get y_s2 first
        mask_list_s2 = self.get_mask_for_s2(B, C, H, W, dtype, device)
        y_s2 = self.get_s1_s2_with_mask(y, mask_list_s2, B, C, H // 2, W // 2, reduce=8)
        # get y_s1 from y_s2
        mask_list_rec_s2 = self.get_mask_for_rec_s2(B, C, H // 2, W // 2, dtype, device)
        y_s1 = self.get_s1_s2_with_mask(y_s2, mask_list_rec_s2, B, C, H // 4, W // 4, reduce=4)

        # get scales_s1 and means_s1, same as getting s1 and s2
        scales_all, means_all = common_params.chunk(2,1)
        scales_s2 = self.get_s1_s2_with_mask(scales_all, mask_list_s2, B, C, H // 2, W // 2, reduce=8)
        scales_s1 = self.get_s1_s2_with_mask(scales_s2, mask_list_rec_s2, B, C, H // 4, W // 4, reduce=4)
        means_s2 = self.get_s1_s2_with_mask(means_all, mask_list_s2, B, C, H // 2, W // 2, reduce=8)
        means_s1 = self.get_s1_s2_with_mask(means_s2, mask_list_rec_s2, B, C, H // 4, W // 4, reduce=4)
        common_params_s1 = torch.cat((scales_s1, means_s1), dim=1)
        context_next = common_params_s1

        mask_list = self.get_mask_two_parts(B, C, H // 4, W // 4, dtype, device)
        y_res_list_s1 = []
        y_q_list_s1 = []
        y_hat_list_s1 = []
        scale_list_s1 = []

        for i in range(2):
            if i == 0:
                y_res_0, y_q_0, y_hat_0, s_hat_0 = self.process_with_mask(y_s1, scales_s1, means_s1, mask_list[i])
                y_res_list_s1.append(y_res_0)
                y_q_list_s1.append(y_q_0)
                y_hat_list_s1.append(y_hat_0)
                scale_list_s1.append(s_hat_0)
            else:
                base = global_hat_s1 if global_hat_s1 is not None else 0
                y_hat_so_far = base + torch.sum(torch.stack(y_hat_list_s1), dim=0)
                params = torch.cat((context_next, y_hat_so_far), dim=1)
                context = y_spatial_prior_s1(y_spatial_prior_adaptor_list_s1[i - 1](params), adaptive_params_list[i - 1])
                context_next = self.attn_s1(context, context_next)
                scales, means = context.chunk(2, 1)
                y_res_1, y_q_1, y_hat_1, s_hat_1 = self.process_with_mask(y_s1, scales, means, mask_list[i])
                y_res_list_s1.append(y_res_1)
                y_q_list_s1.append(y_q_1)
                y_hat_list_s1.append(y_hat_1)
                scale_list_s1.append(s_hat_1)
        
        y_res = torch.sum(torch.stack(y_res_list_s1), dim=0)
        y_q = torch.sum(torch.stack(y_q_list_s1), dim=0)
        y_hat = torch.sum(torch.stack(y_hat_list_s1), dim=0)
        scales_hat = torch.sum(torch.stack(scale_list_s1), dim=0)

        if write:
            y_q_write_list_s1 = [self.combine_for_writing_s1(y_q_list_s1[i]) for i in range(len(y_q_list_s1))]
            scales_hat_write_list_s1 = [self.combine_for_writing_s1(scale_list_s1[i]) for i in range(len(scale_list_s1))]
        
        # up-scaling to s2
        y_res = self.recon_for_s2_s3(y_res, mask_list_rec_s2, B, C, H // 2, W // 2, dtype, device)
        y_q = self.recon_for_s2_s3(y_q, mask_list_rec_s2, B, C, H // 2, W // 2, dtype, device)
        y_hat = self.recon_for_s2_s3(y_hat, mask_list_rec_s2, B, C, H // 2, W // 2, dtype, device)
        scales_hat = self.recon_for_s2_s3(scales_hat, mask_list_rec_s2, B, C, H // 2, W // 2, dtype, device)

        context_next_scales, context_next_means = context_next.chunk(2, 1)
        context_next_scales = self.recon_for_s2_s3(context_next_scales, mask_list_rec_s2, B, C, H // 2, W // 2, dtype, device)
        context_next_means = self.recon_for_s2_s3(context_next_means, mask_list_rec_s2, B, C, H // 2, W // 2, dtype, device)
        context = torch.cat((context_next_scales, context_next_means), dim=1)

        ############### 4-step scale-2 (s2) (2× downsample) coding

        mask_list_s1 = self.get_mask_for_s1(B, C, H, W, dtype, device)
        scales_s2 = self.get_s2_hyper_with_mask(scales_all, mask_list_s1, mask_list_s2, mask_list_rec_s2, B, C, H // 2, W // 2, dtype, device)
        means_s2 = self.get_s2_hyper_with_mask(means_all, mask_list_s1, mask_list_s2, mask_list_rec_s2, B, C, H // 2, W // 2, dtype, device)
        common_params_s2 = torch.cat((scales_s2, means_s2), dim=1)
        context += common_params_s2
        context_next = context_net[0](context)
        
        mask_list = self.get_mask_four_parts(B, C, H // 2, W // 2, dtype, device)[1:]
        y_res_list_s2 = [y_res]
        y_q_list_s2   = [y_q]
        y_hat_list_s2 = [y_hat]
        scale_list_s2 = [scales_hat]

        for i in range(3):
            base = global_hat_s2 if global_hat_s2 is not None else 0
            y_hat_so_far = base + torch.sum(torch.stack(y_hat_list_s2), dim=0)
            params = torch.cat((context_next, y_hat_so_far), dim=1)
            context = y_spatial_prior_s2(y_spatial_prior_adaptor_list_s2[i - 1](params), adaptive_params_list[i + 1])
            context_next = self.attn_s2(context, context_next)
            scales, means = context.chunk(2, 1)
            y_res_1, y_q_1, y_hat_1, s_hat_1 = self.process_with_mask(y_s2, scales, means, mask_list[i])
            y_res_list_s2.append(y_res_1)
            y_q_list_s2.append(y_q_1)
            y_hat_list_s2.append(y_hat_1)
            scale_list_s2.append(s_hat_1)
        
        y_res = torch.sum(torch.stack(y_res_list_s2), dim=0)
        y_q = torch.sum(torch.stack(y_q_list_s2), dim=0)
        y_hat = torch.sum(torch.stack(y_hat_list_s2), dim=0)
        scales_hat = torch.sum(torch.stack(scale_list_s2), dim=0)

        if write:
            y_q_write_list_s2 = [self.combine_for_writing_s2(y_q_list_s2[i]) for i in range(1, len(y_q_list_s2))]
            scales_hat_write_list_s2 = [self.combine_for_writing_s2(scale_list_s2[i]) for i in range(1, len(scale_list_s2))]
       
        # up-scaling to s3
        y_res = self.recon_for_s2_s3(y_res, mask_list_s2, B, C, H, W, dtype, device)
        y_q = self.recon_for_s2_s3(y_q, mask_list_s2, B, C, H, W, dtype, device)
        y_hat = self.recon_for_s2_s3(y_hat, mask_list_s2, B, C, H, W, dtype, device)
        scales_hat = self.recon_for_s2_s3(scales_hat, mask_list_s2, B, C, H, W, dtype, device)

        context_next_scales, context_next_means = context_next.chunk(2, 1)
        context_next_scales = self.recon_for_s2_s3(context_next_scales, mask_list_s2, B, C, H, W, dtype, device)
        context_next_means = self.recon_for_s2_s3(context_next_means, mask_list_s2, B, C, H, W, dtype, device)
        context = torch.cat((context_next_scales, context_next_means), dim=1)

        ############### 8-step scale-3 (s3) coding

        scales_s3 = self.get_s3_hyper_with_mask(scales_all, mask_list_s2, B, C, H, W, dtype, device)
        means_s3 = self.get_s3_hyper_with_mask(means_all, mask_list_s2, B, C, H, W, dtype, device)
        common_params_s3 = torch.cat((scales_s3, means_s3), dim=1)
        context += common_params_s3
        context_next = context_net[1](context)

        mask_list = self.get_mask_eight_parts(B, C, H, W, dtype, device)[2:]
        y_res_list_s3 = [y_res]
        y_q_list_s3   = [y_q]
        y_hat_list_s3 = [y_hat]
        scale_list_s3 = [scales_hat]

        for i in range(6):
            base = global_hat_s3 if global_hat_s3 is not None else 0
            y_hat_so_far = base + torch.sum(torch.stack(y_hat_list_s3), dim=0)
            params = torch.cat((context_next, y_hat_so_far), dim=1)
            context = y_spatial_prior_s3(y_spatial_prior_adaptor_list_s3[i - 1](params), adaptive_params_list[i + 4])
            context_next = self.attn_s3(context, context_next)
            scales, means = context.chunk(2, 1)
            y_res_1, y_q_1, y_hat_1, s_hat_1 = self.process_with_mask(y, scales, means, mask_list[i])
            y_res_list_s3.append(y_res_1)
            y_q_list_s3.append(y_q_1)
            y_hat_list_s3.append(y_hat_1)
            scale_list_s3.append(s_hat_1)

        y_res = torch.sum(torch.stack(y_res_list_s3), dim=0)
        y_q = torch.sum(torch.stack(y_q_list_s3), dim=0)
        y_hat = torch.sum(torch.stack(y_hat_list_s3), dim=0)
        scales_hat = torch.sum(torch.stack(scale_list_s3), dim=0)

        if write:
            y_q_write_list_s3 = [self.combine_for_writing_s3(y_q_list_s3[i]) for i in range(1, len(y_q_list_s3))]
            scales_hat_write_list_s3 = [self.combine_for_writing_s3(scale_list_s3[i]) for i in range(1, len(scale_list_s3))]

            return y_q_write_list_s1 + y_q_write_list_s2 + y_q_write_list_s3, scales_hat_write_list_s1 + scales_hat_write_list_s2 + scales_hat_write_list_s3

        return y_res, y_q, y_hat, scales_hat
    
    def compress(self, x):
        from src.entropy_models import ubransEncoder
        y = self.g_a(x)
        z = self.h_a(y)
        z_res_hat = torch.round(z - self.means_hyper)
        indexes_z = self.build_indexes_z(z_res_hat.size())
        
        encoder_z = ubransEncoder()
        self.compress_symbols(z_res_hat, indexes_z, self.quantized_cdf_z.cpu().numpy(), self.cdf_length_z.cpu().numpy(), self.offset_z.cpu().numpy(), encoder_z)
        z_string = encoder_z.flush()
        
        z_hat = z_res_hat + self.means_hyper

        params = self.h_s(z_hat)

        y_dwt = self.dwt(y)
        y_ll, y_lh, y_hl, y_hh = torch.chunk(y_dwt, 4, dim=1)
        params_dwt = F.avg_pool2d(params, 2, 2)

        # LL hat for conditioning
        _, _, y_hat_ll, _ = self.forward_hpcm(
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

        y_q_write_list_ll, scales_hat_write_list_ll = self.compress_hpcm(
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
        y_q_write_list_lh, scales_hat_write_list_lh = self.compress_hpcm(
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
        y_q_write_list_hl, scales_hat_write_list_hl = self.compress_hpcm(
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
        y_q_write_list_hh, scales_hat_write_list_hh = self.compress_hpcm(
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

        y_q_write_list = (
            y_q_write_list_ll
            + y_q_write_list_lh
            + y_q_write_list_hl
            + y_q_write_list_hh
        )
        scales_hat_write_list = (
            scales_hat_write_list_ll
            + scales_hat_write_list_lh
            + scales_hat_write_list_hl
            + scales_hat_write_list_hh
        )

        encoder_y = ubransEncoder()
        for i in range(len(y_q_write_list)):
            indexes_w = self.build_indexes_conditional(scales_hat_write_list[i])
            self.compress_symbols(y_q_write_list[i], indexes_w, self.quantized_cdf_y.cpu().numpy(), self.cdf_length_y.cpu().numpy(), self.offset_y.cpu().numpy(), encoder_y)
        y_string = encoder_y.flush()
        
        return {"strings": [y_string, z_string], "shape": z_res_hat.size()[2:]}
        
    def decompress(self, strings, shape):
        from src.entropy_models import ubransDecoder
        device = self.quantized_cdf_z.device
        output_size = (1, self.scales_hyper.size(1), *shape)
        indexes_z = self.build_indexes_z(output_size).to(device)
        
        decoder_z = ubransDecoder()
        decoder_z.set_stream(strings[1])
        z_res_hat = self.decompress_symbols(indexes_z, self.quantized_cdf_z.cpu().numpy(), self.cdf_length_z.cpu().numpy(), self.offset_z.cpu().numpy(), decoder_z)
        z_hat = z_res_hat+self.means_hyper
        
        params = self.h_s(z_hat)
        decoder_y = ubransDecoder()
        decoder_y.set_stream(strings[0])

        params_dwt = F.avg_pool2d(params, 2, 2)

        y_hat_ll = self.decompress_hpcm(
            params_dwt,
            self.y_spatial_prior_adaptor_list_s1,
            self.y_spatial_prior_s1_s2,
            self.y_spatial_prior_adaptor_list_s2,
            self.y_spatial_prior_s1_s2,
            self.y_spatial_prior_adaptor_list_s3,
            self.y_spatial_prior_s3,
            self.adaptive_params_list,
            self.context_net,
            decoder_y,
        )
        y_hat_lh = self.decompress_hpcm(
            params_dwt,
            self.y_spatial_prior_adaptor_list_s1,
            self.y_spatial_prior_s1_s2,
            self.y_spatial_prior_adaptor_list_s2,
            self.y_spatial_prior_s1_s2,
            self.y_spatial_prior_adaptor_list_s3,
            self.y_spatial_prior_s3,
            self.adaptive_params_list,
            self.context_net,
            decoder_y,
            global_hat=y_hat_ll,
            skip_s3=self.skip_s3_for_hf,
        )
        y_hat_hl = self.decompress_hpcm(
            params_dwt,
            self.y_spatial_prior_adaptor_list_s1,
            self.y_spatial_prior_s1_s2,
            self.y_spatial_prior_adaptor_list_s2,
            self.y_spatial_prior_s1_s2,
            self.y_spatial_prior_adaptor_list_s3,
            self.y_spatial_prior_s3,
            self.adaptive_params_list,
            self.context_net,
            decoder_y,
            global_hat=y_hat_ll,
            skip_s3=self.skip_s3_for_hf,
        )
        y_hat_hh = self.decompress_hpcm(
            params_dwt,
            self.y_spatial_prior_adaptor_list_s1,
            self.y_spatial_prior_s1_s2,
            self.y_spatial_prior_adaptor_list_s2,
            self.y_spatial_prior_s1_s2,
            self.y_spatial_prior_adaptor_list_s3,
            self.y_spatial_prior_s3,
            self.adaptive_params_list,
            self.context_net,
            decoder_y,
            global_hat=y_hat_ll,
            skip_s3=self.skip_s3_for_hf,
        )

        y_hat_dwt = torch.cat([y_hat_ll, y_hat_lh, y_hat_hl, y_hat_hh], dim=1)
        y_hat = self.idwt(y_hat_dwt)
    
        x_hat = self.g_s(y_hat).clamp_(0,1)
        
        return {"x_hat": x_hat}
    
    def compress_hpcm(self, y, common_params, 
                              y_spatial_prior_adaptor_list_s1, y_spatial_prior_s1, 
                              y_spatial_prior_adaptor_list_s2, y_spatial_prior_s2, 
                              y_spatial_prior_adaptor_list_s3, y_spatial_prior_s3, 
                              adaptive_params_list, context_net, global_hat=None, skip_s3=False
                              ):
        return self.forward_hpcm(y, common_params, 
                              y_spatial_prior_adaptor_list_s1, y_spatial_prior_s1, 
                              y_spatial_prior_adaptor_list_s2, y_spatial_prior_s2, 
                              y_spatial_prior_adaptor_list_s3, y_spatial_prior_s3, 
                              adaptive_params_list, context_net, write=True, global_hat=global_hat, skip_s3=skip_s3
                              )

    def decompress_hpcm(self, common_params, 
                                y_spatial_prior_adaptor_list_s1, y_spatial_prior_s1, 
                                y_spatial_prior_adaptor_list_s2, y_spatial_prior_s2, 
                                y_spatial_prior_adaptor_list_s3, y_spatial_prior_s3, 
                                adaptive_params_list, context_net, decoder_y, global_hat=None, skip_s3=False
                                ):
        scales, means = common_params.chunk(2,1)
        dtype = means.dtype
        device = means.device
        B, C, H, W = means.size()

        global_hat_s1 = global_hat_s2 = global_hat_s3 = None
        if global_hat is not None:
            global_hat_s1 = F.avg_pool2d(global_hat, 4, 4)
            global_hat_s2 = F.avg_pool2d(global_hat, 2, 2)
            global_hat_s3 = global_hat

        ############### 2-step resolution-1 (s1) (4× downsample) coding
        mask_list_s2 = self.get_mask_for_s2(B, C, H, W, dtype, device)
        mask_list_rec_s2 = self.get_mask_for_rec_s2(B, C, H // 2, W // 2, dtype, device)

        # get scales_s1 and means_s1
        scales_all, means_all = common_params.chunk(2,1)
        scales_s2 = self.get_s1_s2_with_mask(scales_all, mask_list_s2, B, C, H // 2, W // 2, reduce=8)
        scales_s1 = self.get_s1_s2_with_mask(scales_s2, mask_list_rec_s2, B, C, H // 4, W // 4, reduce=4)
        means_s2 = self.get_s1_s2_with_mask(means_all, mask_list_s2, B, C, H // 2, W // 2, reduce=8)
        means_s1 = self.get_s1_s2_with_mask(means_s2, mask_list_rec_s2, B, C, H // 4, W // 4, reduce=4)
        common_params_s1 = torch.cat((scales_s1, means_s1), dim=1)
        context_next = common_params_s1

        mask_list = self.get_mask_two_parts(B, C, H // 4, W // 4, dtype, device)

        for i in range(2):
            if i == 0:
                scales_r = self.combine_for_writing_s1(scales_s1 * mask_list[i])
                indexes_r = self.build_indexes_conditional(scales_r)
                y_q_r = self.decompress_symbols(indexes_r, self.quantized_cdf_y.cpu().numpy(), self.cdf_length_y.cpu().numpy(), self.offset_y.cpu().numpy(), decoder_y)
                y_hat_curr_step = (torch.cat([y_q_r for _ in range(2)], dim=1) + means_s1) * mask_list[i]
                base = global_hat_s1 if global_hat_s1 is not None else 0
                y_hat_so_far = y_hat_curr_step + base
            else:
                params = torch.cat((context_next, y_hat_so_far), dim=1)
                context = y_spatial_prior_s1(y_spatial_prior_adaptor_list_s1[i - 1](params), adaptive_params_list[i - 1])
                context_next = self.attn_s1(context, context_next)
                scales, means = context.chunk(2, 1)
                scales_r = self.combine_for_writing_s1(scales * mask_list[i])
                indexes_r = self.build_indexes_conditional(scales_r)
                y_q_r = self.decompress_symbols(indexes_r, self.quantized_cdf_y.cpu().numpy(), self.cdf_length_y.cpu().numpy(), self.offset_y.cpu().numpy(), decoder_y)
                y_hat_curr_step = (torch.cat([y_q_r for _ in range(2)], dim=1) + means) * mask_list[i]
                y_hat_so_far = y_hat_so_far + y_hat_curr_step
        
        y_hat_so_far = self.recon_for_s2_s3(y_hat_so_far, mask_list_rec_s2, B, C, H // 2, W // 2, dtype, device)

        context_next_scales, context_next_means = context_next.chunk(2, 1)
        context_next_scales = self.recon_for_s2_s3(context_next_scales, mask_list_rec_s2, B, C, H // 2, W // 2, dtype, device)
        context_next_means = self.recon_for_s2_s3(context_next_means, mask_list_rec_s2, B, C, H // 2, W // 2, dtype, device)
        context = torch.cat((context_next_scales, context_next_means), dim=1)

        ############### 4-step resolution-2 (s2) (2× downsample) coding
        mask_list_s1 = self.get_mask_for_s1(B, C, H, W, dtype, device)
        scales_s2 = self.get_s2_hyper_with_mask(scales_all, mask_list_s1, mask_list_s2, mask_list_rec_s2, B, C, H // 2, W // 2, dtype, device)
        means_s2 = self.get_s2_hyper_with_mask(means_all, mask_list_s1, mask_list_s2, mask_list_rec_s2, B, C, H // 2, W // 2, dtype, device)
        common_params_s2 = torch.cat((scales_s2, means_s2), dim=1)
        context += common_params_s2
        context_next = context_net[0](context)

        if global_hat_s2 is not None:
            y_hat_so_far = y_hat_so_far + global_hat_s2

        mask_list = self.get_mask_four_parts(B, C, H // 2, W // 2, dtype, device)[1:]

        for i in range(3):
            params = torch.cat((context_next, y_hat_so_far), dim=1)
            context = y_spatial_prior_s2(y_spatial_prior_adaptor_list_s2[i - 1](params), adaptive_params_list[i + 1])
            context_next = self.attn_s2(context, context_next)
            scales, means = context.chunk(2, 1)
            scales_r = self.combine_for_writing_s2(scales * mask_list[i])
            indexes_r = self.build_indexes_conditional(scales_r)
            y_q_r = self.decompress_symbols(indexes_r, self.quantized_cdf_y.cpu().numpy(), self.cdf_length_y.cpu().numpy(), self.offset_y.cpu().numpy(), decoder_y)
            y_hat_curr_step = (torch.cat([y_q_r for _ in range(4)], dim=1) + means) * mask_list[i]
            y_hat_so_far = y_hat_so_far + y_hat_curr_step

        y_hat_so_far = self.recon_for_s2_s3(y_hat_so_far, mask_list_s2, B, C, H, W, dtype, device)

        context_next_scales, context_next_means = context_next.chunk(2, 1)
        context_next_scales = self.recon_for_s2_s3(context_next_scales, mask_list_s2, B, C, H, W, dtype, device)
        context_next_means = self.recon_for_s2_s3(context_next_means, mask_list_s2, B, C, H, W, dtype, device)
        context = torch.cat((context_next_scales, context_next_means), dim=1)

        ############### 8-step resolution-3 (s3) coding
        scales_s3 = self.get_s3_hyper_with_mask(scales_all, mask_list_s2, B, C, H, W, dtype, device)
        means_s3 = self.get_s3_hyper_with_mask(means_all, mask_list_s2, B, C, H, W, dtype, device)
        common_params_s3 = torch.cat((scales_s3, means_s3), dim=1)
        context += common_params_s3
        context_next = context_net[1](context)

        if global_hat_s3 is not None:
            y_hat_so_far = y_hat_so_far + global_hat_s3

        mask_list = self.get_mask_eight_parts(B, C, H, W, dtype, device)[2:]

        for i in range(6):
            params = torch.cat((context_next, y_hat_so_far), dim=1)
            context = y_spatial_prior_s3(y_spatial_prior_adaptor_list_s3[i - 1](params), adaptive_params_list[i + 4])
            context_next = self.attn_s3(context, context_next)
            scales, means = context.chunk(2, 1)
            scales_r = self.combine_for_writing_s3(scales * mask_list[i])
            indexes_r = self.build_indexes_conditional(scales_r)
            y_q_r = self.decompress_symbols(indexes_r, self.quantized_cdf_y.cpu().numpy(), self.cdf_length_y.cpu().numpy(), self.offset_y.cpu().numpy(), decoder_y)
            y_hat_curr_step = (torch.cat([y_q_r for _ in range(8)], dim=1) + means) * mask_list[i]
            y_hat_so_far = y_hat_so_far + y_hat_curr_step
        
        y_hat = y_hat_so_far

        return y_hat
