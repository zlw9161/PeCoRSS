"""
TFoc vanilla + hard top-k mask-draw (v2_revise, standalone).
"""

from __future__ import annotations

import math
import os
from typing import Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F


class DoubleConvBlock(nn.Module):
    def __init__(self, in_ch: int, out_ch: int, k_size: int, pad: int, dil: int):
        super().__init__()
        self.block = nn.Sequential(
            nn.Conv2d(in_ch, out_ch, kernel_size=k_size, padding=pad, dilation=dil),
            nn.BatchNorm2d(out_ch),
            nn.LeakyReLU(inplace=True),
            nn.Conv2d(out_ch, out_ch, kernel_size=k_size, padding=pad, dilation=dil),
            nn.BatchNorm2d(out_ch),
            nn.LeakyReLU(inplace=True),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.block(x)


class ConvBlock(nn.Module):
    def __init__(self, in_ch: int, out_ch: int, k_size: int, pad: int, dil: int):
        super().__init__()
        self.block = nn.Sequential(
            nn.Conv2d(in_ch, out_ch, kernel_size=k_size, padding=pad, dilation=dil),
            nn.BatchNorm2d(out_ch),
            nn.LeakyReLU(inplace=True),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.block(x)


class TREBlock(nn.Module):
    def __init__(self, in_ch: int, out_ch: int, k_size: Tuple[int, int, int], strd: Tuple[int, int, int], pad: Tuple[int, int, int]):
        super().__init__()
        self.block = nn.Sequential(
            nn.Conv3d(in_ch, out_ch, kernel_size=k_size, stride=strd, padding=pad),
            nn.BatchNorm3d(out_ch),
            nn.LeakyReLU(inplace=True),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.block(x)


class Double3DConvBlock(nn.Module):
    def __init__(
        self,
        in_ch: int,
        out_ch: int,
        k_size: Tuple[int, int, int],
        strd: Tuple[int, int, int],
        pad: Tuple[int, int, int],
        btnk_sc: int,
    ):
        super().__init__()
        self.block = nn.Sequential(
            nn.Conv3d(in_ch, out_ch * btnk_sc, kernel_size=k_size, stride=strd, padding=pad),
            nn.BatchNorm3d(out_ch * btnk_sc),
            nn.LeakyReLU(inplace=True),
            nn.Conv3d(out_ch * btnk_sc, out_ch, kernel_size=k_size, stride=strd, padding=pad),
            nn.BatchNorm3d(out_ch),
            nn.LeakyReLU(inplace=True),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.block(x)


class ASPPBlock(nn.Module):
    def __init__(self, in_ch: int, out_ch: int = 256):
        super().__init__()
        self.conv1_1x1 = nn.Conv2d(in_ch, out_ch, kernel_size=1, padding=0, dilation=1)
        self.single_conv_block1_1x1 = ConvBlock(in_ch, out_ch, k_size=1, pad=0, dil=1)
        self.single_conv_block1_3x3 = ConvBlock(in_ch, out_ch, k_size=3, pad=6, dil=6)
        self.single_conv_block2_3x3 = ConvBlock(in_ch, out_ch, k_size=3, pad=12, dil=12)
        self.single_conv_block3_3x3 = ConvBlock(in_ch, out_ch, k_size=3, pad=18, dil=18)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x1 = F.adaptive_avg_pool2d(x, (1, 1))
        x1 = F.interpolate(x1, size=x.shape[2:], align_corners=False, mode="bilinear")
        x1 = self.conv1_1x1(x1)
        x2 = self.single_conv_block1_1x1(x)
        x3 = self.single_conv_block1_3x3(x)
        x4 = self.single_conv_block2_3x3(x)
        x5 = self.single_conv_block3_3x3(x)
        return torch.cat((x2, x3, x4, x5, x1), 1)


class TFocConvV2Revise(nn.Module):
    def __init__(
        self,
        in_ch: int,
        out_ch: int,
        btnk_ch: int,
        search_size: int,
        topk: int = 3,
        topk_temperature: float = 0.7,
        metric_fg_weight: float = 0.5,
        use_straight_through_khot: bool = True,
    ):
        super().__init__()
        self.btnk_ch = int(btnk_ch)
        self.k_n = int(search_size)
        self.pad_size = math.floor(self.k_n / 2)
        self.topk = int(topk)
        self.topk_temperature = float(topk_temperature)
        self.metric_fg_weight = float(metric_fg_weight)
        self.use_straight_through_khot = bool(use_straight_through_khot)

        self.debug_cuda = os.environ.get("TFOC_DEBUG_CUDA", "0") == "1"
        self.debug_strict_index = os.environ.get("TFOC_DEBUG_STRICT_INDEX", "0") == "1"

        self.spatial_conv = Double3DConvBlock(in_ch, self.btnk_ch, (1, 3, 3), (1, 1, 1), (0, 1, 1), 16)
        self.tfoc_prepadding = nn.ReplicationPad3d((self.pad_size, self.pad_size, self.pad_size, self.pad_size, 0, 0))
        self.inner_predict = TREBlock(self.btnk_ch * 2, 4, (1, 1, 1), (1, 1, 1), (0, 0, 0))
        self.temporal_predict = nn.Sequential(
            TREBlock(self.btnk_ch, out_ch, (3, 1, 1), (1, 1, 1), (1, 0, 0)),
            TREBlock(out_ch, out_ch, (3, 1, 1), (1, 1, 1), (1, 0, 0)),
        )
        self._grid_center_cache = {}
        self._grid_local_cache = {}

    def _debug_check_index(self, idx: torch.Tensor, max_inclusive: int, name: str) -> None:
        if not self.debug_cuda:
            return
        idx_min = int(idx.min().item())
        idx_max = int(idx.max().item())
        if idx_min < 0 or idx_max > max_inclusive:
            raise RuntimeError(
                f"[TFOC_DEBUG_CUDA] index out of bounds at {name}: min={idx_min} max={idx_max} allowed=[0,{max_inclusive}]"
            )

    def forward(self, x_seq: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        x_seq = self.spatial_conv(x_seq)
        b, c, t, h, w = x_seq.shape
        n = self.k_n * self.k_n
        k = max(1, min(self.topk, n))

        x_tgt_ori = x_seq[:, :, -1, :, :].unsqueeze(2)
        x_tgt_cand = x_tgt_ori.expand(-1, -1, t - 1, -1, -1).unsqueeze(-1).expand(-1, -1, -1, -1, -1, n)
        x_hst = self.tfoc_prepadding(x_seq[:, :, : t - 1, :, :])
        p_hst = self._gen_global_sample_grid(b, h, w, n, x_seq.device)
        x_hst_cand = self._sample_x(x_hst, p_hst, n)

        pair = torch.cat((x_hst_cand, x_tgt_cand), dim=1).view(b, 2 * c, t - 1, h * w, n)
        logits = self.inner_predict(pair)
        probs_all = torch.softmax(logits, dim=1)
        probs_map = probs_all.max(dim=-1).values.mean(dim=2).view(b, 4, h, w)

        fg_conf = (1.0 - probs_all[:, 0, ...]).view(b, t - 1, h, w, n)
        score = -(x_hst_cand - x_tgt_cand).abs().mean(dim=1) + self.metric_fg_weight * fg_conf
        tau = max(self.topk_temperature, 1e-6)
        topk_val, topk_idx = torch.topk(score, k=k, dim=-1, largest=True, sorted=False)
        self._debug_check_index(topk_idx, n - 1, "topk_idx")

        idx = topk_idx.unsqueeze(1).expand(-1, c, -1, -1, -1, -1)
        x_hst_topk = x_hst_cand.gather(dim=-1, index=idx)
        w_topk = torch.softmax(topk_val / tau, dim=-1)
        x_hst_topk = x_hst_topk * w_topk.unsqueeze(1)

        # IMPORTANT order: t1(k1..kk), t2(k1..kk), ..., t_{t-1}(k1..kk)
        x_hist_seq_hard = x_hst_topk.permute(0, 1, 2, 5, 3, 4).contiguous().view(b, c, (t - 1) * k, h, w)

        if self.use_straight_through_khot:
            w_soft_full = torch.softmax(score / tau, dim=-1)
            x_agg_soft = (x_hst_cand * w_soft_full.unsqueeze(1)).sum(dim=-1)
            x_hist_seq_soft = x_agg_soft.unsqueeze(3).expand(-1, -1, -1, k, -1, -1).reshape(b, c, (t - 1) * k, h, w)
            x_hist_seq = x_hist_seq_hard.detach() + (x_hist_seq_soft - x_hist_seq_soft.detach())
        else:
            x_hist_seq = x_hist_seq_hard

        x_fuse = torch.cat((x_hist_seq, x_tgt_ori), dim=2).contiguous()
        x_ctx = self.temporal_predict(x_fuse)
        x_out = x_ctx[:, :, -1:, :, :]
        return x_out, probs_map

    def _get_grid_center(self, b: int, h: int, w: int, n: int, device: torch.device) -> torch.Tensor:
        key = (h, w, n, device.type, device.index)
        if key in self._grid_center_cache:
            g = self._grid_center_cache[key]
            return g if b == 1 else g.expand(b, -1, -1, -1)
        k_ls = (self.k_n - 1) // 2
        gx, gy = torch.meshgrid(
            torch.arange(k_ls, h + k_ls, device=device),
            torch.arange(k_ls, w + k_ls, device=device),
            indexing="ij",
        )
        gc_x = torch.flatten(gx).view(1, 1, h, w).expand(1, n, h, w)
        gc_y = torch.flatten(gy).view(1, 1, h, w).expand(1, n, h, w)
        grid_c = torch.cat([gc_x, gc_y], 1)
        self._grid_center_cache[key] = grid_c
        return grid_c if b == 1 else grid_c.expand(b, -1, -1, -1)

    def _get_grid_local_search(self, n: int, device: torch.device) -> torch.Tensor:
        key = (n, device.type, device.index)
        if key in self._grid_local_cache:
            return self._grid_local_cache[key]
        k_ls = (self.k_n - 1) // 2
        gl_x, gl_y = torch.meshgrid(
            torch.arange(-k_ls, k_ls + 1, device=device),
            torch.arange(-k_ls, k_ls + 1, device=device),
            indexing="ij",
        )
        grid_ls = torch.cat([torch.flatten(gl_x), torch.flatten(gl_y)], 0).view(1, 2 * n, 1, 1)
        self._grid_local_cache[key] = grid_ls
        return grid_ls

    def _gen_global_sample_grid(self, b: int, h: int, w: int, n: int, device: torch.device) -> torch.Tensor:
        return self._get_grid_center(b, h, w, n, device=device) + self._get_grid_local_search(n, device=device)

    def _sample_x(self, x_pad: torch.Tensor, p: torch.Tensor, n: int) -> torch.Tensor:
        p = p.permute(0, 2, 3, 1)
        b, h, w, _ = p.size()
        h_pad = x_pad.size(3)
        w_pad = x_pad.size(4)
        c = x_pad.size(1)
        t = x_pad.size(2)
        x_pad = x_pad.view(b, c, t, -1)

        px_raw = p[..., :n]
        py_raw = p[..., n:]
        if self.debug_cuda and self.debug_strict_index:
            self._debug_check_index(px_raw, h_pad - 1, "_sample_x:px_raw")
            self._debug_check_index(py_raw, w_pad - 1, "_sample_x:py_raw")

        px = torch.clamp(px_raw, min=0, max=h_pad - 1)
        py = torch.clamp(py_raw, min=0, max=w_pad - 1)
        index = (px * w_pad + py).long().view(b, 1, 1, -1).expand(-1, c, t, -1)
        if self.debug_cuda and self.debug_strict_index:
            self._debug_check_index(index, h_pad * w_pad - 1, "_sample_x:index")
        x_r = x_pad.gather(dim=-1, index=index)
        return x_r.view(b, c, t, h, w, n)


class EncodingBranch(nn.Module):
    def __init__(self, signal_type: str, n_frames: int):
        super().__init__()
        self.n_frames = n_frames
        self.signal_type = signal_type
        self.double_3d_conv_block = Double3DConvBlock(1, 64, (3, 3, 3), (1, 1, 1), (0, 1, 1), 1)
        self.tfoc_conv_block = TFocConvV2Revise(64, 128, 8, 5, 3, 0.7, 0.5)
        self.doppler_max_pool = nn.MaxPool2d(2, stride=(2, 1))
        self.max_pool = nn.MaxPool2d(2, stride=2)
        self.single_conv_block1_3x3 = ConvBlock(128, 128, 3, 1, 1)
        self.single_conv_block2_3x3 = ConvBlock(128, 128, 3, 1, 1)
        self.single_conv_block1_1x1 = ConvBlock(128, 128, 1, 0, 1)
        self.single_conv_block1_3x3_prb = ConvBlock(4, 32, 3, 1, 1)
        self.single_conv_block2_3x3_prb = ConvBlock(32, 32, 3, 1, 1)
        self.single_conv_block1_1x1_prb = ConvBlock(32, 32, 1, 0, 1)

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        x0 = self.double_3d_conv_block(x)
        x1, prb1 = self.tfoc_conv_block(x0)
        x1 = x1.squeeze(dim=2)
        if self.signal_type in ("range_doppler", "angle_doppler"):
            x1_down = self.doppler_max_pool(F.pad(x1, (0, 1, 0, 0), "constant", 0))
            prb1_down = self.doppler_max_pool(F.pad(prb1, (0, 1, 0, 0), "constant", 0))
        else:
            x1_down = self.max_pool(x1)
            prb1_down = self.max_pool(prb1)
        x2 = self.single_conv_block2_3x3(self.single_conv_block1_3x3(x1_down))
        prb2 = self.single_conv_block2_3x3_prb(self.single_conv_block1_3x3_prb(prb1_down))
        if self.signal_type in ("range_doppler", "angle_doppler"):
            x2_down = self.doppler_max_pool(F.pad(x2, (0, 1, 0, 0), "constant", 0))
            prb2_down = self.doppler_max_pool(F.pad(prb2, (0, 1, 0, 0), "constant", 0))
        else:
            x2_down = self.max_pool(x2)
            prb2_down = self.max_pool(prb2)
        x3 = self.single_conv_block1_1x1(x2_down)
        prb3 = self.single_conv_block1_1x1_prb(prb2_down)
        return x2_down, x3, prb2_down, prb3


class TFocNet(nn.Module):
    def __init__(self, n_classes: int, n_frames: int, signal_type: str, zero_doppler_filter: bool = True):
        super().__init__()
        self.n_classes = n_classes
        self.n_frames = n_frames
        self.zero_doppler_filter = zero_doppler_filter
        self.rd_encoding_branch = EncodingBranch("range_doppler", self.n_frames)
        self.ra_encoding_branch = EncodingBranch("range_angle", self.n_frames)
        self.ad_encoding_branch = EncodingBranch("angle_doppler", self.n_frames)

        self.rd_aspp_block = ASPPBlock(128, 128)
        self.ra_aspp_block = ASPPBlock(128, 128)
        self.ad_aspp_block = ASPPBlock(128, 128)
        self.rd_single_conv_block1_1x1 = ConvBlock(640, 128, 1, 0, 1)
        self.ra_single_conv_block1_1x1 = ConvBlock(640, 128, 1, 0, 1)
        self.ad_single_conv_block1_1x1 = ConvBlock(640, 128, 1, 0, 1)
        self.rd_single_conv_block2_1x1 = ConvBlock(384, 128, 1, 0, 1)
        self.ra_single_conv_block2_1x1 = ConvBlock(384, 128, 1, 0, 1)
        self.rd_upconv1 = nn.ConvTranspose2d(384, 128, (2, 1), stride=(2, 1))
        self.ra_upconv1 = nn.ConvTranspose2d(384, 128, 2, stride=2)
        self.rd_double_conv_block1 = DoubleConvBlock(128, 128, 3, 1, 1)
        self.ra_double_conv_block1 = DoubleConvBlock(128, 128, 3, 1, 1)
        self.rd_upconv2 = nn.ConvTranspose2d(128, 128, (2, 1), stride=(2, 1))
        self.ra_upconv2 = nn.ConvTranspose2d(128, 128, 2, stride=2)
        self.rd_double_conv_block2 = DoubleConvBlock(128, 128, 3, 1, 1)
        self.ra_double_conv_block2 = DoubleConvBlock(128, 128, 3, 1, 1)
        self.rd_final = nn.Conv2d(128, n_classes, 1)
        self.ra_final = nn.Conv2d(128, n_classes, 1)

        self.rd_single_conv_block2_1x1_prb = ConvBlock(32 * 3, 32, 1, 0, 1)
        self.ra_single_conv_block2_1x1_prb = ConvBlock(32 * 3, 32, 1, 0, 1)
        self.rd_upconv1_prb = nn.ConvTranspose2d(32 * 3, 32, (2, 1), stride=(2, 1))
        self.rd_upconv2_prb = nn.ConvTranspose2d(32, 32, (2, 1), stride=(2, 1))
        self.ra_upconv1_prb = nn.ConvTranspose2d(32 * 3, 32, 2, stride=2)
        self.ra_upconv2_prb = nn.ConvTranspose2d(32, 32, 2, stride=2)
        self.rd_double_conv_block1_prb = DoubleConvBlock(32, 32, 3, 1, 1)
        self.ra_double_conv_block1_prb = DoubleConvBlock(32, 32, 3, 1, 1)
        self.rd_double_conv_block2_prb = DoubleConvBlock(32, 32, 3, 1, 1)
        self.ra_double_conv_block2_prb = DoubleConvBlock(32, 32, 3, 1, 1)
        self.rd_final_prb = nn.Conv2d(32, n_classes, 1, padding=0)
        self.ra_final_prb = nn.Conv2d(32, n_classes, 1, padding=0)

    def forward(self, x_rd: torch.Tensor, x_ra: torch.Tensor, x_ad: torch.Tensor, lamda: Optional[torch.Tensor] = None):
        ra_features, ra_latent, ra_prb, ra_prb_latent = self.ra_encoding_branch(x_ra)
        if self.zero_doppler_filter:
            x_rd = self._zero_doppler_filter(x_rd)
            x_ad = self._zero_doppler_filter(x_ad)
        rd_features, rd_latent, rd_prb, rd_prb_latent = self.rd_encoding_branch(x_rd)
        ad_features, ad_latent, ad_prb, ad_prb_latent = self.ad_encoding_branch(x_ad)

        x2_rd = self.rd_single_conv_block1_1x1(self.rd_aspp_block(rd_features))
        x2_ra = self.ra_single_conv_block1_1x1(self.ra_aspp_block(ra_features))
        x2_ad = self.ad_single_conv_block1_1x1(self.ad_aspp_block(ad_features))
        x3 = torch.cat((rd_latent, ra_latent, ad_latent), 1)
        x3_rd = self.rd_single_conv_block2_1x1(x3)
        x3_ra = self.ra_single_conv_block2_1x1(x3)
        x4_rd = torch.cat((x2_rd, x3_rd, x2_ad), 1)
        x4_ra = torch.cat((x2_ra, x3_ra, x2_ad), 1)

        x8_rd = self.rd_double_conv_block2(self.rd_upconv2(self.rd_double_conv_block1(self.rd_upconv1(x4_rd))))
        x8_ra = self.ra_double_conv_block2(self.ra_upconv2(self.ra_double_conv_block1(self.ra_upconv1(x4_ra))))
        x9_rd = self.rd_final(x8_rd)
        x9_ra = self.ra_final(x8_ra)

        prb_fusion = torch.cat((rd_prb_latent, ra_prb_latent, ad_prb_latent), 1)
        rd_prb_fusion = self.rd_single_conv_block2_1x1_prb(prb_fusion)
        ra_prb_fusion = self.ra_single_conv_block2_1x1_prb(prb_fusion)
        x4_rd_prb = torch.cat((rd_prb, rd_prb_fusion, ad_prb), 1)
        x4_ra_prb = torch.cat((ra_prb, ra_prb_fusion, ad_prb), 1)
        x8_rd_prb = self.rd_double_conv_block2_prb(self.rd_upconv2_prb(self.rd_double_conv_block1_prb(self.rd_upconv1_prb(x4_rd_prb))))
        x8_ra_prb = self.ra_double_conv_block2_prb(self.ra_upconv2_prb(self.ra_double_conv_block1_prb(self.ra_upconv1_prb(x4_ra_prb))))
        x9_rd_prb = self.rd_final_prb(x8_rd_prb)
        x9_ra_prb = self.ra_final_prb(x8_ra_prb)
        return x9_rd, x9_ra, x9_rd_prb, x9_ra_prb

    def _zero_doppler_filter(self, rd: torch.Tensor) -> torch.Tensor:
        device = rd.device
        mask = torch.cat(
            [
                torch.normal(1, 1e-3, size=(256, 31), device=device),
                torch.normal(1e-8, 1e-3, size=(256, 1), device=device),
                torch.normal(1, 1e-3, size=(256, 32), device=device),
            ],
            dim=-1,
        )
        return rd * mask.unsqueeze(0).unsqueeze(0)
