"""
TFoc vanilla + soft mask-draw (v1_revise)

This file is standalone and does not depend on the v1 model file.
Relative to v1, only TFocConv interaction is revised:
- interleave history/target by aligned time pairs: hst1,tgt1,hst2,tgt2,...
- perform non-linear interaction on candidate axis n while preserving n.
"""

from __future__ import annotations

import math
from typing import Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F


class DoubleConvBlock(nn.Module):
    """(Conv2d => BN => LeakyReLU) * 2"""

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
    """(Conv2d => BN => LeakyReLU)"""

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
    """(Conv3d => BN => LeakyReLU)"""

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
    """(Conv3d => BN => LeakyReLU) * 2"""

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
    """Atrous Spatial Pyramid Pooling"""

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


class TFocConvV1(nn.Module):
    """
    Temporal focusing conv (v1_revise): soft attention over local candidates.
    Revised only in pair interaction:
    - aligned interleaving across time pairs
    - pairwise non-linear mapping over candidate n without collapsing n
    """

    def __init__(
        self,
        in_ch: int,
        out_ch: int,
        btnk_ch: int,
        search_size: int,
        attn_temperature: float = 1.0,
        metric_kernel_n: Optional[int] = None,
    ):
        super().__init__()
        self.btnk_ch = btnk_ch
        self.k_n = int(search_size)
        self.pad_size = math.floor(self.k_n / 2)
        self.attn_temperature = float(attn_temperature)

        self.spatial_conv = Double3DConvBlock(
            in_ch,
            self.btnk_ch,
            k_size=(1, 3, 3),
            strd=(1, 1, 1),
            pad=(0, 1, 1),
            btnk_sc=16,
        )
        self.tfoc_prepadding = nn.ReplicationPad3d((self.pad_size, self.pad_size, self.pad_size, self.pad_size, 0, 0))

        n_candidates = self.k_n * self.k_n
        k_n = metric_kernel_n if metric_kernel_n is not None else n_candidates
        k_n = max(1, min(int(k_n), n_candidates))
        if k_n % 2 == 0:
            k_n = max(1, k_n - 1)
        pad_n = k_n // 2

        self.pair_interact = TREBlock(
            self.btnk_ch,
            self.btnk_ch,
            k_size=(2, 1, k_n),
            strd=(2, 1, 1),
            pad=(0, 0, pad_n),
        )
        self.inner_predict = TREBlock(self.btnk_ch, 4, k_size=(1, 1, 1), strd=(1, 1, 1), pad=(0, 0, 0))

        self.temporal_conv = nn.Sequential(
            TREBlock(self.btnk_ch, out_ch, k_size=(2, 1, 1), strd=(1, 1, 1), pad=(0, 0, 0)),
            TREBlock(out_ch, out_ch, k_size=(1, 1, 1), strd=(1, 1, 1), pad=(0, 0, 0)),
        )
        # Reuse integer sampling grids across forwards to reduce allocator churn.
        self._grid_center_cache = {}
        self._grid_local_cache = {}

    def forward(self, x_seq: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        x_seq = self.spatial_conv(x_seq)
        b, c, t, h, w = x_seq.shape
        device = x_seq.device
        n = self.k_n * self.k_n

        x_tgt_ori = x_seq[:, :, -1, :, :].unsqueeze(2)
        x_tgt_rep = x_tgt_ori.expand(-1, -1, t - 1, -1, -1)
        x_hst = self.tfoc_prepadding(x_seq[:, :, : t - 1, :, :])

        p_hst = self._gen_global_sample_grid(b, h, w, n, device=device)

        x_tgt_cand = x_tgt_rep.unsqueeze(-1).expand(-1, -1, -1, -1, -1, n)
        x_hst_cand = self._sample_x(x_hst, p_hst, n)

        x_tgt_flat = x_tgt_cand.reshape(b, c, t - 1, h * w, n)
        x_hst_flat = x_hst_cand.reshape(b, c, t - 1, h * w, n)

        pair = torch.stack((x_hst_flat, x_tgt_flat), dim=3)
        pair = pair.reshape(b, c, 2 * (t - 1), h * w, n)

        pair_feat = self.pair_interact(pair)
        logits = self.inner_predict(pair_feat)
        probs_all = F.softmax(logits, dim=1)

        attn_score = 1.0 - probs_all[:, 0, ...]
        attn = F.softmax(attn_score / max(self.attn_temperature, 1e-6), dim=-1)

        x_hist_agg = torch.sum(x_hst_flat * attn.unsqueeze(1), dim=-1)

        time_score = attn_score.mean(dim=-1)
        time_w = F.softmax(time_score, dim=1)
        x_hist_agg = torch.sum(x_hist_agg * time_w.unsqueeze(1), dim=2)
        x_hist_agg = x_hist_agg.reshape(b, c, 1, h, w)

        probs_map = probs_all.max(dim=-1).values.mean(dim=2).reshape(b, 4, h, w)

        x_fuse = torch.cat((x_hist_agg, x_tgt_ori), dim=2)
        x_out = self.temporal_conv(x_fuse)
        return x_out, probs_map

    def _get_grid_center(self, b: int, h: int, w: int, n: int, device: torch.device) -> torch.Tensor:
        key = (h, w, n, device.type, device.index)
        if key in self._grid_center_cache:
            grid_c = self._grid_center_cache[key]
            return grid_c if b == 1 else grid_c.expand(b, -1, -1, -1)

        k_ls = (self.k_n - 1) // 2
        gc_x_mesh, gc_y_mesh = torch.meshgrid(
            torch.arange(k_ls, h + k_ls, device=device),
            torch.arange(k_ls, w + k_ls, device=device),
            indexing="ij",
        )
        gc_x = torch.flatten(gc_x_mesh).view(1, 1, h, w).expand(1, n, h, w)
        gc_y = torch.flatten(gc_y_mesh).view(1, 1, h, w).expand(1, n, h, w)
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
        grid_c = self._get_grid_center(b, h, w, n, device=device)
        grid_ls = self._get_grid_local_search(n, device=device)
        return grid_c + grid_ls

    def _sample_x(self, x_pad: torch.Tensor, p: torch.Tensor, n: int) -> torch.Tensor:
        p = p.permute(0, 2, 3, 1)
        b, h, w, _ = p.size()
        w_pad = x_pad.size(4)
        c = x_pad.size(1)
        t = x_pad.size(2)

        x_pad = x_pad.reshape(b, c, t, -1)
        index = p[..., :n] * w_pad + p[..., n:]
        index = index.long().reshape(b, 1, 1, -1).expand(-1, c, t, -1)

        x_r = x_pad.gather(dim=-1, index=index)
        x_r = x_r.reshape(b, c, t, h, w, n)
        return x_r


class EncodingBranch(nn.Module):
    def __init__(self, signal_type: str, n_frames: int):
        super().__init__()
        self.n_frames = n_frames
        self.signal_type = signal_type

        self.double_3d_conv_block = Double3DConvBlock(
            in_ch=1,
            out_ch=64,
            k_size=(3, 3, 3),
            strd=(1, 1, 1),
            pad=(0, 1, 1),
            btnk_sc=1,
        )

        self.tfoc_conv_block = TFocConvV1(
            in_ch=64,
            out_ch=128,
            btnk_ch=8,
            search_size=5,
            attn_temperature=1.0,
        )

        self.doppler_max_pool = nn.MaxPool2d(2, stride=(2, 1))
        self.max_pool = nn.MaxPool2d(2, stride=2)
        self.single_conv_block1_3x3 = ConvBlock(in_ch=128, out_ch=128, k_size=3, pad=1, dil=1)
        self.single_conv_block2_3x3 = ConvBlock(in_ch=128, out_ch=128, k_size=3, pad=1, dil=1)
        self.single_conv_block1_1x1 = ConvBlock(in_ch=128, out_ch=128, k_size=1, pad=0, dil=1)

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        x0 = self.double_3d_conv_block(x)
        x1, _ = self.tfoc_conv_block(x0)
        x1 = x1.squeeze(dim=2)

        if self.signal_type in ("range_doppler", "angle_doppler"):
            x1_pad = F.pad(x1, (0, 1, 0, 0), "constant", 0)
            x1_down = self.doppler_max_pool(x1_pad)
        else:
            x1_down = self.max_pool(x1)

        x2 = self.single_conv_block1_3x3(x1_down)
        x2 = self.single_conv_block2_3x3(x2)

        if self.signal_type in ("range_doppler", "angle_doppler"):
            x2_pad = F.pad(x2, (0, 1, 0, 0), "constant", 0)
            x2_down = self.doppler_max_pool(x2_pad)
        else:
            x2_down = self.max_pool(x2)

        x3 = self.single_conv_block1_1x1(x2_down)
        return x2_down, x3


class TFocNet(nn.Module):
    def __init__(self, n_classes: int, n_frames: int, signal_type: str, zero_doppler_filter: bool = True):
        super().__init__()
        self.n_classes = n_classes
        self.n_frames = n_frames
        self.view_type = signal_type
        self.zero_doppler_filter = zero_doppler_filter

        # Backbone (single-view encoding)
        self.encoding_branch = EncodingBranch(self.view_type, self.n_frames)

        # ASPP
        self.aspp_block = ASPPBlock(in_ch=128, out_ch=128)
        self.single_conv_block1_1x1 = ConvBlock(in_ch=640, out_ch=128, k_size=1, pad=0, dil=1)

        # Latent branch
        self.single_conv_block2_1x1 = ConvBlock(in_ch=128, out_ch=128, k_size=1, pad=0, dil=1)

        # Decoder
        if self.view_type in ("range_doppler", "angle_doppler"):
            self.upconv1 = nn.ConvTranspose2d(256, 128, (2, 1), stride=(2, 1))
            self.upconv2 = nn.ConvTranspose2d(128, 128, (2, 1), stride=(2, 1))
        else:
            self.upconv1 = nn.ConvTranspose2d(256, 128, 2, stride=2)
            self.upconv2 = nn.ConvTranspose2d(128, 128, 2, stride=2)
        self.double_conv_block1 = DoubleConvBlock(in_ch=128, out_ch=128, k_size=3, pad=1, dil=1)
        self.double_conv_block2 = DoubleConvBlock(in_ch=128, out_ch=128, k_size=3, pad=1, dil=1)
        self.final = nn.Conv2d(in_channels=128, out_channels=n_classes, kernel_size=1)

    def forward(self, x: torch.Tensor):
        if self.zero_doppler_filter and self.view_type in ("range_doppler", "angle_doppler"):
            x = self._zero_doppler_filter(x)

        features, latent = self.encoding_branch(x)
        x1 = self.aspp_block(features)
        x2 = self.single_conv_block1_1x1(x1)
        x3 = self.single_conv_block2_1x1(latent)
        x4 = torch.cat((x2, x3), 1)

        x5 = self.upconv1(x4)
        x6 = self.double_conv_block1(x5)
        x7 = self.upconv2(x6)
        x8 = self.double_conv_block2(x7)
        x9 = self.final(x8)
        return x9

    def _zero_doppler_filter(self, rd: torch.Tensor) -> torch.Tensor:
        device = rd.device
        zero_doppler_mask = torch.cat(
            [
                torch.normal(1, 1e-3, size=(256, 31), device=device),
                torch.normal(1e-8, 1e-3, size=(256, 1), device=device),
                torch.normal(1, 1e-3, size=(256, 32), device=device),
            ],
            dim=-1,
        )
        return rd * zero_doppler_mask.unsqueeze(dim=0).unsqueeze(dim=0)
