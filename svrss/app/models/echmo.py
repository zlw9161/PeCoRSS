"""
EchMo Model Implementation
End-to-End Radar Echo Segmentation Model
"""

import os
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.nn.init as init
import scipy.io as scio


# ==================== Utility Functions ====================
def complex_std(x, epsilon=1e-5):
    mu_R = x.real.mean(dim=3, keepdim=True).detach()
    mu_I = x.imag.mean(dim=3, keepdim=True).detach()
    x_R = x.real - mu_R
    x_I = x.imag - mu_I
    sigma_R = torch.sqrt(torch.var(x.real, dim=3, keepdim=True, unbiased=False) + epsilon)
    sigma_I = torch.sqrt(torch.var(x.imag, dim=3, keepdim=True, unbiased=False) + epsilon)
    x_R /= sigma_R
    x_I /= sigma_I
    return x_R + 1j * x_I


def complex_double2float(x):
    x_r = x.real.float()
    x_i = x.imag.float()
    return x_r + 1j * x_i


def filter_coef_loader(path, coef_type=None):
    coef = scio.loadmat(os.path.join(path, 's_tx.mat'))['s_tx']
    coef = torch.tensor(coef)
    
    if coef_type == 'ori_mf':
        coef = complex_double2float(torch.conj(coef))
        zero = torch.complex(torch.zeros([1, 1], dtype=torch.float32), torch.zeros([1, 1], dtype=torch.float32))
        coef = torch.cat((zero, coef), dim=1)
        return coef
    elif coef_type == 'tay_mf':
        win = scio.loadmat(os.path.join(path, 'w_taylor.mat'))['w_taylor']
    elif coef_type == 'che_mf':
        win = scio.loadmat(os.path.join(path, 'w_chebyshev.mat'))['w_chebyshev']
    elif coef_type == 'ham_mf':
        win = scio.loadmat(os.path.join(path, 'w_hamming.mat'))['w_hamming']
    elif coef_type == 'han_mf':
        win = scio.loadmat(os.path.join(path, 'w_hanning.mat'))['w_hanning']
    elif coef_type == 'black_mf':
        win = scio.loadmat(os.path.join(path, 'w_blackman.mat'))['w_blackman']
    elif coef_type == 'cos_mf':
        win = scio.loadmat(os.path.join(path, 'w_cosine.mat'))['w_cosine']
    else:
        win = scio.loadmat(os.path.join(path, 'w_taylor.mat'))['w_taylor']

    win = torch.permute(torch.tensor(win), [1, 0])
    coef = complex_double2float(torch.conj(coef * win))
    zero = torch.complex(torch.zeros([1, 1], dtype=torch.float32), torch.zeros([1, 1], dtype=torch.float32))
    coef = torch.cat((zero, coef), dim=1)
    return coef


# ==================== Model Components ====================
class ComplexBatchNorm2d(nn.Module):
    def __init__(self, num_features):
        super().__init__()
        self.bn_real = nn.BatchNorm2d(num_features)
        self.bn_imag = nn.BatchNorm2d(num_features)
    
    def forward(self, x):
        return torch.complex(self.bn_real(x.real), self.bn_imag(x.imag))


class ComplexReLU(nn.Module):
    def __init__(self, activation='leaky_relu'):
        super(ComplexReLU, self).__init__()
        self.activation = activation
    
    def forward(self, input):
        if self.activation == 'relu':
            return nn.ReLU()(input.real) + 1j * nn.ReLU()(input.imag)
        elif self.activation == 'leaky_relu':
            return F.leaky_relu(input.real) + 1j * F.leaky_relu(input.imag)
        elif self.activation == 'celu':
            return F.celu(input.real) + 1j * F.celu(input.imag)
        else:
            return input


class DoubleConvBlock(nn.Module):
    def __init__(self, in_ch, out_ch, k_size, pad, dil):
        super().__init__()
        self.block = nn.Sequential(
            nn.Conv2d(in_ch, out_ch, kernel_size=k_size, padding=pad, dilation=dil),
            nn.BatchNorm2d(out_ch),
            nn.LeakyReLU(inplace=True),
            nn.Conv2d(out_ch, out_ch, kernel_size=k_size, padding=pad, dilation=dil),
            nn.BatchNorm2d(out_ch),
            nn.LeakyReLU(inplace=True)
        )

    def forward(self, x):
        return self.block(x)


class Double3DConvBlock(nn.Module):
    def __init__(self, in_ch, out_ch, k_size, pad, dil):
        super().__init__()
        self.block = nn.Sequential(
            nn.Conv3d(in_ch, out_ch, kernel_size=k_size, padding=pad, dilation=dil),
            nn.BatchNorm3d(out_ch),
            nn.LeakyReLU(inplace=True),
            nn.Conv3d(out_ch, out_ch, kernel_size=k_size, padding=pad, dilation=dil),
            nn.BatchNorm3d(out_ch),
            nn.LeakyReLU(inplace=True)
        )

    def forward(self, x):
        return self.block(x)


class ConvBlock(nn.Module):
    def __init__(self, in_ch, out_ch, k_size, pad, dil):
        super().__init__()
        self.block = nn.Sequential(
            nn.Conv2d(in_ch, out_ch, kernel_size=k_size, padding=pad, dilation=dil),
            nn.BatchNorm2d(out_ch),
            nn.LeakyReLU(inplace=True)
        )

    def forward(self, x):
        return self.block(x)


class PeakConv2D(nn.Module):
    def __init__(self, in_ch, out_ch, bias, refer_band, guard_band_dop, guard_band, signal_type):
        super(PeakConv2D, self).__init__()
        self.signal_type = signal_type
        if self.signal_type in ('range_doppler', 'angle_doppler'):
            self.guard_band = (guard_band_dop, guard_band_dop)
            self.padding = (self.guard_band[1] + refer_band,
                            self.guard_band[1] + refer_band,
                            self.guard_band[0] + refer_band,
                            self.guard_band[0] + refer_band)
        else:
            self.guard_band = (guard_band, guard_band)
            self.padding = (self.guard_band[1] + refer_band,
                            self.guard_band[1] + refer_band,
                            self.guard_band[0] + refer_band,
                            self.guard_band[0] + refer_band)
        self.refer_band = refer_band
        self.kernel_size = (self.guard_band[0] + self.guard_band[1] + self.refer_band * 2), 4
        self.stride = self.kernel_size
        self.zero_padding = nn.ZeroPad2d(self.padding)
        self.bias = bias
        self.pc_strd = 1
        self.peak_conv = nn.Conv2d(in_ch, out_ch,
                                   kernel_size=self.kernel_size,
                                   padding=0,
                                   stride=self.stride,
                                   bias=self.bias)

    def forward(self, x):
        h, w = x.shape[2], x.shape[3]
        rb = self.refer_band
        gb = self.guard_band
        k_pk = self.kernel_size
        N = k_pk[0] * k_pk[1]
        dtype = x.data.type()
        b = x.size(0)

        x = self.zero_padding(x)
        p_r = self._get_p_r(b, h, w, N, dtype)
        x_r = self._sample_x(x, p_r, N)
        p_c = self._get_p_c(b, h, w, N, dtype)
        x_c = self._sample_x(x, p_c, N)
        x_prf = x_c - x_r
        x_prf = self._reshape_x_prf(k_pk[0], k_pk[1], x_prf)
        out = self.peak_conv(x_prf)
        return out

    def _get_p_c(self, b, h, w, N, dtype):
        p_c_x, p_c_y = torch.meshgrid(
            torch.arange(self.padding[2], h * self.pc_strd + self.padding[2], self.pc_strd),
            torch.arange(self.padding[0], w * self.pc_strd + self.padding[0], self.pc_strd), indexing='ij')
        p_c_x = torch.flatten(p_c_x).view(1, 1, h, w).repeat(1, N, 1, 1)
        p_c_y = torch.flatten(p_c_y).view(1, 1, h, w).repeat(1, N, 1, 1)
        p_c = torch.cat([p_c_x, p_c_y], 1).type(dtype)
        p_c = p_c.repeat(b, 1, 1, 1)
        return p_c

    def _gen_prf_grid(self, rb, gb, N, dtype):
        h_t = -(rb + gb[0])
        h_d = rb + gb[0]
        w_l = -(rb + gb[1])
        w_r = rb + gb[1]
        w_prf = (rb + gb[1]) * 2 + 1
        h_prf = (rb + gb[0]) * 2 + 1

        prf_x_idx, prf_y_idx = torch.meshgrid(
            torch.arange(h_t, h_d + 1),
            torch.arange(w_l, w_r + 1), indexing='ij')

        prf_xt = prf_x_idx[0:rb]
        prf_xr = prf_x_idx[rb:(h_prf - 2 * rb + 1), (w_prf - rb):w_prf]
        prf_xd = prf_x_idx[(h_prf - rb):h_prf]
        prf_xl = prf_x_idx[rb:(h_prf - 2 * rb + 1), 0:rb]
        prf_x = torch.cat([torch.flatten(prf_xt),
                           torch.flatten(prf_xr),
                           torch.flatten(prf_xd),
                           torch.flatten(prf_xl)], 0)
        prf_yt = prf_y_idx[0:rb]
        prf_yr = prf_y_idx[rb:(h_prf - 2 * rb + 1), (w_prf - rb):w_prf]
        prf_yd = prf_y_idx[(h_prf - rb):h_prf]
        prf_yl = prf_y_idx[rb:(h_prf - 2 * rb + 1), 0:rb]
        prf_y = torch.cat([torch.flatten(prf_yt),
                           torch.flatten(prf_yr),
                           torch.flatten(prf_yd),
                           torch.flatten(prf_yl)], 0)

        prf = torch.cat([prf_x, prf_y], 0)
        prf = prf.view(1, 2 * N, 1, 1).type(dtype)
        return prf

    def _get_p_r(self, b, h, w, N, dtype):
        rb = self.refer_band
        gb = self.guard_band
        prf = self._gen_prf_grid(rb, gb, N, dtype)
        p_c = self._get_p_c(b, h, w, N, dtype)
        p_r = p_c + prf
        return p_r

    def _sample_x(self, x_pad, p, N):
        p = p.contiguous().permute(0, 2, 3, 1).floor()
        b, h, w, _ = p.size()
        h_pad = x_pad.size(2)
        w_pad = x_pad.size(3)
        c = x_pad.size(1)
        x_pad = x_pad.contiguous().view(b, c, -1)
        index = p[..., :N] * w_pad + p[..., N:]
        index = index.contiguous().unsqueeze(dim=1).expand(-1, c, -1, -1, -1).contiguous().view(b, c, -1)
        index = index.long().to(x_pad.device)
        x_r = x_pad.gather(dim=-1, index=index).contiguous().view(b, c, h, w, N)
        return x_r

    @staticmethod
    def _reshape_x_prf(k_h, k_w, x_prf):
        b, c, h, w, N = x_prf.size()
        x_prf = torch.cat([x_prf[..., s:s + k_w].contiguous().view(b, c, h, w * k_w) for s in range(0, N, k_w)], dim=-1)
        x_prf = x_prf.contiguous().view(b, c, h * k_h, w * k_w)
        return x_prf


class DoublePeakConv2D(nn.Module):
    def __init__(self, in_ch, out_ch, bias, rb, gb_dop, gb, sig):
        super(DoublePeakConv2D, self).__init__()
        self.bias = bias
        self.refer_band = rb
        self.guard_band_dop = gb_dop
        self.guard_band = gb
        self.signal_type = sig
        self.pk_conv1 = PeakConv2D(in_ch, out_ch, bias=self.bias,
                                   refer_band=self.refer_band,
                                   guard_band_dop=self.guard_band_dop,
                                   guard_band=self.guard_band,
                                   signal_type=self.signal_type)
        self.bn1 = nn.BatchNorm2d(out_ch)
        self.act1 = nn.LeakyReLU(inplace=True)
        self.pk_conv2 = PeakConv2D(out_ch, out_ch, bias=self.bias,
                                   refer_band=self.refer_band,
                                   guard_band_dop=self.guard_band_dop,
                                   guard_band=self.guard_band,
                                   signal_type=self.signal_type)
        self.bn2 = nn.BatchNorm2d(out_ch)
        self.act2 = nn.LeakyReLU(inplace=True)

    def forward(self, x):
        x = self.pk_conv1(x)
        x = self.bn1(x)
        x = self.act1(x)
        x = self.pk_conv2(x)
        x = self.bn2(x)
        x = self.act2(x)
        return x


class ASPPBlock(nn.Module):
    def __init__(self, in_ch, out_ch, shape):
        super().__init__()
        self.shape = shape
        self.global_avg_pool = nn.AvgPool2d(self.shape)
        self.conv1_1x1 = nn.Conv2d(in_ch, out_ch, kernel_size=1, padding=0, dilation=1)
        self.single_conv_block1_1x1 = ConvBlock(in_ch, out_ch, k_size=1, pad=0, dil=1)
        self.single_conv_block1_3x3 = ConvBlock(in_ch, out_ch, k_size=3, pad=6, dil=6)
        self.single_conv_block2_3x3 = ConvBlock(in_ch, out_ch, k_size=3, pad=12, dil=12)
        self.single_conv_block3_3x3 = ConvBlock(in_ch, out_ch, k_size=3, pad=18, dil=18)

    def forward(self, x):
        x1 = F.interpolate(self.global_avg_pool(x), size=self.shape, align_corners=False, mode='bilinear')
        x1 = self.conv1_1x1(x1)
        x2 = self.single_conv_block1_1x1(x)
        x3 = self.single_conv_block1_3x3(x)
        x4 = self.single_conv_block2_3x3(x)
        x5 = self.single_conv_block3_3x3(x)
        x_cat = torch.cat((x2, x3, x4, x5, x1), 1)
        return x_cat


class EncodingBranch(nn.Module):
    def __init__(self, signal_type, n_frames, in_ch=1):
        super().__init__()
        self.signal_type = signal_type
        self.n_frames = n_frames
        self.in_ch = in_ch
        self.double_3dconv_block1 = Double3DConvBlock(in_ch=self.in_ch, out_ch=32,
                                                      k_size=(1+n_frames//2, 3, 3),
                                                      pad=(0, 1, 1), dil=1)
        self.doppler_max_pool = nn.MaxPool2d(2, stride=(1, 2))
        self.max_pool = nn.MaxPool2d(2, stride=2)
        self.double_pkc_block = DoublePeakConv2D(in_ch=32, out_ch=64,
                                                 bias=True, rb=1, gb_dop=1, gb=1, sig=self.signal_type)
        self.single_conv_block1_1x1 = ConvBlock(in_ch=64, out_ch=128, k_size=1, pad=0, dil=1)

    def forward(self, x):
        x1 = self.double_3dconv_block1(x)
        x1 = torch.squeeze(x1, 2)
        x1_pad = F.pad(x1, (0, 0, 0, 0), "constant", 0)
        x1_down = self.max_pool(x1_pad)
        x2 = self.double_pkc_block(x1_down)
        x2_pad = F.pad(x2, (0, 0, 0, 1), "constant", 0)
        x2_down = self.doppler_max_pool(x2_pad)
        x3 = self.single_conv_block1_1x1(x2_down)
        return x2_down, x3


class PKCIn4ANY(nn.Module):
    def __init__(self, n_classes, n_frames, in_ch, w, h):
        super().__init__()
        self.n_classes = n_classes
        self.n_frames = n_frames
        self.in_ch = in_ch
        self.rd_shape = (int(w//2), int(h//4))

        self.rd_encoding_branch = EncodingBranch('range_doppler', self.n_frames, self.in_ch)
        self.rd_aspp_block = ASPPBlock(in_ch=64, out_ch=128, shape=self.rd_shape)
        self.rd_single_conv_block1_1x1 = ConvBlock(in_ch=640, out_ch=128, k_size=1, pad=0, dil=1)

        self.rd_upconv1 = nn.ConvTranspose2d(256, 128, (2, 2), stride=(2, 2))
        self.rd_double_conv_block1 = DoubleConvBlock(in_ch=128, out_ch=128, k_size=3, pad=1, dil=1)
        self.rd_upconv2 = nn.ConvTranspose2d(128, 128, (1, 2), stride=(1, 2))
        self.rd_double_conv_block2 = DoubleConvBlock(in_ch=128, out_ch=128, k_size=3, pad=1, dil=1)
        self.rd_final = nn.Conv2d(in_channels=128, out_channels=n_classes, kernel_size=1)

    def forward(self, x_rd):
        rd_features, rd_latent = self.rd_encoding_branch(x_rd)
        x2_rd = self.rd_aspp_block(rd_features)
        x2_rd = self.rd_single_conv_block1_1x1(x2_rd)
        x4_rd = torch.cat((x2_rd, rd_latent), 1)
        x4_rd = self.rd_upconv1(x4_rd)
        x4_rd = self.rd_double_conv_block1(x4_rd)
        x4_rd = self.rd_upconv2(x4_rd)
        x4_rd = self.rd_double_conv_block2(x4_rd)
        x4_rd = self.rd_final(x4_rd)
        return x4_rd


def _init_translator_weights(m):
    if isinstance(m, nn.Linear):
        torch.nn.init.xavier_uniform_(m.weight)
        nn.init.constant_(m.bias, 0.)
    elif isinstance(m, nn.Conv2d):
        torch.nn.init.xavier_uniform_(m.weight)
        if m.bias is not None:
            nn.init.constant_(m.bias, 0.)
        elif isinstance(m, nn.BatchNorm2d):
            nn.init.uniform_(m.weight, 0., 1.)
            nn.init.constant_(m.bias, 0.)


class Leanable_Match_FilterV5(nn.Module):
    def __init__(self, k_size, in_ch, out_ch, lmf_coef, activation='relu', bias=True):
        super(Leanable_Match_FilterV5, self).__init__()
        self.k_size = k_size
        self.stride = 1
        self.in_ch = in_ch
        self.out_ch = out_ch
        self.activation = activation
        self.bias = bias
        self.lmf_real = nn.Conv2d(self.in_ch, self.out_ch,
                                  kernel_size=(1, k_size), stride=(1, 1),
                                  bias=self.bias)
        self.lmf_imag = nn.Conv2d(self.in_ch, self.out_ch,
                                  kernel_size=(1, k_size), stride=(1, 1),
                                  bias=self.bias)
        init.kaiming_uniform_(self.lmf_real.weight)
        init.kaiming_uniform_(self.lmf_imag.weight)
        if self.bias:
            init.zeros_(self.lmf_real.bias)
            init.zeros_(self.lmf_imag.bias)
        self.cause_padding = torch.nn.ZeroPad2d(padding=(0, self.k_size - 1, 0, 0))
        self.cmplx_relu = ComplexReLU(self.activation)

    def forward(self, x):
        inputR = self.cause_padding(x.real)
        inputI = self.cause_padding(x.imag)
        x_RR = self.lmf_real(inputR)
        x_II = self.lmf_imag(inputI)
        x_RI = self.lmf_imag(inputR)
        x_IR = self.lmf_real(inputI)
        x = x_RR - x_II + 1j * (x_RI + x_IR)
        x = self.cmplx_relu(x)
        return x


class LMF_Chain(nn.Module):
    def __init__(self, configs):
        super(LMF_Chain, self).__init__()
        self.lmf_coef = filter_coef_loader(os.path.join(configs.dataset_path, configs.coef_path), configs.lmf_coefs)
        self.in_ch = 1
        self.Chain = nn.Sequential()
        for l in range(configs.lmf_layers):
            self.Chain.add_module('lmfV5-{0}'.format(l),
                                  Leanable_Match_FilterV5(k_size=self.lmf_coef.shape[1],
                                                          in_ch=self.in_ch,
                                                          out_ch=configs.lmf_channels,
                                                          lmf_coef=self.lmf_coef,
                                                          activation=configs.activation,
                                                          bias=configs.bias))
            self.in_ch = configs.lmf_channels
        
    def forward(self, x):
        x = self.Chain(x)
        return x


class Multi_Order_LDC(nn.Module):
    def __init__(self, in_ch, out_ch, orders, activation='relu', bias=True):
        super(Multi_Order_LDC, self).__init__()
        self.in_ch = in_ch
        self.out_ch = out_ch
        self.orders = orders
        self.activation = activation
        self.bias = bias
        self.ldc_list = nn.ModuleList()
        self.pad_list = nn.ModuleList()
        for n in range(self.orders):
            ldc = nn.Conv2d(self.in_ch, self.out_ch, kernel_size=(n+2, 1), bias=self.bias)
            pad = torch.nn.ZeroPad2d(padding=(0, 0, n+1, 0))
            self.in_ch = self.out_ch
            self.ldc_list.append(ldc)
            self.pad_list.append(pad)
        self.cmplx_relu = ComplexReLU(self.activation)
        self._initialize_MLDC_weights()

    def _initialize_MLDC_weights(self):
        for n, conv in enumerate(self.ldc_list):
            nn.init.kaiming_normal_(conv.weight)
            if self.bias:
                nn.init.zeros_(conv.bias)

    def forward(self, x):
        B, P, R = x.shape[0], x.shape[2], x.shape[3]
        x = x.unsqueeze(dim=1).broadcast_to(B, self.orders, x.shape[1], P, R)
        y_r = torch.zeros(B, self.orders, self.out_ch, P, R).to(x.device)
        y_i = torch.zeros(B, self.orders, self.out_ch, P, R).to(x.device)
        for n in range(self.orders):
            x_n_real = self.ldc_list[n](self.pad_list[n](x[:,n,...].real))
            x_n_imag = self.ldc_list[n](self.pad_list[n](x[:,n,...].imag))
            y_r[:,n,...] = x_n_real
            y_i[:,n,...] = x_n_imag
        y_r = y_r.view(B, self.orders*self.out_ch, P, R)
        y_i = y_i.view(B, self.orders*self.out_ch, P, R)
        y = torch.complex(y_r, y_i)
        y = self.cmplx_relu(y)
        return y


class Group_Purturble_DF(nn.Module):
    def __init__(self, in_ch, out_ch, k_size, groups, activation='relu', bias=True):
        super(Group_Purturble_DF, self).__init__()
        self.k_size = k_size
        self.in_ch = in_ch
        self.out_ch = out_ch
        self.groups = groups
        self.activation = activation
        self.bias = bias
        self.gdf_cos = nn.Conv2d(self.in_ch*self.groups, self.out_ch*self.groups,
                                 kernel_size=(k_size, 1), stride=(k_size, 1),
                                 padding=0, groups=self.groups, bias=self.bias)
        self.gdf_sin = nn.Conv2d(self.in_ch*self.groups, self.out_ch*self.groups,
                                 kernel_size=(k_size, 1), stride=(k_size, 1),
                                 padding=0, groups=self.groups, bias=self.bias)
        self._initialize_GDF_weights()
        self.cmplx_relu = ComplexReLU(self.activation)

    def _initialize_GDF_weights(self):
        init.kaiming_uniform_(self.gdf_cos.weight)
        init.kaiming_uniform_(self.gdf_sin.weight)
        if self.bias:
            init.zeros_(self.gdf_cos.bias)
            init.zeros_(self.gdf_sin.bias)

    def forward(self, x):
        B, R = x.shape[0], x.shape[3]
        x_real, x_imag = x.real.to(x.device), x.imag.to(x.device)
        x_rr = self.gdf_cos(x_real)
        x_ii = self.gdf_sin(x_imag)
        x_ri = self.gdf_sin(x_real)
        x_ir = self.gdf_cos(x_imag)
        x = x_rr + x_ii - 1j * x_ri + 1j * x_ir
        x = self.cmplx_relu(x)
        x = x.view(B, self.groups, self.out_ch, 1, R).squeeze(dim=3)
        return x


class EchMo(nn.Module):
    def __init__(self, configs):
        super(EchMo, self).__init__()
        self.cfg = configs
        self.task_name = configs.task_name
        self.n_range = configs.n_range
        self.n_pulse = configs.n_pulse
        self.n_classes = configs.n_classes
        self.n_frames = configs.n_frames
        self.scale = configs.scale
        self.activation = configs.activation
        self.bias = configs.bias
        self.lmf_chains = configs.lmf_chains
        self.lmf_channels = configs.lmf_channels
        self.lmf_layers = configs.lmf_layers
        self.lmf_coefs = configs.lmf_coefs
        self.ldc_orders = configs.ldc_orders
        self.ldc_channels = configs.ldc_channels
        self.df_groups = configs.ldc_orders
        self.df_channels = configs.df_channels
        
        self.LMFChain = nn.ModuleList([LMF_Chain(configs=self.cfg) for _ in range(self.lmf_chains)])
        self.MLDC = Multi_Order_LDC(in_ch=self.lmf_channels*self.lmf_chains,
                                    out_ch=self.ldc_channels,
                                    orders=self.ldc_orders,
                                    activation=self.activation,
                                    bias=self.bias)
        self.GDF = Group_Purturble_DF(in_ch=self.ldc_channels,
                                      out_ch=self.df_channels,
                                      k_size=self.n_pulse,
                                      groups=self.df_groups,
                                      activation=self.activation,
                                      bias=self.bias)
        self.RDTranslator = PKCIn4ANY(n_classes=self.n_classes,
                                      n_frames=self.n_frames,
                                      in_ch=self.df_groups,
                                      w=self.df_channels,
                                      h=self.n_range)
        self.RDTranslator.apply(_init_translator_weights)

    def forward(self, x):
        if self.scale == 'instance':
            x = complex_double2float(complex_std(x, epsilon=1e-8))
        elif self.scale == 'global':
            x = complex_double2float(x)

        for i in range(self.lmf_chains):
            if i == 0:
                x_k = self.LMFChain[i](x)
            else:
                x_ki = self.LMFChain[i](x)
                x_k = torch.concatenate([x_k, x_ki], dim=1)
        x_h = self.MLDC(x_k)
        x_d = self.GDF(x_h)
        gdf_out = x_d
        x_d = torch.abs(x_d).unsqueeze(dim=2)
        y = self.RDTranslator(x_d)
        return y, x_k, x_h, gdf_out
