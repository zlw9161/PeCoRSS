import torch
import torch.nn as nn
import torch.nn.functional as F

from svrss.models.model_utils import PeakConv2D

class DoubleConvBlock(nn.Module):
    """ (2D conv => BN => LeakyReLU) * 2 """

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
        x = self.block(x)
        return x


class Double3DConvBlock(nn.Module):
    """ (3D conv => BN => LeakyReLU) * 2 """

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
        x = self.block(x)
        return x


class ConvBlock(nn.Module):
    """ (2D conv => BN => LeakyReLU) """

    def __init__(self, in_ch, out_ch, k_size, pad, dil):
        super().__init__()
        self.block = nn.Sequential(
            nn.Conv2d(in_ch, out_ch, kernel_size=k_size, padding=pad, dilation=dil),
            nn.BatchNorm2d(out_ch),
            nn.LeakyReLU(inplace=True)
        )

    def forward(self, x):
        x = self.block(x)
        return x

# double PeakConv2D block definition zlw @20220621
class DoublePeakConv2D(nn.Module):
    """ (2D PeakConv => BN => LeakyReLU) * 2 """

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
        # print("x:", type(x))
        x = self.bn1(x)
        x = self.act1(x)
        x = self.pk_conv2(x)
        x = self.bn2(x)
        x = self.act2(x)
        return x


class ASPPBlock(nn.Module):
    """Atrous Spatial Pyramid Pooling
    Parallel conv blocks with different dilation rate
    """

    def __init__(self, in_ch, out_ch, shape):
        super().__init__()
        self.shape = shape
        self.global_avg_pool = nn.AvgPool2d(self.shape) # 62, 512 for cwr
        self.conv1_1x1 = nn.Conv2d(in_ch, out_ch, kernel_size=1, padding=0, dilation=1)
        self.single_conv_block1_1x1 = ConvBlock(in_ch, out_ch, k_size=1, pad=0, dil=1)
        self.single_conv_block1_3x3 = ConvBlock(in_ch, out_ch, k_size=3, pad=6, dil=6)
        self.single_conv_block2_3x3 = ConvBlock(in_ch, out_ch, k_size=3, pad=12, dil=12)
        self.single_conv_block3_3x3 = ConvBlock(in_ch, out_ch, k_size=3, pad=18, dil=18)

    def forward(self, x):
        x1 = F.interpolate(self.global_avg_pool(x), size=self.shape, align_corners=False,
                           mode='bilinear')
        x1 = self.conv1_1x1(x1)
        x2 = self.single_conv_block1_1x1(x)
        x3 = self.single_conv_block1_3x3(x)
        x4 = self.single_conv_block2_3x3(x)
        x5 = self.single_conv_block3_3x3(x)
        x_cat = torch.cat((x2, x3, x4, x5, x1), 1)
        return x_cat


class EncodingBranch(nn.Module):
    """
    Encoding branch for a single radar view

    PARAMETERS
    ----------
    signal_type: str
        Type of radar view.
        Supported: 'range_doppler', 'range_angle' and 'angle_doppler'
    """

    def __init__(self, signal_type, n_frames, in_ch):
        super().__init__()
        self.signal_type = signal_type
        self.n_frames = n_frames
        self.in_ch = in_ch
        self.double_3dconv_block1 = Double3DConvBlock(in_ch=self.in_ch, out_ch=32, 
                                                      k_size=(1+n_frames//2, 3, 3),
                                                      pad=(0, 1, 1), dil=1)
        self.doppler_max_pool = nn.MaxPool2d(2, stride=(1, 2))
        self.max_pool = nn.MaxPool2d(2, stride=2)
        # self.double_conv_block2 = DoubleConvBlock(in_ch=128, out_ch=128, k_size=3,
        #                                           pad=1, dil=1)

        self.double_pkc_block = DoublePeakConv2D(in_ch=32, out_ch=64,
                                                 bias=True,
                                                 rb=1,
                                                 gb_dop=1,
                                                 gb=1,
                                                 sig=self.signal_type)
        self.single_conv_block1_1x1 = ConvBlock(in_ch=64, out_ch=128, k_size=1,
                                                pad=0, dil=1)

    def forward(self, x):
        x1 = self.double_3dconv_block1(x)
        x1 = torch.squeeze(x1, 2)  # remove temporal dimension

        x1_pad = F.pad(x1, (0, 0, 0, 0), "constant", 0)
        x1_down = self.max_pool(x1_pad)

        x2 = self.double_pkc_block(x1_down)
        x2_pad = F.pad(x2, (0, 0, 0, 1), "constant", 0)
        x2_down = self.doppler_max_pool(x2_pad)

        x3 = self.single_conv_block1_1x1(x2_down)
        # return input of ASPP block + latent features
        return x2_down, x3


class PKCIn4ANY_cuda(nn.Module):
    """
    PKCIn-Net w/ ReDA-PeakConv for KuRALS-PDR data

    PARAMETERS
    ----------
    n_classes: int
        Number of classes used for the semantic segmentation task
    n_frames: int
        Total numer of frames used as a sequence
    """

    def __init__(self, n_classes, n_frames, in_ch, w, h):
        super().__init__()
        self.n_classes = n_classes
        self.n_frames = n_frames
        self.in_ch = in_ch
        self.rd_shape = (int(w//2), int(h//4))

        # Backbone (encoding)
        self.rd_encoding_branch = EncodingBranch('range_doppler', self.n_frames, self.in_ch)

        # ASPP Blocks
        self.rd_aspp_block = ASPPBlock(in_ch=64, out_ch=128, shape=self.rd_shape)
        self.rd_single_conv_block1_1x1 = ConvBlock(in_ch=640, out_ch=128, k_size=1, pad=0, dil=1)

        # Decoding
        # self.rd_single_conv_block2_1x1 = ConvBlock(in_ch=128, out_ch=128, k_size=1, pad=0, dil=1)

        # Pallel range-Doppler (RD) and range-angle (RA) decoding branches
        self.rd_upconv1 = nn.ConvTranspose2d(256, 128, (2, 2), stride=(2, 2))
        self.rd_double_conv_block1 = DoubleConvBlock(in_ch=128, out_ch=128, k_size=3,
                                                     pad=1, dil=1)
        self.rd_upconv2 = nn.ConvTranspose2d(128, 128, (1, 2), stride=(1, 2))
        self.rd_double_conv_block2 = DoubleConvBlock(in_ch=128, out_ch=128, k_size=3,
                                                     pad=1, dil=1)

        # spatio information
        # shape of (1, 2, h, w)
        # self.pc = torch.stack(torch.meshgrid(torch.arange(0, 124, 1), torch.arange(0, 2048, 1)), dim=0).unsqueeze_(0)
        
        # Final 1D convs
        self.rd_final = nn.Conv2d(in_channels=128, out_channels=n_classes, kernel_size=1)

    def forward(self, x_rd):
        # Backbone
        rd_features, rd_latent = self.rd_encoding_branch(x_rd)

        # ASPP blocks
        x2_rd = self.rd_aspp_block(rd_features)
        x2_rd = self.rd_single_conv_block1_1x1(x2_rd)

        # Latent Space
        # Features join either the RD or the RA branch
        # x3 = rd_latent
        # x3_rd = self.rd_single_conv_block2_1x1(x3)

        # Latent Space + ASPP features
        x4_rd = torch.cat((x2_rd, rd_latent), 1)

        # Parallel decoding branches with upconvs
        x4_rd = self.rd_upconv1(x4_rd)
        x4_rd = self.rd_double_conv_block1(x4_rd)

        x4_rd = self.rd_upconv2(x4_rd)
        x4_rd = self.rd_double_conv_block2(x4_rd)

        # adding spatio info
        # x8_rd = torch.cat([self.pc.to(x8_rd.device).expand(x8_rd.shape[0], -1, -1, -1), x8_rd], dim=1)
        
        # Final 1D convolutions
        x4_rd = self.rd_final(x4_rd)

        return x4_rd