#### zlw@20230818 ####
#### Single View Baseline Model for Carrada ####
# import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torch.nn.functional as F



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


class TREBlock(nn.Module):
    """ (3D conv => BN => LeakyReLU) """
    """ Temporal Relation Embeddings """

    def __init__(self, in_ch, out_ch, k_size, strd, pad):
        super().__init__()
        self.block = nn.Sequential(
            nn.Conv3d(in_ch, out_ch, kernel_size=k_size, stride=strd, padding=pad),
            nn.BatchNorm3d(out_ch),
            nn.LeakyReLU(inplace=True))

    def forward(self, x):
        x = self.block(x)
        return x


class ASPPBlock(nn.Module):
    """Atrous Spatial Pyramid Pooling
    Parallel conv blocks with different dilation rate
    """

    def __init__(self, in_ch, out_ch=256):
        super().__init__()
        self.global_avg_pool = nn.AvgPool2d((64, 64))
        self.conv1_1x1 = nn.Conv2d(in_ch, out_ch, kernel_size=1, padding=0, dilation=1)
        self.single_conv_block1_1x1 = ConvBlock(in_ch, out_ch, k_size=1, pad=0, dil=1)
        self.single_conv_block1_3x3 = ConvBlock(in_ch, out_ch, k_size=3, pad=6, dil=6)
        self.single_conv_block2_3x3 = ConvBlock(in_ch, out_ch, k_size=3, pad=12, dil=12)
        self.single_conv_block3_3x3 = ConvBlock(in_ch, out_ch, k_size=3, pad=18, dil=18)

    def forward(self, x):
        x1 = F.interpolate(self.global_avg_pool(x), size=(64, 64), align_corners=False,
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
    Backbone Encoding branch for a single radar view

    PARAMETERS
    ----------
    signal_type: str
        Type of radar view.
        Supported: 'range_doppler', 'range_angle' and 'angle_doppler'
        Tensor shape: time x range(angle) x doppler(angle)
    """

    def __init__(self, signal_type, n_frames):
        super().__init__()
        self.n_frames = n_frames
        self.signal_type = signal_type
        self.single_3d_conv_block_1 = TREBlock(in_ch=1, out_ch=128,
                                               k_size=(3, 3, 3),
                                               strd=(1, 1, 1),
                                               pad=(0, 1, 1))
        self.single_3d_conv_block_2 = TREBlock(in_ch=128, out_ch=128,
                                               k_size=(3, 3, 3),
                                               strd=(1, 1, 1),
                                               pad=(0, 1, 1))
        self.doppler_max_pool3d = nn.MaxPool3d(kernel_size=(1, 2, 2),
                                               stride=(1, 2, 1))
        self.max_pool3d = nn.MaxPool3d(kernel_size=(1, 2, 2),
                                       stride=(1, 2, 2))
        self.doppler_max_pool = nn.MaxPool2d(2, stride=(2, 1))
        self.max_pool = nn.MaxPool2d(2, stride=2)
        self.single_conv_block1_3x3 = ConvBlock(in_ch=128, out_ch=128, k_size=3, pad=1, dil=1)
        self.single_conv_block2_3x3 = ConvBlock(in_ch=128, out_ch=128, k_size=3, pad=1, dil=1)
        self.single_conv_block1_1x1 = ConvBlock(in_ch=128, out_ch=128, k_size=1, pad=0, dil=1)
        self.single_2d_conv_block_1 = TREBlock(in_ch=128, out_ch=128,
                                               k_size=(1, 3, 3),
                                               strd=(1, 1, 1),
                                               pad=(0, 1, 1))
        self.single_2d_conv_block_2 = TREBlock(in_ch=128, out_ch=128,
                                               k_size=(1, 3, 3),
                                               strd=(1, 1, 1),
                                               pad=(0, 1, 1))

    def forward(self, x):
        x0 = self.single_3d_conv_block_1(x)
        x0 = self.single_3d_conv_block_2(x0)  # [b,16,(n_frames-3+1-3+1),h,w]


        x1 = self.single_2d_conv_block_1(x0)  # [b,128,1,h,w]
        x1 = self.single_2d_conv_block_2(x1)

        if self.signal_type in ('range_doppler', 'angle_doppler'):
            # The Doppler dimension requires a specific processing
            # padding the last two dimentions of the tensor
            x1_pad = F.pad(x1, (0, 1, 0, 0), "constant", 0)  # F.pad(top, bottom, left, right)
            x1_down = self.doppler_max_pool3d(x1_pad)
        else:
            x1_down = self.max_pool3d(x1)  # [2,128,128,128]
        x1_down = x1_down.squeeze(dim=2)  # zlw@20230816

        x2 = self.single_conv_block1_3x3(x1_down)  # [2,128,128,128]
        x2 = self.single_conv_block2_3x3(x2)  # [2,128,128,128]

        if self.signal_type in ('range_doppler', 'angle_doppler'):
            # The Doppler dimension requires a specific processing
            x2_pad = F.pad(x2, (0, 1, 0, 0), "constant", 0)
            x2_down = self.doppler_max_pool(x2_pad)
        else:
            x2_down = self.max_pool(x2)  # [2,128,64,64]

        x3 = self.single_conv_block1_1x1(x2_down)  # [2,128,64,64]
        # return input of TAM (x2_down) & ASPP block (x3) & Latent space (x3)
        return x2_down, x3


def _zero_doppler_filter(rd):
    zero_doppler_mask = torch.cat([torch.normal(1, 1e-3, size=(256, 31)),
                                   torch.normal(1e-8, 1e-3, size=(256, 1)),
                                   torch.normal(1, 1e-3, size=(256, 32))], dim=-1).cuda()
    rd = rd * zero_doppler_mask.unsqueeze(dim=0).unsqueeze(dim=0)
    return rd


class BaseLine3D(nn.Module):
    """
    Single View Baseline Model w/ 3D Conv for Carrada.

    PARAMETERS
    ----------
    n_classes: int, required
        Number of classes used for the semantic segmentation task
    n_frames: int, required
        Total numer of frames used as a sequence
    signal_type: str, required
        range_doppler or range_angle
    zero_doppler: bool
        whether filtering the doppler band w/ zero Freq
    """

    def __init__(self, n_classes, n_frames, signal_type, zero_doppler_filter=False):
        super().__init__()
        self.n_classes = n_classes
        self.n_frames = n_frames
        self.view_type = signal_type
        self.zero_doppler_filter = zero_doppler_filter

        # Backbone (encoding)
        self.encoding_branch = EncodingBranch(self.view_type, self.n_frames)

        # ASPP Blocks
        self.aspp_block = ASPPBlock(in_ch=128, out_ch=128)
        self.single_conv_block1_1x1 = ConvBlock(in_ch=640, out_ch=128, k_size=1, pad=0, dil=1)

        # Decoding
        self.single_conv_block2_1x1 = ConvBlock(in_ch=128, out_ch=128, k_size=1, pad=0, dil=1)

        # decoding branch
        if self.view_type in 'range_doppler':
            self.upconv1 = nn.ConvTranspose2d(256, 128, (2, 1), stride=(2, 1))
            self.upconv2 = nn.ConvTranspose2d(128, 128, (2, 1), stride=(2, 1))
        else:
            self.upconv1 = nn.ConvTranspose2d(256, 128, 2, stride=2)
            self.upconv2 = nn.ConvTranspose2d(128, 128, 2, stride=2)
        self.double_conv_block1 = DoubleConvBlock(in_ch=128, out_ch=128, k_size=3,
                                                  pad=1, dil=1)
        self.double_conv_block2 = DoubleConvBlock(in_ch=128, out_ch=128, k_size=3,
                                                  pad=1, dil=1)

        # Final 1D convs
        self.final = nn.Conv2d(in_channels=128, out_channels=n_classes, kernel_size=1)

    def forward(self, x):
        # Backbone
        # x_rd = self._zero_doppler_filter(x_rd)
        # x_ad = self._zero_doppler_filter(x_ad)
        # plt.figure()
        # plt.subplot(2, 1, 1)
        # plt.pcolor(x_ad[0, 0, 0, ...].cpu().numpy(), label='x_ad')
        # plt.legend()
        # plt.subplot(2, 1, 2)
        # plt.pcolor(x_rd[0, 0, 0, ...].cpu().numpy(), label='x_rd')
        # plt.legend()
        # plt.show()
        if self.zero_doppler_filter & bool(self.view_type in 'range_doppler'):
            x = _zero_doppler_filter(x)
        features, latent = self.encoding_branch(x)

        # ASPP blocks
        x1 = self.aspp_block(features)
        x2 = self.single_conv_block1_1x1(x1)

        # Latent Space Embedding
        # Features join either the RD or the RA branch
        x3 = self.single_conv_block2_1x1(latent)

        # Latent Space + ASPP features
        x4 = torch.cat((x2, x3), 1)

        # Decoding branch with upconvs
        x5 = self.upconv1(x4)
        x6 = self.double_conv_block1(x5)

        x7 = self.upconv2(x6)
        x8 = self.double_conv_block2(x7)

        # Final 1D convolutions
        x9 = self.final(x8)

        return x9
