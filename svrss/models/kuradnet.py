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
            nn.ReLU(inplace=True),
            nn.Conv2d(out_ch, out_ch, kernel_size=k_size, padding=pad, dilation=dil),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True)
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
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        x = self.block(x)
        return x

class EncodingBranch(nn.Module):
    """
    Encoding branch for a single radar view

    PARAMETERS
    ----------
    signal_type: str
        Type of radar view.
        Supported: 'range_doppler', 'range_angle' and 'angle_doppler'
    """

    def __init__(self, signal_type):
        super().__init__()
        self.signal_type = signal_type
        self.doubleconv_block1 = DoubleConvBlock(in_ch=1, out_ch=64, k_size=3, pad=1, dil=1)
        self.doppler_max_pool = nn.MaxPool2d(2, stride=(1, 2))

        self.doubleconv_block2 = DoubleConvBlock(in_ch=64, out_ch=128, k_size=3, pad=1, dil=1)
        
        self.single_conv_block1_1x1 = ConvBlock(in_ch=128, out_ch=256, k_size=3, pad=1, dil=1)

    def forward(self, x):
        x1 = self.doubleconv_block1(x)

        x1_pad = F.pad(x1, (0, 0, 0, 1), "constant", 0)
        x1_down = self.doppler_max_pool(x1_pad)

        x2 = self.doubleconv_block2(x1_down)
        x2_pad = F.pad(x2, (0, 0, 0, 1), "constant", 0)
        x2_down = self.doppler_max_pool(x2_pad)

        x3 = self.single_conv_block1_1x1(x2_down)

        return x3


class KuRADNet(nn.Module):
    """
    Baseline model for KuRALS-PDR dataset.

    PARAMETERS
    ----------
    n_classes: int
        Number of classes used for the semantic segmentation task
    n_frames: int
        Total numer of frames used as a sequence
    """

    def __init__(self, n_classes, n_frames):
        super().__init__()
        self.n_classes = n_classes
        self.n_frames = n_frames

        # encoding
        self.doubleconv_block1 = DoubleConvBlock(in_ch=1, out_ch=64, k_size=3, pad=1, dil=1)
        self.doppler_max_pool = nn.MaxPool2d(2, stride=(1, 2))

        self.doubleconv_block2 = DoubleConvBlock(in_ch=64, out_ch=128, k_size=3, pad=1, dil=1)
        
        self.single_conv_block1_1x1 = ConvBlock(in_ch=128, out_ch=256, k_size=3, pad=1, dil=1)

        # decoding
        self.rd_upconv1 = nn.ConvTranspose2d(256, 128, (1, 2), stride=(1, 2))
        self.rd_double_conv_block1 = DoubleConvBlock(in_ch=128, out_ch=128, k_size=3,
                                                     pad=1, dil=1)
        self.rd_upconv2 = nn.ConvTranspose2d(128, 64, (1, 2), stride=(1, 2))
        self.rd_double_conv_block2 = DoubleConvBlock(in_ch=64, out_ch=64, k_size=3,
                                                     pad=1, dil=1)
        
        # Final 1D convs
        self.rd_final = nn.Conv2d(in_channels=64, out_channels=n_classes, kernel_size=1)

    def forward(self, x):
        # encoding
        x1 = self.doubleconv_block1(x)

        x1_pad = F.pad(x1, (0, 0, 0, 1), "constant", 0)
        x1_down = self.doppler_max_pool(x1_pad)

        x2 = self.doubleconv_block2(x1_down)
        x2_pad = F.pad(x2, (0, 0, 0, 1), "constant", 0)
        x2_down = self.doppler_max_pool(x2_pad)

        x3 = self.single_conv_block1_1x1(x2_down)

        # decoding
        x4_rd = self.rd_upconv1(x3)
        x4_rd = self.rd_double_conv_block1(x4_rd)

        x4_rd = self.rd_upconv2(x4_rd)
        x4_rd = self.rd_double_conv_block2(x4_rd)
        
        # Final 1D convolutions
        x4_rd = self.rd_final(x4_rd)

        return x4_rd

class KuRADNet_WSkip(nn.Module):
    """
    Baseline model for KuRALS-PDR dataset.

    PARAMETERS
    ----------
    n_classes: int
        Number of classes used for the semantic segmentation task
    n_frames: int
        Total numer of frames used as a sequence
    """

    def __init__(self, n_classes, n_frames):
        super().__init__()
        self.n_classes = n_classes
        self.n_frames = n_frames

        # encoding
        self.doubleconv_block1 = DoubleConvBlock(in_ch=1, out_ch=64, k_size=3, pad=1, dil=1)
        self.doppler_max_pool = nn.MaxPool2d(2, stride=(1, 2))

        self.doubleconv_block2 = DoubleConvBlock(in_ch=64, out_ch=128, k_size=3, pad=1, dil=1)
        
        self.single_conv_block1_1x1 = ConvBlock(in_ch=128, out_ch=256, k_size=3, pad=1, dil=1)

        # decoding
        self.rd_upconv1 = nn.ConvTranspose2d(256, 128, (1, 2), stride=(1, 2))
        self.rd_double_conv_block1 = DoubleConvBlock(in_ch=256, out_ch=128, k_size=3,
                                                     pad=1, dil=1)
        self.rd_upconv2 = nn.ConvTranspose2d(128, 64, (1, 2), stride=(1, 2))
        self.rd_double_conv_block2 = DoubleConvBlock(in_ch=128, out_ch=64, k_size=3,
                                                     pad=1, dil=1)
        
        # Final 1D convs
        self.rd_final = nn.Conv2d(in_channels=64, out_channels=n_classes, kernel_size=1)

    def forward(self, x):
        # encoding
        x1 = self.doubleconv_block1(x)

        x1_pad = F.pad(x1, (0, 0, 0, 1), "constant", 0)
        x1_down = self.doppler_max_pool(x1_pad)

        x2 = self.doubleconv_block2(x1_down)
        x2_pad = F.pad(x2, (0, 0, 0, 1), "constant", 0)
        x2_down = self.doppler_max_pool(x2_pad)

        x3 = self.single_conv_block1_1x1(x2_down)

        # decoding
        x4 = self.rd_upconv1(x3)
        x4 = torch.concat([x2, x4], dim=1)
        x4 = self.rd_double_conv_block1(x4)

        x4 = self.rd_upconv2(x4)
        x4 = torch.concat([x1, x4], dim=1)
        x4 = self.rd_double_conv_block2(x4)
        
        # Final 1D convolutions
        x4 = self.rd_final(x4)

        return x4