#### zyc@231207 ####
#### TFoc + vanilla + maskdraw ####
#### ASPP: using TFoc output as input
#### LSE: using latent features(TFoc + 1x1Conv) as input
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torch.nn.functional as F
import math
import os

os.environ['CUDA_VISIBLE_DEVICES'] = "0,1"



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
            nn.LeakyReLU(inplace=True),
            nn.Dropout2d(p=0.1)
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
            nn.LeakyReLU(inplace=True),
            nn.Dropout2d(p=0.1)
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
            nn.LeakyReLU(inplace=True),
            nn.Dropout3d(p=0.1))

    def forward(self, x):
        x = self.block(x)
        return x

class TFocBlock(nn.Module):
    def __init__(self, in_ch, out_ch, k_size, strd, pad):
        super().__init__()
        self.layer1 = TREBlock(in_ch, out_ch, k_size=(k_size, 1, 1), strd=(strd, strd, strd), pad=(0, 0, 0))
        self.layer2 = TREBlock(out_ch, out_ch, k_size=(1, k_size, k_size), strd=(strd, strd, strd), pad=(0, pad, pad))
        #self.layer3 = TREBlock(out_ch, out_ch, k_size=(k_size, 1, 1), strd=(strd, strd, strd), pad=(0, 0, 0))

    def forward(self, x):
        #b, c, t, h, w = x.size(dim=0), x.size(dim=1), x.size(dim=2), x.size(dim=3), x.size(dim=4)
        x1 = self.layer1(x) #(b,c,t-4,h,w)
        x = self.layer2(x1) + x1 #(b,c,t-4,h,w)
        # if self.layer3 is None:
        #     self.layer3 = torch.nn.Linear(t*h*w, t*h*w, True).to(x.device)
        #     self.register_module('Tfoc_fc', self.layer3)
        # x1 = self.layer3(x.reshape(b*c, -1)).reshape(b, c, t, h, w) ##(b*c, t-4 * h*w) -> (b, c, t, h, w)
        return x

class Double3DConvBlock(nn.Module):
    """ (3D conv => BN => LeakyReLU) * 2 (temporal shared 2D Conv)"""

    def __init__(self, in_ch, out_ch, k_size, strd, pad, btnk_sc):
        super().__init__()

        self.block = nn.Sequential(
            nn.Conv3d(in_ch, out_ch*btnk_sc, kernel_size=k_size, stride=strd, padding=pad),
            nn.BatchNorm3d(out_ch*btnk_sc),
            nn.LeakyReLU(inplace=True),
            nn.Conv3d(out_ch*btnk_sc, out_ch, kernel_size=k_size, stride=strd, padding=pad),
            nn.BatchNorm3d(out_ch),
            nn.LeakyReLU(inplace=True),
            nn.Dropout3d(p=0.1)
        )

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


class TemporalGroupedDilatedConv(nn.Module):
    def __init__(self, in_channels: int, t_len: int, kernel_size: int = 3):
        """
        初始化t维度分组空洞卷积
        :param in_channels: 输入通道数（对应x的c维度）
        :param t_len: t维度的长度（即分组数）
        :param kernel_size: 前t-1个卷积的核大小（最后一个固定为1）
        """
        super().__init__()
        self.t_len = t_len
        self.in_channels = in_channels

        # 为每个t维度单独创建一个卷积层（支持不同空洞率）
        self.conv_layers = nn.ModuleList()
        for i in range(t_len):
            # 1. 计算当前t的空洞率：从大到小递减
            dilation = t_len - 1 - i  # i=0时最大，i=t_len-1时为1

            # 2. 最后一个t维度：固定kernel=1，dilation=1，保证感受野=1
            if i == t_len - 1:
                current_kernel = 1
                current_dilation = 1
            else:
                current_kernel = kernel_size
                current_dilation = dilation

            # 3. 计算padding，保证h/w维度卷积后尺寸不变
            padding = current_dilation * (current_kernel - 1) // 2

            # 4. 创建卷积层（输入输出通道数一致，保证c维度不变）
            self.conv_layers.append(
                nn.Conv2d(
                    in_channels=in_channels,
                    out_channels=in_channels,
                    kernel_size=current_kernel,
                    dilation=current_dilation,
                    padding=padding,
                    bias=False  # 可选，根据需求调整
                )
            )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        前向传播
        :param x: 输入张量，形状 (b, c, t, h, w)
        :return: 输出张量，形状与输入一致 (b, c, t, h, w)
        """
        b, c, t, h, w = x.shape
        assert t == self.t_len, f"输入t维度长度({t})与初始化t_len({self.t_len})不匹配"

        # 存储每个t维度的卷积结果
        output_list = []
        for i in range(t):
            # 取出第i个t维度的特征：形状 (b, c, h, w)
            x_t = x[:, :, i, :, :]
            # 对该t维度做卷积
            out_t = self.conv_layers[i](x_t)
            # 恢复t维度：形状 (b, c, 1, h, w)
            out_t = out_t.unsqueeze(2)
            output_list.append(out_t)

        # 拼接所有t维度：形状 (b, c, t, h, w)
        return torch.cat(output_list, dim=2)


class TFocConv(nn.Module):
    """Temporal Focusing Conv Block w/ single historic frame
    input: x_seq = Conv3D(x) BxCx2xHxW tensor
    output: x_tfc BxCxHxW tensor
    """

    def __init__(self, in_ch, out_ch, btnk_ch, search_size):
        super(TFocConv, self).__init__()
        self.btnk_ch = btnk_ch
        self.k_n = search_size
        self.pad_size = math.floor(search_size / 2)
        self.spatial_conv = Double3DConvBlock(in_ch,
                                              self.btnk_ch,
                                              k_size=(1,3,3),
                                              strd=(1,1,1),
                                              pad=(0,1,1),
                                              btnk_sc=16)
        self.tempdila_conv = TemporalGroupedDilatedConv(in_channels=self.btnk_ch, t_len=5, kernel_size=3)  # 前t-1个卷积的核大小

        self.inner_predict = TREBlock(self.btnk_ch,
                                      4,
                                      k_size=(1, 1, 1),
                                      strd=(1, 1, 1),
                                      pad=(0, 0, 0))
        self.softmax = nn.Softmax(dim=1)

        self.tfoc_conv = TFocBlock(self.btnk_ch, out_ch, k_size=3, strd=1, pad=1)
        self.temporal_conv = TREBlock(out_ch,
                                      out_ch,
                                      k_size=(3, 1, 1),
                                      strd=(1, 1, 1),
                                      pad=(0, 0, 0))

    def forward(self, x_seq): #x_mask(b,c,t,h,w)
        x_seq = self.spatial_conv(x_seq)

        # 1 temporal dilation convolution
        x_seq = self.tempdila_conv(x_seq) #b,c,t,h,w,

        # 2 trac_loss
        x_seq1 = self.inner_predict(x_seq)  # (b,4,t,h,w)
        x_seq1 = self.softmax(x_seq1)  # (b,4,t,h,w)
        probs = torch.prod(x_seq1, dim=2)  # (b,4,h,w)

        x_tfoc = self.tfoc_conv(x_seq)  # (b,c,t-2,h,w)
        x_tfoc = self.temporal_conv(x_tfoc)# (b,c,t-4,h,w)
        return x_tfoc, probs



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

    def __init__(self, signal_type, n_frames, search_size):
        super().__init__()
        self.n_frames = n_frames
        self.signal_type = signal_type
        self.search_size = search_size

        self.double_3d_conv_block = Double3DConvBlock(in_ch=1,
                                                      out_ch=64,
                                                      k_size=(3,3,3),
                                                      strd=(1,1,1),
                                                      pad=(0,1,1),
                                                      btnk_sc=1)
        self.tfoc_conv_block = TFocConv(in_ch=64,
                                        out_ch=128,
                                        btnk_ch=8, #out_ch/btnk_sc
                                        search_size=self.search_size)
        self.doppler_max_pool = nn.MaxPool2d(2, stride=(2, 1))
        self.max_pool = nn.MaxPool2d(2, stride=2)
        self.single_conv_block1_3x3 = ConvBlock(in_ch=128, out_ch=128, k_size=3, pad=1, dil=1)
        self.single_conv_block2_3x3 = ConvBlock(in_ch=128, out_ch=128, k_size=3, pad=1, dil=1)
        self.single_conv_block1_1x1 = ConvBlock(in_ch=128, out_ch=128, k_size=1, pad=0, dil=1)
        self.single_conv_block1_3x3_prb = ConvBlock(in_ch=4, out_ch=32, k_size=3, pad=1, dil=1)
        self.single_conv_block2_3x3_prb = ConvBlock(in_ch=32, out_ch=32, k_size=3, pad=1, dil=1)
        self.single_conv_block1_1x1_prb = ConvBlock(in_ch=32, out_ch=32, k_size=1, pad=0, dil=1)

    def forward(self, x):
        x0 = self.double_3d_conv_block(x)  # [b,128,(n_frames-3+1-3+1),h,w]
        x1, prb1 = self.tfoc_conv_block(x0)  # x1:[b,128,1,h,w] prb1:[b,4,h,w]
        x1 = x1.squeeze(dim=2)
        if self.signal_type in ('range_doppler', 'angle_doppler'):
            # The Doppler dimension requires a specific processing
            # padding the last two dimentions of the tensor
            x1_pad = F.pad(x1, (0, 1, 0, 0), "constant", 0)  # F.pad(top, bottom, left, right)
            x1_down = self.doppler_max_pool(x1_pad)
            # zyc1114
            prb1_pad = F.pad(prb1, (0, 1, 0, 0), "constant", 0)  # F.pad(top, bottom, left, right)
            prb1_down = self.doppler_max_pool(prb1_pad)
            ##
        else:
            # zyc1114
            x1_down = self.max_pool(x1)  # [2,128,128,128]
            prb1_down = self.max_pool(prb1)

        x2 = self.single_conv_block1_3x3(x1_down)  # [2,128,128,128]
        x2 = self.single_conv_block2_3x3(x2)  # [2,128,128,128]
        # zyc1114
        prb2 = self.single_conv_block1_3x3_prb(prb1_down)  # [2,16,128,128]
        prb2 = self.single_conv_block2_3x3_prb(prb2)  # [2,16,128,128]

        if self.signal_type in ('range_doppler', 'angle_doppler'):
            # The Doppler dimension requires a specific processing
            x2_pad = F.pad(x2, (0, 1, 0, 0), "constant", 0)
            x2_down = self.doppler_max_pool(x2_pad)
            # zyc1114
            prb2_pad = F.pad(prb2, (0, 1, 0, 0), "constant", 0)
            prb2_down = self.doppler_max_pool(prb2_pad)
            ##
        else:
            x2_down = self.max_pool(x2)  # [2,128,64,64]
            # zyc
            prb2_down = self.max_pool(prb2)  # [2,128,64,64]
            ##

        x3 = self.single_conv_block1_1x1(x2_down)  # [2,128,64,64]
        prb3 = self.single_conv_block1_1x1_prb(prb2_down)  # [2,128,64,64]
        # return input of TAM (x2_down) & ASPP block (x3) & Latent space (x3)
        return x2_down, x3, prb2_down, prb3


class TFocNet(nn.Module):
    """
    Temporal Multi-View with ASPP Network (TMVA-Net)

    PARAMETERS
    ----------
    n_classes: int
        Number of classes used for the semantic segmentation task
    n_frames: int
        Total numer of frames used as a sequence
    """

    def __init__(self, n_classes, n_frames, search_size, zero_doppler_filter=True):
        super().__init__()
        self.n_classes = n_classes
        self.n_frames = n_frames
        self.zero_doppler_filter = zero_doppler_filter
        self.search_size_rd = search_size[0]
        self.search_size_ra = search_size[1]


        # Backbone (encoding)
        self.rd_encoding_branch = EncodingBranch('range_doppler', self.n_frames, self.search_size_rd)
        self.ra_encoding_branch = EncodingBranch('range_angle', self.n_frames, self.search_size_ra)
        self.ad_encoding_branch = EncodingBranch('angle_doppler', self.n_frames, self.search_size_rd)

        # ASPP Blocks
        self.rd_aspp_block = ASPPBlock(in_ch=128, out_ch=128)
        self.ra_aspp_block = ASPPBlock(in_ch=128, out_ch=128)
        self.ad_aspp_block = ASPPBlock(in_ch=128, out_ch=128)
        self.rd_single_conv_block1_1x1 = ConvBlock(in_ch=640, out_ch=128, k_size=1, pad=0, dil=1)
        self.ra_single_conv_block1_1x1 = ConvBlock(in_ch=640, out_ch=128, k_size=1, pad=0, dil=1)
        self.ad_single_conv_block1_1x1 = ConvBlock(in_ch=640, out_ch=128, k_size=1, pad=0, dil=1)

        # Decoding
        self.rd_single_conv_block2_1x1 = ConvBlock(in_ch=384, out_ch=128, k_size=1, pad=0, dil=1)
        self.ra_single_conv_block2_1x1 = ConvBlock(in_ch=384, out_ch=128, k_size=1, pad=0, dil=1)

        # Pallel range-Doppler (RD) and range-angle (RA) decoding branches
        self.rd_upconv1 = nn.ConvTranspose2d(384, 128, (2, 1), stride=(2, 1))
        self.ra_upconv1 = nn.ConvTranspose2d(384, 128, 2, stride=2)
        self.rd_double_conv_block1 = DoubleConvBlock(in_ch=128, out_ch=128, k_size=3,
                                                     pad=1, dil=1)
        self.ra_double_conv_block1 = DoubleConvBlock(in_ch=128, out_ch=128, k_size=3,
                                                     pad=1, dil=1)
        self.rd_upconv2 = nn.ConvTranspose2d(128, 128, (2, 1), stride=(2, 1))
        self.ra_upconv2 = nn.ConvTranspose2d(128, 128, 2, stride=2)
        self.rd_double_conv_block2 = DoubleConvBlock(in_ch=128, out_ch=128, k_size=3,
                                                     pad=1, dil=1)
        self.ra_double_conv_block2 = DoubleConvBlock(in_ch=128, out_ch=128, k_size=3,
                                                     pad=1, dil=1)

        # Final 1D convs
        self.rd_final = nn.Conv2d(in_channels=128, out_channels=n_classes, kernel_size=1)
        self.ra_final = nn.Conv2d(in_channels=128, out_channels=n_classes, kernel_size=1)

        #zyc@20231207

        # self.rd_single_conv_block1_1x1_prb = ConvBlock(in_ch=32, out_ch=32, k_size=1, pad=0, dil=1)
        # self.ra_single_conv_block1_1x1_prb = ConvBlock(in_ch=32, out_ch=32, k_size=1, pad=0, dil=1)
        # self.ad_single_conv_block1_1x1_prb = ConvBlock(in_ch=32, out_ch=32, k_size=1, pad=0, dil=1)

        # Decoding
        self.rd_single_conv_block2_1x1_prb = ConvBlock(in_ch=32 * 3, out_ch=32, k_size=1, pad=0, dil=1)
        self.ra_single_conv_block2_1x1_prb = ConvBlock(in_ch=32 * 3, out_ch=32, k_size=1, pad=0, dil=1)

        # Pallel range-Doppler (RD) and range-angle (RA) decoding branches
        self.rd_upconv1_prb = nn.ConvTranspose2d(32 * 3, 32, (2, 1), stride=(2, 1))
        self.rd_upconv2_prb = nn.ConvTranspose2d(32, 32, (2, 1), stride=(2, 1))

        self.ra_upconv1_prb = nn.ConvTranspose2d(32 * 3, 32, 2, stride=2)
        self.ra_upconv2_prb = nn.ConvTranspose2d(32, 32, 2, stride=2)

        self.rd_double_conv_block1_prb = DoubleConvBlock(in_ch=32, out_ch=32, k_size=3,
                                                         pad=1, dil=1)
        self.ra_double_conv_block1_prb = DoubleConvBlock(in_ch=32, out_ch=32, k_size=3,
                                                         pad=1, dil=1)

        self.rd_double_conv_block2_prb = DoubleConvBlock(in_ch=32, out_ch=32, k_size=3,
                                                         pad=1, dil=1)
        self.ra_double_conv_block2_prb = DoubleConvBlock(in_ch=32, out_ch=32, k_size=3,
                                                         pad=1, dil=1)

        # Final 1D convs
        self.rd_final_prb = nn.Conv2d(in_channels=32, out_channels=n_classes, kernel_size=1, padding=0)
        self.ra_final_prb = nn.Conv2d(in_channels=32, out_channels=n_classes, kernel_size=1, padding=0)

    def forward(self, x_rd, x_ra, x_ad):
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
        ra_features, ra_latent, ra_prb, ra_prb_latent = self.ra_encoding_branch(x_ra)
        if self.zero_doppler_filter:
            x_rd = self._zero_doppler_filter(x_rd)
            x_ad = self._zero_doppler_filter(x_ad)
        rd_features, rd_latent, rd_prb, rd_prb_latent = self.rd_encoding_branch(x_rd)
        ad_features, ad_latent, ad_prb, ad_prb_latent = self.ad_encoding_branch(x_ad)

        # ASPP blocks
        x1_rd = self.rd_aspp_block(rd_features)
        x1_ra = self.ra_aspp_block(ra_features)
        x1_ad = self.ad_aspp_block(ad_features)
        x2_rd = self.rd_single_conv_block1_1x1(x1_rd)
        x2_ra = self.ra_single_conv_block1_1x1(x1_ra)
        x2_ad = self.ad_single_conv_block1_1x1(x1_ad)

        # Latent Space
        # Features join either the RD or the RA branch
        x3 = torch.cat((rd_latent, ra_latent, ad_latent), 1)
        x3_rd = self.rd_single_conv_block2_1x1(x3)
        x3_ra = self.ra_single_conv_block2_1x1(x3)

        # Latent Space + ASPP features
        x4_rd = torch.cat((x2_rd, x3_rd, x2_ad), 1)
        x4_ra = torch.cat((x2_ra, x3_ra, x2_ad), 1)

        # Parallel decoding branches with upconvs
        x5_rd = self.rd_upconv1(x4_rd)
        x5_ra = self.ra_upconv1(x4_ra)
        x6_rd = self.rd_double_conv_block1(x5_rd)
        x6_ra = self.ra_double_conv_block1(x5_ra)

        x7_rd = self.rd_upconv2(x6_rd)
        x7_ra = self.ra_upconv2(x6_ra)
        x8_rd = self.rd_double_conv_block2(x7_rd)
        x8_ra = self.ra_double_conv_block2(x7_ra)

        # Final 1D convolutions
        x9_rd = self.rd_final(x8_rd)
        x9_ra = self.ra_final(x8_ra)

        #zyc@20231207 ra_prb, ra_prb_latent
        # x2_rd_prb = self.rd_single_conv_block1_1x1_prb(rd_prb)
        # x2_ra_prb = self.ra_single_conv_block1_1x1_prb(ra_prb)
        # x2_ad_prb = self.ad_single_conv_block1_1x1_prb(ad_prb)

        # Latent Space Embedding
        # Features join either the RD or the RA branch
        prb_fusion = torch.cat((rd_prb_latent, ra_prb_latent, ad_prb_latent), 1)

        rd_prb_fusion = self.rd_single_conv_block2_1x1_prb(prb_fusion)
        ra_prb_fusion = self.ra_single_conv_block2_1x1_prb(prb_fusion)

        # Latent Space + ASPP features
        x4_rd_prb = torch.cat((rd_prb, rd_prb_fusion, ad_prb), 1)
        x4_ra_prb = torch.cat((ra_prb, ra_prb_fusion, ad_prb), 1)

        # Parallel decoding branches with upconvs
        x5_rd_prb = self.rd_upconv1_prb(x4_rd_prb)
        x5_ra_prb = self.ra_upconv1_prb(x4_ra_prb)
        x6_rd_prb = self.rd_double_conv_block1_prb(x5_rd_prb)
        x6_ra_prb = self.ra_double_conv_block1_prb(x5_ra_prb)

        x7_rd_prb = self.rd_upconv2_prb(x6_rd_prb)
        x7_ra_prb = self.ra_upconv2_prb(x6_ra_prb)
        x8_rd_prb = self.rd_double_conv_block2_prb(x7_rd_prb)
        x8_ra_prb = self.ra_double_conv_block2_prb(x7_ra_prb)

        # Final 1D convolutions
        x9_rd_prb = self.rd_final_prb(x8_rd_prb)
        x9_ra_prb = self.ra_final_prb(x8_ra_prb)

        return x9_rd, x9_ra, x9_rd_prb, x9_ra_prb

    def _zero_doppler_filter(self, rd):
        zero_doppler_mask = torch.cat([torch.normal(1, 1e-3, size=(256, 31)),
                                       torch.normal(1e-8, 1e-3, size=(256, 1)),
                                       torch.normal(1, 1e-3, size=(256, 32))], dim=-1).cuda()
        rd = rd * zero_doppler_mask.unsqueeze(dim=0).unsqueeze(dim=0)
        return rd

