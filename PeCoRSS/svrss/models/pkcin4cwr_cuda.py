import torch
import torch.nn as nn
import torch.nn.functional as F
import prf_div

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

# reda peak_conv zlw @20220624
# Peak Conv with fixed guard banwidth (gb) and dense reference receptive field (RRF)
class PeakConv2D(nn.Module):
    def __init__(self, in_ch, out_ch, bias, refer_band, guard_band_dop, guard_band, signal_type):
        """
        Args:
            in_ch为32;
            out_ch为64;
            refer_band: a single integer value or integer vector of two values.
            guard_band_dop: guard bandwidth for doppler input
            guard_band: guard banwidth for non-doppler input
            signal_type in ('range_doppler', 'angle_doppler', 'range_angle').
            k_size, padding and stride are computed using the input fixed guard bandwidth
            input: x in the shape of BxCxHxW
        """
        super(PeakConv2D, self).__init__()
        self.signal_type = signal_type # 'range_doppler'
        if self.signal_type in ('range_doppler', 'angle_doppler'):
            # guard_band should be bi-lateral TODO: ?????????? 这是什么意思
            self.guard_band = (guard_band_dop, guard_band_dop) # （1，1）
            # padding order: l-r-t-b
            self.padding = (self.guard_band[1] + refer_band,
                            self.guard_band[1] + refer_band,
                            self.guard_band[0] + refer_band,
                            self.guard_band[0] + refer_band) # (2,2,2,2)
        else:
            self.guard_band = (guard_band, guard_band)
            # padding order: l-r-t-b
            self.padding = (self.guard_band[1] + refer_band,
                            self.guard_band[1] + refer_band,
                            self.guard_band[0] + refer_band,
                            self.guard_band[0] + refer_band) # ()
        self.refer_band = refer_band # 1 
        self.kernel_size = (self.guard_band[0] + self.guard_band[1] + self.refer_band * 2), 4 # (1+1+1*2, 4) TODO: 很疑惑，这里是如何计算的？？？？，为什么直接固定为4
        self.stride = self.kernel_size # 步长为kernel_size， 互补干扰好吧， 4
        # padding order: l-r-t-b
        self.zero_padding = nn.ZeroPad2d(self.padding) # 进行padding, TODO: 目前不确定在什么上进行操作padding
        self.bias = bias # TODO: 不知道干什么的？应该卷积里也有吧
        self.pc_strd = 1 # TODO: pc_strd是什么？？？？
        # conv block for peak receptive field (PRF) in x
        # take PRF as input and output the center point output of PRF;
        # kernel channel = in_ch*2: depth_concat(p_c, p_r)
        # since each p_c is expanded as a tensor of kernel scale RF, the stride = kernel_size 不知道
        self.peak_conv = nn.Conv2d(in_ch, out_ch, # 32, 64
                                   kernel_size=self.kernel_size, # (4,4)
                                   padding=0,
                                   stride=self.stride, # (4,4)
                                   bias=self.bias)
        #self.peak_conv.register_backward_hook(self._set_lr)

    @staticmethod
    # set learning rate for peak_conv
    def _set_lr(module, grad_input, grad_output):
        grad_input = (grad_input[i] * 0.1 for i in range(len(grad_input)))
        grad_output = (grad_output[i] * 0.1 for i in range(len(grad_output)))

    def forward(self, x):
        # x shape: torch.Size([2, 32, 32, 400]) 32是通道数目
        h, w = x.shape[2], x.shape[3] # 32, 400
        rb = self.refer_band # 1
        gb = self.guard_band # (1,1)
        k_pk = self.kernel_size # (4,4) pk是不是peak的缩写，TODO: 所以peak部分是指外面那一圈吗
        N = k_pk[0] * k_pk[1] # 16 参与数量，我猜这就是peak的数量，但我觉得这个值是否具有通用性（这个peak数量是这么计算出来的吗，还是只有3’5配置才适用）呢？
        dtype = x.data.type() # 
        b = x.size(0) # batch_size 2
        # 输入x形状为torch.Size([2, 32, 36, 400])
        x = self.zero_padding(x) # torch.Size([2, 32, 40, 404])， padding为2，这一步是都适用的，为了让边界点也可以作为中心点
        
        x_prf = self.get_prf(x, kernel_size=5, stride=1) # 定死里面的参数

        
        
        x_prf = self._reshape_x_prf(k_pk[0], k_pk[1], x_prf) # torch.Size([2, 32, 144, 1600])
        # 4, 4, torch.Size([2, 32, 128, 1600]) TODO: 不知道它是怎么做的 torch.Size([2, 32, 128, 1600]) 
        # torch.Size([2, 32, 32 * 4, 400 * 4]) = torch.Size([2, 32, 128, 1600])

        # out: (b, c, h, w)
        out = self.peak_conv(x_prf) # 就是一个conv2d，前面是关键部分吗
        return out
    
    @staticmethod
    def get_prf(x, kernel_size, stride):
        batch_size = x.shape[0]
        c = x.shape[1]
        h = x.shape[2] - 4
        w = x.shape[3] - 4
        output_tensor = torch.empty((batch_size, c, h, w, 16), device='cuda')
        prf_div.prf_forward(
            x,
            output_tensor,
            kernel_size,
            stride
        )
        return output_tensor

    @staticmethod
    # reshape the x_prf
    def _reshape_x_prf(k_h, k_w, x_prf):
        # x_prf shape: 
        b, c, h, w, N = x_prf.size()
        x_prf = torch.cat([x_prf[..., s:s + k_w].contiguous().view(b, c, h, w * k_w) for s in range(0, N, k_w)], dim=-1)
        x_prf = x_prf.contiguous().view(b, c, h * k_h, w * k_w)
        return x_prf

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

    def __init__(self, in_ch, out_ch=256):
        super().__init__()
        self.global_avg_pool = nn.AvgPool2d((62, 512))
        self.conv1_1x1 = nn.Conv2d(in_ch, out_ch, kernel_size=1, padding=0, dilation=1)
        self.single_conv_block1_1x1 = ConvBlock(in_ch, out_ch, k_size=1, pad=0, dil=1)
        self.single_conv_block1_3x3 = ConvBlock(in_ch, out_ch, k_size=3, pad=6, dil=6)
        self.single_conv_block2_3x3 = ConvBlock(in_ch, out_ch, k_size=3, pad=12, dil=12)
        self.single_conv_block3_3x3 = ConvBlock(in_ch, out_ch, k_size=3, pad=18, dil=18)

    def forward(self, x):
        x1 = F.interpolate(self.global_avg_pool(x), size=(62, 512), align_corners=False,
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

    def __init__(self, signal_type, n_frames):
        super().__init__()
        self.signal_type = signal_type
        self.n_frames = n_frames
        self.double_3dconv_block1 = Double3DConvBlock(in_ch=1, out_ch=32, 
                                                      k_size=(1+n_frames//2, 3, 3),
                                                      pad=(0, 1, 1), dil=1) # 首先进入一个3d卷积层，TODO：不知道是在什么维度上操作
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
        x1 = torch.squeeze(x1, 2)  # remove temporal dimension 是有时间维度的，删除了，TODO: 难道这部分输出后维度为1？

        x1_pad = F.pad(x1, (0, 0, 0, 0), "constant", 0) # 进行pad， 这是什么操作，TODO: 4个0？
        x1_down = self.max_pool(x1_pad)

        x2 = self.double_pkc_block(x1_down) # 调用了pkc卷积块，TODO: 需要明确其卷积块
        x2_pad = F.pad(x2, (0, 0, 0, 1), "constant", 0) # 这部分的pad是做什么， TODO: 是因为上面的卷积会改变输入张量形状吗
        x2_down = self.doppler_max_pool(x2_pad) # 

        x3 = self.single_conv_block1_1x1(x2_down)
        # return input of ASPP block + latent features
        return x2_down, x3


class PKCIn4CWR_cuda(nn.Module):
    """
    PKCIn-Net w/ ReDA-PeakConv for KuRALS-CWR data

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

        # Backbone (encoding)
        self.rd_encoding_branch = EncodingBranch('range_doppler', self.n_frames)

        # ASPP Blocks
        self.rd_aspp_block = ASPPBlock(in_ch=64, out_ch=128)
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