#### zyc@231207 ####
#### TFoc + vanilla + maskdraw ####
#### ASPP: using TFoc output as input
#### LSE: using latent features(TFoc + 1x1Conv) as input
from __future__ import annotations
import torch
import torch.nn as nn
import torch.nn.functional as F
import math
import os

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
            nn.LeakyReLU(inplace=True)
           )

    def forward(self, x):
        x = self.block(x)
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
            nn.LeakyReLU(inplace=True)

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

    def __init__(self, in_ch, out_ch, btnk_ch, search_size, topk):
        super(TFocConv, self).__init__()
        self.btnk_ch = btnk_ch
        self.k_n = search_size
        self.pad_size = math.floor(search_size / 2)
        self.topk = topk
        self.spatial_conv = Double3DConvBlock(in_ch,
                                              self.btnk_ch,
                                              k_size=(1,3,3),
                                              strd=(1,1,1),
                                              pad=(0,1,1),
                                              btnk_sc=16)
        self.tfoc_prepadding = nn.ReplicationPad3d((self.pad_size, self.pad_size, self.pad_size, self.pad_size, 0, 0))
        self.inner_predict = TREBlock(self.btnk_ch*2,
                                      4,
                                      k_size=(1, 1, 1),
                                      strd=(1,1,1),
                                      pad=(0,0,0))
        self.softmax = nn.Softmax(dim=1)
        self.probpooling = nn.MaxPool3d(kernel_size=(1,1,search_size*search_size), stride=(1,1,1), padding=(0,0,0))
        self.tfoc_conv = TREBlock(self.btnk_ch,
                                  self.btnk_ch,
                                  k_size=(1, 1, topk + 1),
                                  strd=(1,1,1),
                                  pad=(0,0,0))
        #self.tfoc_postpadding = nn.ReplicationPad3d((2, 2, 2, 2, 0, 0))  # zlw@20230816
        self.temporal_conv = Double3DConvBlock(self.btnk_ch,
                                               out_ch,
                                               k_size=(3,1,1),
                                               strd=(1,1,1),
                                               pad=(0,0,0),
                                               btnk_sc=1)
        self.tempdila_conv = TemporalGroupedDilatedConv(in_channels=self.btnk_ch, t_len=5, kernel_size=3)  # 前t-1个卷积的核大小

    def forward(self, x_seq, lamda): #x_mask(b,c,t,h,w)
        x_seq = self.spatial_conv(x_seq)
        b, c, t, h, w = x_seq.shape[0], x_seq.shape[1], x_seq.shape[2], x_seq.shape[3], x_seq.shape[4]
        k_n = self.k_n
        n = k_n * k_n
        dtype = x_seq.data.type()

        # 0 temporal dilation convolution
        x_temdila = self.tempdila_conv(x_seq)  # b,c,t,h,w,

        # 1 extract x_tgt, repeat t-1 times in time dim and padding
        x_tgt_ori = x_seq[:, :, -1, :, :]  # (b,c,h,w)
        x_tgt_ori = torch.unsqueeze(x_tgt_ori, 2)  # (b,c,1,h,w)
        x_tgt = self.tfoc_prepadding(x_tgt_ori.repeat(1, 1, t - 1, 1, 1))  # (b,c,t-1,h_p,w_p)

        # 2 extrac x_hst (change x_hst as x_seq)
        x_hst = self.tfoc_prepadding(x_seq[:, :, :t-1, :, :])  # (b,c,t-1,h_p,w_p)
        # 3 generate center points in target maps coordinates
        p_tgt, p_tgt_pos = self._get_grid_center(b, h, w, n, dtype)  # cpu int64
        # 4 sample center points in target map x_tgt (b, c, t-1, h, w, N) using p_tgt
        x_tgt = self._sample_x(x_tgt, p_tgt, n)  # (b,c,t-1,h,w,n)

        # 5 generate search point in history maps  coordinates
        p_hst = self._gen_global_sample_grid(b, h, w, n, dtype)  # (b,2n,h,w) cpu int64
        # 6 sample search point  in history maps x_hst(b, c, t-1,h, w, N) using p_hst
        x_hst = self._sample_x(x_hst, p_hst, n)  # (b,c,t-1,h,w,n) #cuda0 florat32

        # 7 computing distance matrix
        # track_loss
        x_seq = torch.cat((x_hst, x_tgt), dim=1)  # (b,2c,t-1,h,w,n)
        x_seq = self.inner_predict(x_seq.view(b, 2*c, t - 1, h*w, n))  # (b,4,t-1,h*w,n)
        x_seq = self.softmax(x_seq) # (b,4,t-1,h*w,n)

        # prepare for ce loss
        probs = torch.prod(self.probpooling(x_seq).squeeze(dim=-1).view(b, 4, t - 1, h, w), dim=2)  # (b,4,h,w)

        # mask-draw
        _, x_clas_index = torch.max(x_seq, dim=1) #(b,t-1,h*w,n)
        # calculate the number of each value on n dimention
        x_tem_onehot = F.one_hot( x_clas_index.reshape(-1), num_classes=4)#(b*(t-1)*h*w*n,4)
        x_tem_onehot = x_tem_onehot.reshape(-1, n, 4) #(b*(t-1)*h*w,n,4)
        x_tem_onehot = torch.sum(x_tem_onehot, dim=1) #(b*(t-1)*h*w,4)
        _, x_clas_index = torch.max(x_tem_onehot, dim=1) #(b*(t-1)*h*w)

        x_clas_index = x_clas_index.reshape(b,t-1,h*w,1).repeat(1,1,1,n).unsqueeze(dim=1) #(b,1,(t-1),h*w,n)
        x_clas_val = torch.gather(x_seq, dim=1, index=x_clas_index).squeeze(dim=1) #(b,(t-1),h*w,n)

        # plt.figure()
        # plt.subplot(4, 2, 1)
        # plt.pcolor(x_clas_val[1, 0, :,:,i].detach().cpu().numpy(), label='hst1')
        # plt.subplot(4, 2, 2)
        # plt.pcolor(x_clas_val[1, 1, ...,i].detach().cpu().numpy(), label='hst2')
        # plt.subplot(4, 2, 3)
        # plt.pcolor(x_clas_val[1, 2, ...,i].detach().cpu().numpy(), label='hst3')
        # plt.subplot(4, 2, 4)
        # plt.pcolor(x_clas_val[1, 3, ...,i].detach().cpu().numpy(), label='hst4')
        # plt.subplot(4, 2, 5)
        # plt.pcolor(x_clas_val[1, 4, ...,i].detach().cpu().numpy(), label='hst5')
        # plt.subplot(4, 2, 6)
        # plt.pcolor(x_clas_val[1, 5, ...,i].detach().cpu().numpy(), label='hst6')
        # plt.subplot(4, 2, 7)
        # plt.pcolor(x_clas_val[1, 6, ...,i].detach().cpu().numpy(), label='hst7')
        # plt.subplot(4, 2, 8)
        # plt.pcolor(x_clas_val[1, 7, ...,i].detach().cpu().numpy(), label='hst8')
        # plt.legend()
        # plt.show()
       
        # plt.figure()
        # plt.subplot(4, 2, 1)
        # plt.pcolor(x_clas_val[1, 0, ...,i].detach().cpu().numpy(), label='hst1')
        # plt.subplot(4, 2, 2)
        # plt.pcolor(x_clas_val[1, 1, ...,i].detach().cpu().numpy(), label='hst2')
        # plt.subplot(4, 2, 3)
        # plt.pcolor(x_clas_val[1, 2, ...,i].detach().cpu().numpy(), label='hst3')
        # plt.subplot(4, 2, 4)
        # plt.pcolor(x_clas_val[1, 3, ...,i].detach().cpu().numpy(), label='hst4')
        # plt.subplot(4, 2, 5)
        # plt.pcolor(x_clas_val[1, 4, ...,i].detach().cpu().numpy(), label='hst5')
        # plt.subplot(4, 2, 6)
        # plt.pcolor(x_clas_val[1, 5, ...,i].detach().cpu().numpy(), label='hst6')
        # plt.subplot(4, 2, 7)
        # plt.pcolor(x_clas_val[1, 6, ...,i].detach().cpu().numpy(), label='hst7')
        # plt.subplot(4, 2, 8)
        # plt.pcolor(x_clas_val[1, 7, ...,i].detach().cpu().numpy(), label='hst8')
        # plt.legend()
        # plt.show()

        # 10 similarity
        val_resample, index_resample = torch.topk(x_clas_val, k=self.topk, dim=3, largest=True, sorted=True)  # (b,t-1,h*w,8)
        index_resample, _ = torch.sort(index_resample, dim=-1, descending=False, stable=True)
        val_resample[val_resample > 0] = 1
        index_resample = val_resample * index_resample

        device = index_resample.device
        unsample_flag = ((n - 1) / 2) * torch.ones(index_resample.shape, device=device)
        index_resample = torch.where(index_resample == 0, unsample_flag.long(), index_resample.long())

        # index_resample = torch.argmin(mat_dist, dim=3) #(b,t-1,h*w)
        location_reference = torch.arange(0, index_resample.size(-2), 1, device=device) * n  # (h*w)
        location_reference = location_reference.unsqueeze(dim=-1).repeat(1, self.topk)  # (h*w,8)
        location_reference = location_reference.unsqueeze(dim=0).repeat(t - 1, 1, 1)  # (t-1, h*w, 8)
        location_reference = location_reference.unsqueeze(dim=0).repeat(b, 1, 1, 1)  # (b, t-1, h*w, 8)
        index_resample = index_resample + location_reference  # (b,t-1,h*w, 8)

        index_resample= index_resample.contiguous().view(b, t-1, -1) #(b,t-1,h*w*8)
        index_resample = index_resample.contiguous().unsqueeze(dim=2).repeat(1, 1, c, 1)  # (b,t-1,c,h*w*8)

        # 11 topk sampling
        x_resample = torch.gather(x_hst.contiguous().view(b, t-1, c, -1), dim=-1, index=index_resample.long())

        x_resample = x_resample.contiguous().view(b, t-1, c, h, w, self.topk)  # (b,t-1,c,h,w,8)
        x_resample = x_resample.permute(0, 2, 1, 3, 4, 5)  # (b,c,t-1,h,w,8)

        # 12 enhance branch
        x_resample = torch.cat((x_resample, x_tgt_ori.repeat(1, 1, t-1, 1, 1).unsqueeze(dim=-1)),
                               dim=-1) #(b,c,t-1,h,w,9)
        x_resample = x_resample.contiguous().view(b, c, t-1, h*w, -1) #(b,c,(t-1),h*w,9)
        x_tfoc = self.tfoc_conv(x_resample).squeeze(dim=-1) #(b,c,t-1,h*w,1)-->(b,c,t-1,h*w)

        x_tfoc = x_tfoc.view(b, -1, t-1, h, w)
        #x_tfoc = lamda * torch.cat((x_tfoc, x_tgt_ori), dim= 2) + (1-lamda) * x_temdila #(b,c,t,h,w)
        x_tfoc = torch.cat((x_tfoc, x_tgt_ori), dim=2) + x_temdila  # (b,c,t,h,w)
        x_tfoc = self.temporal_conv(x_tfoc) # [b, c, t-4, h, w]
        return x_tfoc, probs

    # getting center grid from the padded grid of input x_tgt
    def _get_grid_center(self, b, h, w, n, dtype):
        # generating grid for center point
        k_ls = (self.k_n - 1) // 2
        gc_x_mesh, gc_y_mesh = torch.meshgrid(
            torch.arange(k_ls, h + k_ls, 1),
            torch.arange(k_ls, w + k_ls, 1))
        gc_x = torch.flatten(gc_x_mesh).view(1, 1, h, w).repeat(1, n, 1, 1)
        gc_y = torch.flatten(gc_y_mesh).view(1, 1, h, w).repeat(1, n, 1, 1)
        grid_c = torch.cat([gc_x, gc_y], 1).type(dtype)  # (1,2n,h,w)
        grid_c = grid_c.repeat(b, 1, 1, 1)  # (b,2n,h,w)
        grid_c_pos = torch.cat([torch.flatten(gc_x_mesh).view(1, h, w), torch.flatten(gc_y_mesh).view(1, h, w)],
                               dim=0)  # (2,h,w)
        return grid_c, grid_c_pos

    # getting local search grid: (1,2n,1,1)
    def _get_grid_local_search(self, n, dtype):
        k_ls = (self.k_n - 1) // 2
        gl_x, gl_y = torch.meshgrid(
            torch.arange(-k_ls, k_ls + 1),
            torch.arange(-k_ls, k_ls + 1))
        grid_ls = torch.cat([torch.flatten(gl_x), torch.flatten(gl_y)], 0)  # (2n, 1)
        grid_ls = grid_ls.view(1, 2 * n, 1, 1).type(dtype)
        return grid_ls

    def _gen_global_sample_grid(self, b, h, w, n, dtype):
        grid_c, _ = self._get_grid_center(b, h, w, n, dtype)  # (b,2n,h,w)
        grid_ls = self._get_grid_local_search(n, dtype)  # (1,2n,1,1)
        # print(grid_c.device, grid_ls.device)
        grid_gs = grid_c + grid_ls  # (b,2n,h,w)
        return grid_gs

    # expanding hitoric map using grid_gs
    def _sample_x(self, x_pad, p, N):
        # p:(b,2N,h,w)  x_pad:(b,c,t-1,h_p,w_p)
        p = p.contiguous().permute(0, 2, 3, 1)  # (b,h,w,2n)
        b, h, w, _ = p.size()
        h_pad = x_pad.size(3)
        w_pad = x_pad.size(4)
        c = x_pad.size(1)
        t = x_pad.size(2)
        # (b, c, t-1, h_pad*w_pad)
        # strech each spatial channel of x_padas 1-D vector
        x_pad = x_pad.contiguous().view(b, c, t, -1)  # cuda0 float32
        # (b, h_pad, w_pad, n)
        # transform spatial coord of p into the 1-D index
        index = p[..., :N] * w_pad + p[..., N:]  # cpu,int64
        # index_x = torch.clamp(p[..., :N], 0, h_pad-1)
        # index_y = torch.clamp(p[..., N:], 0, w_pad-1)
        # index = index_x * w_pad + index_y
        # (b, c, t-1, h*w*N)
        index = index.contiguous().unsqueeze(dim=1).expand(-1, t, -1, -1, -1)
        index = index.contiguous().unsqueeze(dim=1).expand(-1, c, -1, -1, -1, -1)
        index = index.contiguous().view(b, c, t, -1)
        index = index.long().to(x_pad.device)  # int64 [b,c,t-1,h*w*N]
        x_r = x_pad.gather(dim=-1, index=index)
        x_r = x_r.contiguous().view(b, c, t, h, w, N)
        return x_r  # cuda0 float32

    # resampling expanded historic map with resampling index
    # def _resample_x(self, x_exp, idx, n):


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
        self.double_3d_conv_block = Double3DConvBlock(in_ch=1,
                                                      out_ch=64,
                                                      k_size=(3,3,3),
                                                      strd=(1,1,1),
                                                      pad=(0,1,1),
                                                      btnk_sc=1)
        self.tfoc_conv_block = TFocConv(in_ch=64,
                                        out_ch=128,
                                        btnk_ch=8, #out_ch/btnk_sc
                                        search_size=7,
                                        topk=8)
        self.doppler_max_pool = nn.MaxPool2d(2, stride=(2, 1))
        self.max_pool = nn.MaxPool2d(2, stride=2)
        self.single_conv_block1_3x3 = ConvBlock(in_ch=128, out_ch=128, k_size=3, pad=1, dil=1)
        self.single_conv_block2_3x3 = ConvBlock(in_ch=128, out_ch=128, k_size=3, pad=1, dil=1)
        self.single_conv_block1_1x1 = ConvBlock(in_ch=128, out_ch=128, k_size=1, pad=0, dil=1)
        self.single_conv_block1_3x3_prb = ConvBlock(in_ch=4, out_ch=32, k_size=3, pad=1, dil=1)
        self.single_conv_block2_3x3_prb = ConvBlock(in_ch=32, out_ch=32, k_size=3, pad=1, dil=1)
        self.single_conv_block1_1x1_prb = ConvBlock(in_ch=32, out_ch=32, k_size=1, pad=0, dil=1)

    def forward(self, x, lamda):
        x0 = self.double_3d_conv_block(x)  # [b,128,(n_frames-3+1-3+1),h,w]
        x1, prb1 = self.tfoc_conv_block(x0, lamda)  # x1:[b,128,1,h,w] prb1:[b,4,h,w]
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

    def __init__(self, n_classes, n_frames, signal_type, zero_doppler_filter=True):
        super().__init__()
        self.n_classes = n_classes
        self.n_frames = n_frames
        self.zero_doppler_filter = zero_doppler_filter

        # Backbone (encoding)
        self.rd_encoding_branch = EncodingBranch('range_doppler', self.n_frames)
        self.ra_encoding_branch = EncodingBranch('range_angle', self.n_frames)
        self.ad_encoding_branch = EncodingBranch('angle_doppler', self.n_frames)

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

    def forward(self, x_rd, x_ra, x_ad, lamda):
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
        ra_features, ra_latent, ra_prb, ra_prb_latent = self.ra_encoding_branch(x_ra, lamda)
        if self.zero_doppler_filter:
            x_rd = self._zero_doppler_filter(x_rd)
            x_ad = self._zero_doppler_filter(x_ad)
        rd_features, rd_latent, rd_prb, rd_prb_latent = self.rd_encoding_branch(x_rd, lamda)
        ad_features, ad_latent, ad_prb, ad_prb_latent = self.ad_encoding_branch(x_ad, lamda)

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
        device = rd.device
        zero_doppler_mask = torch.cat([torch.normal(1, 1e-3, size=(256, 31), device=device),
                                       torch.normal(1e-8, 1e-3, size=(256, 1), device=device),
                                       torch.normal(1, 1e-3, size=(256, 32), device=device)], dim=-1)
        rd = rd * zero_doppler_mask.unsqueeze(dim=0).unsqueeze(dim=0)
        return rd

