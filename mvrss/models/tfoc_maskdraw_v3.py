#### zyc@20231130 ####
#### Multi-View TFoc_Ultra_Maskdraw Ad_w/o_tfoc ####
import matplotlib.pyplot as plt

import torch
import torch.nn as nn
import torch.nn.functional as F
import math
import os


sdc = 128

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

class Double3DConvBlock_Ad(nn.Module):
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

class ATPPBlock(nn.Module):
    """Atrous Temporal Pyramid Pooling
    Parallel conv blocks with different dilation rate
    """

    def __init__(self, in_ch, out_ch=32):
        super().__init__()
        self.conv3d_1_5x1x1 = nn.Conv3d(in_ch, out_ch, kernel_size=(5, 1, 1), stride=1, padding=(0,0,0), dilation=1)
        self.conv3d_2_4x1x1 = nn.Conv3d(in_ch, out_ch, kernel_size=(4, 1, 1), stride=1, padding=(1,0,0), dilation=2)
        self.conv3d_3_3x1x1 = nn.Conv3d(in_ch, out_ch, kernel_size=(2, 1, 1), stride=1, padding=(0,0,0), dilation=3)

    def forward(self, x):
        x1 = self.conv3d_1_5x1x1(x)
        x2 = self.conv3d_2_4x1x1(x)
        x3 = self.conv3d_3_3x1x1(x)
        x_cat = torch.cat((x1, x2, x3[:,:,0,:,:].unsqueeze(dim=2)), dim = 1)
        return x_cat


class TFocConv(nn.Module):
    """Temporal Focusing Conv Block w/ single historic frame
    input: x_seq = Conv3D(x) BxCx2xHxW tensor
    output: x_tfc BxCxHxW tensor
    """

    def __init__(self, in_ch, out_ch, btnk_ch, search_size, n_frames, pos_enc, att_mask, topk, dist_type, mask):
        super(TFocConv, self).__init__()
        self.k_n = search_size
        self.pad_size = math.floor(search_size / 2)
        self.dist_type = dist_type
        self.n_frames = n_frames
        self.pos_enc = pos_enc
        self.att_mask = att_mask
        self.topk = topk
        self.pe_size = 16
        self.btnk_ch = btnk_ch
        self.mask = mask
        # self.tfoc_padding = nn.ZeroPad2d((self.pad_size, self.pad_size, self.pad_size, self.pad_size))


        # self.Layernorm = nn.LayerNorm([4,256,256])
        # self.atpp = ATPPBlock(in_ch, in_ch)
        # self.hst_avg_pool = nn.AvgPool2d([4,1])

        # if self.pos_enc:
        #     self.tfoc_pos_embedding = TREBlock(7, self.pe_size,
        #                                        k_size=(1, 1, 1),
        #                                        strd=(1, 1, 1),
        #                                        pad=(0, 0, 0))
        #     self.tfoc_conv_1 = TREBlock(in_ch + self.pe_size, out_ch,
        #                                 k_size=(3, 3, 3),
        #                                 strd=(1, 1, 1),
        #                                 pad=(0, 0, 0))
        # else:
        #     self.tfoc_conv_1 = TREBlock(in_ch, out_ch,
        #                                 k_size=(3, 3, 3),
        #                                 strd=(1, 1, 1),
        #                                 pad=(0, 0, 0))
        # self.tfoc_conv_2 = TREBlock(out_ch, out_ch,
        #                             k_size=(3, 3, 3),
        #                             strd=(1, 1, 1),
        #                             pad=(0, 0, 0))
        # self.spatial_conv = TREBlock(in_ch, out_ch, k_size=(1,3,3), strd=(1,1,1), pad=(0,1,1))
        self.spatial_conv = Double3DConvBlock(in_ch,
                                              self.btnk_ch,
                                              k_size=(1,3,3),
                                              strd=(1,1,1),
                                              pad=(0,1,1),
                                              btnk_sc=16)
        self.tfoc_prepadding = nn.ReplicationPad3d((self.pad_size, self.pad_size, self.pad_size, self.pad_size, 0, 0))
        self.kernel_pre = TREBlock(self.btnk_ch*2, self.topk+1, k_size=(1, 1, search_size*search_size), strd=(1,1,1), pad=(0,0,0))
        self.inner_predict = TREBlock(self.btnk_ch*2, 4, k_size=(1, 1, 1), strd=(1,1,1), pad=(0,0,0))

        self.probpooling = nn.MaxPool3d(kernel_size=(1,1,search_size*search_size), stride=(1,1,1), padding=(0,0,0))
        self.tfoc_conv = TREBlock(self.btnk_ch,
                                   self.btnk_ch,
                                   k_size=(1, 1, topk + 1),
                                   strd=(1,1,1),
                                   pad=(0,0,0))
        self.tfoc_postpadding = nn.ReplicationPad3d((2, 2, 2, 2, 0, 0))  # zlw@20230816
        self.temporal_conv = Double3DConvBlock(self.btnk_ch,
                                               out_ch,
                                               k_size=(3,1,1),
                                               strd=(1,1,1),
                                               pad=(0,0,0),
                                               btnk_sc=1)

        # self.tfoc_pooling = nn.AvgPool3d((in_ch - 1, 1, 1), stride=(1, 1, 1))

    # distance matrix
    def _compute_dist(self, x_tgt, x_hst):  ##(b,t-1,c,w*h,n)
        if self.dist_type == 'l2':
            x_diff = x_tgt - x_hst
            mat_dist = torch.norm(x_diff, p='fro', dim=2)  # (b,t-1,w*h,n)
        elif self.dist_type == 'l1':
            x_diff = x_tgt - x_hst
            mat_dist = torch.norm(x_diff, p=1, dim=2)
        elif self.dist_type == 'cos':
            mat_dist = -nn.functional.cosine_similarity(x_tgt, x_hst, dim=2, eps=1e-6)
        if self.att_mask:
            scaler = math.sqrt(x_tgt.shape[2])
            x_t = x_tgt.contiguous().permute(0, 1, 3, 4, 2).unsqueeze(dim=-1)  ## (b,t-1,w*h,n,c,1)
            x_h = x_hst.contiguous().permute(0, 1, 3, 4, 2).unsqueeze(dim=-2)  ## (b,t-1,w*h,n,1,c)
            att_dist = torch.matmul(x_h, x_t).contiguous().view(x_tgt.shape[0],
                                                                x_tgt.shape[1],
                                                                x_tgt.shape[3],
                                                                x_tgt.shape[4])
            att_dist = torch.sigmoid(att_dist / scaler)
            return att_dist, mat_dist
        return mat_dist

    def _get_relative_pos(self, x_resample_pos, p_tgt_pos):
        b, tt, c, h, w = x_resample_pos.shape[0], x_resample_pos.shape[1], x_resample_pos.shape[2], \
                         x_resample_pos.shape[3], x_resample_pos.shape[4]

        p_tgt_pos = p_tgt_pos.contiguous().unsqueeze(dim=0).unsqueeze(dim=0)  # (1, 1, 2, h, w)
        p_tgt_pos = p_tgt_pos.contiguous().repeat(b, 1, 1, 1, 1).cuda()  # (b, 1, 2, h, w)
        delta_pos = x_resample_pos - p_tgt_pos.repeat(1, tt, 1, 1, 1)  # (b, t-1, 2, h, w)
        pos_dist = torch.norm(delta_pos, p='fro', dim=2).contiguous().unsqueeze(dim=2)  # (b,t-1,1,h,w)
        # pos_angle = delta_pos / pos_dist.contiguous().repeat(1, 1, 2, 1, 1)  # (b, t-1, 2, h, w)
        # pos_angle[:, :, 0, :, :] = torch.where(torch.isnan(pos_angle[:, :, 0, :, :]),
        #                                        torch.full_like(pos_angle[:, :, 0, :, :], 0), pos_angle[:, :, 0, :, :])
        # pos_angle[:, :, 1, :, :] = torch.where(torch.isnan(pos_angle[:, :, 1, :, :]),
        #                                        torch.full_like(pos_angle[:, :, 1, :, :], 1), pos_angle[:, :, 1, :, :])
        # relative_pos = torch.cat([x_resample_pos, delta_pos, pos_dist, pos_angle], dim=2)  # (b, t-1, 7, h, w)
        # p_tgt_pos = torch.cat([p_tgt_pos, torch.zeros([b, 1, 5, h, w]).cuda()], dim=2)  # (b,1,7 h,w)
        # relative_pos = torch.cat([relative_pos, p_tgt_pos], dim=1).contiguous().permute(0, 2, 1, 3, 4)  # (b,7,t,h,w)
        # relative_pos_max = relative_pos.max(dim=-1).values.max(dim=-1).values.max(dim=-1).values
        # relative_pos = relative_pos / relative_pos_max.unsqueeze(dim=-1).unsqueeze(dim=-1).unsqueeze(dim=-1)
        return pos_dist

    def forward(self, x_seq): #x_mask(b,c,t,h,w)
        x_seq = self.spatial_conv(x_seq)
        b, c, t, h, w = x_seq.shape[0], x_seq.shape[1], x_seq.shape[2], x_seq.shape[3], x_seq.shape[4]
        k_n = self.k_n
        n = k_n * k_n
        dtype = x_seq.data.type()

        # 1 extract x_tgt, repeat t-1 times in time dim and padding
        x_tgt_ori = x_seq[:, :, -1, :, :]  # (b,c,h,w)
        x_tgt_ori = torch.unsqueeze(x_tgt_ori, 2)  # (b,c,1,h,w)
        x_tgt = self.tfoc_prepadding(x_tgt_ori.repeat(1, 1, t - 1, 1, 1))  # (b,c,t-1,h_p,w_p)

        # 2 extrac x_hst (change x_hst as x_seq)
        x_hst = self.tfoc_prepadding(x_seq[:, :, :t - 1, :, :])  # (b,c,t-1,h_p,w_p)

        # 3 generate center points in target maps coordinates
        p_tgt, p_tgt_pos = self._get_grid_center(b, h, w, n, dtype)  # cpu int64
        # 4 sample center points in target map x_tgt (b, c, t-1, h, w, N) using p_tgt
        x_tgt = self._sample_x(x_tgt, p_tgt, n)  # (b,c,t-1,h,w,n)

        # 5 generate search point in history maps  coordinates
        p_hst = self._gen_global_sample_grid(b, h, w, n, dtype)  # (b,2n,h,w) cpu int64
        # 6 sample search point  in history maps x_hst(b, c, t-1,h, w, N) using p_hst
        x_hst = self._sample_x(x_hst, p_hst, n)  # (b,c,t-1,h,w,n) #cuda0 florat32

        # 7 computing distance matrix
        x_seq = torch.cat((x_hst, x_tgt), dim=1)  # (b,2c,t-1,h,w,n)

        # channel concat and squeeze to c=4

        x_seq = self.inner_predict(x_seq.view(b, 2 * c, t - 1, h * w, n))  # (b,4,t-1,h*w,n)

        # prepare for ce loss
        x_nbr = self.probpooling(x_seq).squeeze(dim=-1)  # (b,4,t-1,h*w)
        x_pre = torch.prod(x_nbr.view(b, 4, t - 1, h, w), dim=2)  # (b,4,h,w)

        # mask-draw
        x_clas_val, x_clas_index = torch.max(x_seq, dim=1)  # (b,t-1,h*w,n)
        # _, x_clas_index = torch.max(x_seq, dim=1)  # (b,t-1,h*w,n)
        # x_clas_val = self.mf(x_seq)
        # i=10
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
        x_clas_index[x_clas_index > 0] = 1
        x_clas_val = x_clas_val * x_clas_index
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
        val_resample, index_resample = torch.topk(x_clas_val, k=self.topk, dim=3, largest=False,
                                                  sorted=True)  # (b,t-1,h*w,8)
        val_resample[val_resample > 0] = 1
        index_resample = val_resample * index_resample
        unsample_flag = ((n - 1) / 2) * torch.ones(index_resample.shape)
        index_resample = torch.where(index_resample == 0, unsample_flag.long().cuda(), index_resample.long())
        # index_resample = torch.argmin(mat_dist, dim=3) #(b,t-1,h*w)
        location_reference = torch.arange(0, index_resample.size(-2), 1) * n  # (h*w)
        location_reference = location_reference.unsqueeze(dim=-1).repeat(1, self.topk)  # (h*w,8)
        location_reference = location_reference.unsqueeze(dim=0).repeat(t - 1, 1, 1)  # (t-1, h*w, 8)
        location_reference = location_reference.unsqueeze(dim=0).repeat(b, 1, 1, 1)  # (b, t-1, h*w, 8)
        index_resample = index_resample + location_reference.cuda()  # (b,t-1,h*w, 8)
        index_resample = index_resample.contiguous().view(b, t - 1, -1)  # (b,t-1,h*w*8)
        index_resample = index_resample.contiguous().unsqueeze(dim=2).repeat(1, 1, c, 1)  # (b,t-1,c,h*w*8)

        # 11 topk sampling
        x_resample = torch.gather(x_hst.contiguous().view(b, t - 1, c, -1), dim=-1, index=index_resample.long())
        x_resample = x_resample.contiguous().view(b, t - 1, c, h, w, self.topk)  # (b,t-1,c,h,w,8)

        if self.att_mask:
            x_resample = (mask * x_resample).permute(0, 2, 1, 3, 4, 5)  # (b,c,t-1,h,w,8)
        else:
            x_resample = x_resample.permute(0, 2, 1, 3, 4, 5)  # (b,c,t-1,h,w,8)

        # 12 enhance branch
        x_resample = torch.cat((x_resample, x_tgt_ori.repeat(1, 1, t - 1, 1, 1).unsqueeze(dim=-1)),
                               dim=-1)  # (b,c,t-1,h,w,9)
        x_resample = x_resample.contiguous().view(b, c, t - 1, h * w, -1)  # (b,c,(t-1),h*w,9)
        x_tfoc = self.tfoc_conv(x_resample).squeeze(dim=-1)  # (b,c,t-1,h*w,1)-->(b,c,t-1,h*w)
        x_tfoc = x_tfoc.view(b, -1, t - 1, h, w)
        x_tfoc = torch.cat((x_tfoc, x_tgt_ori), dim=2)  # (b,c,t,h,w)

        x_tfc = self.temporal_conv(x_tfoc)  # [b, c, t-2, h, w]
        return x_tfc, x_pre

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
        index = index.long().cuda()  # cuda0 int64 [b,128,t-1,h*w*N]
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
        self.tfoc_conv_block_1 = TFocConv(in_ch=1, out_ch=128, btnk_ch=4,
                                          search_size=5,
                                          pos_enc=False,
                                          att_mask=False,
                                          topk=8,
                                          dist_type='l2',
                                          n_frames=self.n_frames,
                                          mask = True)
        self.tfoc_conv_block_2 = TFocConv(in_ch=128, out_ch=128, btnk_ch=8,
                                          search_size=3,
                                          pos_enc=False,
                                          att_mask=False,
                                          topk=4,
                                          dist_type='l2',
                                          n_frames=self.n_frames,
                                          mask = False)
        self.doppler_max_pool3d = nn.MaxPool3d(kernel_size=(1, 2, 2),
                                               stride=(1, 2, 1))
        self.max_pool3d = nn.MaxPool3d(kernel_size=(1, 2, 2),
                                       stride=(1, 2, 2))
        self.doppler_max_pool = nn.MaxPool2d(2, stride=(2, 1))
        self.max_pool = nn.MaxPool2d(2, stride=2)
        self.single_conv_block1_3x3 = ConvBlock(in_ch=128, out_ch=128, k_size=3, pad=1, dil=1)
        self.single_conv_block2_3x3 = ConvBlock(in_ch=128, out_ch=128, k_size=3, pad=1, dil=1)
        self.single_conv_block1_1x1 = ConvBlock(in_ch=128, out_ch=128, k_size=1, pad=0, dil=1)

    def forward(self, x):

        x0, _ = self.tfoc_conv_block_1(x)
        x1, x_pre = self.tfoc_conv_block_2(x0)

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
        return x2_down, x3, x_pre

class EncodingBranch_Ad(nn.Module):
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
        self.double_3dconv_block1 = Double3DConvBlock_Ad(in_ch=1, out_ch=128, k_size=3,
                                                      pad=(0, 1, 1), dil=1)
        self.double_3dconv_block2 = Double3DConvBlock_Ad(in_ch=128, out_ch=128, k_size=3,
                                                         pad=(0, 1, 1), dil=1)
        self.doppler_max_pool3d = nn.MaxPool3d(kernel_size=(1, 2, 2),
                                               stride=(1, 2, 1))
        self.max_pool3d = nn.MaxPool3d(kernel_size=(1, 2, 2),
                                       stride=(1, 2, 2))
        self.doppler_max_pool = nn.MaxPool2d(2, stride=(2, 1))
        self.max_pool = nn.MaxPool2d(2, stride=2)
        self.single_conv_block1_3x3 = ConvBlock(in_ch=128, out_ch=128, k_size=3, pad=1, dil=1)
        self.single_conv_block2_3x3 = ConvBlock(in_ch=128, out_ch=128, k_size=3, pad=1, dil=1)
        self.single_conv_block1_1x1 = ConvBlock(in_ch=128, out_ch=128, k_size=1, pad=0, dil=1)

    def forward(self, x):
        x0 = self.double_3dconv_block1(x)
        x1 = self.double_3dconv_block2(x0)
        x1_pad = F.pad(x1, (0, 1, 0, 0), "constant", 0)  # F.pad(top, bottom, left, right)
        x1_down = self.doppler_max_pool3d(x1_pad)

        x1_down = x1_down.squeeze(dim=2)  # zlw@20230816

        x2 = self.single_conv_block1_3x3(x1_down)  # [2,128,128,128]
        x2 = self.single_conv_block2_3x3(x2)  # [2,128,128,128]

        x2_pad = F.pad(x2, (0, 1, 0, 0), "constant", 0)
        x2_down = self.doppler_max_pool(x2_pad)

        x3 = self.single_conv_block1_1x1(x2_down)  # [2,128,64,64]
        # return input of TAM (x2_down) & ASPP block (x3) & Latent space (x3)
        return x2_down, x3

def _zero_doppler_filter(rd):
    zero_doppler_mask = torch.cat([torch.normal(1, 1e-3, size=(256, 31)),
                                   torch.normal(1e-8, 1e-3, size=(256, 1)),
                                   torch.normal(1, 1e-3, size=(256, 32))], dim=-1).cuda()
    rd = rd * zero_doppler_mask.unsqueeze(dim=0).unsqueeze(dim=0)
    return rd


class TFocNet(nn.Module):
    """
    Temporal Focus Net Single View

    PARAMETERS
    ----------
    n_classes: int
        Number of classes used for the semantic segmentation task
    n_frames: int
        Total numer of frames used as a sequence
    """

    def __init__(self, n_classes, n_frames, signal_type, zero_doppler_filter=False):
        super().__init__()
        self.n_classes = n_classes
        self.n_frames = n_frames
        self.zero_doppler_filter = zero_doppler_filter
        # self.view_type = signal_type

        # Backbone (encoding)
        self.rd_encoding_branch = EncodingBranch('range_doppler', self.n_frames)
        self.ra_encoding_branch = EncodingBranch('range_angle', self.n_frames)
        self.ad_encoding_branch = EncodingBranch_Ad('angle_doppler', self.n_frames)

        # ASPP Blocks
        self.rd_aspp_block = ASPPBlock(in_ch=sdc, out_ch=sdc)
        self.ra_aspp_block = ASPPBlock(in_ch=sdc, out_ch=sdc)
        self.ad_aspp_block = ASPPBlock(in_ch=sdc, out_ch=sdc)
        self.rd_single_conv_block1_1x1 = ConvBlock(in_ch=sdc * 5, out_ch=sdc, k_size=1, pad=0, dil=1)
        self.ra_single_conv_block1_1x1 = ConvBlock(in_ch=sdc * 5, out_ch=sdc, k_size=1, pad=0, dil=1)
        self.ad_single_conv_block1_1x1 = ConvBlock(in_ch=sdc * 5, out_ch=sdc, k_size=1, pad=0, dil=1)

        # Decoding
        self.rd_single_conv_block2_1x1 = ConvBlock(in_ch=sdc*3, out_ch=sdc, k_size=1, pad=0, dil=1)
        self.ra_single_conv_block2_1x1 = ConvBlock(in_ch=sdc*3, out_ch=sdc, k_size=1, pad=0, dil=1)

        # Pallel range-Doppler (RD) and range-angle (RA) decoding branches
        self.rd_upconv1 = nn.ConvTranspose2d(sdc * 3, sdc, (2, 1), stride=(2, 1))
        self.rd_upconv2 = nn.ConvTranspose2d(sdc, sdc, (2, 1), stride=(2, 1))

        self.ra_upconv1 = nn.ConvTranspose2d(sdc * 3, sdc, 2, stride=2)
        self.ra_upconv2 = nn.ConvTranspose2d(sdc, sdc, 2, stride=2)

        self.rd_double_conv_block1 = DoubleConvBlock(in_ch=sdc, out_ch=sdc, k_size=3,
                                                     pad=1, dil=1)
        self.ra_double_conv_block1 = DoubleConvBlock(in_ch=sdc, out_ch=sdc, k_size=3,
                                                     pad=1, dil=1)


        self.rd_double_conv_block2 = DoubleConvBlock(in_ch=sdc, out_ch=sdc, k_size=3,
                                                     pad=1, dil=1)
        self.ra_double_conv_block2 = DoubleConvBlock(in_ch=sdc, out_ch=sdc, k_size=3,
                                                     pad=1, dil=1)

        # Final 1D convs
        self.rd_final = nn.Conv2d(in_channels=sdc, out_channels=n_classes, kernel_size=1, padding=0)
        self.ra_final = nn.Conv2d(in_channels=sdc, out_channels=n_classes, kernel_size=1, padding=0)

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
        # if self.zero_doppler_filter:
        #     x_rd = _zero_doppler_filter(x_rd)

        ra_features, ra_latent, x_pre_ra = self.ra_encoding_branch(x_ra)
        rd_features, rd_latent, x_pre_rd = self.rd_encoding_branch(x_rd)
        ad_features, ad_latent = self.ad_encoding_branch(x_ad)

        # ASPP blocks
        x1_rd = self.rd_aspp_block(rd_features)
        x1_ra = self.ra_aspp_block(ra_features)
        x1_ad = self.ad_aspp_block(ad_features)
        x2_rd = self.rd_single_conv_block1_1x1(x1_rd)
        x2_ra = self.ra_single_conv_block1_1x1(x1_ra)
        x2_ad = self.ad_single_conv_block1_1x1(x1_ad)

        # Latent Space Embedding
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
        return x9_rd, x9_ra, x_pre_rd, x_pre_ra
