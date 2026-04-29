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


class TFocConv(nn.Module):
    """Temporal Focusing Conv Block w/ single historic frame
    input: x_seq = Conv3D(x) BxCx2xHxW tensor
    output: x_tfc BxCxHxW tensor
    """

    def __init__(self, in_ch, out_ch, btnk_ch, search_size, chunk_size):
        super(TFocConv, self).__init__()
        self.btnk_ch = btnk_ch
        self.k_n = search_size
        self.pad_size = math.floor(search_size / 2)
        self.chunk_size = chunk_size

        self.spatial_conv = Double3DConvBlock(in_ch,
                                              self.btnk_ch,
                                              k_size=(1,3,3),
                                              strd=(1,1,1),
                                              pad=(0,1,1),
                                              btnk_sc=16)
        self.tfoc_prepadding = nn.ReplicationPad3d((self.pad_size, self.pad_size, self.pad_size, self.pad_size, 0, 0))
        self.tfoc_predict = TREBlock(self.btnk_ch,
                                      self.btnk_ch,
                                      k_size=(2, search_size, search_size),
                                      strd=(2, 1, 1),
                                      pad=(0, self.pad_size, self.pad_size))
        self.tfoc_cmpr= ConvBlock(search_size*search_size,
                                  search_size*search_size,
                                  k_size=(1, 1),
                                  pad =(0, 0),
                                  dil=1)
        self.inner_predict = TREBlock(self.btnk_ch * 2,
                                      4,
                                      k_size=(1, 1, 1),
                                      strd=(1, 1, 1),
                                      pad=(0, 0, 0))
        self.softmax = nn.Softmax(dim=1)
        self.probpooling = nn.MaxPool3d(kernel_size=(1, 1, search_size*search_size), stride=(1,1,1), padding=(0,0,0))
        self.tfoc_conv = TFocBlock(self.btnk_ch, out_ch, k_size=3, strd=1, pad=1)
        self.temporal_conv = TREBlock(out_ch,
                                      out_ch,
                                      k_size=(3, 1, 1),
                                      strd=(1, 1, 1),
                                      pad=(0, 0, 0))
        #self.tfoc_postpadding = nn.ReplicationPad3d((2, 2, 2, 2, 0, 0))  # zlw@20230816


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
        x_hst = self.tfoc_prepadding(x_seq[:, :, :t-1, :, :])  # (b,c,t-1,h_p,w_p)
        h_p, w_p = x_seq.shape[3], x_seq.shape[4]
        # 3 generate center points in target maps coordinates
        p_tgt, p_tgt_pos = self._get_grid_center(b, h, w, n, dtype)  # cpu int64
        # 4 sample center points in target map x_tgt (b, c, t-1, h, w, N) using p_tgt
        x_tgt = self._sample_x(x_tgt, p_tgt, n)  # (b,c,t-1,h,w,n)
        
        
        # 5 generate search point in history maps  coordinates
        p_hst = self._gen_global_sample_grid(b, h, w, n, dtype)  # (b,2n,h,w) cpu int64
        # 6 sample search point  in history maps x_hst(b, c, t-1,h, w, N) using p_hst
        x_hst = self._sample_x(x_hst, p_hst, n)  # (b,c,t-1,h,w,n) #cuda0 florat32
        
        
        # 7 computing distance matrix
        #track_loss
        x_seq1 = torch.cat((x_hst, x_tgt), dim=1)  # (b,2c,t-1,h,w,n)
        x_seq1 = self.inner_predict(x_seq1.view(b, 2 * c, t - 1, h * w, n))  # (b,4,t-1,h*w,n)
        x_seq1 = self.softmax(x_seq1)  # (b,4,t-1,h*w,n)
        x_nbr = self.probpooling(x_seq1).squeeze(dim=-1)  # (b,4,t-1,h*w)
        probs = torch.prod(x_nbr.view(b, 4, t - 1, h, w), dim=2)  # (b,4,h,w)
      
        
        #zyc@251017 tfoc: cross_concat version
        x_hst = x_hst.permute(0, 1, 2, 5, 3, 4) #(b,c,t-1,n,h,w)
        x_tgt = x_tgt.permute(0, 1, 2, 5, 3, 4) #(b,c,t-1,n,h,w)
        x_seq = torch.stack([x_tgt, x_hst], dim=4) #(b,c,t-1,n,2,h,w)
      
        
        x_seq = x_seq.reshape(b, c, t-1, -1, h_p, w_p).permute(0, 2, 1, 3, 4, 5) #(b,c,t-1,2n,h,w)->#(b,t-1,c,2n,h,w)
        x_seq = self.tfoc_predict(x_seq.reshape(-1, c, 2*n, h_p, w_p))  # (b*t-1,c,n, h, w)
        x_seq = self.tfoc_cmpr(x_seq.reshape(-1, n, h_p, w_p)) # (b*t-1*c,n, h, w)
        x_seq = x_seq.sum(dim=1).reshape(b, t-1, c, h, w).permute(0, 2, 1, 3, 4) # (b,t-1,c, h, w)->(b, c, t-1, h, w)
        x_tfoc = torch.cat((x_seq, x_tgt_ori), dim=2) #(b,c,t,h,w)
        
        x_tfoc = self.tfoc_conv(x_tfoc)  # (b,c,t-2,h,w)
        x_tfoc = self.temporal_conv(x_tfoc)# (b,c,t-4,h,w)
        return x_tfoc, probs

    # ##original version
    # def forward(self, x_seq): #x_mask(b,c,t,h,w)
    #     x_seq = self.spatial_conv(x_seq)
    #     b, c, t, h, w = x_seq.shape[0], x_seq.shape[1], x_seq.shape[2], x_seq.shape[3], x_seq.shape[4]
    #     k_n = self.k_n
    #     n = k_n * k_n
    #     dtype = x_seq.data.type()
    #
    #     # 1 extract x_tgt, repeat t-1 times in time dim and padding
    #     x_tgt_ori = x_seq[:, :, -1, :, :]  # (b,c,h,w)
    #     x_tgt_ori = torch.unsqueeze(x_tgt_ori, 2)  # (b,c,1,h,w)
    #     x_tgt = self.tfoc_prepadding(x_tgt_ori.repeat(1, 1, t - 1, 1, 1))  # (b,c,t-1,h_p,w_p)
    #
    #     # 2 extrac x_hst (change x_hst as x_seq)
    #     x_hst = self.tfoc_prepadding(x_seq[:, :, :t-1, :, :])  # (b,c,t-1,h_p,w_p)
    #     h_p, w_p = x_seq.shape[3], x_seq.shape[4]
    #     # 3 generate center points in target maps coordinates
    #     p_tgt, p_tgt_pos = self._get_grid_center(b, h, w, n, dtype)  # cpu int64
    #     # 4 sample center points in target map x_tgt (b, c, t-1, h, w, N) using p_tgt
    #     x_tgt = self._sample_x(x_tgt, p_tgt, n)  # (b,c,t-1,h,w,n)
    #     # 5 generate search point in history maps  coordinates
    #     p_hst = self._gen_global_sample_grid(b, h, w, n, dtype)  # (b,2n,h,w) cpu int64
    #     # 6 sample search point  in history maps x_hst(b, c, t-1,h, w, N) using p_hst
    #     x_hst = self._sample_x(x_hst, p_hst, n)  # (b,c,t-1,h,w,n) #cuda0 florat32
    #     # 7 computing distance matrix
    #
    #     #track_loss
    #     x_seq1 = torch.cat((x_hst, x_tgt), dim=1)  # (b,2c,t-1,h,w,n)
    #     x_seq1 = self.inner_predict(x_seq1.view(b, 2 * c, t - 1, h * w, n))  # (b,4,t-1,h*w,n)
    #     x_seq1 = self.softmax(x_seq1)  # (b,4,t-1,h*w,n)
    #     x_nbr = self.probpooling(x_seq1).squeeze(dim=-1)  # (b,4,t-1,h*w)
    #     probs = torch.prod(x_nbr.view(b, 4, t - 1, h, w), dim=2)  # (b,4,h,w)
    #
    #
    #     #zyc@251017 tfoc: cross_concat version
    #     x_hst = x_hst.permute(0, 1, 2, 5, 3, 4) #(b,c,t-1,n,h,w)
    #     x_tgt = x_tgt.permute(0, 1, 2, 5, 3, 4) #(b,c,t-1,n,h,w)
    #     x_seq = torch.stack([x_tgt, x_hst], dim=4) #(b,c,t-1,n,2,h,w)
    #     x_seq = x_seq.reshape(b, c, t-1, -1, h_p, w_p).permute(0, 2, 1, 3, 4, 5) #(b,c,t-1,2n,h,w)->#(b,t-1,c,2n,h,w)
    #     pad_h, pad_w = k_n-h_p%k_n, k_n-h_p%k_n
    #     x_seq = F.pad(x_seq.reshape(-1, c, 2*n,  h_p, w_p), pad=(0, pad_w, 0, pad_h, 0, 0, 0, 0), mode='constant')
    #     x_hst = F.pad(x_hst.reshape(-1, t-1, n,  h_p, w_p), pad=(0, pad_w, 0, pad_h, 0, 0, 0, 0), mode='constant') #(16,4,49,259,259)
    #     x_hst = x_hst.reshape(b, c, t-1, n, h_p+pad_h, w_p+pad_w) #
    #     x_seq = self.tfoc_predict(x_seq)# (b*t-1,c,n,h/search_size,w/search_size)
    #     _, x_blcidx = x_seq.max(dim=2)# (b*t-1,c,h/search_size,w/search_size)
    #     x_blcidx = x_blcidx.reshape(b, t-1, c, x_seq.shape[-2], x_seq.shape[-1]).permute(0, 2, 1, 3, 4) # (b,t-1,c,h/search_size,w/search_size)->(b,c,t-1,h/search_size,w/search_size)
    #     x_blcidx = x_blcidx.reshape(b, c, t-1, -1) #(b,c,t-1,h/search_size*w/search_size)
    #     # using x_blcidx as index, pick out suitable block from x_hst
    #     index_sbb = torch.arange(x_blcidx.size(dim=-1), device=x_blcidx.device)
    #     index_sbb = index_sbb.unsqueeze(0).unsqueeze(0).unsqueeze(0).unsqueeze(-1)
    #     index_sbb = index_sbb.repeat(b,c,t-1,1,1)
    #     x_blcidx = torch.cat((x_blcidx.unsqueeze(-1), index_sbb), dim=-1) #(b,c,t-1, h/search_size*w/search_size, 2)
    #     h_new, w_new = int((h_p+pad_h)/k_n), int((w_p+pad_w)/k_n)
    #     x_hst = x_hst.reshape(b, c, t-1, n, h_new, k_n, w_new, k_n).permute(0, 1, 2, 3, 4, 6, 5, 7).reshape(b, c, t-1, n, -1, k_n, k_n) #(b, c, t-1, n, -1, k_n, k_n)
    #     x_blc_val = x_hst[x_blcidx, :, :]#(b, c, t-1,  h/search_size*w/search_size,  k_n, k_n)
    #     x_blc_val = x_blc_val.view(b, c, t-1,  h_new, w_new, k_n, k_n).permute(0, 1, 2, 3, 5, 4, 6).reshape(b, c, t-1,  h_new,  k_n, -1)
    #     x_blc_val = x_blc_val.permute(0, 1, 2, 4, 3, 5).reshape(b, c, t-1, -1, w_new*k_n) #(b, c, t-1, h, w)
    #     x_tfoc = torch.cat((x_blc_val, x_tgt_ori), dim=2) #(b,c,t,h,w)
    #     x_tfoc = self.tfoc_conv(x_tfoc) #(b,c,t,h,w)
    #     x_tfc = self.temporal_conv(x_tfoc)  # [b, c, t-4, h, w]
    #
    #     return x_tfc, probs


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

    def __init__(self, signal_type, n_frames, search_size, chunk_size):
        super().__init__()
        self.n_frames = n_frames
        self.signal_type = signal_type
        self.search_size = search_size
        self.chunk_size = chunk_size
        self.double_3d_conv_block = Double3DConvBlock(in_ch=1,
                                                      out_ch=64,
                                                      k_size=(3,3,3),
                                                      strd=(1,1,1),
                                                      pad=(0,1,1),
                                                      btnk_sc=1)
        self.tfoc_conv_block = TFocConv(in_ch=64,
                                        out_ch=128,
                                        btnk_ch=8, #out_ch/btnk_sc
                                        search_size=self.search_size,
                                        chunk_size=self.chunk_size)
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

    def __init__(self, n_classes, n_frames, search_size, chunk_size, zero_doppler_filter=True):
        super().__init__()
        self.n_classes = n_classes
        self.n_frames = n_frames
        self.zero_doppler_filter = zero_doppler_filter
        self.search_size = search_size
        self.chunk_size = chunk_size

        # Backbone (encoding)
        self.rd_encoding_branch = EncodingBranch('range_doppler', self.n_frames, self.search_size, self.chunk_size)
        self.ra_encoding_branch = EncodingBranch('range_angle', self.n_frames, self.search_size, self.chunk_size)
        self.ad_encoding_branch = EncodingBranch('angle_doppler', self.n_frames, self.search_size, self.chunk_size)

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

