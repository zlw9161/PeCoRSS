#### zlw@20221229 ####
#### MVANet w/ TFoc Conv ####
#### ASPP: using TFoc output as input
#### LSE: using latent features(TFoc + 1x1Conv) as input
import torch
import torch.nn as nn
import torch.nn.functional as F
import math
import os
#os.environ['CUDA_LAUNCH_BLOCKING'] = '1'

# os.environ['CUDA_VISIBLE_DEVICES'] = '0,1'

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
    """ (3D conv => BN => LeakyReLU) * 2 (temporal shared 2D Conv)"""

    def __init__(self, in_ch, out_ch, k_size, strd, pad, dil):
        super().__init__()

        self.block = nn.Sequential(
            nn.Conv3d(in_ch, out_ch, kernel_size=k_size, stride=strd, padding=pad, dilation=dil),
            nn.BatchNorm3d(out_ch),
            nn.LeakyReLU(inplace=True),
            nn.Conv3d(out_ch, out_ch, kernel_size=k_size, stride=strd, padding=pad, dilation=dil),
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
        self.block1 = nn.Conv2d(in_ch, out_ch, kernel_size=k_size, padding=pad, dilation=dil)
        self.block2 =  nn.BatchNorm2d(out_ch)
        self.block3 = nn.LeakyReLU(inplace=True)


    def forward(self, x):
        x = self.block1(x)
        x = self.block2(x)

        #print('std:', self.block2.running_var.sum())
        x = self.block3(x)
        return x, self.block2.running_mean

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
        x2,_  = self.single_conv_block1_1x1(x)
        x3,_  = self.single_conv_block1_3x3(x)
        x4 ,_ = self.single_conv_block2_3x3(x)
        x5,_  = self.single_conv_block3_3x3(x)
        x_cat = torch.cat((x2, x3, x4, x5, x1), 1)
        return x_cat


class TFocConv(nn.Module):
    """Temporal Focusing Conv Block w/ single historic frame
    input: x_seq = Conv3D(x) BxCx2xHxW tensor
    output: x_tfc BxCxHxW tensor
    """

    def __init__(self, in_ch, out_ch, search_size, n_frames, dist_type):
        super(TFocConv, self).__init__()
        self.k_n = search_size
        self.pad_size = math.floor(search_size/2)
        self.dist_type = dist_type
        self.n_frames = n_frames
        self.tfoc_padding = nn.ZeroPad2d((self.pad_size, self.pad_size, self.pad_size, self.pad_size))
        self.tfoc_conv_1 = TREBlock(in_ch, out_ch,
                                    k_size=(self.n_frames, 3, 3),
                                    strd=(1, 1, 1),
                                    pad=(0, 1, 1))
        self.tfoc_conv_2 = ConvBlock(in_ch, out_ch,
                                     k_size=3, pad=1, dil=1)

    # distance matrix
    def _compute_dist(self, x_tgt, x_hst): ##(b,t-1,c,w*h,n)
        if self.dist_type == 'l2':
            x_diff = x_tgt - x_hst
            mat_dist = torch.norm(x_diff, p='fro', dim=2) #(b,t-1,w*h,n)
        elif self.dist_type == 'l1':
            x_diff = x_tgt - x_hst
            mat_dist = torch.norm(x_diff, p=1, dim=2)
        elif self.dist_type == 'cos':
            mat_dist = -nn.functional.cosine_similarity(x_tgt, x_hst, dim=2, eps=1e-6)
        return mat_dist

    def forward(self, x_seq):
        b, c, t, h, w = x_seq.shape[0], x_seq.shape[1], x_seq.shape[2], x_seq.shape[3], x_seq.shape[4]
        k_n = self.k_n
        n = k_n * k_n
        dtype = x_seq.data.type()




        # 1 extract x_tgt, repeat t-1 times in time dim and padding
        x_tgt_ori = x_seq[:, :, -1, :, :] # (b,c,h,w)
        x_tgt_ori = torch.unsqueeze(x_tgt_ori, 2)  # (b,c,1,h,w)
        x_tgt = x_tgt_ori.repeat(1, 1, t - 1, 1, 1)  # (b,c,t-1,h,w)
        x_tgt = self.tfoc_padding(x_tgt) #(b,c,t-1,h_p,w_p)



        # 2 extrac x_hst
        x_hst = self.tfoc_padding(x_seq[:, :, :t-1, :, :]) #(b,c,t-1,h,w)

        # 5 generate center points in target maps coordinates
        p_tgt = self._get_grid_center(b, h, w, n, dtype)  # cpu int64
        # 6 sample center points in target map x_tgt (b, c, t-1, h, w, N) using p_tgt
        x_tgt_smp = self._sample_x(x_tgt, p_tgt, n)  # (b,c,t-1,h,w,n)



        # 3 generate search point in history maps  coordinates
        p_hst = self._gen_global_sample_grid(b, h, w, n, dtype) #(b,2n,h,w) b, h, w, n, cpu int64
        # 4 sample search point  in history maps x_hst(b, c, t-1,h, w, N) using p_hst
        x_hst_smp = self._sample_x(x_hst, p_hst, n) #(b,c,t-1,h,w,n) #cuda0 florat32




        # 7 computing distance matrix
        x_hst_trans = x_hst_smp.permute(0,2,1,3,4,5) #(b,t-1,c,h,w,n)
        x_hst_trans = x_hst_trans.view(b, t-1, c, -1, n) #(b,t-1,c,h*w,n) calculate distance between vectors of dim (xx,xx,c,h*w,n), get(xx,xx,h*w,n）then argmax on dim n to get (xx,xx,h*w)
        x_tgt_trans = x_tgt_smp.permute(0, 2, 1, 3, 4, 5)  # (b,t-1,c,h,w,n)
        x_tgt_trans = x_tgt_trans.view(b, t - 1, c, -1, n) #(b,t-1,c,h*w,n)
        mat_dist = self._compute_dist(x_hst_trans, x_tgt_trans)  #(b,t-1,h*w,n)


        # 8 generating resampling index
        index_resample = torch.argmin(mat_dist, dim=3) #(b,t-1,h*w)
        location_reference = torch.arange(0, index_resample.size(-1), 1) * n #(h*w)
        location_reference = location_reference.unsqueeze(dim=0).repeat(t-1, 1) #(t-1, h*w)
        location_reference = location_reference.unsqueeze(dim=0).repeat(b, 1, 1)  # (b, t-1, h*w)
        index_resample = index_resample + location_reference.cuda() #(b,t-1,h*w)
        index_resample = index_resample.contiguous().unsqueeze(dim = 2).repeat(1, 1, c, 1) # (b,t-1,c,h*w)

        #6 resample x_hst from x_hst_exp
        x_hst_exp = x_hst_trans.contiguous().view(b,t-1,c,-1)  #(b,t-1,c,h*w*n)
        x_resample = torch.gather(x_hst_exp, dim=-1, index=index_resample).contiguous().view(b, t-1, c, h, -1) #(b,t-1,c,h,w)
        x_resample = x_resample.contiguous().permute(0, 2, 1, 3, 4) #(b,c,t-1,h,w)

        #7 TFocConv
        x_tfc = self.tfoc_conv_1(torch.cat((x_resample, x_tgt_ori), dim=2)) #(b,c',1,h,w) [2,128,1,128,128]
        x_tfc = torch.squeeze(x_tfc, 2) #(b,c,h,w)
        x_tfc, _ = self.tfoc_conv_2(x_tfc)
        return x_tfc

    # getting center grid from the padded grid of input x_tgt
    def _get_grid_center(self, b, h, w, n, dtype):
        # generating grid for center point
        k_ls = (self.k_n-1) // 2
        gc_x, gc_y = torch.meshgrid(
            torch.arange(k_ls, h+k_ls, 1),
            torch.arange(k_ls, w+k_ls, 1))
        gc_x = torch.flatten(gc_x).view(1, 1, h, w).repeat(1, n, 1, 1)
        gc_y = torch.flatten(gc_y).view(1, 1, h, w).repeat(1, n, 1, 1)
        grid_c = torch.cat([gc_x, gc_y], 1).type(dtype) # (1,2n,h,w)
        grid_c = grid_c.repeat(b, 1, 1, 1) # (b,2n,h,w)
        return grid_c

    # getting local search grid: (1,2n,1,1)
    def _get_grid_local_search(self, n, dtype):
        k_ls = (self.k_n-1) // 2
        gl_x, gl_y = torch.meshgrid(
            torch.arange(-k_ls, k_ls+1),
            torch.arange(-k_ls, k_ls+1))
        grid_ls = torch.cat([torch.flatten(gl_x), torch.flatten(gl_y)], 0) # (2n, 1)
        grid_ls = grid_ls.view(1, 2*n, 1, 1).type(dtype)
        return grid_ls

    def _gen_global_sample_grid(self, b, h, w, n, dtype):
        grid_c = self._get_grid_center(b, h, w, n, dtype) # (b,2n,h,w)
        grid_ls = self._get_grid_local_search(n, dtype) # (1,2n,1,1)
        # print(grid_c.device, grid_ls.device)
        grid_gs = grid_c + grid_ls # (b,2n,h,w)
        return grid_gs

    # expanding hitoric map using grid_gs
    def _sample_x(self, x_pad, p, N):
        # p:(b,2N,h,w)  x_pad:(b,c,t-1,h_p,w_p)
        p = p.contiguous().permute(0, 2, 3, 1) # (b,h,w,2n)
        b, h, w, _ = p.size()
        h_pad = x_pad.size(3)
        w_pad = x_pad.size(4)
        c = x_pad.size(1)
        t = x_pad.size(2)
        # (b, c, t-1, h_pad*w_pad)
        # strech each spatial channel of x_padas 1-D vector
        x_pad = x_pad.contiguous().view(b, c, t, -1) #cuda0 float32
        # (b, h_pad, w_pad, n)
        # transform spatial coord of p into the 1-D index
        index = p[..., :N]*w_pad + p[..., N:] #cpu,int64
        # index_x = torch.clamp(p[..., :N], 0, h_pad-1)
        # index_y = torch.clamp(p[..., N:], 0, w_pad-1)
        # index = index_x * w_pad + index_y
        # (b, c, t-1, h*w*N)
        index = index.contiguous().unsqueeze(dim=1).expand(-1, t, -1, -1, -1)
        index = index.contiguous().unsqueeze(dim=1).expand(-1, c, -1, -1, -1, -1)
        index = index.contiguous().view(b, c, t, -1)
        index = index.long().cuda() #cuda0 int64 [b,128,t-1,h*w*N]
        x_r = x_pad.gather(dim=-1, index=index)
        x_r = x_r.contiguous().view(b, c, t, h, w, N)
        return x_r #cuda0 float32

    # resampling expanded historic map with resampling index
    #def _resample_x(self, x_exp, idx, n):



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
        self.double_3dconv_block1 = Double3DConvBlock(in_ch=1, out_ch=128,
                                                      k_size=(1, 3, 3),
                                                      strd=(1, 1, 1),
                                                      pad=(0, 1, 1), dil=1)
        self.doppler_max_pool3d = nn.MaxPool3d(kernel_size=(1, 2, 2),
                                               stride=(1, 2, 1))
        self.max_pool3d = nn.MaxPool3d(kernel_size=(1, 2, 2),
                                       stride=(1, 2, 2))
        self.tfoc_conv_block = TFocConv(in_ch=128, out_ch=128,
                                        search_size=9,
                                        dist_type='l2',
                                        n_frames=self.n_frames)
        self.doppler_max_pool = nn.MaxPool2d(2, stride=(2, 1))
        self.max_pool = nn.MaxPool2d(2, stride=2)
        self.single_conv_block1_1x1 = ConvBlock(in_ch=128, out_ch=128, k_size=1, pad=0, dil=1)

    def forward(self, x):
        print('x:', x.size(), '\t')
        x1 = self.double_3dconv_block1(x) #[2,128,3,256,256]

        if self.signal_type in ('range_doppler', 'angle_doppler'):
            # The Doppler dimension requires a specific processing
            # padding the last two dimentions of the tensor
            x1_pad = F.pad(x1, (0, 1, 0, 0), "constant", 0) # F.pad(top, bottom, left, right)
            x1_down = self.doppler_max_pool3d(x1_pad)
        else:
            x1_down = self.max_pool3d(x1) #[2,128,3,128,128]


        x2 = self.tfoc_conv_block(x1_down) #[2,128,128,128]
        if self.signal_type in ('range_doppler', 'angle_doppler'):
            # The Doppler dimension requires a specific processing
            x2_pad = F.pad(x2, (0, 1, 0, 0), "constant", 0)
            x2_down = self.doppler_max_pool(x2_pad)
        else:
            x2_down = self.max_pool(x2) #[2,128,64,64]

        x3, bn_print  = self.single_conv_block1_1x1(x2_down) #[2,128,64,64]
        print('########')
        print('bn_print', bn_print, x3[0,0,0,0])
        print('>>>>>>>>')
        # return input of TAM (x2_down) & ASPP block (x3) & Latent space (x3)
        return x2_down, x3


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

    def __init__(self, n_classes, n_frames):
        super().__init__()
        self.n_classes = n_classes
        self.n_frames = n_frames

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

    def forward(self, x_rd, x_ra, x_ad):
        # Backbone
        ra_features, ra_latent = self.ra_encoding_branch(x_ra)
        rd_features, rd_latent = self.rd_encoding_branch(x_rd)
        ad_features, ad_latent = self.ad_encoding_branch(x_ad)

        # ASPP blocks
        x1_rd = self.rd_aspp_block(rd_features)
        x1_ra = self.ra_aspp_block(ra_features)
        x1_ad = self.ad_aspp_block(ad_features)
        x2_rd, _ = self.rd_single_conv_block1_1x1(x1_rd)
        x2_ra, _ = self.ra_single_conv_block1_1x1(x1_ra)
        x2_ad, _ = self.ad_single_conv_block1_1x1(x1_ad)

        # Latent Space
        # Features join either the RD or the RA branch
        x3 = torch.cat((rd_latent, ra_latent, ad_latent), 1)
        x3_rd, _ = self.rd_single_conv_block2_1x1(x3)
        x3_ra, _ = self.ra_single_conv_block2_1x1(x3)

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

        return x9_rd, x9_ra
