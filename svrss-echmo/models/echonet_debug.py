# End-to-end RSS network
# zlw@20240810
# reference model: LMFNetV4.py zlw@20240416
import torch
import torch.nn as nn
import torch.nn.functional as F
from model_utils import ComplexBatchNorm2d, ComplexReLU, ComplexConv2d
from model_utils.echo_tools import complex_std, complex_double2float, filter_coef_loader
import os


class Leanable_Match_FilterV2(nn.Module):
    def __init__(self, k_size, mf_coef, activation='relu', bias=True):
        super(Leanable_Match_FilterV2, self).__init__()
        self.k_size = k_size
        self.stride = 1
        self.in_ch = 1
        self.out_ch = 1
        self.mf = mf_coef.view(self.in_ch, self.out_ch, 1, self.k_size)
        self.activation = activation
        self.bias = bias
        self.kernel_real, self.kernel_imag, self.bias_real, self.bias_imag = self.init_kernels()

        # causule conv only padding one side
        self.cause_padding = torch.nn.ZeroPad2d(padding=(0, self.k_size-1, 0, 0))
        self.batch_norm = ComplexBatchNorm2d(num_features=self.out_ch)
        self.cmplx_relu = ComplexReLU(self.activation)

    # weight initialized w/ mfTime, bias (optional) 
    def init_kernels(self):
        # real_kernels = torch.randn(1, self.k_size, 1)
        # imag_kernels = torch.randn(1, self.k_size, 1)
        # real_kernels = self.mf.real
        # imag_kernels = self.mf.imag
        real_kernel = self.mf.real
        imag_kernel = self.mf.imag
        real_kernel = nn.Parameter(real_kernel)
        imag_kernel = nn.Parameter(imag_kernel)
        # print('kernel in LMF:', real_kernel.shape, real_kernel.dtype)
        if self.bias:
            real_bias = nn.Parameter(torch.randn(self.out_ch).type(torch.float32))
            imag_bias = nn.Parameter(torch.randn(self.out_ch).type(torch.float32))
        else:
            real_bias = None
            imag_bias = None
        return real_kernel, imag_kernel, real_bias, imag_bias

    # input: (B, C, numRange, numPulse) --> (B, C, numRange + k_size - 1, numPulse)
    def forward(self, x):
        print('x in LMF:', x.shape, x.dtype)
        inputR = self.cause_padding(x.real)
        inputI = self.cause_padding(x.imag)
        lmf_RR = F.conv2d(inputR,
                          self.kernel_real,
                          self.bias_real,
                          stride=(self.stride, self.stride),
                          padding=0)
        lmf_II = F.conv2d(inputI,
                          self.kernel_imag,
                          self.bias_imag,
                          stride=(self.stride, self.stride),
                          padding=0)
        lmf_RI = F.conv2d(inputR,
                          self.kernel_imag,
                          self.bias_imag,
                          stride=(self.stride, self.stride),
                          padding=0)
        lmf_IR = F.conv2d(inputI,
                          self.kernel_real,
                          self.bias_real,
                          stride=(self.stride, self.stride),
                          padding=0)
        lmf_res = lmf_RR - lmf_II + 1j * (lmf_RI + lmf_IR)

        import matplotlib.pyplot as plt
        import numpy as np
        fig_lmf = plt.figure('LMFV2 [activation:none, bias:False]', figsize=(6, 12))
        lmf_vis = lmf_res[0,0,:,:].detach().numpy()
        x_echo = x[0,0,:,:].detach().numpy()
        grid_x, grid_y = np.arange(lmf_vis.shape[1]), np.arange(lmf_vis.shape[0])
        grid_x, grid_y = np.meshgrid(grid_x, grid_y)
        ax1 = fig_lmf.add_subplot(211, projection='3d')
        ax1.plot_surface(grid_x, grid_y, abs(x_echo), cmap='viridis')
        ax1.set_title('input echo amp.')
        ax2 = fig_lmf.add_subplot(212, projection='3d')
        ax2.plot_surface(grid_x, grid_y, abs(lmf_vis), cmap='viridis')
        ax2.set_title('lmf output')
        # for i in range(128):
        #     print('lmf_res_max_coord_', i, np.argmax(abs(lmf_vis)[i,:]))

        lmf_res = self.batch_norm(lmf_res)
        lmf_res = self.cmplx_relu(lmf_res)
        return lmf_res


class Leanable_Coherent_Integration(nn.Module):
    # learnable MTD or FFT
    '''
    X(f) = \sum_{t=0}^{T-1} x(t)(cos(2*\pi*f*t/T) - j*sin(2*\pi*f*t/T))
    out_ch denotes the number of Doppler channels (f in the Equation)
    k_size denotes the number of pulses (t in the Equation)
    the kernel in Conv: out_ch * in_ch * k_h * k_w
    then the kernel for fourier basis: doppler * in_ch * n_pulse * 1
    '''
    def __init__(self, in_ch, out_ch, k_size, activation='relu', bias=True):
        super(Leanable_Coherent_Integration, self).__init__()
        self.k_size = k_size
        self.stride = (k_size, 1)
        self.in_ch = in_ch
        self.out_ch = out_ch
        self.activation = activation
        self.bias = bias
        # self.doppler_padding = torch.nn.ZeroPad2d(padding=(self.k_size-n_pulse, 0, 0, 0))
        self.cos_kernel, self.sin_kernel, self.cos_bias, self.sin_bias = self.init_fourier_kernels()
        # self.batch_norm = ComplexBatchNorm2d(num_features=self.out_ch)
        self.cmplx_relu = ComplexReLU(self.activation)

    # weight initialized w/ mfTime, bias (optional) 
    def init_fourier_kernels(self):
        # broadcasting to in_ch, out_ch
        # cos_kernel = torch.randn(self.in_ch, self.out_ch, self.k_size, 1)
        # sin_kernel = torch.randn(self.in_ch, self.out_ch, self.k_size, 1)
        coefs = torch.zeros(self.out_ch, self.k_size)
        f, t = torch.meshgrid(torch.arange(1, self.out_ch + 1),
                              torch.arange(1, self.k_size + 1), indexing='ij')
        coefs = 2 * torch.pi * f * (t / self.k_size)
        # for f in range(self.out_ch):
        #     for t in range(self.k_size):
        #         coefs[f,t] = 2*torch.pi*f*(t/self.k_size)
        cos_kernel = torch.broadcast_to(torch.cos(coefs).unsqueeze(dim=1),
                                        (self.out_ch, self.in_ch, self.k_size)).unsqueeze(dim=3)
        sin_kernel = torch.broadcast_to(torch.sin(coefs).unsqueeze(dim=1),
                                        (self.out_ch, self.in_ch, self.k_size)).unsqueeze(dim=3)
        cos_kernel = nn.Parameter(cos_kernel)
        sin_kernel = nn.Parameter(sin_kernel)
        print('cos_kernel:', cos_kernel.shape)
        print('sin_kernel:', sin_kernel.shape)
        if self.bias:
            cos_bias = nn.Parameter(torch.randn(self.out_ch).type(torch.float32))
            sin_bias = nn.Parameter(torch.randn(self.out_ch).type(torch.float32))
        else:
            cos_bias = None
            sin_bias = None
        return cos_kernel, sin_kernel, cos_bias, sin_bias

    # input: (B, C, numPulse, numRange) --> (B, C, numDoppler, numRange)
    def forward(self, x):
        # inputR = self.doppler_padding(x.real)
        # inputI = self.doppler_padding(x.imag)
        inputR, inputI = x.real, x.imag
        lci_RR = F.conv2d(inputR,
                          self.cos_kernel,
                          self.cos_bias,
                          stride=self.stride,
                          padding=0)
        lci_II = F.conv2d(inputI,
                          self.sin_kernel,
                          self.sin_bias,
                          stride=self.stride,
                          padding=0)
        lci_RI = F.conv2d(inputR,
                          self.sin_kernel,
                          self.sin_bias,
                          stride=self.stride,
                          padding=0)
        lci_IR = F.conv2d(inputI,
                          self.cos_kernel,
                          self.cos_bias,
                          stride=self.stride,
                          padding=0)
        lci_res = lci_RR + lci_II - 1j * lci_RI + 1j * lci_IR
        # lci_R = F.conv2d(inputR,
        #                  self.cos_kernel,
        #                  self.cos_bias,
        #                  stride=self.stride,
        #                  padding=0)
        # lci_I = F.conv2d(inputI,
        #                  self.sin_kernel,
        #                  self.sin_bias,
        #                  stride=self.stride,
        #                  padding=0)
        # lci_res = lci_R + 1j * lci_I
        # lci_res = self.batch_norm(lci_res)
        lci_res = self.cmplx_relu(lci_res)
        # folding the doppler dim into the h_size
        lci_res = lci_res.permute(0, 2, 1, 3)
        return lci_res

class LMF_Chain(nn.Module):
    def __init__(self, configs):
        super(LMF_Chain, self).__init__()
        self.configs = configs
        self.mf_coef = filter_coef_loader(os.path.join(configs.root_path, configs.data_path),
                                          configs.coef_type)
        # self.mf_coef = torch.tensor(self.mf_coef.copy()).to()
        self.Chain = nn.Sequential()
        for l in range(configs.e_layers):
            self.Chain.add_module('lmfV2-{0}'.format(l),
                                  Leanable_Match_FilterV2(k_size=self.mf_coef.shape[1], 
                                                          mf_coef=self.mf_coef,
                                                          activation=configs.activation,
                                                          bias=configs.bias))
    
    def forward(self, x):
        o = self.Chain(x)
        return o


class EchoNet(nn.Module):
    def __init__(self, configs):
        super(EchoNet, self).__init__()
        self.configs = configs
        self.task_name = configs.task_name
        self.activation = configs.activation
        self.bias = configs.bias
        self.n_chains = configs.n_chains
        self.n_channels = configs.n_channels
        self.n_pulse = configs.n_pulse
        self.n_doppler = configs.n_doppler
        self.loss = configs.loss
        self.scale = configs.scale
        self.LMFChain = nn.ModuleList([LMF_Chain(configs=self.configs)
                                      for _ in range(self.n_chains)])
        self.LCI = Leanable_Coherent_Integration(in_ch=self.n_chains,
                                                 out_ch=self.n_doppler,
                                                 k_size=self.n_pulse,
                                                 activation=self.activation,
                                                 bias=self.bias)
        self.Decoder = nn.Sequential(            
            ComplexConv2d(in_channels=self.n_chains, 
                          out_channels=self.n_channels, 
                          k_size=(1, 1), 
                          bias=True, 
                          init_weight=True),
            ComplexBatchNorm2d(num_features=self.n_channels),
            ComplexReLU(self.activation),
            ComplexConv2d(in_channels=self.n_channels, 
                          out_channels=1, 
                          k_size=(1, 1), 
                          bias=True, 
                          init_weight=True),
            ComplexBatchNorm2d(num_features=1),
            ComplexReLU(self.activation),
        )
        
    def forward(self, x):
        # e = torch.randn([x.shape[0],
        #                  self.n_chains,
        #                  x.shape[2],
        #                  x.shape[3]], requires_grad=True, dtype=x.dtype).cuda(self.gpu)
        if self.scale == 'instance':
            x = complex_std(x, epsilon=1e-5)
        elif self.scale == 'global':
            None

        for i in range(self.n_chains):
            if i == 0:
                e = self.LMFChain[i](x)
            else:
                e_i = self.LMFChain[i](x)
                e = torch.concatenate([e, e_i], dim=1)

        # MTI w/o learning
        e = torch.cat((e[:,:,0,:].unsqueeze(dim=2), e[:,:,1:,:] - e[:,:,:-1,:]), dim=2)

        import matplotlib.pyplot as plt
        import numpy as np
        fig_net = plt.figure('MTI-LCI [activation:none, bias:False]', figsize=(6, 12))
        mti_vis = e[0,0,:,:].detach().numpy()
        grid_x, grid_y = np.arange(mti_vis.shape[1]), np.arange(mti_vis.shape[0])
        grid_x, grid_y = np.meshgrid(grid_x, grid_y)
        ax1 = fig_net.add_subplot(211, projection='3d')
        ax1.plot_surface(grid_x, grid_y, abs(mti_vis), cmap='viridis')
        ax1.set_title('mti amp. in echonet')

        # Learnable MTD, a.k.a LCI
        # e_fft = torch.fft.fft(e, dim=2)
        e = self.LCI(e)

        mtd_vis = e[0,0,:,:].detach().numpy()
        grid_x, grid_y = np.arange(mtd_vis.shape[1]), np.arange(mtd_vis.shape[0])
        grid_x, grid_y = np.meshgrid(grid_x, grid_y)
        ax2 = fig_net.add_subplot(212, projection='3d')
        ax2.plot_surface(grid_x, grid_y, abs(mtd_vis), cmap='viridis')
        ax2.set_title('learnable mtd amp.')

        d = self.Decoder(e)
        # zlw@20240401 torch.sqrt(x.real**2 + x.imag**2) == torch.abs(x)
        # outpout amplititude and phase
        if self.loss == 'MSE':
            o = torch.sqrt(d.real**2 + d.imag**2) # B, 1, P, R
        else:
            o_amp = torch.abs(d) # B, 1, P, R
            o_pha = torch.angle(d) # B, 1, P, R
            o = torch.concat([o_amp, o_pha], dim=1) # B, 2, P, R
        return o


if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--task_name', default='Segmentation', 
                        help='Options: LMF; MTD; Segmentation.', type=str)
    parser.add_argument('--n_chains', default=1,
                        help='The number of LMF chains (or heads).', type=int)
    parser.add_argument('--n_channels', default=8,
                        help='The default output channels of the whole model.', type=int)
    parser.add_argument('--e_layers', default=1,
                        help='The encoding layers (chain depth) of the LMFNet.', type=int)
    parser.add_argument('--scale', default='instance', 
                        help='normalization type, normalize with instance-wise stats or global stats', type=str)
    parser.add_argument('--coef_type', default='tay_mf',
                        help='MF coef type for LMF kernel initialization', type=str)
    parser.add_argument('--n_pulse', default=128,
                        help='Number of input pulses for one CPI', type=int)
    parser.add_argument('--n_doppler', default=128,
                        help='Number of Doppler bins for one CPI', type=int)
    parser.add_argument('--loss', default='MSE', 
                        help='Loss for the LMFNet, need to be discard later.', type=str)
    parser.add_argument('--activation', default=None, help='Activation type for echonet.', type=str)
    parser.add_argument('--bias', default=False, help='bias switch for echonet.', type=bool)
    parser.add_argument('--data_path', default='coefs', help='data path', type=str)
    parser.add_argument('--root_path', default='/media/liwenzhang/Data/DataSets/NuaaSim', help='root path', type=str)
    
    configs = parser.parse_args()
    print(configs)
    
    ## input echos construction
    # B = 2
    # C = 1
    # R = 100
    # x_real = torch.randn([B, C, configs.n_pulse, R], dtype=torch.float32)
    # x_imag = torch.randn([B, C, configs.n_pulse, R], dtype=torch.float32)
    # x = torch.complex(x_real, x_imag)
    # print(x.shape)

    # load inputs
    import numpy as np
    seq = 'npy/data_1000m_O_18dB'
    eco = 'range_doppler_echo/000025.npy'
    dpc = 'range_doppler_dpc/000025.npy'
    mti = 'range_doppler_mti/000025.npy'
    mtd = 'range_doppler_mtd/000025.npy'
    x_eco = torch.tensor(np.load(os.path.join(configs.root_path, seq, eco))) # [128, 1024] complex128
    x_dpc = torch.tensor(np.load(os.path.join(configs.root_path, seq, dpc))) # [128, 1024] complex128
    x_mti = torch.tensor(np.load(os.path.join(configs.root_path, seq, mti))) # [128, 1024] complex128
    x_mtd = torch.tensor(np.load(os.path.join(configs.root_path, seq, mtd))) # [128, 1024] float64
    x_mtd = x_mtd[0,:,:] + 1j*x_mtd[1,:,:]
    x_eco = complex_double2float(x_eco)
    x_dpc = complex_double2float(x_dpc)
    x_mti = complex_double2float(x_mti)
    x_mtd = complex_double2float(x_mtd)
    print('|', 'eco', x_eco.dtype, '|',
          'dpc', x_dpc.dtype, '|',
          'mti', x_mti.dtype, '|',
          'mtd', x_mtd.dtype, '|')
    print('|', 'eco', x_eco.shape, '|',
          'dpc', x_dpc.shape, '|',
          'mti', x_mti.shape, '|',
          'mtd', x_mtd.shape, '|')
    B, C = 1, 1
    x = x_eco.view(B, C, x_eco.shape[0], x_eco.shape[1])
    print('input shape:', x.shape)

    # model init
    model = EchoNet(configs=configs)
    # model running
    o = model(x)
    print('output shape:', o.shape)

    # Reference Visualization
    import matplotlib.pyplot as plt
    s_ori = filter_coef_loader('/media/liwenzhang/Data/DataSets/NuaaSim/coefs', None) # [1, 163]
    s_win = filter_coef_loader('/media/liwenzhang/Data/DataSets/NuaaSim/coefs', 'tay_mf') # [1, 163]

    fig_ref = plt.figure('Reference Signals', figsize=(24, 12))

    plt.subplot(2, 3, 1)
    plt.plot(range(s_ori.shape[1]), s_ori.real[0, :], 'b.-')
    plt.plot(range(s_ori.shape[1]), s_ori.imag[0, :], 'r*-')
    plt.plot(range(s_ori.shape[1]), abs(s_ori)[0, :], 'g.-')
    plt.legend(['real', 'imag', 'amp.'])
    plt.title('ref sig')
    plt.subplot(2, 3, 2)
    plt.plot(range(s_win.shape[1]), s_win.real[0, :], 'b.-')
    plt.plot(range(s_win.shape[1]), s_win.imag[0, :], 'r*-')
    plt.plot(range(s_win.shape[1]), abs(s_win)[0, :], 'g.-')
    plt.legend(['real', 'imag', 'amp.'])
    plt.title('ref sig w/ taylor window')

    grid_x, grid_y = np.arange(x_eco.shape[1]), np.arange(x_eco.shape[0])
    grid_x, grid_y = np.meshgrid(grid_x, grid_y)
    ax1 = fig_ref.add_subplot(233, projection='3d')
    ax1.plot_surface(grid_x, grid_y, abs(x_eco), cmap='viridis')
    ax1.set_title('ref echo amp.')
    ax2 = fig_ref.add_subplot(234, projection='3d')
    ax2.plot_surface(grid_x, grid_y, abs(x_dpc), cmap='viridis')
    ax2.set_title('ref dpc amp.')
    # for i in range(128):
    #     print('x_dpc_max_coord_', i,  torch.argmax(abs(x_dpc)[i,:]))

    ax3 = fig_ref.add_subplot(235, projection='3d')
    ax3.plot_surface(grid_x, grid_y, abs(x_mti), cmap='viridis')
    ax3.set_title('ref mti amp.')
    ax4 = fig_ref.add_subplot(236, projection='3d')
    ax4.plot_surface(grid_x, grid_y, abs(x_mtd), cmap='viridis')
    ax4.set_title('ref mtd amp.')
    plt.show()
    