# End-to-end RSS network
# Echo in Mask out: EchMo zlw@20250317
# Echo Coherent Semantic Segmantation ECoSS zlw@20240810
# Reference Model: LMFNetV4.py zlw@20240416
import torch
import torch.nn as nn
import torch.nn.functional as F
from svrss.models.model_utils.complex_layers import ComplexBatchNorm2d, ComplexReLU
from svrss.models.model_utils.echo_tools import complex_std, complex_double2float, filter_coef_loader
from svrss.models import PKCIn4ANY_cuda
import os


def _init_translator_weights(m):
    if isinstance(m, nn.Linear):
        torch.nn.init.xavier_uniform_(m.weight)
        nn.init.constant_(m.bias, 0.)
    elif isinstance(m, nn.Conv2d):
        torch.nn.init.xavier_uniform_(m.weight)
        if m.bias is not None:
            nn.init.constant_(m.bias, 0.)
        elif isinstance(m, nn.BatchNorm2d):
            nn.init.uniform_(m.weight, 0., 1.)
            nn.init.constant_(m.bias, 0.)

class Leanable_Match_FilterV2(nn.Module):
    def __init__(self, k_size, in_ch, out_ch, lmf_coef, activation='relu', bias=True):
        super(Leanable_Match_FilterV2, self).__init__()
        self.k_size = k_size
        self.stride = 1
        self.in_ch = in_ch
        self.out_ch = out_ch
        self.mf = lmf_coef.view(1, 1, 1, self.k_size)
        self.activation = activation
        self.bias = bias
        self.kernel_real, self.kernel_imag, self.bias_real, self.bias_imag = self.init_kernels()

        # causule conv only padding one side
        self.cause_padding = torch.nn.ZeroPad2d(padding=(self.k_size - 1, 0, 0, 0))
        self.batch_norm = ComplexBatchNorm2d(num_features=self.out_ch)
        self.cmplx_relu = ComplexReLU(self.activation)

    # weight initialized w/ mfTime, bias (optional)
    def init_kernels(self):
        real_kernel = self.mf.real
        imag_kernel = self.mf.imag
        real_kernel = nn.Parameter(real_kernel.broadcast_to(self.out_ch, self.in_ch, 1, self.k_size))
        imag_kernel = nn.Parameter(imag_kernel.broadcast_to(self.out_ch, self.in_ch, 1, self.k_size))
        if self.bias:
            real_bias = nn.Parameter(torch.randn(self.out_ch).type(torch.float32))
            imag_bias = nn.Parameter(torch.randn(self.out_ch).type(torch.float32))
        else:
            real_bias = None
            imag_bias = None
        return real_kernel, imag_kernel, real_bias, imag_bias

    # input: (B, C, numRange, numPulse) --> (B, C, numRange + k_size - 1, numPulse)
    def forward(self, x):
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
        fig_lmf = plt.figure('LMFV5 [activation:none, bias:False]', figsize=(6, 12))
        lmf_vis = lmf_res.abs()[0,0,:,:].detach().numpy()
        x_echo = x[0,0,:,:].detach().numpy()
        grid_x, grid_y = np.arange(lmf_vis.shape[1]), np.arange(lmf_vis.shape[0])
        grid_x, grid_y = np.meshgrid(grid_x, grid_y)
        ax1 = fig_lmf.add_subplot(211, projection='3d')
        ax1.plot_surface(grid_x, grid_y, abs(x_echo), cmap='viridis')
        ax1.set_title('input echo amp.')
        ax2 = fig_lmf.add_subplot(212, projection='3d')
        ax2.plot_surface(grid_x, grid_y, abs(lmf_vis), cmap='viridis')
        ax2.set_title('lmf output')

        lmf_res = self.batch_norm(lmf_res)
        lmf_res = self.cmplx_relu(lmf_res)
        return lmf_res


class Leanable_Match_FilterV5(nn.Module):
    def __init__(self, k_size, in_ch, out_ch, lmf_coef, activation='relu', bias=True):
        super(Leanable_Match_FilterV5, self).__init__()
        self.k_size = k_size
        self.stride = 1
        self.in_ch = in_ch
        self.out_ch = out_ch
        self.mf = lmf_coef.view(1, 1, 1, self.k_size)
        self.activation = activation
        self.bias = bias
        kernel_real, kernel_imag = self.init_kernels()
        self.kernel_real = nn.Parameter(kernel_real)
        self.kernel_imag = nn.Parameter(kernel_imag)
        if self.bias:
            self.bias_real = nn.Parameter(torch.zeros(self.out_ch, dtype=torch.float32))
            self.bias_imag = nn.Parameter(torch.zeros(self.out_ch, dtype=torch.float32))
        else:
            self.register_parameter('bias_real', None)
            self.register_parameter('bias_imag', None)

        # causule conv only padding one side
        self.cause_padding = torch.nn.ZeroPad2d(padding=(0, self.k_size - 1, 0, 0))
        # self.batch_norm = ComplexBatchNorm2d(num_features=self.out_ch)
        self.cmplx_relu = ComplexReLU(self.activation)

    # weight initialized w/ mfTime, bias (optional)
    def init_kernels(self):
        real_kernel = self.mf.real
        imag_kernel = self.mf.imag
        real_kernel = real_kernel.broadcast_to(self.out_ch, self.in_ch, 1, self.k_size)
        imag_kernel = imag_kernel.broadcast_to(self.out_ch, self.in_ch, 1, self.k_size)
        return real_kernel, imag_kernel

    # input: (B, C, numRange, numPulse) --> (B, C, numRange + k_size - 1, numPulse)
    def forward(self, x):
        # x_echo = x
        inputR = self.cause_padding(x.real)
        inputI = self.cause_padding(x.imag)
        x_RR = F.conv2d(inputR, self.kernel_real, self.bias_real, stride=(1, 1), padding=0)
        x_II = F.conv2d(inputI, self.kernel_imag, self.bias_imag, stride=(1, 1), padding=0)
        x_RI = F.conv2d(inputR, self.kernel_imag, self.bias_imag, stride=(1, 1), padding=0)
        x_IR = F.conv2d(inputI, self.kernel_real, self.bias_real, stride=(1, 1), padding=0)
        x = x_RR - x_II + 1j * (x_RI + x_IR)

        # import matplotlib.pyplot as plt
        # import numpy as np
        # fig_lmf = plt.figure('LMFV5 [activation:none, bias:False]', figsize=(6, 12))
        # lmf_vis = x[0,0,:,:].detach().numpy()
        # x_echo = x_echo[0,0,:,:].detach().numpy()
        # grid_x, grid_y = np.arange(lmf_vis.shape[1]), np.arange(lmf_vis.shape[0])
        # grid_x, grid_y = np.meshgrid(grid_x, grid_y)
        # ax1 = fig_lmf.add_subplot(211, projection='3d')
        # ax1.plot_surface(grid_x, grid_y, abs(x_echo), cmap='viridis')
        # ax1.set_title('input echo amp.')
        # ax2 = fig_lmf.add_subplot(212, projection='3d')
        # ax2.plot_surface(grid_x, grid_y, abs(lmf_vis), cmap='viridis')
        # ax2.set_title('lmf output')

        # x = self.batch_norm(x)
        x = self.cmplx_relu(x)
        return x


class LMF_Chain(nn.Module):
    def __init__(self, configs):
        super(LMF_Chain, self).__init__()
        self.lmf_coef = filter_coef_loader(os.path.join(configs.dataset_path, configs.coef_path),
                                          configs.lmf_coefs)
        self.in_ch = 1
        # self.mf_coef = torch.tensor(self.mf_coef.copy()).to()
        self.Chain = nn.Sequential()
        for l in range(configs.lmf_layers):
            self.Chain.add_module('lmfV5-{0}'.format(l),
                                  Leanable_Match_FilterV5(k_size=self.lmf_coef.shape[1],
                                                          in_ch=self.in_ch,
                                                          out_ch=configs.lmf_channels,
                                                          lmf_coef=self.lmf_coef,
                                                          activation=configs.activation,
                                                          bias=configs.bias))
            self.in_ch = configs.lmf_channels

    def forward(self, x):
        x = self.Chain(x)
        return x



class Multi_Order_LDC(nn.Module):
    # Multi-order Learnable Delay Canceler (MLDC)
    def __init__(self, in_ch, out_ch, orders, activation='relu', bias=True):
        super(Multi_Order_LDC, self).__init__()
        self.in_ch = in_ch
        self.out_ch = out_ch
        self.orders = orders
        self.activation = activation
        self.bias = bias
        self.hp_coefs = self.get_highpass_coefs()
        self.hp_kernels = self.init_highpass_kernels()
        self.pad_list = nn.ModuleList()
        if self.bias:
            self.hp_biases = nn.ParameterList([
                nn.Parameter(torch.zeros(self.out_ch, dtype=torch.float32))
                for _ in range(self.orders)
            ])
        else:
            self.hp_biases = None
        for n in range(self.orders):
            pad = torch.nn.ZeroPad2d(padding=(0, 0, n+1, 0))
            self.pad_list.append(pad)
        # self.batch_norm = ComplexBatchNorm2d(num_features=self.orders*self.out_ch)
        self.cmplx_relu = ComplexReLU(self.activation)

    # LDC kernel initialized w/ high-pass filters, bias (optional)
    def get_highpass_coefs(self):
        import math
        hp_coefs = []
        for j in range(self.orders):
            n = j + 1
            hp_coefs_n = torch.zeros(n+1, 1)
            for i in range(n+1):
                if i == 0:
                    hp_coefs_n[i] = 1
                else:
                    # (-1)^i \cdot \frac{n!}{i!(n-i)!}
                    hp_coefs_n[i] = (-1)**(i) * math.factorial(n) // (math.factorial(i) * math.factorial(n-i))
            hp_coefs.append(hp_coefs_n)
        return hp_coefs
        
    def init_highpass_kernels(self):
        hp_kernels = nn.ParameterList()
        for n in range(len(self.hp_coefs)):
            hp_kernels_n = self.hp_coefs[n].unsqueeze(dim=0).unsqueeze(dim=0).broadcast_to(self.out_ch,
                                                                                           self.in_ch,
                                                                                           n+2,
                                                                                           1)
            hp_kernels.append(nn.Parameter(hp_kernels_n.clone().contiguous()))
        return hp_kernels
    
    def forward(self, x):
        # input: \hat{X}: (B, C_K, P, R)
        # output: \tilde{X}: (B, (orders+1)*C_H, P, R)
        B, P, R = x.shape[0], x.shape[2], x.shape[3]
        x = x.unsqueeze(dim=1).broadcast_to(B, self.orders, 
                                            x.shape[1], P, R)
        # 20250807
        # y_r = torch.zeros(B, self.orders, self.out_ch, P, R).cuda()
        # y_i = torch.zeros(B, self.orders, self.out_ch, P, R).cuda()
        y_r = torch.zeros(B, self.orders, self.out_ch, P, R).to(x.device)
        y_i = torch.zeros(B, self.orders, self.out_ch, P, R).to(x.device)
        for n in range(self.orders):
            bias = self.hp_biases[n] if self.bias else None
            x_n_real = F.conv2d(self.pad_list[n](x[:, n, ...].real),
                                self.hp_kernels[n],
                                bias=bias,
                                stride=(1, 1),
                                padding=0)
            x_n_imag = F.conv2d(self.pad_list[n](x[:, n, ...].imag),
                                self.hp_kernels[n],
                                bias=bias,
                                stride=(1, 1),
                                padding=0)
            y_r[:,n,...] = x_n_real
            y_i[:,n,...] = x_n_imag # x_n_real + 1j * x_n_imag

            # import matplotlib.pyplot as plt
            # import numpy as np
            # fig_ldc = plt.figure('LDC-'+str(n+1), figsize=(6, 12))
            # ldc_vis = x_n_real[0,0,:,:].detach().numpy()
            # ldc_input = x[0,n,0,...].real.detach().numpy()
            # grid_x, grid_y = np.arange(ldc_vis.shape[1]), np.arange(ldc_vis.shape[0])
            # grid_x, grid_y = np.meshgrid(grid_x, grid_y)
            # ax1 = fig_ldc.add_subplot(211, projection='3d')
            # print('ldc_input:', ldc_input.shape)
            # ax1.plot_surface(grid_x, grid_y, ldc_input, cmap='viridis')
            # ax1.set_title('ldc input.')
            # ax2 = fig_ldc.add_subplot(212, projection='3d')
            # ax2.plot_surface(grid_x, grid_y, ldc_vis, cmap='viridis')
            # ax2.set_title('ldc output')
        
        y_r = y_r.view(B, self.orders*self.out_ch, P, R)
        y_i = y_i.view(B, self.orders*self.out_ch, P, R)
        y = torch.complex(y_r, y_i)
        # y = self.batch_norm(y)
        y = self.cmplx_relu(y)
        return y
              


class Group_Purturble_DF(nn.Module):
    # Group-purturble Doppler filters (GDF)
    '''
    X(f) = \sum_{t=0}^{T-1} x(t)(cos(2*\pi*f*t/T) - j*sin(2*\pi*f*t/T))
    out_ch denotes the number of Doppler channels (f in the Equation)
    k_size denotes the number of pulses (t in the Equation)
    the kernel in Conv: out_ch * in_ch * k_h * k_w
    then the kernel for fourier basis: doppler * in_ch * n_pulse * 1
    '''
    def __init__(self, in_ch, out_ch, k_size, groups, activation='relu', bias=True):
        super(Group_Purturble_DF, self).__init__()
        self.k_size = k_size
        self.in_ch = in_ch
        self.out_ch = out_ch
        self.groups = groups
        self.activation = activation
        self.bias = bias
        self.stride = (k_size, 1)
        # self.doppler_padding = torch.nn.ZeroPad2d(padding=(self.k_size-n_pulse, 0, 0, 0))
        cos_kernel, sin_kernel = self.init_fourier_kernels()
        self.cos_kernel = nn.Parameter(cos_kernel)
        self.sin_kernel = nn.Parameter(sin_kernel)
        if self.bias:
            self.cos_bias = nn.Parameter(torch.zeros(self.out_ch * self.groups, dtype=torch.float32))
            self.sin_bias = nn.Parameter(torch.zeros(self.out_ch * self.groups, dtype=torch.float32))
        else:
            self.register_parameter('cos_bias', None)
            self.register_parameter('sin_bias', None)
        # self.batch_norm = ComplexBatchNorm2d(num_features=self.out_ch*self.groups)
        self.cmplx_relu = ComplexReLU(self.activation)

    # weight initialized w/ mfTime, bias (optional)
    def init_fourier_kernels(self):
        coefs = torch.zeros(self.out_ch, self.k_size)
        f, t = torch.meshgrid(torch.arange(1, self.out_ch + 1),
                              torch.arange(1, self.k_size + 1), indexing='ij')
        coefs = 2 * torch.pi * f * (t / self.k_size)
        # fit each conv
        # (D, P) --> (D, 1, P) --> (D, C_in, P) --> (D, C_in, P, 1)
        cos_kernel = torch.broadcast_to(torch.cos(coefs).unsqueeze(dim=1),
                                        (self.out_ch, self.in_ch, self.k_size)).unsqueeze(dim=3)
        sin_kernel = torch.broadcast_to(torch.sin(coefs).unsqueeze(dim=1),
                                        (self.out_ch, self.in_ch, self.k_size)).unsqueeze(dim=3)
        cos_kernel = torch.broadcast_to(cos_kernel.unsqueeze(dim=0),
                                        (self.groups, self.out_ch, self.in_ch, self.k_size, 1))
        sin_kernel = torch.broadcast_to(sin_kernel.unsqueeze(dim=0),
                                        (self.groups, self.out_ch, self.in_ch, self.k_size, 1))
        cos_kernel = cos_kernel.reshape(self.out_ch*self.groups, self.in_ch, self.k_size, 1)
        sin_kernel = sin_kernel.reshape(self.out_ch*self.groups, self.in_ch, self.k_size, 1)
        return cos_kernel, sin_kernel

    # input: (B, C, numPulse, numRange) --> (B, C, numDoppler, numRange)
    def forward(self, x):
        # input: \tilde{X}: (B, groups*C_H, P, R)
        # output: \dot{X}: (B, groups*C_D, 1, R) --> (B, groups, C_D, R)
        B, R = x.shape[0], x.shape[3]
        # x_real, x_imag = x.real.cuda(), x.imag.cuda()
        # 20250807
        x_real, x_imag = x.real.to(x.device), x.imag.to(x.device)
        # x_real, x_imag = x.real, x.imag
        x_rr = F.conv2d(x_real, self.cos_kernel, self.cos_bias, stride=self.stride, padding=0, groups=self.groups)
        x_ii = F.conv2d(x_imag, self.sin_kernel, self.sin_bias, stride=self.stride, padding=0, groups=self.groups)
        x_ri = F.conv2d(x_real, self.sin_kernel, self.sin_bias, stride=self.stride, padding=0, groups=self.groups)
        x_ir = F.conv2d(x_imag, self.cos_kernel, self.cos_bias, stride=self.stride, padding=0, groups=self.groups)
        x = x_rr + x_ii - 1j * x_ri + 1j * x_ir
        # x = self.batch_norm(x)
        x = self.cmplx_relu(x)
        # folding the doppler dim into the h_size
        x = x.view(B, self.groups, self.out_ch, 1, R).squeeze(dim=3)
        return x
    


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
        coefs = torch.zeros(self.out_ch, self.k_size)
        f, t = torch.meshgrid(torch.arange(1, self.out_ch + 1),
                              torch.arange(1, self.k_size + 1), indexing='ij')
        coefs = 2 * torch.pi * f * (t / self.k_size)
        cos_kernel = torch.broadcast_to(torch.cos(coefs).unsqueeze(dim=1),
                                        (self.out_ch, self.in_ch, self.k_size)).unsqueeze(dim=3)
        sin_kernel = torch.broadcast_to(torch.sin(coefs).unsqueeze(dim=1),
                                        (self.out_ch, self.in_ch, self.k_size)).unsqueeze(dim=3)
        cos_kernel = nn.Parameter(cos_kernel)
        sin_kernel = nn.Parameter(sin_kernel)
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
        # lci_res = self.batch_norm(lci_res)
        lci_res = self.cmplx_relu(lci_res)
        # folding the doppler dim into the h_size
        lci_res = lci_res.permute(0, 2, 1, 3)
        return lci_res


class EchMo(nn.Module):
    def __init__(self, configs):
        super(EchMo, self).__init__()
        self.cfg = configs
        self.task_name = configs.task_name
        self.n_range = configs.n_range
        self.n_pulse = configs.n_pulse
        self.n_classes = configs.n_classes
        self.n_frames = configs.n_frames
        self.scale = configs.scale
        # net global cfg
        self.activation = configs.activation
        self.bias = configs.bias
        # lmf cfg
        self.lmf_chains = configs.lmf_chains
        self.lmf_channels = configs.lmf_channels
        self.lmf_layers = configs.lmf_layers
        self.lmf_coefs = configs.lmf_coefs
        # mldc cfg
        self.ldc_orders = configs.ldc_orders
        self.ldc_channels = configs.ldc_channels
        # gdf cfg
        self.df_groups = configs.ldc_orders
        self.df_channels = configs.df_channels
        
        self.LMFChain = nn.ModuleList([LMF_Chain(configs=self.cfg)
                                       for _ in range(self.lmf_chains)])
        
        self.MLDC = Multi_Order_LDC(in_ch=self.lmf_channels*self.lmf_chains,
                                    out_ch=self.ldc_channels,
                                    orders=self.ldc_orders,
                                    activation=self.activation,
                                    bias=self.bias)
        self.GDF = Group_Purturble_DF(in_ch=self.ldc_channels,
                                      out_ch=self.df_channels,
                                      k_size=self.n_pulse,
                                      groups=self.df_groups,
                                      activation=self.activation,
                                      bias=self.bias)
        # for amp+phase+I/Q                              
        self.RDTranslator = PKCIn4ANY_cuda(n_classes=self.n_classes,
                                           n_frames=self.n_frames,
                                           in_ch=self.df_groups,
                                           w=self.df_channels,
                                           h=self.n_range)
        self.RDTranslator.apply(_init_translator_weights)

       
    def forward(self, x):
        if self.scale == 'instance':
            x = complex_double2float(complex_std(x, epsilon=1e-8))
        elif self.scale == 'global':
            x = complex_double2float(x)


        for i in range(self.lmf_chains):
            if i == 0:
                x_k = self.LMFChain[i](x)
            else:
                x_ki = self.LMFChain[i](x)
                x_k = torch.concatenate([x_k, x_ki], dim=1)
        # # MTI w/o learning
        # x_h = torch.cat((x_k[:, :, 0, :].unsqueeze(dim=2), x_k[:, :, 1:, :] - x_k[:, :, :-1, :]), dim=2)
        # print('x_h', x_h.shape)
        ## MTD w/o learning
        # x_d = torch.fft.fft(x_h, dim=2)

        # Learnable MTI and MTD
        x_h = self.MLDC(x_k)
        # x_d = self.LCI(x_h)
        x_d = self.GDF(x_h)
        
        
        # import matplotlib.pyplot as plt
        # import numpy as np
        # fig_gdf = plt.figure('gdf', figsize=(6, 12))
        # gdf_vis = torch.abs(x_d)[0,0,:,:].detach().numpy()
        # gdf_input = x_h[0,0,...].r eal.detach().numpy()
        # grid_x, grid_y = np.arange(gdf_vis.shape[1]), np.arange(gdf_vis.shape[0])
        # grid_x, grid_y = np.meshgrid(grid_x, grid_y)
        # ax1 = fig_gdf.add_subplot(211, projection='3d')
        # print('gdf_input:', gdf_input.shape)
        # ax1.plot_surface(grid_x, grid_y, gdf_input, cmap='viridis')
        # ax1.set_title('ldc input.')
        # ax2 = fig_gdf.add_subplot(212, projection='3d')
        # ax2.plot_surface(grid_x, grid_y, gdf_vis, cmap='viridis')
        # ax2.set_title('gdf output (avg)')

        # convert e to real, and add temporal dim
        x_d = torch.abs(x_d).unsqueeze(dim=2) # amp-only
        # x_d = torch.cat((x_d.real, x_d.imag), dim=1).unsqueeze(dim=2) # I/Q
        # x_d = torch.cat((torch.angle(x_d), torch.abs(x_d)), dim=1).unsqueeze(dim=2) # phase+amp (pham)
        # x_d = torch.cat((torch.abs(x_d), torch.angle(x_d), x_d.real, x_d.imag), dim=1).unsqueeze(dim=2) # amp+phase+I+Q
        y = self.RDTranslator(x_d)
        return y


if __name__ == '__main__':
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument('--task_name', default='Segmentation',
                        help='Options: LMF; MTD; Segmentation.', type=str)
    parser.add_argument('--coef_path', default='coefs', help='data path', type=str)
    parser.add_argument('--dataset_path', default='/data/PeCoLab/DataSets/NuaaSim', help='dataset path', type=str)
    # data setting
    parser.add_argument('--n_classes', default=2,
                        help='Number of target classes, depend on dataset', type=int)
    parser.add_argument('--n_frames', default=1,
                        help='Number of CPIs', type=int)
    parser.add_argument('--n_range', default=1024,
                        help='Number of input range units for each echo', type=int)
    parser.add_argument('--n_pulse', default=128,
                        help='Number of input pulses for one CPI', type=int)
    parser.add_argument('--scale', default='instance',
                        help='normalization type, normalize with instance-wise stats or global stats', type=str)
    # lmf config
    parser.add_argument('--lmf_chains', default=1,
                        help='The number of LMF chains (or heads).', type=int)
    parser.add_argument('--lmf_channels', default=1,
                        help='The output channels of the LMF.', type=int)
    parser.add_argument('--lmf_layers', default=1,
                        help='The encoding layers (chain depth) of the LMFNet.', type=int)
    parser.add_argument('--lmf_coefs', default='tay_mf',
                        help='MF coef type for LMF kernel initialization', type=str)
    # mldc config
    parser.add_argument('--ldc_channels', default=1,
                        help='Number of output channels for each LDC', type=int)
    parser.add_argument('--ldc_orders', default=3,
                        help='Number of output channels for MLDC', type=int)
    # gdf config
    parser.add_argument('--df_channels', default=128,
                        help='Number of Doppler bins for one CPI', type=int)
    # parser.add_argument('--df_groups', default=4,
    #                     help='Number of groups for GDF, it is equal to the orders of MLDC + 1', type=int)
    # global settings
    parser.add_argument('--activation', default=None, help='Activation type for echonet.', type=str)
    parser.add_argument('--bias', default=False, help='bias switch for echonet.', type=bool)

    configs = parser.parse_args()

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
    x_eco = torch.tensor(np.load(os.path.join(configs.dataset_path, seq, eco)))  # [128, 1024] complex128
    x_dpc = torch.tensor(np.load(os.path.join(configs.dataset_path, seq, dpc)))  # [128, 1024] complex128
    x_mti = torch.tensor(np.load(os.path.join(configs.dataset_path, seq, mti)))  # [128, 1024] complex128
    x_mtd = torch.tensor(np.load(os.path.join(configs.dataset_path, seq, mtd)))  # [128, 1024] float64
    x_mtd = x_mtd[0, :, :] + 1j * x_mtd[1, :, :]
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
    model = EchMo(configs=configs)
    print('Model CFG:', configs)
    for name, module in model.named_modules():
        if isinstance(module, nn.Conv2d) or isinstance(module, nn.Linear) or isinstance(module, nn.Parameter) or isinstance(module, nn.Conv3d):
            param_count = sum(p.numel() for p in module.parameters() if p.requires_grad)
            print(f"Layer: {name}, #Params: {param_count}")
    total_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Total #Params in EchMo: {total_params}")
    from thop import profile
    flops, _ = profile(model, inputs=(x,))
    print(f"EchMo Computing Cosuming (FLOPs): {flops}")
    
    # model running
    o = model(x)
    print('output shape:', o.shape)

    # Reference Visualization
    import matplotlib.pyplot as plt

    s_ori = filter_coef_loader('E:/DataSets/NuaaSim/coefs', None)  # [1, 163]
    s_win = filter_coef_loader('E:/DataSets/NuaaSim/coefs', 'tay_mf')  # [1, 163]

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
