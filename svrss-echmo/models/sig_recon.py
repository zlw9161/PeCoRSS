import torch
import torch.nn as nn
import torch.nn.functional as F

class ComplexReLU(nn.Module):
    def __init__(self, mode='split'):
        super().__init__()
        self.mode = mode
        if mode == 'modrelu':
            self.bias = nn.Parameter(torch.zeros(1))  
        elif mode == 'crelu':
            self.threshold = nn.Parameter(torch.tensor(0.5))
    
    def forward(self, z):
        if self.mode == 'modrelu':
            # 对复数模长施加ReLU，同时添加可学习相位偏置
            magnitude = torch.abs(z)
            phase = torch.angle(z) + self.bias
            return F.relu(magnitude) * torch.exp(1j * phase)
        
        elif self.mode == 'crelu':
            # 基于可学习阈值过滤低能量信号
            mask = (torch.abs(z) > self.threshold).float()
            return z * mask
        
        elif self.mode == 'split':
            # 分离式ReLU（独立处理实虚部）
            return torch.complex(F.relu(z.real), F.relu(z.imag))

class Radar_Encoder(nn.Module):
    def __init__(self, in_ch, out_ch, bias=True):
        super().__init__()
        self.stride = 1
        self.in_ch = in_ch
        self.out_ch = out_ch
        self.bias = bias
        self.lmf_real = nn.Conv2d(self.in_ch, self.out_ch,
                                  kernel_size=(1, 3), stride=(1, 1),
                                  bias=self.bias)
        self.lmf_imag = nn.Conv2d(self.in_ch, self.out_ch,
                                  kernel_size=(1, 3), stride=(1, 1),
                                  bias=self.bias)
        # self.batch_norm = ComplexBatchNorm2d(num_features=self.out_ch)
        self.cmplx_relu = ComplexReLU()

    # input: (B, C, numRange, numPulse) --> (B, C, numRange + k_size - 1, numPulse)
    def forward(self, x):
        x_RR = self.lmf_real(x.real)
        x_II = self.lmf_imag(x.imag)
        x_RI = self.lmf_imag(x.real)
        x_IR = self.lmf_real(x.imag)
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
    
class Radar_Decoder(nn.Module):
    def __init__(self, in_ch, out_ch):
        super().__init__()
        self.k_size = 3
        self.tconv_real = nn.ConvTranspose2d(in_ch, out_ch, kernel_size=(1,3),
                                           stride=(1,1))
        self.tconv_imag = nn.ConvTranspose2d(in_ch, out_ch, kernel_size=(1,3),
                                           stride=(1,1))
        # self.norm = ComplexBatchNorm(out_ch)
        self.act = ComplexReLU()

    def forward(self, x):
        x_RR = self.tconv_real(x.real)
        x_II = self.tconv_imag(x.imag)
        x_RI = self.tconv_imag(x.real)
        x_IR = self.tconv_real(x.imag)

        x_output = x_RR - x_II + 1j * (x_RI + x_IR)

        return x_output


class Sig_Recon(nn.Module):
    def __init__(self):
        super().__init__()
        self.encoding_branch = Radar_Encoder(1, 128, bias=True)
        self.decoding_branch = Radar_Decoder(128, 1)
        self._init_weights()
        
    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, (nn.Conv2d, nn.ConvTranspose2d)):
                # 复数卷积核实虚部分别初始化
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, ComplexReLU):
                # 自定义激活参数初始化
                if m.mode == 'modrelu':
                    nn.init.uniform_(m.bias, -0.5, 0.5)

    def forward(self, x):
        x2 = self.encoding_branch(x)
        x3 = self.decoding_branch(x2)
        return x3


if __name__ == '__main__':
    data = torch.randn(2,1,128,1024)
    imaginary_part = torch.zeros_like(data)
    complex_data = torch.complex(data, imaginary_part)
    model = Sig_Recon()
    pre = model(complex_data)
    print(pre)