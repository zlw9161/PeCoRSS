import torch
import torch.nn as nn
import torch.nn.functional as F

class ComplexInterpolate_phase_mag(nn.Module):
     # 幅相分离处理（保持相位连续性）
    def __init__(self, phase_mode='nearest'):
        super().__init__()
        self.phase_mode = phase_mode  # 相位插值模式

    def forward(self, x, target_size):
       
        magnitude = torch.abs(x)
        phase = torch.angle(x)
        
        mag_interp = F.interpolate(magnitude, size=target_size, 
                                 mode='bilinear', align_corners=False)
        phase_interp = F.interpolate(phase, size=target_size, 
                                   mode=self.phase_mode)
        
        return mag_interp * torch.exp(1j * phase_interp)
    

class ComplexInterpolate_real_img(nn.Module):
     # 实虚部分离处理（减少计算资源）
    def __init__(self, phase_mode='nearest'):
        super().__init__()
        self.phase_mode = phase_mode  # 相位插值模式

    def forward(self, x, target_size=(256,1024)):
       
        real = F.interpolate(x.real, size=target_size, mode='bilinear', align_corners=False)
        imag = F.interpolate(x.imag, size=target_size, mode= self.phase_mode)

        return torch.complex(real, imag)


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
        
class ComplexBatchNorm2d(nn.Module):
    def __init__(self, num_features):
        super().__init__()
        self.bn_real = nn.BatchNorm2d(num_features)
        self.bn_imag = nn.BatchNorm2d(num_features)

    def forward(self, x):
        real = self.bn_real(x.real)
        imag = self.bn_imag(x.imag)
        return torch.complex(real, imag)
    
class ComplexConv2d(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, padding=0):
        super().__init__()
        # 实部卷积分支
        self.conv_r = nn.Conv2d(in_channels, out_channels, kernel_size, padding=padding)
        # 虚部卷积分支
        self.conv_i = nn.Conv2d(in_channels, out_channels, kernel_size, padding=padding)

    def forward(self, x):
        real = self.conv_r(x.real) - self.conv_i(x.imag)
        imag = self.conv_r(x.imag) + self.conv_i(x.real)
        return torch.complex(real, imag)

class Radar_Encoder(nn.Module):
    def __init__(self, in_ch, out_ch, target_size=(256,1024)):
        super().__init__()
        self.stride = 1
        self.in_ch = in_ch
        self.out_ch = out_ch
        self.target_size = target_size

        # self.lmf_real = nn.Conv2d(self.in_ch, self.out_ch,
        #                           kernel_size=(1, 3), stride=(1, 1),
        #                           bias=True)
        # self.lmf_imag = nn.Conv2d(self.in_ch, self.out_ch,
        #                           kernel_size=(1, 3), stride=(1, 1),
        #                           bias=True)
        self.lmf_real = nn.Conv2d(self.in_ch, self.out_ch,
                                  kernel_size=(3, 3), stride=(1, 1),padding=(1,1),
                                  bias=True)
        self.lmf_imag = nn.Conv2d(self.in_ch, self.out_ch,
                                  kernel_size=(3, 3), stride=(1, 1),padding=(1,1),
                                  bias=True)
        self.AdaptiveAvgPool=nn.AdaptiveAvgPool2d(target_size)
        # # self.interpolate = ComplexInterpolate_real_img()
        # self.interpolate = ComplexInterpolate_phase_mag()
        # self.phase_align = nn.Conv2d(out_ch, out_ch, 3, padding=1)  # 相位对齐卷积

        # self.batch_norm = ComplexBatchNorm2d(num_features=self.out_ch)
        self.cmplx_relu = ComplexReLU()

    # input: (B, C, numRange, numPulse) --> (B, C, numRange + k_size - 1, numPulse)
    def forward(self, x):
        self.original_size = x.shape[2:] 

        x_RR = self.lmf_real(x.real)
        x_II = self.lmf_imag(x.imag)
        x_RI = self.lmf_imag(x.real)
        x_IR = self.lmf_real(x.imag)
        x = x_RR - x_II + 1j * (x_RI + x_IR)

        # x = self.interpolate(x, target_size = self.target_size)
        # x = self.phase_align(x.real) + 1j * self.phase_align(x.imag)  # 相位对齐
        
        x_r = self.AdaptiveAvgPool(x.real)
        x_i = self.AdaptiveAvgPool(x.imag)

        x = torch.complex(x_r,x_i)

        # x = self.batch_norm(x)
        x = self.cmplx_relu(x)
        return x
    
class Radar_Decoder(nn.Module):
    def __init__(self, in_ch, out_ch,encoder):
        super().__init__()
        self.encoder_ref = encoder 
        # self.tconv_real = nn.ConvTranspose2d(in_ch, out_ch, kernel_size=(1,3),
        #                                    stride=(1,1),bias=True)
        # self.tconv_imag = nn.ConvTranspose2d(in_ch, out_ch, kernel_size=(1,3),
        #                                    stride=(1,1),bias=True)
        
        # self.tconv_real = nn.ConvTranspose2d(in_ch, out_ch, kernel_size=(3,3), 
        #                                 stride=(1,1), padding=(1,1), bias=True)
        # self.tconv_imag = nn.ConvTranspose2d(in_ch, out_ch, kernel_size=(3,3), 
        #                                 stride=(1,1), padding=(1,1), bias=True)
        
        # 高度保持 128，宽度从 256 → 1024
        self.tconv_real = nn.ConvTranspose2d(
            in_channels=in_ch,
            out_channels=out_ch,
            kernel_size=(3, 4),      # H方向用3，W方向用4
            stride=(1, 4),           # H步幅1（不放大），W步幅4（4倍放大）
            padding=(1, 0),          # H填充1，W不填充
            output_padding=(0, 0),   # 无补偿
            bias=True
        )
        self.tconv_imag = nn.ConvTranspose2d(
            in_ch, out_ch,
            kernel_size=(3, 4),
            stride=(1, 4),
            padding=(1, 0),
            output_padding=(0, 0),
            bias=True
        )
        # # self.norm = ComplexBatchNorm(out_ch)
        # self.act = ComplexReLU()
        
        # # 特征精修层
        # self.refine_conv = nn.Sequential(
        #     ComplexBatchNorm2d(1),
        #     ComplexReLU(),
        #     ComplexConv2d(1, 1, 3, padding=1)  # 替换为复数卷积
        # )
        # 动态插值模块
        # self.interpolate = ComplexInterpolate_phase_mag()

    def forward(self, x):

        target_size =  self.encoder_ref.original_size

        x_RR = self.tconv_real(x.real)
        x_II = self.tconv_imag(x.imag)
        x_RI = self.tconv_imag(x.real)
        x_IR = self.tconv_real(x.imag)

        x_cat = x_RR - x_II + 1j * (x_RI + x_IR)

        x_output = x_cat

        # x_refine = self.refine_conv(x_cat)
        # x_output = self.interpolate(x_refine, target_size=target_size)

        return x_output


class Sig_Recon(nn.Module):
    def __init__(self):
        super().__init__()
        self.encoding_branch = Radar_Encoder(in_ch=1, out_ch=32, target_size=(128,256))
        self.decoding_branch = Radar_Decoder(in_ch=32, out_ch=1, encoder=self.encoding_branch)
        self._init_weights()

    def forward(self, x):
        x2 = self.encoding_branch(x)
        x3 = self.decoding_branch(x2)
        return x3
    
    def _init_weights(self):
        for m in self.modules():
            # 编码器初始化
            if isinstance(m, Radar_Encoder):
                # 实部卷积核：Kaiming初始化
                nn.init.kaiming_normal_(m.lmf_real.weight, mode='fan_out', nonlinearity='relu')
                # 虚部卷积核：正交初始化
                nn.init.orthogonal_(m.lmf_imag.weight)
                if m.lmf_real.bias is not None:
                    nn.init.constant_(m.lmf_real.bias, 0)
                    nn.init.constant_(m.lmf_imag.bias, 0.1)  # 虚部偏置小量初始化
            
            # 解码器初始化
            elif isinstance(m, Radar_Decoder):
                # 转置卷积实部：Kaiming初始化
                nn.init.kaiming_normal_(m.tconv_real.weight, mode='fan_in', nonlinearity='relu')
                # 转置卷积虚部：正交初始化
                nn.init.orthogonal_(m.tconv_imag.weight)
                if m.tconv_real.bias is not None:
                    nn.init.constant_(m.tconv_real.bias, 0)
                    nn.init.constant_(m.tconv_imag.bias, 0.1)
            
            # 标准实数卷积层初始化
            elif isinstance(m, (nn.Conv2d, nn.ConvTranspose2d)):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            
            # 复数激活参数初始化（改进相位偏移范围）
            elif isinstance(m, ComplexReLU):
                if m.mode == 'modrelu':
                    nn.init.uniform_(m.bias, -0.25, 0.25)  # 缩小初始相位偏移范围
                elif m.mode == 'crelu':
                    nn.init.constant_(m.threshold, 0.707)  # 按复数模长平均值初始化


if __name__ == '__main__':
    data = torch.randn(2,1,128,1024)
    imaginary_part = torch.zeros_like(data)
    complex_data = torch.complex(data, imaginary_part)
    model = Sig_Recon()
    pre = model(complex_data)
    print(pre)