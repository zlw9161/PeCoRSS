# Layers for complex numbers
# zlw@20240221
# Acknowledge to https://github.com/wavefrontshaping/complexPyTorch
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional

class _ComplexBatchNorm(nn.Module):
    running_mean: Optional[torch.Tensor]

    def __init__(
        self,
        num_features,
        eps=1e-5,
        momentum=0.1,
        affine=True,
        track_running_stats=False,
    ):
        super(_ComplexBatchNorm, self).__init__()
        self.num_features = num_features
        self.eps = eps
        self.momentum = momentum
        self.affine = affine
        self.track_running_stats = track_running_stats
        if self.affine:
            self.weight = nn.Parameter(torch.Tensor(num_features, 3))
            self.bias = nn.Parameter(torch.Tensor(num_features, 2))
        else:
            self.register_parameter("weight", None)
            self.register_parameter("bias", None)
        if self.track_running_stats:
            self.register_buffer(
                "running_mean", torch.zeros(num_features, dtype=torch.complex64)
            )
            self.register_buffer("running_covar", torch.zeros(num_features, 3))
            self.running_covar[:, 0] = 1.4142135623730951
            self.running_covar[:, 1] = 1.4142135623730951
            self.register_buffer(
                "num_batches_tracked", torch.tensor(0, dtype=torch.long)
            )
        else:
            self.register_parameter("running_mean", None)
            self.register_parameter("running_covar", None)
            self.register_parameter("num_batches_tracked", None)
        self.reset_parameters()

    def reset_running_stats(self):
        if self.track_running_stats:
            self.running_mean.zero_()
            self.running_covar.zero_()
            self.running_covar[:, 0] = 1.4142135623730951
            self.running_covar[:, 1] = 1.4142135623730951
            self.num_batches_tracked.zero_()

    def reset_parameters(self):
        self.reset_running_stats()
        if self.affine:
            nn.init.constant_(self.weight[:, :2], 1.4142135623730951)
            nn.init.zeros_(self.weight[:, 2])
            nn.init.zeros_(self.bias)
            

class ComplexBatchNorm2d(_ComplexBatchNorm):
    def forward(self, inp):
        exponential_average_factor = 0.0

        if self.training and self.track_running_stats:
            if self.num_batches_tracked is not None:
                self.num_batches_tracked += 1
                if self.momentum is None:  # use cumulative moving average
                    exponential_average_factor = 1.0 / \
                        float(self.num_batches_tracked)
                else:  # use exponential moving average
                    exponential_average_factor = self.momentum

        if self.training or (not self.track_running_stats):
            # calculate mean of real and imaginary part
            # mean does not support automatic differentiation for outputs with complex dtype.
            mean_r = inp.real.mean([0, 2, 3]).type(torch.complex64)
            mean_i = inp.imag.mean([0, 2, 3]).type(torch.complex64)
            mean = mean_r + 1j * mean_i
        else:
            mean = self.running_mean

        if self.training and self.track_running_stats:
            # update running mean
            with torch.no_grad():
                self.running_mean = (
                    exponential_average_factor * mean
                    + (1 - exponential_average_factor) * self.running_mean
                )

        inp = inp - mean[None, :, None, None]

        if self.training or (not self.track_running_stats):
            # Elements of the covariance matrix (biased for train)
            n = inp.numel() / inp.size(1)
            Crr = 1.0 / n * inp.real.pow(2).sum(dim=[0, 2, 3]) + self.eps
            Cii = 1.0 / n * inp.imag.pow(2).sum(dim=[0, 2, 3]) + self.eps
            Cri = (inp.real.mul(inp.imag)).mean(dim=[0, 2, 3])
        else:
            Crr = self.running_covar[:, 0] + self.eps
            Cii = self.running_covar[:, 1] + self.eps
            Cri = self.running_covar[:, 2]  # +self.eps

        if self.training and self.track_running_stats:
            with torch.no_grad():
                self.running_covar[:, 0] = (
                    exponential_average_factor * Crr * n / (n - 1)  #
                    + (1 - exponential_average_factor) * self.running_covar[:, 0]
                )

                self.running_covar[:, 1] = (
                    exponential_average_factor * Cii * n / (n - 1)
                    + (1 - exponential_average_factor) * self.running_covar[:, 1]
                )

                self.running_covar[:, 2] = (
                    exponential_average_factor * Cri * n / (n - 1)
                    + (1 - exponential_average_factor) * self.running_covar[:, 2]
                )

        # calculate the inverse square root the covariance matrix
        det = Crr * Cii - Cri.pow(2)
        s = torch.sqrt(det)
        t = torch.sqrt(Cii + Crr + 2 * s)
        inverse_st = 1.0 / (s * t)
        Rrr = (Cii + s) * inverse_st
        Rii = (Crr + s) * inverse_st
        Rri = -Cri * inverse_st

        inp = (
            Rrr[None, :, None, None] * inp.real + Rri[None, :, None, None] * inp.imag
        ).type(torch.complex64) + 1j * (
            Rii[None, :, None, None] * inp.imag + Rri[None, :, None, None] * inp.real
        ).type(
            torch.complex64
        )

        if self.affine:
            inp = (
                self.weight[None, :, 0, None, None] * inp.real
                + self.weight[None, :, 2, None, None] * inp.imag
                + self.bias[None, :, 0, None, None]
            ).type(torch.complex64) + 1j * (
                self.weight[None, :, 2, None, None] * inp.real
                + self.weight[None, :, 1, None, None] * inp.imag
                + self.bias[None, :, 1, None, None]
            ).type(
                torch.complex64
            )
        return inp


class ComplexBatchNorm1d(_ComplexBatchNorm):
    def forward(self, inp):

        exponential_average_factor = 0.0

        if self.training and self.track_running_stats:
            if self.num_batches_tracked is not None:
                self.num_batches_tracked += 1
                if self.momentum is None:  # use cumulative moving average
                    exponential_average_factor = 1.0 / \
                        float(self.num_batches_tracked)
                else:  # use exponential moving average
                    exponential_average_factor = self.momentum

        if self.training or (not self.track_running_stats):
            # calculate mean of real and imaginary part
            mean_r = inp.real.mean(dim=0).type(torch.complex64)
            mean_i = inp.imag.mean(dim=0).type(torch.complex64)
            mean = mean_r + 1j * mean_i
        else:
            mean = self.running_mean

        if self.training and self.track_running_stats:
            # update running mean
            with torch.no_grad():
                self.running_mean = (
                    exponential_average_factor * mean
                    + (1 - exponential_average_factor) * self.running_mean
                )

        inp = inp - mean[None, ...]

        if self.training or (not self.track_running_stats):
            # Elements of the covariance matrix (biased for train)
            n = inp.numel() / inp.size(1)
            Crr = inp.real.var(dim=0, unbiased=False) + self.eps
            Cii = inp.imag.var(dim=0, unbiased=False) + self.eps
            Cri = (inp.real.mul(inp.imag)).mean(dim=0)
        else:
            Crr = self.running_covar[:, 0] + self.eps
            Cii = self.running_covar[:, 1] + self.eps
            Cri = self.running_covar[:, 2]

        if self.training and self.track_running_stats:
            with torch.no_grad():
                self.running_covar[:, 0] = (
                    exponential_average_factor * Crr * n / (n - 1)
                    + (1 - exponential_average_factor) * self.running_covar[:, 0]
                )

                self.running_covar[:, 1] = (
                    exponential_average_factor * Cii * n / (n - 1)
                    + (1 - exponential_average_factor) * self.running_covar[:, 1]
                )

                self.running_covar[:, 2] = (
                    exponential_average_factor * Cri * n / (n - 1)
                    + (1 - exponential_average_factor) * self.running_covar[:, 2]
                )

        # calculate the inverse square root the covariance matrix
        det = Crr * Cii - Cri.pow(2)
        s = torch.sqrt(det)
        t = torch.sqrt(Cii + Crr + 2 * s)
        inverse_st = 1.0 / (s * t)
        Rrr = (Cii + s) * inverse_st
        Rii = (Crr + s) * inverse_st
        Rri = -Cri * inverse_st

        inp = (Rrr[None, :] * inp.real + Rri[None, :] * inp.imag).type(
            torch.complex64
        ) + 1j * (Rii[None, :] * inp.imag + Rri[None, :] * inp.real).type(
            torch.complex64
        )

        if self.affine:
            inp = (
                self.weight[None, :, 0] * inp.real
                + self.weight[None, :, 2] * inp.imag
                + self.bias[None, :, 0]
            ).type(torch.complex64) + 1j * (
                self.weight[None, :, 2] * inp.real
                + self.weight[None, :, 1] * inp.imag
                + self.bias[None, :, 1]
            ).type(
                torch.complex64
            )

        del Crr, Cri, Cii, Rrr, Rii, Rri, det, s, t
        return inp


class ComplexReLU(nn.Module):
    def __init__(self, activation='leaky_relu'):
        super(ComplexReLU, self).__init__()
        self.activation = activation
    
    def forward(self, input):
        _relu = nn.ReLU()
        if self.activation == 'relu':
            return _relu(input.real) + 1j * _relu(input.imag)
        elif self.activation == 'crelu':
            input = input.unsqueeze(0)
            o_real = torch.cat((_relu(input.real), _relu(-input.real)))
            o_imag = torch.cat((_relu(input.imag), _relu(-input.imag)))
            return o_real + 1j * o_imag
        elif self.activation == 'leaky_relu':
            return F.leaky_relu(input.real) + 1j * F.leaky_relu(input.imag)
        elif self.activation == 'celu':
            return F.celu(input.real) + 1j * F.celu(input.imag)
        else:
            return input


class ComplexConv2d(nn.Module):
    def __init__(self, in_channels, out_channels, k_size, bias, init_weight=True, activation='leaky_relu'):
        super(ComplexConv2d, self).__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = k_size
        self.bias = bias
        self.activation = activation
        self.conv_real = nn.Conv2d(self.in_channels,
                                   self.out_channels,
                                   self.kernel_size,
                                   stride=1,
                                   padding=0,
                                   bias=self.bias)
        self.conv_imag = nn.Conv2d(self.in_channels,
                                   self.out_channels,
                                   self.kernel_size,
                                   stride=1,
                                   padding=0,
                                   bias=self.bias)
        if init_weight:
            self._initialize_weights()

    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                # nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity=self.activation)
                nn.init.xavier_normal_(m.weight, gain=1.)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)

    def forward(self, x):
        o_RR = self.conv_real(x.real)
        o_RI = self.conv_imag(x.real)
        o_IR = self.conv_real(x.imag)
        o_II = self.conv_imag(x.imag)
        return o_RR - o_II + 1j * (o_RI + o_IR)