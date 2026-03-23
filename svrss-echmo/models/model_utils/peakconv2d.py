import torch
import torch.nn as nn
import torch.nn.functional as F
import importlib.util

def is_prf_cuda_install():
    return importlib.util.find_spec('prf_cuda') is not None

prf_cuda_installed = is_prf_cuda_install()

if prf_cuda_installed:
    import prf_cuda
    class PRFFunction(torch.autograd.Function):
        @staticmethod
        def forward(ctx, input, kernel_size, stride):
            output = torch.empty(
                (input.size(0), input.size(1),
                (input.size(2) - kernel_size) // stride + 1,
                (input.size(3) - kernel_size) // stride + 1, 
                16), 
                device=input.device
            )
            b = input.detach()
            output = prf_cuda.prf_forward(b, kernel_size, stride)
            ctx.input_shape = input.shape
            ctx.kernel_size = kernel_size
            ctx.stride = stride
            return output
        
        @staticmethod
        def backward(ctx, grad_output):
            input_shape = ctx.input_shape
            kernel_size = ctx.kernel_size
            stride = ctx.stride
            
            
            grad_input = prf_cuda.prf_backward(
                grad_output,
                input_shape[2],
                input_shape[3],
                grad_output.size(2),
                grad_output.size(3),
                grad_output.size(4),
                kernel_size,
                stride
            )
            return grad_input, None, None
        
    class PRFModule(nn.Module):
        def __init__(self, kernel_size, stride):
            super(PRFModule, self).__init__()
            self.kernel_size = kernel_size
            self.stride = stride
            
        def forward(self, input):
            return PRFFunction.apply(input, self.kernel_size, self.stride)
    
    # reda peak_conv zlw @20220624
    # Peak Conv with fixed guard banwidth (gb) and dense reference receptive field (RRF)
    class PeakConv2D(nn.Module):
        def __init__(self, in_ch, out_ch, bias, refer_band, guard_band_dop, guard_band, signal_type):
            print("use cuda PeakConv2D")
            """
            Args:
                refer_band: a single integer value or integer vector of two values.
                guard_band_dop: guard bandwidth for doppler input
                guard_band: guard banwidth for non-doppler input
                signal_type in ('range_doppler', 'angle_doppler', 'range_angle').
                k_size, padding and stride are computed using the input fixed guard bandwidth
                input: x in the shape of BxCxHxW
            """
            super(PeakConv2D, self).__init__()
            self.signal_type = signal_type
            if self.signal_type in ('range_doppler', 'angle_doppler'):
                # guard_band should be bi-lateral
                self.guard_band = (guard_band_dop, guard_band_dop)
                # padding order: l-r-t-b
                self.padding = (self.guard_band[1] + refer_band,
                                self.guard_band[1] + refer_band,
                                self.guard_band[0] + refer_band,
                                self.guard_band[0] + refer_band)
            else:
                self.guard_band = (guard_band, guard_band)
                # padding order: l-r-t-b
                self.padding = (self.guard_band[1] + refer_band,
                                self.guard_band[1] + refer_band,
                                self.guard_band[0] + refer_band,
                                self.guard_band[0] + refer_band)
            self.refer_band = refer_band
            self.kernel_size = (self.guard_band[0] + self.guard_band[1] + self.refer_band * 2), 4
            self.stride = self.kernel_size
            # padding order: l-r-t-b
            self.zero_padding = nn.ZeroPad2d(self.padding)
            self.bias = bias
            self.pc_strd = 1
            # conv block for peak receptive field (PRF) in x
            # take PRF as input and output the center point output of PRF;
            # kernel channel = in_ch*2: depth_concat(p_c, p_r)
            # since each p_c is expanded as a tensor of kernel scale RF, the stride = kernel_size
            
            # TODO: to be modified to a variable
            self.prf = PRFModule(kernel_size=5, stride=1)
            self.peak_conv = nn.Conv2d(in_ch, out_ch,
                                    kernel_size=self.kernel_size,
                                    padding=0,
                                    stride=self.stride,
                                    bias=self.bias)
            #self.peak_conv.register_backward_hook(self._set_lr)

        @staticmethod
        # set learning rate for peak_conv
        def _set_lr(module, grad_input, grad_output):
            grad_input = (grad_input[i] * 0.1 for i in range(len(grad_input)))
            grad_output = (grad_output[i] * 0.1 for i in range(len(grad_output)))

        def forward(self, x):
            h, w = x.shape[2], x.shape[3]
            rb = self.refer_band
            gb = self.guard_band
            k_pk = self.kernel_size
            N = k_pk[0] * k_pk[1]
            dtype = x.data.type()
            b = x.size(0)

            x = self.zero_padding(x)

            # prf_cuda does not support BF16, so keep this custom op in FP32.
            with torch.autocast(device_type='cuda', enabled=False):
                x_prf = self.prf(x.float())
            # (b, 2c, h*k_pk[0], w*k_pk[1])
            x_prf = self._reshape_x_prf(k_pk[0], k_pk[1], x_prf)

            # out: (b, c, h, w)
            out = self.peak_conv(x_prf)
            return out

        @staticmethod
        # reshape the x_prf
        def _reshape_x_prf(k_h, k_w, x_prf):
            b, c, h, w, N = x_prf.size()
            x_prf = torch.cat([x_prf[..., s:s + k_w].contiguous().view(b, c, h, w * k_w) for s in range(0, N, k_w)], dim=-1)
            x_prf = x_prf.contiguous().view(b, c, h * k_h, w * k_w)
            return x_prf
else:
    # reda peak_conv zlw @20220624
    # Peak Conv with fixed guard banwidth (gb) and dense reference receptive field (RRF)
    class PeakConv2D(nn.Module):
        def __init__(self, in_ch, out_ch, bias, refer_band, guard_band_dop, guard_band, signal_type):
            print("not use cuda peakconv2d")
            """
            Args:
                refer_band: a single integer value or integer vector of two values.
                guard_band_dop: guard bandwidth for doppler input
                guard_band: guard banwidth for non-doppler input
                signal_type in ('range_doppler', 'angle_doppler', 'range_angle').
                k_size, padding and stride are computed using the input fixed guard bandwidth
                input: x in the shape of BxCxHxW
            """
            super(PeakConv2D, self).__init__()
            self.signal_type = signal_type
            if self.signal_type in ('range_doppler', 'angle_doppler'):
                # guard_band should be bi-lateral
                self.guard_band = (guard_band_dop, guard_band_dop)
                # padding order: l-r-t-b
                self.padding = (self.guard_band[1] + refer_band,
                                self.guard_band[1] + refer_band,
                                self.guard_band[0] + refer_band,
                                self.guard_band[0] + refer_band)
            else:
                self.guard_band = (guard_band, guard_band)
                # padding order: l-r-t-b
                self.padding = (self.guard_band[1] + refer_band,
                                self.guard_band[1] + refer_band,
                                self.guard_band[0] + refer_band,
                                self.guard_band[0] + refer_band)
            self.refer_band = refer_band
            self.kernel_size = (self.guard_band[0] + self.guard_band[1] + self.refer_band * 2), 4
            self.stride = self.kernel_size
            # padding order: l-r-t-b
            self.zero_padding = nn.ZeroPad2d(self.padding)
            self.bias = bias
            self.pc_strd = 1
            # conv block for peak receptive field (PRF) in x
            # take PRF as input and output the center point output of PRF;
            # kernel channel = in_ch*2: depth_concat(p_c, p_r)
            # since each p_c is expanded as a tensor of kernel scale RF, the stride = kernel_size
            self.peak_conv = nn.Conv2d(in_ch, out_ch,
                                    kernel_size=self.kernel_size,
                                    padding=0,
                                    stride=self.stride,
                                    bias=self.bias)
            #self.peak_conv.register_backward_hook(self._set_lr)

        @staticmethod
        # set learning rate for peak_conv
        def _set_lr(module, grad_input, grad_output):
            grad_input = (grad_input[i] * 0.1 for i in range(len(grad_input)))
            grad_output = (grad_output[i] * 0.1 for i in range(len(grad_output)))

        def forward(self, x):
            h, w = x.shape[2], x.shape[3]
            rb = self.refer_band
            gb = self.guard_band
            k_pk = self.kernel_size
            N = k_pk[0] * k_pk[1]
            dtype = x.data.type()
            b = x.size(0)

            x = self.zero_padding(x)

            # let p_r denote the reference point positions
            # p_r shape: Bx2NxHxW
            p_r = self._get_p_r(b, h, w, N, dtype)
            # sample x_r (b, c, h, w, N) using p_r
            x_r = self._sample_x(x, p_r, N)
            # sample x_c (b, c, h, w, N) using p_c
            p_c = self._get_p_c(b, h, w, N, dtype)
            x_c = self._sample_x(x, p_c, N)

            # getting x_prf for peak_conv
            # (b, c, h, w, N) ==> (b, 2c, h, w, N)
            # x_prf = torch.cat((x_c, x_r), 1)
            x_prf = x_c - x_r
            # (b, 2c, h*k_pk[0], w*k_pk[1])
            x_prf = self._reshape_x_prf(k_pk[0], k_pk[1], x_prf)

            # out: (b, c, h, w)
            out = self.peak_conv(x_prf)
            return out

        # getting p_c (the center point coords) from the padded grid of input x
        def _get_p_c(self, b, h, w, N, dtype):
            # generating pc_grid
            p_c_x, p_c_y = torch.meshgrid(
                torch.arange(self.padding[2], h * self.pc_strd + self.padding[2], self.pc_strd),
                torch.arange(self.padding[0], w * self.pc_strd + self.padding[0], self.pc_strd))
            p_c_x = torch.flatten(p_c_x).view(1, 1, h, w).repeat(1, N, 1, 1)
            p_c_y = torch.flatten(p_c_y).view(1, 1, h, w).repeat(1, N, 1, 1)
            # p_c: 1x2NxHxW
            p_c = torch.cat([p_c_x, p_c_y], 1).type(dtype)
            # (B, 2N, h, w)
            p_c = p_c.repeat(b, 1, 1, 1)
            return p_c

        # generating peak receptive field grid
        def _gen_prf_grid(self, rb, gb, N, dtype):
            h_t = -(rb + gb[0])
            h_d = rb + gb[0]
            w_l = -(rb + gb[1])
            w_r = rb + gb[1]
            w_prf = (rb + gb[1]) * 2 + 1
            h_prf = (rb + gb[0]) * 2 + 1

            prf_x_idx, prf_y_idx = torch.meshgrid(
                torch.arange(h_t, h_d + 1),
                torch.arange(w_l, w_r + 1))

            # grid_size*2, 1 grid_size = (h_d*2+1) * (w_r*2+1)
            # taking positions clockwise
            prf_xt = prf_x_idx[0:rb]
            prf_xr = prf_x_idx[rb:(h_prf - 2 * rb + 1), (w_prf - rb):w_prf]
            prf_xd = prf_x_idx[(h_prf - rb):h_prf]
            prf_xl = prf_x_idx[rb:(h_prf - 2 * rb + 1), 0:rb]
            prf_x = torch.cat([torch.flatten(prf_xt),
                            torch.flatten(prf_xr),
                            torch.flatten(prf_xd),
                            torch.flatten(prf_xl)], 0)
            prf_yt = prf_y_idx[0:rb]
            prf_yr = prf_y_idx[rb:(h_prf - 2 * rb + 1), (w_prf - rb):w_prf]
            prf_yd = prf_y_idx[(h_prf - rb):h_prf]
            prf_yl = prf_y_idx[rb:(h_prf - 2 * rb + 1), 0:rb]
            prf_y = torch.cat([torch.flatten(prf_yt),
                            torch.flatten(prf_yr),
                            torch.flatten(prf_yd),
                            torch.flatten(prf_yl)], 0)

            prf = torch.cat([prf_x, prf_y], 0)
            prf = prf.view(1, 2 * N, 1, 1).type(dtype)
            return prf

        # getting p_r positions from each p_c
        def _get_p_r(self, b, h, w, N, dtype):
            rb = self.refer_band
            gb = self.guard_band
            # (1, 2N, 1, 1)
            prf = self._gen_prf_grid(rb, gb, N, dtype)
            # (B, 2N, h, w)
            p_c = self._get_p_c(b, h, w, N, dtype)
            # (B, 2N, h, w)
            p_r = p_c + prf
            return p_r

        # sampling x using p_r or p_c
        def _sample_x(self, x_pad, p, N):
            p = p.contiguous().permute(0, 2, 3, 1).floor()
            b, h, w, _ = p.size()
            h_pad = x_pad.size(2)
            w_pad = x_pad.size(3)
            c = x_pad.size(1)
            # (b, c, h_pad*w_pad)
            # strech each spatial channel of x_pad as 1-D vector
            x_pad = x_pad.contiguous().view(b, c, -1)
            # (b, h_pad, w_pad, N)
            # transform spatial coord of p into the 1-D index
            index = p[..., :N] * w_pad + p[..., N:]
            # index_x = torch.clamp(p[..., :N], 0, h_pad-1)
            # index_y = torch.clamp(p[..., N:], 0, w_pad-1)
            # index = index_x * w_pad + index_y
            # (b, c, h*w*N)
            index = index.contiguous().unsqueeze(dim=1).expand(-1, c, -1, -1, -1).contiguous().view(b, c, -1)
            index = index.long().to(x_pad.device)
            x_r = x_pad.gather(dim=-1, index=index).contiguous().view(b, c, h, w, N)
            return x_r

        @staticmethod
        # reshape the x_prf
        def _reshape_x_prf(k_h, k_w, x_prf):
            b, c, h, w, N = x_prf.size()
            x_prf = torch.cat([x_prf[..., s:s + k_w].contiguous().view(b, c, h, w * k_w) for s in range(0, N, k_w)], dim=-1)
            x_prf = x_prf.contiguous().view(b, c, h * k_h, w * k_w)
            return x_prf
    