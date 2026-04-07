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

class Learnable_Match_FilterV2(nn.Module):
    def __init__(self, k_size, in_ch, out_ch, lmf_coef, activation='relu', bias=True):
        super(Learnable_Match_FilterV2, self).__init__()
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


class Learnable_Match_FilterV5(nn.Module):
    def __init__(self, k_size, in_ch, out_ch, lmf_coef, activation='relu', bias=True):
        super(Learnable_Match_FilterV5, self).__init__()
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


class Learnable_Match_FilterV6(nn.Module):
    # LMF with learnable kernels, initialized w/ MF, optional bias, activation, and residual connection. 
    # Residual mode requires in_ch == out_ch.
    # Added residual_mode flag to enable/disable residual connection, and corresponding kernel initialization.
    # When residual_mode is True, kernels are initialized to zero and learn to deviate from identity mapping. 
    # When False, kernels are initialized to MF and learn freely.
    # LMF_V6 requires fine-tuning strategy to achieve good performance. 
    # Curriculum learning is recommended:
    # - starting with freezing the LMF kernels and gradually unfreezing them,
    # - while also adjusting the learning rates for the prior and residual components if using residual mode. 
    # lmf_curriculum_enable: whether to use curriculum learning for LMF_V6, which includes hyperparameters:
    # - freeze: if epoch < total_epochs * freeze_ratio, freeze the LMF kernels; else unfreeze.
    # - transition: if epoch < total_epochs * transition_ratio, scale the lr of LMF kernels by a factor (e.g., 0.1); else use the base lr.
    # - learning rate scaling:
    #   - prior: if residual mode, use a smaller learning rate for the prior (initialized with MF) to prevent it from 
    #       deviating too much from the useful MF initialization.
    #   - residual: if residual mode, you may want to start with a smaller learning rate for the residual kernels
    #       since they are initialized to zero and need to learn from scratch, 
    #       and then increase it as training progresses. @zlw 20260325
    # NOTE: LMF_V6 = LMF_V5 + optional residual connection + curriculum-compatible attributes, when lmf_layers = 1, LMF_V5 == LMF_V6 (w/o residual connection).
    def __init__(self, k_size, in_ch, out_ch, lmf_coef, activation='relu', bias=True, residual_mode=False):
        super(Learnable_Match_FilterV6, self).__init__()
        self.k_size = k_size
        self.stride = 1
        self.in_ch = in_ch
        self.out_ch = out_ch
        self.mf = lmf_coef.view(1, 1, 1, self.k_size)
        self.activation = activation
        self.bias = bias
        self.residual_mode = residual_mode
        self.residual_scale = 1.0
        if self.residual_mode:
            if self.in_ch != self.out_ch:
                raise ValueError('Residual LMF layers require in_ch == out_ch.')
            kernel_real, kernel_imag = self.init_residual_kernels()
        else:
            kernel_real, kernel_imag = self.init_prior_kernels()
        self.kernel_real = nn.Parameter(kernel_real)
        self.kernel_imag = nn.Parameter(kernel_imag)
        if not self.residual_mode:
            self.register_buffer('prior_kernel_real', kernel_real.detach().clone())
            self.register_buffer('prior_kernel_imag', kernel_imag.detach().clone())
        else:
            self.prior_kernel_real = None
            self.prior_kernel_imag = None
        if self.bias:
            self.bias_real = nn.Parameter(torch.zeros(self.out_ch, dtype=torch.float32))
            self.bias_imag = nn.Parameter(torch.zeros(self.out_ch, dtype=torch.float32))
        else:
            self.register_parameter('bias_real', None)
            self.register_parameter('bias_imag', None)

        self.cause_padding = torch.nn.ZeroPad2d(padding=(0, self.k_size - 1, 0, 0))
        self.cmplx_relu = ComplexReLU(self.activation)

    def init_prior_kernels(self):
        real_kernel = self.mf.real.broadcast_to(self.out_ch, self.in_ch, 1, self.k_size)
        imag_kernel = self.mf.imag.broadcast_to(self.out_ch, self.in_ch, 1, self.k_size)
        return real_kernel.clone().contiguous(), imag_kernel.clone().contiguous()

    def init_residual_kernels(self):
        shape = (self.out_ch, self.in_ch, 1, self.k_size)
        real_kernel = torch.zeros(shape, dtype=torch.float32)
        imag_kernel = torch.zeros(shape, dtype=torch.float32)
        return real_kernel, imag_kernel

    def _complex_conv(self, x):
        inputR = self.cause_padding(x.real)
        inputI = self.cause_padding(x.imag)
        x_RR = F.conv2d(inputR, self.kernel_real, self.bias_real, stride=(1, 1), padding=0)
        x_II = F.conv2d(inputI, self.kernel_imag, self.bias_imag, stride=(1, 1), padding=0)
        x_RI = F.conv2d(inputR, self.kernel_imag, self.bias_imag, stride=(1, 1), padding=0)
        x_IR = F.conv2d(inputI, self.kernel_real, self.bias_real, stride=(1, 1), padding=0)
        return x_RR - x_II + 1j * (x_RI + x_IR)

    def forward(self, x):
        x_filtered = self._complex_conv(x)
        if self.residual_mode:
            x_residual = self.cmplx_relu(x_filtered)
            return x + self.residual_scale * x_residual
        return self.cmplx_relu(x_filtered)


class Learnable_Match_FilterV6_ReXAttn(nn.Module):
    """LMF-V6 with lightweight cross-attention residual enhancement.

    Main path keeps prior-initialized complex convolution from V6.
    Residual branch uses cross-attention where:
    - query comes from prior kernel tokens (or current kernel in residual mode)
    - key/value come from input echo tokens
    """

    def __init__(
        self,
        k_size,
        in_ch,
        out_ch,
        lmf_coef,
        activation='relu',
        bias=True,
        residual_mode=False,
        rexattn_dim=16,
        rexattn_scale=0.1,
    ):
        super(Learnable_Match_FilterV6_ReXAttn, self).__init__()
        self.k_size = k_size
        self.stride = 1
        self.in_ch = in_ch
        self.out_ch = out_ch
        self.mf = lmf_coef.view(1, 1, 1, self.k_size)
        self.activation = activation
        self.bias = bias
        self.residual_mode = residual_mode
        self.residual_scale = 1.0

        if self.residual_mode:
            if self.in_ch != self.out_ch:
                raise ValueError('Residual LMF layers require in_ch == out_ch.')
            kernel_real, kernel_imag = self.init_residual_kernels()
        else:
            kernel_real, kernel_imag = self.init_prior_kernels()

        self.kernel_real = nn.Parameter(kernel_real)
        self.kernel_imag = nn.Parameter(kernel_imag)
        if not self.residual_mode:
            self.register_buffer('prior_kernel_real', kernel_real.detach().clone())
            self.register_buffer('prior_kernel_imag', kernel_imag.detach().clone())
        else:
            self.prior_kernel_real = None
            self.prior_kernel_imag = None

        if self.bias:
            self.bias_real = nn.Parameter(torch.zeros(self.out_ch, dtype=torch.float32))
            self.bias_imag = nn.Parameter(torch.zeros(self.out_ch, dtype=torch.float32))
        else:
            self.register_parameter('bias_real', None)
            self.register_parameter('bias_imag', None)

        self.rexattn_dim = int(rexattn_dim)
        self.rexattn_scale = float(rexattn_scale)
        token_dim = 2 * self.in_ch
        self.q_proj = nn.Linear(token_dim, self.rexattn_dim, bias=False)
        self.k_proj = nn.Linear(token_dim, self.rexattn_dim, bias=False)
        self.v_proj = nn.Linear(token_dim, self.rexattn_dim, bias=False)
        self.o_proj = nn.Linear(self.rexattn_dim, 2, bias=False)

        self.cause_padding = torch.nn.ZeroPad2d(padding=(0, self.k_size - 1, 0, 0))
        self.cmplx_relu = ComplexReLU(self.activation)

    def init_prior_kernels(self):
        real_kernel = self.mf.real.broadcast_to(self.out_ch, self.in_ch, 1, self.k_size)
        imag_kernel = self.mf.imag.broadcast_to(self.out_ch, self.in_ch, 1, self.k_size)
        return real_kernel.clone().contiguous(), imag_kernel.clone().contiguous()

    def init_residual_kernels(self):
        shape = (self.out_ch, self.in_ch, 1, self.k_size)
        real_kernel = torch.zeros(shape, dtype=torch.float32)
        imag_kernel = torch.zeros(shape, dtype=torch.float32)
        return real_kernel, imag_kernel

    def _complex_conv(self, x):
        inputR = self.cause_padding(x.real)
        inputI = self.cause_padding(x.imag)
        x_RR = F.conv2d(inputR, self.kernel_real, self.bias_real, stride=(1, 1), padding=0)
        x_II = F.conv2d(inputI, self.kernel_imag, self.bias_imag, stride=(1, 1), padding=0)
        x_RI = F.conv2d(inputR, self.kernel_imag, self.bias_imag, stride=(1, 1), padding=0)
        x_IR = F.conv2d(inputI, self.kernel_real, self.bias_real, stride=(1, 1), padding=0)
        return x_RR - x_II + 1j * (x_RI + x_IR)

    def _query_tokens(self, dtype, device):
        k_real = self.prior_kernel_real if self.prior_kernel_real is not None else self.kernel_real
        k_imag = self.prior_kernel_imag if self.prior_kernel_imag is not None else self.kernel_imag
        # Compress kernel taps into one query token per output channel.
        q_real = k_real.squeeze(2).mean(dim=-1)
        q_imag = k_imag.squeeze(2).mean(dim=-1)
        q_tokens = torch.cat((q_real, q_imag), dim=1).contiguous().to(device=device, dtype=dtype)
        return q_tokens

    def _cross_attention_residual(self, x):
        # x: [B, C, P, R], output: [B, out_ch, P, R]
        bsz, channels, pulse, width = x.shape
        torch._assert(
            channels == self.in_ch,
            f'Unexpected ReXAttn input channels: got {channels}, expected {self.in_ch}.',
        )

        q_tokens = self._query_tokens(dtype=x.real.dtype, device=x.device)
        q = self.q_proj(q_tokens)  # [O, D]

        kv_in = torch.cat((x.real, x.imag), dim=1).permute(0, 2, 3, 1).contiguous()  # [B, P, R, 2C]
        k = self.k_proj(kv_in)  # [B, P, R, D]
        v = self.v_proj(kv_in)  # [B, P, R, D]

        scale = self.rexattn_dim ** -0.5
        logits = torch.einsum('od,bprd->bopr', q, k) * scale  # [B, O, P, R]
        attn = torch.sigmoid(logits)

        ctx = attn.unsqueeze(-1) * v.unsqueeze(1)  # [B, O, P, R, D]
        out_ri = self.o_proj(ctx)  # [B, O, P, R, 2]
        attn_res = torch.complex(out_ri[..., 0], out_ri[..., 1])
        return attn_res

    def forward(self, x):
        x_filtered = self._complex_conv(x)
        x_residual = self.cmplx_relu(x_filtered)
        if self.rexattn_scale != 0.0:
            x_attn = self._cross_attention_residual(x)
            x_residual = x_residual + self.rexattn_scale * self.cmplx_relu(x_attn)

        if self.residual_mode:
            return x + self.residual_scale * x_residual
        return x_residual


class Learnable_Match_FilterV7(nn.Module):
    """Latency-optimized independent LMF implementation.

    Learnable_Match_FilterV7 keeps V6 behavior (prior/residual modes + curriculum-compatible
    attributes) while reducing 4 real-valued conv calls to 1 by packing
    complex convolution into a single real conv2d weight bank.
    """

    def __init__(self, k_size, in_ch, out_ch, lmf_coef, activation='relu', bias=True, residual_mode=False):
        super(Learnable_Match_FilterV7, self).__init__()
        self.k_size = k_size
        self.stride = 1
        self.in_ch = in_ch
        self.out_ch = out_ch
        self.mf = lmf_coef.view(1, 1, 1, self.k_size)
        self.activation = activation
        self.bias = bias
        self.residual_mode = residual_mode
        self.residual_scale = 1.0

        if self.residual_mode:
            if self.in_ch != self.out_ch:
                raise ValueError('Residual LMF layers require in_ch == out_ch.')
            kernel_real, kernel_imag = self.init_residual_kernels()
        else:
            kernel_real, kernel_imag = self.init_prior_kernels()

        self.kernel_real = nn.Parameter(kernel_real)
        self.kernel_imag = nn.Parameter(kernel_imag)

        if not self.residual_mode:
            self.register_buffer('prior_kernel_real', kernel_real.detach().clone())
            self.register_buffer('prior_kernel_imag', kernel_imag.detach().clone())
        else:
            self.prior_kernel_real = None
            self.prior_kernel_imag = None

        if self.bias:
            self.bias_real = nn.Parameter(torch.zeros(self.out_ch, dtype=torch.float32))
            self.bias_imag = nn.Parameter(torch.zeros(self.out_ch, dtype=torch.float32))
        else:
            self.register_parameter('bias_real', None)
            self.register_parameter('bias_imag', None)

        self.cause_padding = torch.nn.ZeroPad2d(padding=(0, self.k_size - 1, 0, 0))
        self.cmplx_relu = ComplexReLU(self.activation)

    def init_prior_kernels(self):
        real_kernel = self.mf.real.broadcast_to(self.out_ch, self.in_ch, 1, self.k_size)
        imag_kernel = self.mf.imag.broadcast_to(self.out_ch, self.in_ch, 1, self.k_size)
        return real_kernel.clone().contiguous(), imag_kernel.clone().contiguous()

    def init_residual_kernels(self):
        shape = (self.out_ch, self.in_ch, 1, self.k_size)
        real_kernel = torch.zeros(shape, dtype=torch.float32)
        imag_kernel = torch.zeros(shape, dtype=torch.float32)
        return real_kernel, imag_kernel

    def _build_kernel_bank(self):
        # Complex conv mapping:
        # y_real = conv([R, I], [K_real, -K_imag])
        # y_imag = conv([R, I], [K_imag,  K_real])
        w_real = torch.cat((self.kernel_real, -self.kernel_imag), dim=1)
        w_imag = torch.cat((self.kernel_imag, self.kernel_real), dim=1)
        return torch.cat((w_real, w_imag), dim=0)

    def _build_bias_bank(self):
        if not self.bias:
            return None
        b_real = self.bias_real - self.bias_imag
        b_imag = self.bias_real + self.bias_imag
        return torch.cat((b_real, b_imag), dim=0)

    def _complex_conv(self, x):
        inputR = self.cause_padding(x.real)
        inputI = self.cause_padding(x.imag)
        input_cat = torch.cat((inputR, inputI), dim=1)

        kernel_bank = self._build_kernel_bank()
        bias_bank = self._build_bias_bank()
        y_cat = F.conv2d(input_cat, kernel_bank, bias=bias_bank, stride=(1, 1), padding=0)
        y_real, y_imag = torch.chunk(y_cat, chunks=2, dim=1)
        return torch.complex(y_real, y_imag)

    def forward(self, x):
        x_filtered = self._complex_conv(x)
        if self.residual_mode:
            x_residual = self.cmplx_relu(x_filtered)
            return x + self.residual_scale * x_residual
        return self.cmplx_relu(x_filtered)


class LMF_Chain(nn.Module):
    def __init__(self, configs):
        super(LMF_Chain, self).__init__()
        self.lmf_coef = filter_coef_loader(os.path.join(configs.dataset_path, configs.coef_path),
                                          configs.lmf_coefs)
        self.lmf_impl = str(getattr(configs, 'lmf_impl', 'v6')).lower()
        self.in_ch = 1
        # self.mf_coef = torch.tensor(self.mf_coef.copy()).to()
        self.Chain = nn.Sequential()
        for l in range(configs.lmf_layers):
            residual_mode = (self.lmf_impl in ('v6', 'v6_rexattn', 'v7')) and (l > 0)
            if self.lmf_impl == 'v7':
                module_name = 'lmfV7-res-{0}'.format(l) if residual_mode else 'lmfV7-prior-{0}'.format(l)
                module_cls = Learnable_Match_FilterV7
            elif self.lmf_impl == 'v6':
                module_name = 'lmfV6-res-{0}'.format(l) if residual_mode else 'lmfV6-prior-{0}'.format(l)
                module_cls = Learnable_Match_FilterV6
            elif self.lmf_impl == 'v6_rexattn':
                module_name = 'lmfV6ReXAttn-res-{0}'.format(l) if residual_mode else 'lmfV6ReXAttn-prior-{0}'.format(l)
                module_cls = Learnable_Match_FilterV6_ReXAttn
            elif self.lmf_impl == 'v5':
                module_name = 'lmfV5-{0}'.format(l)
                module_cls = Learnable_Match_FilterV5
            else:
                raise ValueError(f'Unsupported lmf_impl: {self.lmf_impl}. Use "v5", "v6", "v6_rexattn" or "v7".')

            module_kwargs = dict(
                k_size=self.lmf_coef.shape[1],
                in_ch=self.in_ch,
                out_ch=configs.lmf_channels,
                lmf_coef=self.lmf_coef,
                activation=configs.activation,
                bias=configs.bias,
            )
            if self.lmf_impl in ('v6', 'v6_rexattn', 'v7'):
                module_kwargs['residual_mode'] = residual_mode

            self.Chain.add_module(module_name,
                                  module_cls(**module_kwargs))
            self.in_ch = configs.lmf_channels

    def forward(self, x):
        x = self.Chain(x)
        return x


class LMF_Layer_Parallel(nn.Module):
    """Parallel layer for all LMF chains at the same depth.

    Input/Output layout:
    - input:  [B, chains, C_in, P, R]
    - output: [B, chains, C_out, P, R]

    Notes:
    - This layer is chain-parallel and compatible with both `lmf_impl=v6` and
            `lmf_impl=v7` at the architecture level.
    - For runtime efficiency, both paths share the packed-conv execution style
      that is mathematically equivalent to the 4-conv complex form.
    """

    def __init__(self, k_size, in_ch, out_ch, lmf_coef, num_chains, activation='relu', bias=True, residual_mode=False, lmf_impl='v7'):
        super(LMF_Layer_Parallel, self).__init__()
        self.k_size = k_size
        self.in_ch = in_ch
        self.out_ch = out_ch
        self.num_chains = num_chains
        self.mf = lmf_coef.view(1, 1, 1, self.k_size)
        self.activation = activation
        self.bias = bias
        self.residual_mode = residual_mode
        self.residual_scale = 1.0
        self.lmf_impl = str(lmf_impl).lower()

        if self.lmf_impl not in ('v5', 'v6', 'v7'):
            raise ValueError(f'Unsupported lmf_impl: {self.lmf_impl}. Use "v5", "v6" or "v7".')

        if self.residual_mode and self.in_ch != self.out_ch:
            raise ValueError('Residual LMF layers require in_ch == out_ch.')

        kernel_real, kernel_imag = self._init_kernels()
        self.kernel_real = nn.Parameter(kernel_real)
        self.kernel_imag = nn.Parameter(kernel_imag)

        if not self.residual_mode:
            self.register_buffer('prior_kernel_real', kernel_real.detach().clone())
            self.register_buffer('prior_kernel_imag', kernel_imag.detach().clone())
        else:
            self.prior_kernel_real = None
            self.prior_kernel_imag = None

        if self.bias:
            self.bias_real = nn.Parameter(torch.zeros(self.num_chains, self.out_ch, dtype=torch.float32))
            self.bias_imag = nn.Parameter(torch.zeros(self.num_chains, self.out_ch, dtype=torch.float32))
        else:
            self.register_parameter('bias_real', None)
            self.register_parameter('bias_imag', None)

        self.cause_padding = torch.nn.ZeroPad2d(padding=(0, self.k_size - 1, 0, 0))
        self.cmplx_relu = ComplexReLU(self.activation)

    def _init_kernels(self):
        shape = (self.num_chains, self.out_ch, self.in_ch, 1, self.k_size)
        if self.residual_mode:
            real_kernel = torch.zeros(shape, dtype=torch.float32)
            imag_kernel = torch.zeros(shape, dtype=torch.float32)
        else:
            base_real = self.mf.real.broadcast_to(self.out_ch, self.in_ch, 1, self.k_size)
            base_imag = self.mf.imag.broadcast_to(self.out_ch, self.in_ch, 1, self.k_size)
            real_kernel = base_real.unsqueeze(0).broadcast_to(shape).clone().contiguous()
            imag_kernel = base_imag.unsqueeze(0).broadcast_to(shape).clone().contiguous()
        return real_kernel, imag_kernel

    def _build_kernel_bank(self):
        # Complex conv mapping for each chain group:
        # y_real = conv([R, I], [K_real, -K_imag])
        # y_imag = conv([R, I], [K_imag,  K_real])
        w_real = torch.cat((self.kernel_real, -self.kernel_imag), dim=2)
        w_imag = torch.cat((self.kernel_imag, self.kernel_real), dim=2)
        w = torch.cat((w_real, w_imag), dim=1)
        return w.reshape(self.num_chains * 2 * self.out_ch, 2 * self.in_ch, 1, self.k_size).contiguous()

    def _build_bias_bank(self):
        if not self.bias:
            return None
        b_real = self.bias_real - self.bias_imag
        b_imag = self.bias_real + self.bias_imag
        return torch.cat((b_real, b_imag), dim=1).reshape(self.num_chains * 2 * self.out_ch).contiguous()

    def _complex_conv(self, x):
        B, chains, C_in, P, R = x.shape
        torch._assert(
            chains == self.num_chains and C_in == self.in_ch,
            f'Unexpected input shape for parallel LMF layer: {tuple(x.shape)}',
        )

        xr = self.cause_padding(x.real.reshape(B * chains, C_in, P, R))
        xi = self.cause_padding(x.imag.reshape(B * chains, C_in, P, R))
        xr = xr.reshape(B, chains, C_in, P, xr.shape[-1])
        xi = xi.reshape(B, chains, C_in, P, xi.shape[-1])

        x_cat = torch.cat((xr, xi), dim=2).reshape(B, chains * 2 * C_in, P, xr.shape[-1]).contiguous()
        y_cat = F.conv2d(
            x_cat,
            self._build_kernel_bank(),
            bias=self._build_bias_bank(),
            stride=(1, 1),
            padding=0,
            groups=self.num_chains,
        )
        y_cat = y_cat.reshape(B, chains, 2 * self.out_ch, P, R)
        y_real, y_imag = torch.chunk(y_cat, chunks=2, dim=2)
        return torch.complex(y_real, y_imag)

    def forward(self, x):
        x_filtered = self._complex_conv(x)
        if self.residual_mode:
            x_residual = self.cmplx_relu(x_filtered)
            return x + self.residual_scale * x_residual
        return self.cmplx_relu(x_filtered)


class LMF_Chain_Parallel(nn.Module):
    """Chain-parallel LMF pipeline.

    Computes all chains together at each depth, then outputs concatenated channels
    with shape [B, chains * lmf_channels, P, R].
    """

    def __init__(self, configs):
        super(LMF_Chain_Parallel, self).__init__()
        self.lmf_coef = filter_coef_loader(os.path.join(configs.dataset_path, configs.coef_path),
                                           configs.lmf_coefs)
        self.lmf_impl = str(getattr(configs, 'lmf_impl', 'v7')).lower()
        self.num_chains = int(configs.lmf_chains)
        self.out_ch = int(configs.lmf_channels)
        self.layers = nn.ModuleList()

        in_ch = 1
        for l in range(configs.lmf_layers):
            residual_mode = (self.lmf_impl in ('v6', 'v7')) and (l > 0)
            if self.lmf_impl not in ('v5', 'v6', 'v7'):
                if self.lmf_impl == 'v6_rexattn':
                    raise ValueError('lmf_impl="v6_rexattn" is only supported with lmf_chain_impl="serial".')
                raise ValueError(f'Unsupported lmf_impl: {self.lmf_impl}. Use "v5", "v6", "v6_rexattn" or "v7".')
            self.layers.append(
                LMF_Layer_Parallel(
                    k_size=self.lmf_coef.shape[1],
                    in_ch=in_ch,
                    out_ch=self.out_ch,
                    lmf_coef=self.lmf_coef,
                    num_chains=self.num_chains,
                    activation=configs.activation,
                    bias=configs.bias,
                    residual_mode=residual_mode,
                    lmf_impl=self.lmf_impl,
                )
            )
            in_ch = self.out_ch

    def forward(self, x):
        # Broadcast one shared input to all chain slots.
        x = x.unsqueeze(1).expand(-1, self.num_chains, -1, -1, -1)
        for layer in self.layers:
            x = layer(x)
        B, chains, C, P, R = x.shape
        return x.reshape(B, chains * C, P, R)


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
        
        y_r = y_r.view(B, self.orders*self.out_ch, P, R)
        y_i = y_i.view(B, self.orders*self.out_ch, P, R)
        y = torch.complex(y_r, y_i)
        # y = self.batch_norm(y)
        y = self.cmplx_relu(y)
        return y


class MLDC_Parallel(nn.Module):
    """Parallel draft for MLDC.

    This module preserves Multi_Order_LDC semantics but reduces kernel launch count
    by packing all order-specific filters into one shared-height conv kernel bank,
    then running only two conv2d calls (real/imag).
    """

    def __init__(self, in_ch, out_ch, orders, activation='relu', bias=True):
        super(MLDC_Parallel, self).__init__()
        self.in_ch = in_ch
        self.out_ch = out_ch
        self.orders = orders
        self.activation = activation
        self.bias = bias
        # Order n uses kernel height (n+2), max kernel height is orders+1.
        self.max_kh = self.orders + 1

        self.hp_coefs = self.get_highpass_coefs()
        self.hp_kernels = self.init_highpass_kernels()
        if self.bias:
            self.hp_biases = nn.ParameterList([
                nn.Parameter(torch.zeros(self.out_ch, dtype=torch.float32))
                for _ in range(self.orders)
            ])
        else:
            self.hp_biases = None
        self.cmplx_relu = ComplexReLU(self.activation)

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
                    hp_coefs_n[i] = (-1)**(i) * math.factorial(n) // (math.factorial(i) * math.factorial(n-i))
            hp_coefs.append(hp_coefs_n)
        return hp_coefs

    def init_highpass_kernels(self):
        hp_kernels = nn.ParameterList()
        for n in range(len(self.hp_coefs)):
            hp_kernels_n = self.hp_coefs[n].unsqueeze(dim=0).unsqueeze(dim=0).broadcast_to(
                self.out_ch,
                self.in_ch,
                n+2,
                1,
            )
            # Store each kernel directly at full height to avoid backward views
            # with padded strides (which trigger DDP grad/bucket stride warnings).
            full_kernel = torch.zeros(
                self.out_ch,
                self.in_ch,
                self.max_kh,
                1,
                dtype=hp_kernels_n.dtype,
            )
            full_kernel[:, :, - (n + 2):, :] = hp_kernels_n
            hp_kernels.append(nn.Parameter(full_kernel.contiguous()))
        return hp_kernels

    def _build_parallel_weight_bank(self):
        # Kernels are already stored with full max_kh height.
        return torch.cat([self.hp_kernels[n] for n in range(self.orders)], dim=0)

    def _build_parallel_bias(self):
        if not self.bias:
            return None
        return torch.cat([self.hp_biases[n] for n in range(self.orders)], dim=0)

    def forward(self, x):
        # Input: [B, C_in, P, R], output: [B, orders*out_ch, P, R]
        # One-time top padding for all orders.
        x_real = F.pad(x.real, (0, 0, self.orders, 0))
        x_imag = F.pad(x.imag, (0, 0, self.orders, 0))

        weight_bank = self._build_parallel_weight_bank()
        bias_bank = self._build_parallel_bias()

        y_r = F.conv2d(x_real, weight_bank, bias=bias_bank, stride=(1, 1), padding=0)
        y_i = F.conv2d(x_imag, weight_bank, bias=bias_bank, stride=(1, 1), padding=0)

        y = torch.complex(y_r, y_i)
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
    def __init__(self, in_ch, out_ch, k_size, groups, activation='relu', bias=True, impl='conv'):
        super(Group_Purturble_DF, self).__init__()
        self.k_size = k_size
        self.in_ch = in_ch
        self.out_ch = out_ch
        self.groups = groups
        self.activation = activation
        self.bias = bias
        self.impl = str(impl).lower()
        if self.impl not in ['conv', 'einsum']:
            raise ValueError(f'Unsupported gdf impl: {self.impl}. Use "conv" or "einsum".')
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
    def _forward_conv(self, x):
        # input: \tilde{X}: (B, groups*C_H, P, R)
        # output: \dot{X}: (B, groups*C_D, 1, R) --> (B, groups, C_D, R)
        B, R = x.shape[0], x.shape[3]
        x_real, x_imag = x.real.to(x.device), x.imag.to(x.device)
        x_rr = F.conv2d(x_real, self.cos_kernel, self.cos_bias, stride=self.stride, padding=0, groups=self.groups)
        x_ii = F.conv2d(x_imag, self.sin_kernel, self.sin_bias, stride=self.stride, padding=0, groups=self.groups)
        x_ri = F.conv2d(x_real, self.sin_kernel, self.sin_bias, stride=self.stride, padding=0, groups=self.groups)
        x_ir = F.conv2d(x_imag, self.cos_kernel, self.cos_bias, stride=self.stride, padding=0, groups=self.groups)
        x = x_rr + x_ii - 1j * x_ri + 1j * x_ir
        x = self.cmplx_relu(x)
        x = x.view(B, self.groups, self.out_ch, 1, R).squeeze(dim=3)
        return x

    def _forward_einsum(self, x):
        bsz, channels, pulse, width = x.shape
        torch._assert(
            channels == self.groups * self.in_ch,
            f'Unexpected GDF input channels: got {channels}, expected {self.groups * self.in_ch}.',
        )
        torch._assert(
            pulse == self.k_size,
            f'Unexpected GDF pulse dimension: got {pulse}, expected {self.k_size}.',
        )

        x_real = x.real.reshape(bsz, self.groups, self.in_ch, pulse, width)
        x_imag = x.imag.reshape(bsz, self.groups, self.in_ch, pulse, width)
        cos_kernel = self.cos_kernel.reshape(self.groups, self.out_ch, self.in_ch, self.k_size)
        sin_kernel = self.sin_kernel.reshape(self.groups, self.out_ch, self.in_ch, self.k_size)

        real_part = torch.einsum('bgcpr,gdcp->bgdr', x_real, cos_kernel)
        real_part = real_part + torch.einsum('bgcpr,gdcp->bgdr', x_imag, sin_kernel)
        imag_part = torch.einsum('bgcpr,gdcp->bgdr', x_imag, cos_kernel)
        imag_part = imag_part - torch.einsum('bgcpr,gdcp->bgdr', x_real, sin_kernel)

        if self.bias:
            cos_bias = self.cos_bias.reshape(1, self.groups, self.out_ch, 1)
            sin_bias = self.sin_bias.reshape(1, self.groups, self.out_ch, 1)
            real_part = real_part + cos_bias + sin_bias
            imag_part = imag_part + cos_bias - sin_bias

        x = torch.complex(real_part, imag_part)
        x = self.cmplx_relu(x)
        return x

    def forward(self, x):
        if self.impl == 'einsum':
            return self._forward_einsum(x)
        return self._forward_conv(x)
    

class Learnable_Coherent_Integration(nn.Module):
    # learnable MTD or FFT
    '''
    X(f) = \sum_{t=0}^{T-1} x(t)(cos(2*\pi*f*t/T) - j*sin(2*\pi*f*t/T))
    out_ch denotes the number of Doppler channels (f in the Equation)
    k_size denotes the number of pulses (t in the Equation)
    the kernel in Conv: out_ch * in_ch * k_h * k_w
    then the kernel for fourier basis: doppler * in_ch * n_pulse * 1
    '''
    def __init__(self, in_ch, out_ch, k_size, activation='relu', bias=True):
        super(Learnable_Coherent_Integration, self).__init__()
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


Leanable_Coherent_Integration = Learnable_Coherent_Integration


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
        self.lmf_impl = str(getattr(configs, 'lmf_impl', 'v6')).lower()
        self.lmf_chain_impl = str(getattr(configs, 'lmf_chain_impl', 'serial')).lower()
        # mldc cfg
        self.ldc_orders = configs.ldc_orders
        self.ldc_channels = configs.ldc_channels
        self.mldc_impl = str(getattr(configs, 'mldc_impl', 'serial')).lower()
        self.gdf_impl = str(getattr(configs, 'gdf_impl', 'conv')).lower()
        # gdf cfg
        self.df_groups = configs.ldc_orders
        self.df_channels = configs.df_channels
        
        if self.lmf_chain_impl == 'parallel':
            self.LMFChainParallel = LMF_Chain_Parallel(configs=self.cfg)
            self.LMFChain = None
        elif self.lmf_chain_impl == 'serial':
            self.LMFChain = nn.ModuleList([LMF_Chain(configs=self.cfg)
                                           for _ in range(self.lmf_chains)])
            self.LMFChainParallel = None
        else:
            raise ValueError(f'Unsupported lmf_chain_impl: {self.lmf_chain_impl}. Use "serial" or "parallel".')
        mldc_cls = Multi_Order_LDC
        if self.mldc_impl == 'parallel':
            mldc_cls = MLDC_Parallel
        elif self.mldc_impl != 'serial':
            raise ValueError(f'Unsupported mldc_impl: {self.mldc_impl}. Use "serial" or "parallel".')

        self.MLDC = mldc_cls(in_ch=self.lmf_channels*self.lmf_chains,
                             out_ch=self.ldc_channels,
                             orders=self.ldc_orders,
                             activation=self.activation,
                             bias=self.bias)
        self.GDF = Group_Purturble_DF(in_ch=self.ldc_channels,
                                      out_ch=self.df_channels,
                                      k_size=self.n_pulse,
                                      groups=self.df_groups,
                                      activation=self.activation,
                                      bias=self.bias,
                                      impl=self.gdf_impl)
        # for amp+phase+I/Q                              
        self.RDTranslator = PKCIn4ANY_cuda(n_classes=self.n_classes,
                                           n_frames=self.n_frames,
                                           in_ch=self.df_groups*4,
                                           w=self.df_channels,
                                           h=self.n_range)
        self.RDTranslator.apply(_init_translator_weights)


    def normalize_input(self, x):
        if self.scale == 'instance':
            x = complex_double2float(complex_std(x, epsilon=1e-8))
        elif self.scale == 'global':
            x = complex_double2float(x)
        return x

    def forward_lmf(self, x):
        if self.lmf_chain_impl == 'parallel':
            x_k = self.LMFChainParallel(x)
        else:
            x_k_list = []
            for i in range(self.lmf_chains):
                x_k_list.append(self.LMFChain[i](x))
            x_k = torch.cat(x_k_list, dim=1)
        return x_k

    def forward_mldc(self, x_k):
        return self.MLDC(x_k)

    def forward_gdf(self, x_h):
        return self.GDF(x_h)

    def pack_features(self, x_d):
        return torch.cat((torch.abs(x_d), torch.angle(x_d), x_d.real, x_d.imag), dim=1).unsqueeze(dim=2)

    def forward_frontend(self, x):
        x = self.normalize_input(x)
        x_k = self.forward_lmf(x)

        # # MTI w/o learning
        # x_h = torch.cat((x_k[:, :, 0, :].unsqueeze(dim=2), x_k[:, :, 1:, :] - x_k[:, :, :-1, :]), dim=2)
        # print('x_h', x_h.shape)
        ## MTD w/o learning
        # x_d = torch.fft.fft(x_h, dim=2)

        # Learnable MTI and MTD
        x_h = self.forward_mldc(x_k)
        # x_d = self.LCI(x_h)
        x_d = self.forward_gdf(x_h)
        
        
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
        # x_d = torch.abs(x_d).unsqueeze(dim=2) # amp-only
        # x_d = torch.cat((x_d.real, x_d.imag), dim=1).unsqueeze(dim=2) # I/Q
        # x_d = torch.cat((torch.angle(x_d), torch.abs(x_d)), dim=1).unsqueeze(dim=2) # phase+amp (pham)
        return self.pack_features(x_d) # amp+phase+I+Q

    def forward_head(self, x_d):
        return self.RDTranslator(x_d)

    def forward(self, x):
        x_d = self.forward_frontend(x)
        y = self.forward_head(x_d)
        return y


if __name__ == '__main__':
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument('--task_name', default='Segmentation',
                        help='Options: LMF; MTD; Segmentation.', type=str)
    parser.add_argument('--coef_path', default='coefs', help='data path', type=str)
    parser.add_argument('--dataset_path', default='E:/DataSets/NuaaSim', help='dataset path', type=str)
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
