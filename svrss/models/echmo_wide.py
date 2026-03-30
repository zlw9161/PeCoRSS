# End-to-end RSS network (wide scalable variant)
# Keep LMF/MLDC/GDF design, add grouped processing + low-rank channel fusion.
import os
import torch
import torch.nn as nn

from svrss.models.model_utils.echo_tools import complex_std, complex_double2float, filter_coef_loader
from svrss.models.pkcin4any_cuda import PKCIn4ANY_cuda
from svrss.models.echmo_debug import LMF_Chain, LMF_Chain_Parallel, Multi_Order_LDC, Group_Purturble_DF, _init_translator_weights


class ComplexLowRankMixer(nn.Module):
    """Low-rank 1x1 mixer for complex feature maps via real/imag decoupled convs."""

    def __init__(self, channels, rank, use_residual=True):
        super().__init__()
        rank = max(1, min(int(rank), int(channels)))
        self.use_residual = use_residual
        self.down = nn.Conv2d(channels, rank, kernel_size=1, bias=False)
        self.up = nn.Conv2d(rank, channels, kernel_size=1, bias=False)

    def forward(self, x):
        xr = self.up(self.down(x.real))
        xi = self.up(self.down(x.imag))
        y = torch.complex(xr, xi)
        if self.use_residual:
            return x + y
        return y


class EchMoWide(nn.Module):
    """EchMo scalable variant with grouped MLDC and low-rank fusion.

    Notes
    -----
    - LMF stage keeps original prior/residual design (LMF_V6 in LMF_Chain).
    - Grouping happens after LMF chains: channels are partitioned and each group
      runs MLDC independently.
    - GDF/translator stages remain unchanged in semantics.
    """

    def __init__(self, configs):
        super().__init__()
        self.cfg = configs
        self.task_name = configs.task_name
        self.n_range = configs.n_range
        self.n_pulse = configs.n_pulse
        self.n_classes = configs.n_classes
        self.n_frames = configs.n_frames
        self.scale = configs.scale

        # global
        self.activation = configs.activation
        self.bias = configs.bias

        # lmf
        self.lmf_chains = configs.lmf_chains
        self.lmf_channels = configs.lmf_channels
        self.lmf_layers = configs.lmf_layers
        self.lmf_coefs = configs.lmf_coefs
        self.lmf_chain_impl = str(getattr(configs, 'lmf_chain_impl', 'serial')).lower()

        # mldc / gdf
        self.ldc_orders = configs.ldc_orders
        self.ldc_channels = configs.ldc_channels
        self.df_groups = configs.ldc_orders
        self.df_channels = configs.df_channels

        # wide configs (fallback defaults keep compatibility)
        self.wide_groups = max(1, int(getattr(configs, 'wide_groups', 4)))
        self.wide_rank = int(getattr(configs, 'wide_rank', 32))
        self.wide_share_mldc = bool(getattr(configs, 'wide_share_mldc', True))
        self.wide_fuse_residual = bool(getattr(configs, 'wide_fuse_residual', True))

        self.total_lmf_channels = self.lmf_channels * self.lmf_chains
        if self.total_lmf_channels % self.wide_groups != 0:
            raise ValueError('lmf_channels * lmf_chains must be divisible by wide_groups.')
        if self.ldc_channels % self.wide_groups != 0:
            raise ValueError('ldc_channels must be divisible by wide_groups.')

        self.group_lmf_in = self.total_lmf_channels // self.wide_groups
        self.group_ldc_out = self.ldc_channels // self.wide_groups

        if self.lmf_chain_impl == 'parallel':
            self.LMFChainParallel = LMF_Chain_Parallel(configs=self.cfg)
            self.LMFChain = None
        elif self.lmf_chain_impl == 'serial':
            self.LMFChain = nn.ModuleList([LMF_Chain(configs=self.cfg)
                                           for _ in range(self.lmf_chains)])
            self.LMFChainParallel = None
        else:
            raise ValueError(f'Unsupported lmf_chain_impl: {self.lmf_chain_impl}. Use "serial" or "parallel".')

        self.ChannelMixer = ComplexLowRankMixer(
            channels=self.total_lmf_channels,
            rank=self.wide_rank,
            use_residual=self.wide_fuse_residual,
        )

        if self.wide_share_mldc:
            self.SharedMLDC = Multi_Order_LDC(
                in_ch=self.group_lmf_in,
                out_ch=self.group_ldc_out,
                orders=self.ldc_orders,
                activation=self.activation,
                bias=self.bias,
            )
            self.GroupMLDC = None
        else:
            self.GroupMLDC = nn.ModuleList([
                Multi_Order_LDC(
                    in_ch=self.group_lmf_in,
                    out_ch=self.group_ldc_out,
                    orders=self.ldc_orders,
                    activation=self.activation,
                    bias=self.bias,
                )
                for _ in range(self.wide_groups)
            ])
            self.SharedMLDC = None

        self.GDF = Group_Purturble_DF(
            in_ch=self.ldc_channels,
            out_ch=self.df_channels,
            k_size=self.n_pulse,
            groups=self.df_groups,
            activation=self.activation,
            bias=self.bias,
        )

        self.RDTranslator = PKCIn4ANY_cuda(
            n_classes=self.n_classes,
            n_frames=self.n_frames,
            in_ch=self.df_groups * 4,
            w=self.df_channels,
            h=self.n_range,
        )
        self.RDTranslator.apply(_init_translator_weights)

    def forward(self, x):
        if self.scale == 'instance':
            x = complex_double2float(complex_std(x, epsilon=1e-8))
        elif self.scale == 'global':
            x = complex_double2float(x)

        # LMF multi-chain aggregation
        if self.lmf_chain_impl == 'parallel':
            x_k = self.LMFChainParallel(x)
        else:
            x_k_list = []
            for i in range(self.lmf_chains):
                x_k_list.append(self.LMFChain[i](x))
            x_k = torch.cat(x_k_list, dim=1)

        # low-rank cross-channel fusion before grouped temporal modeling
        x_k = self.ChannelMixer(x_k)

        # grouped MLDC
        x_groups = torch.chunk(x_k, self.wide_groups, dim=1)
        x_h_groups = []
        if self.wide_share_mldc:
            for x_g in x_groups:
                x_h_groups.append(self.SharedMLDC(x_g))
        else:
            for i, x_g in enumerate(x_groups):
                x_h_groups.append(self.GroupMLDC[i](x_g))
        x_h = torch.cat(x_h_groups, dim=1)

        # keep original GDF -> translator pipeline
        x_d = self.GDF(x_h)
        x_d = torch.cat((torch.abs(x_d), torch.angle(x_d), x_d.real, x_d.imag), dim=1).unsqueeze(dim=2)
        y = self.RDTranslator(x_d)
        return y
