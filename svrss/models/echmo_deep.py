# End-to-end RSS network (deep scalable variant)
# Multi-resolution staged EchMo while keeping LMF/MLDC/GDF design.
import torch
import torch.nn as nn
import torch.nn.functional as F

from svrss.models.echmo_debug import EchMo


class _CfgView:
    """Lightweight config clone for stage-wise EchMo construction."""

    pass


def _clone_cfg(cfg, **overrides):
    c = _CfgView()
    c.__dict__.update(cfg.__dict__)
    for k, v in overrides.items():
        setattr(c, k, v)
    return c


def _resize_complex(x, out_h, out_w):
    xr = F.interpolate(x.real, size=(out_h, out_w), mode='bilinear', align_corners=False)
    xi = F.interpolate(x.imag, size=(out_h, out_w), mode='bilinear', align_corners=False)
    return torch.complex(xr, xi)


class EchMoDeep(nn.Module):
    """Two-stage multi-resolution EchMo.

    Stage-1 runs at full resolution.
    Stage-2 runs at reduced resolution and its logits are upsampled then fused.
    """

    def __init__(self, configs):
        super().__init__()
        self.cfg = configs
        self.n_classes = configs.n_classes
        self.n_pulse = configs.n_pulse
        self.n_range = configs.n_range

        self.deep_stage2_enable = bool(getattr(configs, 'deep_stage2_enable', True))
        self.deep_stage2_range_scale = float(getattr(configs, 'deep_stage2_range_scale', 0.5))
        self.deep_stage2_pulse_scale = float(getattr(configs, 'deep_stage2_pulse_scale', 1.0))
        self.deep_stage2_chain_scale = float(getattr(configs, 'deep_stage2_chain_scale', 1.0))
        self.deep_stage2_lmf_channel_scale = float(getattr(configs, 'deep_stage2_lmf_channel_scale', 1.0))
        self.deep_stage2_ldc_channel_scale = float(getattr(configs, 'deep_stage2_ldc_channel_scale', 1.0))

        self.stage1 = EchMo(configs=configs)

        if self.deep_stage2_enable:
            pulse2 = max(8, int(round(self.n_pulse * self.deep_stage2_pulse_scale)))
            range2 = max(64, int(round(self.n_range * self.deep_stage2_range_scale)))
            chains2 = max(1, int(round(configs.lmf_chains * self.deep_stage2_chain_scale)))
            lmf_ch2 = max(1, int(round(configs.lmf_channels * self.deep_stage2_lmf_channel_scale)))
            ldc_ch2 = max(1, int(round(configs.ldc_channels * self.deep_stage2_ldc_channel_scale)))

            stage2_cfg = _clone_cfg(
                configs,
                n_pulse=pulse2,
                n_range=range2,
                lmf_chains=chains2,
                lmf_channels=lmf_ch2,
                ldc_channels=ldc_ch2,
            )
            self.stage2 = EchMo(configs=stage2_cfg)
            self.stage2_weight = nn.Parameter(torch.tensor(0.5, dtype=torch.float32))
        else:
            self.stage2 = None
            self.register_parameter('stage2_weight', None)

        self.logit_fuser = nn.Conv2d(self.n_classes * 2, self.n_classes, kernel_size=1, bias=True)

    def forward(self, x):
        y1 = self.stage1(x)
        if not self.deep_stage2_enable:
            return y1

        x2 = _resize_complex(x, self.stage2.n_pulse, self.stage2.n_range)
        y2 = self.stage2(x2)
        y2_up = F.interpolate(y2, size=y1.shape[-2:], mode='bilinear', align_corners=False)

        y_sum = y1 + self.stage2_weight * y2_up
        y_cat = torch.cat([y1, y2_up], dim=1)
        y_fused = self.logit_fuser(y_cat)

        # Residual fusion keeps stage-1 as the optimization anchor.
        return y_sum + y_fused
