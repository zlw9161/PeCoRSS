# End-to-end RSS network (memo scalable variant)
# Cross-CPI memory on top of EchMo while keeping LMF/MLDC/GDF design intact.
import torch
import torch.nn as nn

from svrss.models.echmo import EchMo


class _CfgView:
    """Lightweight config clone for memo backbone construction."""

    pass


def _clone_cfg(cfg, **overrides):
    c = _CfgView()
    c.__dict__.update(cfg.__dict__)
    for k, v in overrides.items():
        setattr(c, k, v)
    return c


class EchMoMemo(nn.Module):
    """EchMo with cross-CPI memory accumulation.

    Input can contain multiple CPI frames packed on channel-like axis.
    For each CPI frame, EchMo backbone produces logits; an EMA-style memory
    state is updated and injected into output logits.
    """

    def __init__(self, configs):
        super().__init__()
        self.cfg = configs
        self.n_classes = configs.n_classes

        # Keep backbone behavior unchanged; memory is built outside EchMo.
        backbone_cfg = _clone_cfg(configs, n_frames=1)
        self.backbone = EchMo(configs=backbone_cfg)

        self.memo_enable = bool(getattr(configs, 'memo_enable', True))
        self.memo_beta = float(getattr(configs, 'memo_beta', 0.8))
        self.memo_gamma = float(getattr(configs, 'memo_gamma', 0.4))
        self.memo_use_state_proj = bool(getattr(configs, 'memo_use_state_proj', True))

        if self.memo_use_state_proj:
            self.state_proj = nn.Conv2d(self.n_classes, self.n_classes, kernel_size=1, bias=True)
        else:
            self.state_proj = nn.Identity()

    def _split_cpi_frames(self, x):
        """Normalize input to a list of CPI tensors with shape [B, 1, P, R]."""
        if x.ndim == 4:
            # [B, T, P, R] (T treated as CPI count) or [B, 1, P, R]
            if x.shape[1] == 1:
                return [x]
            return [x[:, t:t + 1, ...] for t in range(x.shape[1])]

        if x.ndim == 5:
            # Common temporal format: [B, 1, T, P, R]
            if x.shape[1] == 1:
                return [x[:, :, t, ...] for t in range(x.shape[2])]
            # Fallback: [B, T, 1, P, R]
            if x.shape[2] == 1:
                return [x[:, t, :, ...] for t in range(x.shape[1])]

        raise ValueError(f'Unsupported input shape for EchMoMemo: {tuple(x.shape)}')

    def forward(self, x):
        frames = self._split_cpi_frames(x)
        if len(frames) == 1 or not self.memo_enable:
            return self.backbone(frames[-1])

        beta = max(0.0, min(self.memo_beta, 1.0))
        gamma = max(0.0, self.memo_gamma)

        h = None
        y_out = None
        for frame in frames:
            y = self.backbone(frame)
            if h is None:
                h = y
            else:
                h = beta * h + (1.0 - beta) * y
            h_proj = self.state_proj(h)
            y_out = y + gamma * h_proj

        return y_out
