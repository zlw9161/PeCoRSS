# End-to-end RSS network (MoE scalable variant)
# Mixture-of-Experts on top of EchMo while preserving LMF/MLDC/GDF design.
import torch
import torch.nn as nn

from svrss.models.echmo_debug import EchMo


class _CfgView:
    """Lightweight config clone for expert construction."""

    pass


def _clone_cfg(cfg, **overrides):
    c = _CfgView()
    c.__dict__.update(cfg.__dict__)
    for k, v in overrides.items():
        setattr(c, k, v)
    return c


class EchMo4MoE(nn.Module):
    """EchMo Mixture-of-Experts model.

    - Experts are standard EchMo backbones.
    - A lightweight gate predicts expert weights from input statistics.
    - Top-k routing is applied on weights (dense forward, sparse weighting).
    """

    def __init__(self, configs):
        super().__init__()
        self.cfg = configs
        self.n_classes = configs.n_classes

        self.moe_num_experts = max(1, int(getattr(configs, 'moe_num_experts', 4)))
        self.moe_topk = max(1, int(getattr(configs, 'moe_topk', 2)))
        self.moe_gate_hidden = max(8, int(getattr(configs, 'moe_gate_hidden', 32)))
        self.moe_temperature = float(getattr(configs, 'moe_temperature', 1.0))
        self.moe_dropout = float(getattr(configs, 'moe_dropout', 0.0))

        self.moe_topk = min(self.moe_topk, self.moe_num_experts)
        self.moe_temperature = max(self.moe_temperature, 1e-6)

        expert_cfg = _clone_cfg(configs)
        self.experts = nn.ModuleList([EchMo(configs=expert_cfg) for _ in range(self.moe_num_experts)])

        # Gate uses compact global statistics from complex input.
        self.gate = nn.Sequential(
            nn.Linear(6, self.moe_gate_hidden),
            nn.GELU(),
            nn.Dropout(self.moe_dropout),
            nn.Linear(self.moe_gate_hidden, self.moe_num_experts),
        )

    def _extract_gate_features(self, x):
        dims = tuple(range(1, x.ndim))
        amp = torch.abs(x)
        phase = torch.angle(x)
        real = x.real
        imag = x.imag

        mean_amp = amp.mean(dim=dims)
        std_amp = amp.std(dim=dims, unbiased=False)
        max_amp = amp.amax(dim=dims)
        mean_real = real.mean(dim=dims)
        mean_imag = imag.mean(dim=dims)
        std_phase = phase.std(dim=dims, unbiased=False)

        feats = torch.stack([mean_amp, std_amp, max_amp, mean_real, mean_imag, std_phase], dim=1)
        return feats

    def _get_expert_weights(self, x):
        feats = self._extract_gate_features(x)
        logits = self.gate(feats) / self.moe_temperature
        weights = torch.softmax(logits, dim=-1)

        if self.moe_topk < self.moe_num_experts:
            top_vals, top_idx = torch.topk(weights, k=self.moe_topk, dim=-1)
            mask = torch.zeros_like(weights)
            mask.scatter_(1, top_idx, 1.0)
            weights = weights * mask
            weights = weights / (weights.sum(dim=-1, keepdim=True) + 1e-8)

        return weights

    def forward(self, x):
        weights = self._get_expert_weights(x)  # [B, E]
        expert_logits = [expert(x) for expert in self.experts]  # E * [B, K, P, R]
        stacked = torch.stack(expert_logits, dim=1)  # [B, E, K, P, R]
        fused = (stacked * weights[:, :, None, None, None]).sum(dim=1)
        return fused
