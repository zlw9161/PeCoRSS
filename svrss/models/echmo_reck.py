# EchMo with optional ReCK controls (non-intrusive to baseline EchMo).
import math

import torch
import torch.nn as nn

from svrss.models.echmo import EchMo


class EchMoReCK(EchMo):
    """EchMo variant with configurable ReCK modes on frontend prior modules.

    ReCK mode definitions:
    - 0: freeze selected prior modules.
    - 1: scale selected prior module gradients by alpha in (0, 1).
    - 2: default EchMo behavior (no extra constraint/relaxation).
    - 3: random initialize selected prior modules (ignore prior init).
    """

    _MODULE_KEYS = ("lmf", "mldc", "gdf")

    def __init__(self, configs):
        super().__init__(configs)
        self._reck_hooks = []

        self.reck_mode = self._parse_reck_mode(
            getattr(configs, "reck_mode", getattr(configs, "ReCK_MODE", "reck2"))
        )
        self.reck_modules = self._parse_reck_modules(
            getattr(configs, "reck_modules", getattr(configs, "ReCK_Module", "all"))
        )
        self.reck_alpha = float(getattr(configs, "reck_alpha", getattr(configs, "ReCK_Alpha", 0.3)))
        self.reck_alpha_lmf = float(getattr(configs, "reck_alpha_lmf", self.reck_alpha))
        self.reck_alpha_mldc = float(getattr(configs, "reck_alpha_mldc", self.reck_alpha))
        self.reck_alpha_gdf = float(getattr(configs, "reck_alpha_gdf", self.reck_alpha))

        self.reck_alpha_map = {
            "lmf": self.reck_alpha_lmf,
            "mldc": self.reck_alpha_mldc,
            "gdf": self.reck_alpha_gdf,
        }
        self._apply_reck_policy()

    def _parse_reck_mode(self, mode):
        text = str(mode).strip().lower()
        mapping = {
            "0": 0,
            "reck0": 0,
            "freeze": 0,
            "1": 1,
            "reck1": 1,
            "slow": 1,
            "2": 2,
            "reck2": 2,
            "default": 2,
            "none": 2,
            "3": 3,
            "reck3": 3,
            "random": 3,
        }
        if text not in mapping:
            raise ValueError(f"Unsupported ReCK mode: {mode}. Use one of 0/1/2/3 or reck0/1/2/3.")
        return mapping[text]

    def _parse_reck_modules(self, modules):
        text = str(modules).strip().lower()
        if text in ("all", "frontend", "lmf,mldc,gdf"):
            return set(self._MODULE_KEYS)
        if text in ("none", "", "null"):
            return set()

        parsed = set()
        for token in text.replace("|", ",").split(","):
            key = token.strip()
            if not key:
                continue
            if key not in self._MODULE_KEYS:
                raise ValueError(
                    f"Unsupported ReCK module key: {key}. Use a subset of {self._MODULE_KEYS}."
                )
            parsed.add(key)
        return parsed

    def _iter_target_modules(self):
        if "lmf" in self.reck_modules:
            if getattr(self, "LMFChainParallel", None) is not None:
                yield "lmf", self.LMFChainParallel
            elif getattr(self, "LMFChain", None) is not None:
                yield "lmf", self.LMFChain
        if "mldc" in self.reck_modules and getattr(self, "MLDC", None) is not None:
            yield "mldc", self.MLDC
        if "gdf" in self.reck_modules and getattr(self, "GDF", None) is not None:
            yield "gdf", self.GDF

    def _random_reinit_module(self, module):
        for param in module.parameters():
            if not param.is_floating_point():
                continue
            if param.ndim >= 2:
                nn.init.kaiming_uniform_(param, a=math.sqrt(5))
            else:
                nn.init.zeros_(param)
            param.requires_grad = True

    def _freeze_module(self, module):
        for param in module.parameters():
            param.requires_grad = False

    def _register_grad_scale_hooks(self, module, alpha):
        scale = float(alpha)
        if not (0.0 < scale < 1.0):
            raise ValueError(f"ReCK-1 requires alpha in (0, 1), got {alpha}.")

        for param in module.parameters():
            if not param.requires_grad:
                param.requires_grad = True
            handle = param.register_hook(lambda grad, s=scale: grad * s if grad is not None else grad)
            self._reck_hooks.append(handle)

    def _apply_reck_policy(self):
        if self.reck_mode == 2 or not self.reck_modules:
            return

        for key, module in self._iter_target_modules():
            if self.reck_mode == 0:
                self._freeze_module(module)
            elif self.reck_mode == 1:
                self._register_grad_scale_hooks(module, self.reck_alpha_map[key])
            elif self.reck_mode == 3:
                self._random_reinit_module(module)

    def reck_info(self):
        return {
            "mode": self.reck_mode,
            "modules": sorted(list(self.reck_modules)),
            "alpha": self.reck_alpha,
            "alpha_lmf": self.reck_alpha_lmf,
            "alpha_mldc": self.reck_alpha_mldc,
            "alpha_gdf": self.reck_alpha_gdf,
        }
