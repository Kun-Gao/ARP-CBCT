from __future__ import annotations

from collections import OrderedDict

import torch

from .arp_cbct_v3 import ARPCBCTV3
from .hierarchical_decoder_balanced import BalancedHierarchicalDecoder3D


class ARPCBCTV4Balanced(ARPCBCTV3):
    """Minimal balanced multi-scale primitive fusion; all ray semantics stay V3."""

    def __init__(self, cfg: dict):
        super().__init__(cfg)
        coarse = int(cfg.get("v3_coarse_channels", cfg["primitive_dim"])); mid = int(cfg.get("v3_mid_channels", 64))
        self.decoder = BalancedHierarchicalDecoder3D(
            coarse_channels=coarse, mid_channels=mid, high_channels=int(cfg.get("v3_high_channels", 24)),
            coarse_blocks=int(cfg.get("v3_coarse_blocks", 10)), mid_blocks=int(cfg.get("v3_mid_blocks", 4)),
            high_blocks=int(cfg.get("v3_high_blocks", 2)), gradient_checkpointing=bool(cfg.get("decoder_gradient_checkpointing", False)),
            max_fullres_channels=int(cfg.get("v3_max_fullres_channels", 32)),
        )

    def load_state_dict(self, state_dict, strict: bool = True):
        """Split the immutable V3 concat convolution into two branch convolutions."""
        state = OrderedDict(state_dict)
        old = "decoder.mid_fusion.0.weight"
        if old in state:
            weight = state.pop(old); split = weight.shape[1] // 2
            state["decoder.mid_coarse.0.weight"] = weight[:, :split].clone()
            state["decoder.mid_primitive.0.weight"] = weight[:, split:].clone()
            for suffix in ("weight", "bias"):
                value = state.pop(f"decoder.mid_fusion.1.{suffix}")
                state[f"decoder.mid_coarse.1.{suffix}"] = value.clone()
                state[f"decoder.mid_primitive.1.{suffix}"] = value.clone()
            state["decoder.gamma_mid"] = torch.ones_like(self.decoder.gamma_mid)
        return super().load_state_dict(state, strict=strict)

    def forward(self, *args, **kwargs):
        output = super().forward(*args, **kwargs)
        output.update(self.decoder.last_balance_diagnostics)
        return output
