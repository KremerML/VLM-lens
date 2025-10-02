"""
GEGLU Projector: 192 → 256 with 1/sqrt(2) Residual

Bridges the ViT visual tokens (D_v = 192) to the text stack width (D_t = 256).

Design:
- Pre-LN on the 192-dim input for stability
- GEGLU gating to produce a 256-d representation
- Linear residual path projecting 192 → 256
- Residual merge scaled by 1/sqrt(2)
- Optional dropout after GEGLU
- Lightweight hooks for mech interp (pre-activations)

Shapes:
  input  : [B, M, 192]
  output : [B, M, 256]

References:
- Shazeer, "GLU Variants Improve Transformer" (2020)
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Dict, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F


# ----------------------
# Config
# ----------------------

@dataclass
class ProjectorConfig:
    in_dim: int = 192
    out_dim: int = 256
    dropout: float = 0.0
    use_layernorm: bool = True
    residual_scale: float = 1 / math.sqrt(2.0)


# ----------------------
# GEGLU primitive
# ----------------------

class GEGLU(nn.Module):
    """Gated GELU Linear Unit.

    Given x ∈ R^{*, d_in}, compute:
      a, b = x W_a + b_a, x W_b + b_b  (both in R^{*, d_out})
      y = a ⊙ GELU(b)

    Returns y and (a, b) for hooks.
    """

    def __init__(self, in_dim: int, out_dim: int, bias: bool = True):
        super().__init__()
        self.proj = nn.Linear(in_dim, out_dim * 2, bias=bias)

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, Tuple[torch.Tensor, torch.Tensor]]:
        a_b = self.proj(x)
        a, b = a_b.chunk(2, dim=-1)
        y = a * F.gelu(b)
        return y, (a, b)


# ----------------------
# Projector block
# ----------------------

class GEGLUProjector(nn.Module):
    """Project [B, M, 192] → [B, M, 256] with a GEGLU MLP and residual.

    Output = ( GEGLU(LN(x))  +  Linear_skip(x) ) * (1/√2)
    """

    def __init__(self, config: ProjectorConfig = ProjectorConfig()):
        super().__init__()
        self.config = config

        self.norm = nn.LayerNorm(config.in_dim, eps=1e-6) if config.use_layernorm else nn.Identity()
        self.geglu = GEGLU(config.in_dim, config.out_dim, bias=True)
        self.dropout = nn.Dropout(config.dropout)
        # Residual path to match dimensions
        self.residual = nn.Linear(config.in_dim, config.out_dim, bias=True)

        self._init_weights()

    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.trunc_normal_(m.weight, std=0.02)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
            elif isinstance(m, nn.LayerNorm):
                nn.init.ones_(m.weight)
                nn.init.zeros_(m.bias)

    @torch.no_grad()
    def flops(self, seq_len: int) -> int:
        """Rough FLOPs for one forward pass with sequence length M.
        Two linear layers of size in→2*out and in→out.
        """
        in_d, out_d = self.config.in_dim, self.config.out_dim
        return seq_len * (in_d * 2 * out_d + in_d * out_d)

    def forward(self, x: torch.Tensor, return_hooks: bool = False) -> Tuple[torch.Tensor, Optional[Dict[str, torch.Tensor]]]:
        """
        Args:
            x: [B, M, in_dim=192] visual tokens
            return_hooks: if True, returns dict with pre-activations for SAE
        Returns:
            y: [B, M, out_dim=256]
            hooks (optional): {
                'pre_a': a,           # [B, M, 256] pre-gate value branch
                'pre_b': b,           # [B, M, 256] pre-gate gate branch
                'geglu_out': geglu,   # [B, M, 256] post-gating before dropout
            }
        """
        assert x.dim() == 3 and x.size(-1) == self.config.in_dim, (
            f"Expected [B, M, {self.config.in_dim}], got {tuple(x.shape)}"
        )

        z = self.norm(x)
        geglu_out, (a, b) = self.geglu(z)           # [B, M, 256]
        geglu_out = self.dropout(geglu_out)

        skip = self.residual(x)                      # [B, M, 256]
        y = (geglu_out + skip) * self.config.residual_scale

        hooks = None
        if return_hooks:
            hooks = {
                'pre_a': a.detach(),
                'pre_b': b.detach(),
                'geglu_out': geglu_out.detach(),
            }
        return y, hooks


# ----------------------
# Self-test
# ----------------------

def _test():
    print("Testing GEGLUProjector…")
    cfg = ProjectorConfig()
    mod = GEGLUProjector(cfg)
    x = torch.randn(2, 32, 192)
    y, hooks = mod(x, return_hooks=True)
    assert y.shape == (2, 32, 256)
    print("Output:", y.shape)
    for k, v in hooks.items():
        print(k, v.shape)
    print("✓ Projector test passed")


if __name__ == "__main__":
    _test()
