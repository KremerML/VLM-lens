"""
ViT-Tiny Vision Encoder with SAE Hook Points

Architecture:
- 6 transformer layers
- d_model = 192, n_heads = 3, head_dim = 64
- Patch size 16x16 → 14x14 = 196 patches
- 2D sinusoidal positional encoding
- [VCLS] token prepended
- Hook points at MLP pre-activations (layers 2, 4) and attention K/V
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import math
from typing import Optional, Dict, Tuple, List
from dataclasses import dataclass


@dataclass
class ViTConfig:
    """ViT-tiny configuration"""
    image_size: int = 224
    patch_size: int = 16
    in_channels: int = 3
    dim: int = 192
    depth: int = 6
    heads: int = 3
    mlp_ratio: int = 4  # hidden_dim = dim * 4 = 768
    dropout: float = 0.0
    
    @property
    def num_patches(self) -> int:
        return (self.image_size // self.patch_size) ** 2  # 196
    
    @property
    def head_dim(self) -> int:
        return self.dim // self.heads  # 64


class PatchEmbed(nn.Module):
    """Convert image to patch embeddings"""
    
    def __init__(self, config: ViTConfig):
        super().__init__()
        self.config = config
        
        # Conv2d acts as patch projection
        self.proj = nn.Conv2d(
            config.in_channels,
            config.dim,
            kernel_size=config.patch_size,
            stride=config.patch_size,
        )
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: [B, 3, 224, 224]
        Returns:
            patches: [B, 196, 192]
        """
        x = self.proj(x)  # [B, 192, 14, 14]
        x = x.flatten(2)  # [B, 192, 196]
        x = x.transpose(1, 2)  # [B, 196, 192]
        return x


class Attention(nn.Module):
    """Multi-head self-attention with hook points"""
    
    def __init__(self, config: ViTConfig):
        super().__init__()
        self.config = config
        self.num_heads = config.heads
        self.head_dim = config.head_dim
        self.scale = self.head_dim ** -0.5
        
        # QKV projection
        self.qkv = nn.Linear(config.dim, config.dim * 3, bias=False)
        self.proj = nn.Linear(config.dim, config.dim, bias=False)
        self.dropout = nn.Dropout(config.dropout)
        
    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
        """
        Args:
            x: [B, L, D] where L = 197 (1 cls + 196 patches)
        Returns:
            output: [B, L, D]
            hooks: dict with 'k', 'v', 'attn_logits' for SAE capture
        """
        B, L, D = x.shape
        
        # Compute Q, K, V
        qkv = self.qkv(x)  # [B, L, 3*D]
        qkv = qkv.reshape(B, L, 3, self.num_heads, self.head_dim)
        qkv = qkv.permute(2, 0, 3, 1, 4)  # [3, B, H, L, head_dim]
        q, k, v = qkv[0], qkv[1], qkv[2]
        
        # Attention scores
        attn_logits = (q @ k.transpose(-2, -1)) * self.scale  # [B, H, L, L]
        attn = F.softmax(attn_logits, dim=-1)
        attn = self.dropout(attn)
        
        # Weighted sum
        out = attn @ v  # [B, H, L, head_dim]
        out = out.transpose(1, 2).reshape(B, L, D)  # [B, L, D]
        out = self.proj(out)
        out = self.dropout(out)
        
        # Hook points for SAE
        hooks = {
            'k': k.detach(),  # [B, H, L, head_dim]
            'v': v.detach(),  # [B, H, L, head_dim]
            'attn_logits': attn_logits.detach(),  # [B, H, L, L]
        }
        
        return out, hooks


class MLP(nn.Module):
    """MLP block with GELU activation and pre-activation hook"""
    
    def __init__(self, config: ViTConfig):
        super().__init__()
        hidden_dim = config.dim * config.mlp_ratio
        self.fc1 = nn.Linear(config.dim, hidden_dim, bias=True)
        self.fc2 = nn.Linear(hidden_dim, config.dim, bias=True)
        self.dropout = nn.Dropout(config.dropout)
    
    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Args:
            x: [B, L, D]
        Returns:
            output: [B, L, D]
            pre_act: [B, L, hidden_dim] - PRE-GELU activations for SAE
        """
        pre_act = self.fc1(x)  # [B, L, hidden_dim] - CRITICAL: before GELU
        h = F.gelu(pre_act)
        h = self.dropout(h)
        out = self.fc2(h)
        out = self.dropout(out)
        
        return out, pre_act.detach()  # detach hook to save memory


class TransformerBlock(nn.Module):
    """Standard transformer block with pre-LN and hooks"""
    
    def __init__(self, config: ViTConfig):
        super().__init__()
        self.norm1 = nn.LayerNorm(config.dim, eps=1e-6)
        self.attn = Attention(config)
        self.norm2 = nn.LayerNorm(config.dim, eps=1e-6)
        self.mlp = MLP(config)
    
    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
        """
        Args:
            x: [B, L, D]
        Returns:
            output: [B, L, D]
            hooks: dict with attention and MLP hooks
        """
        # Attention with residual
        attn_out, attn_hooks = self.attn(self.norm1(x))
        x = x + attn_out
        
        # MLP with residual
        mlp_out, mlp_pre_act = self.mlp(self.norm2(x))
        x = x + mlp_out
        
        # Combine hooks
        hooks = {
            **attn_hooks,
            'mlp_pre_act': mlp_pre_act,  # [B, L, hidden_dim]
        }
        
        return x, hooks


class ViTTiny(nn.Module):
    """Complete ViT-tiny with hook collection"""
    
    def __init__(self, config: ViTConfig):
        super().__init__()
        self.config = config
        
        # Patch embedding
        self.patch_embed = PatchEmbed(config)
        
        # Class token
        self.cls_token = nn.Parameter(torch.zeros(1, 1, config.dim))
        nn.init.normal_(self.cls_token, std=0.02)
        
        # Positional encoding (2D sinusoidal)
        self.pos_embed = self._build_2d_sinusoidal_position_embedding()
        
        # Transformer blocks
        self.blocks = nn.ModuleList([
            TransformerBlock(config) for _ in range(config.depth)
        ])
        
        # Final layer norm
        self.norm = nn.LayerNorm(config.dim, eps=1e-6)
        
        # Initialize weights
        self.apply(self._init_weights)
    
    def _build_2d_sinusoidal_position_embedding(self) -> nn.Parameter:
        """Create 2D sinusoidal positional embeddings for patches + cls token"""
        num_patches = self.config.num_patches
        dim = self.config.dim
        
        # Grid size
        grid_size = int(math.sqrt(num_patches))  # 14
        
        # Create 2D grid
        y_pos = torch.arange(grid_size, dtype=torch.float32)
        x_pos = torch.arange(grid_size, dtype=torch.float32)
        y_grid, x_grid = torch.meshgrid(y_pos, x_pos, indexing='ij')
        
        # Flatten to [196, 2]
        grid = torch.stack([y_grid.flatten(), x_grid.flatten()], dim=-1)
        
        # Sinusoidal encoding for each dimension
        div_term = torch.exp(torch.arange(0, dim // 2, dtype=torch.float32) * 
                            (-math.log(10000.0) / (dim // 2)))
        
        pe = torch.zeros(num_patches, dim)
        pe[:, 0::2] = torch.sin(grid[:, 0:1] * div_term)
        pe[:, 1::2] = torch.cos(grid[:, 1:2] * div_term)
        
        # Add cls token position (all zeros)
        cls_pe = torch.zeros(1, dim)
        pe = torch.cat([cls_pe, pe], dim=0)  # [197, 192]
        
        return nn.Parameter(pe.unsqueeze(0), requires_grad=False)  # [1, 197, 192]
    
    def _init_weights(self, m):
        """Initialize weights"""
        if isinstance(m, nn.Linear):
            nn.init.trunc_normal_(m.weight, std=0.02)
            if m.bias is not None:
                nn.init.zeros_(m.bias)
        elif isinstance(m, nn.LayerNorm):
            nn.init.ones_(m.weight)
            nn.init.zeros_(m.bias)
    
    def forward(
        self, 
        x: torch.Tensor,
        return_hooks: bool = False,
        hook_layers: Optional[List[int]] = None,
    ) -> Tuple[torch.Tensor, Optional[Dict]]:
        """
        Args:
            x: [B, 3, 224, 224]
            return_hooks: If True, return hooks from specified layers
            hook_layers: List of layer indices to capture (e.g., [2, 4] for SAE)
        Returns:
            patches: [B, 197, 192] (includes [VCLS] at position 0)
            hooks: Optional dict of {layer_idx: {k, v, attn_logits, mlp_pre_act}}
        """
        B = x.shape[0]
        
        # Patch embedding
        x = self.patch_embed(x)  # [B, 196, 192]
        
        # Prepend class token
        cls_tokens = self.cls_token.expand(B, -1, -1)  # [B, 1, 192]
        x = torch.cat([cls_tokens, x], dim=1)  # [B, 197, 192]
        
        # Add positional encoding
        x = x + self.pos_embed  # [B, 197, 192]
        
        # Collect hooks if requested
        all_hooks = {} if return_hooks else None
        
        # Transformer blocks
        for layer_idx, block in enumerate(self.blocks):
            x, hooks = block(x)
            
            # Store hooks for specified layers
            if return_hooks and (hook_layers is None or layer_idx in hook_layers):
                all_hooks[layer_idx] = hooks
        
        # Final norm
        x = self.norm(x)  # [B, 197, 192]
        
        return x, all_hooks


def test_vit():
    """Test ViT-tiny with hook capture"""
    print("Testing ViT-tiny...")
    
    config = ViTConfig()
    model = ViTTiny(config)
    
    # Count parameters
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Total params: {total_params:,}")
    print(f"Trainable params: {trainable_params:,}")
    
    # Test forward pass
    x = torch.randn(2, 3, 224, 224)
    
    # Without hooks
    patches, _ = model(x, return_hooks=False)
    print(f"\nOutput shape: {patches.shape}")  # [2, 197, 192]
    
    # With hooks (layers 2 and 4)
    patches, hooks = model(x, return_hooks=True, hook_layers=[2, 4])
    print(f"\nHook layers captured: {list(hooks.keys())}")
    
    if 2 in hooks:
        print(f"Layer 2 hooks:")
        for key, val in hooks[2].items():
            print(f"  {key}: {val.shape}")
    
    print("\n✓ ViT-tiny test passed!")


if __name__ == '__main__':
    test_vit()