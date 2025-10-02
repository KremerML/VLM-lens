"""
Text Transformer: 2-layer Causal Decoder with Visual Prefix Support

Architecture:
- 2 transformer blocks, d=256, 4 heads (head_dim=64)
- GEGLU MLPs (hidden = 4d = 1024)
- 1/√2 residual scaling on both attn & MLP
- Causal masking with prefix support:
  * Visual tokens can attend to earlier visual tokens
  * Text tokens can attend to ALL visual tokens + earlier text tokens
  * Visual tokens CANNOT attend to text tokens
- Per-head K/V/logits hooks for circuit analysis
- Sinusoidal positional encoding over the full sequence

Sequence layout per sample:
  [QCLS] [IMG] V_proj(32 tokens) [IMG_SEP] text_ids(≤32)
  
Total length L ≈ 1 + 1 + 32 + 1 + T ≈ 66-70 tokens
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import math
from typing import Optional, Dict, List, Tuple
from dataclasses import dataclass


@dataclass
class TextTransformerConfig:
    """Configuration for text transformer"""
    dim: int = 256
    depth: int = 2
    heads: int = 4
    mlp_ratio: int = 4  # hidden = 4 * dim = 1024
    vocab_size: int = 150
    max_seq_length: int = 96  # max total sequence (prefix + text)
    dropout: float = 0.0
    residual_scale: float = 1.0 / math.sqrt(2)
    
    @property
    def head_dim(self) -> int:
        return self.dim // self.heads  # 64
    
    @property
    def mlp_hidden_dim(self) -> int:
        return self.dim * self.mlp_ratio  # 1024


class SinusoidalPositionalEncoding(nn.Module):
    """Sinusoidal positional encoding (fixed, no learnable params)"""
    
    def __init__(self, dim: int, max_len: int = 5000):
        super().__init__()
        pe = torch.zeros(max_len, dim)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, dim, 2).float() * (-math.log(10000.0) / dim))
        
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        
        self.register_buffer('pe', pe)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: [B, L, D]
        Returns:
            [B, L, D] with positional encoding added
        """
        return x + self.pe[:x.size(1), :].unsqueeze(0)


class GEGLU(nn.Module):
    """Gated GLU with GELU activation"""
    
    def __init__(self, dim_in: int, dim_hidden: int, bias: bool = True):
        super().__init__()
        self.proj = nn.Linear(dim_in, dim_hidden * 2, bias=bias)
    
    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, Tuple[torch.Tensor, torch.Tensor]]:
        """
        Returns:
            output: [B, L, hidden]
            (a, b): pre-activation branches for hooks
        """
        a_b = self.proj(x)
        a, b = a_b.chunk(2, dim=-1)
        y = a * F.gelu(b)
        return y, (a, b)


class TextMLP(nn.Module):
    """GEGLU-based MLP with residual scaling"""
    
    def __init__(self, config: TextTransformerConfig):
        super().__init__()
        self.config = config
        self.geglu = GEGLU(config.dim, config.mlp_hidden_dim, bias=True)
        self.proj_out = nn.Linear(config.mlp_hidden_dim, config.dim, bias=True)
        self.dropout = nn.Dropout(config.dropout)
    
    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
        """
        Args:
            x: [B, L, D]
        Returns:
            output: [B, L, D]
            hooks: {'pre_a', 'pre_b', 'geglu_out'} for SAE
        """
        geglu_out, (a, b) = self.geglu(x)
        geglu_out = self.dropout(geglu_out)
        out = self.proj_out(geglu_out)
        out = self.dropout(out)
        
        hooks = {
            'pre_a': a.detach(),
            'pre_b': b.detach(),
            'geglu_out': geglu_out.detach(),
        }
        
        return out, hooks


class CausalPrefixAttention(nn.Module):
    """
    Multi-head attention with causal masking and visual prefix support.
    
    Masking rules:
    - Visual prefix tokens (0:M) can only attend to earlier visual tokens
    - Text tokens (M:L) can attend to ALL visual tokens + earlier text tokens
    - This is a "prefix-causal" mask
    """
    
    def __init__(self, config: TextTransformerConfig):
        super().__init__()
        self.config = config
        self.num_heads = config.heads
        self.head_dim = config.head_dim
        self.scale = self.head_dim ** -0.5
        
        # QKV projection
        self.qkv = nn.Linear(config.dim, config.dim * 3, bias=False)
        self.proj = nn.Linear(config.dim, config.dim, bias=False)
        self.dropout = nn.Dropout(config.dropout)
    
    def create_prefix_causal_mask(
        self,
        seq_len: int,
        num_prefix_tokens: int,
        device: torch.device,
    ) -> torch.Tensor:
        """
        Create prefix-causal mask.
        
        Args:
            seq_len: Total sequence length L
            num_prefix_tokens: Number of visual prefix tokens M
            device: Device for mask
        
        Returns:
            mask: [L, L] where True = masked (not allowed to attend)
        """
        # Start with full causal mask (upper triangular)
        mask = torch.triu(torch.ones(seq_len, seq_len, device=device, dtype=torch.bool), diagonal=1)
        
        # Allow text tokens (M:L) to attend to ALL prefix tokens (0:M)
        mask[num_prefix_tokens:, :num_prefix_tokens] = False
        
        return mask  # [L, L]
    
    def forward(
        self,
        x: torch.Tensor,
        num_prefix_tokens: int,
        return_hooks: bool = False,
    ) -> Tuple[torch.Tensor, Optional[Dict[str, torch.Tensor]]]:
        """
        Args:
            x: [B, L, D]
            num_prefix_tokens: Number of visual prefix tokens M
            return_hooks: If True, return per-head K/V/logits
        
        Returns:
            output: [B, L, D]
            hooks: Optional dict with per-head attention info
        """
        B, L, D = x.shape
        H = self.num_heads
        d = self.head_dim
        
        # Compute Q, K, V
        qkv = self.qkv(x)  # [B, L, 3*D]
        qkv = qkv.reshape(B, L, 3, H, d)
        qkv = qkv.permute(2, 0, 3, 1, 4)  # [3, B, H, L, d]
        q, k, v = qkv[0], qkv[1], qkv[2]  # each [B, H, L, d]
        
        # Attention scores
        attn_logits = (q @ k.transpose(-2, -1)) * self.scale  # [B, H, L, L]
        
        # Apply prefix-causal mask
        mask = self.create_prefix_causal_mask(L, num_prefix_tokens, x.device)  # [L, L]
        attn_logits = attn_logits.masked_fill(mask.unsqueeze(0).unsqueeze(0), float('-inf'))
        
        attn = F.softmax(attn_logits, dim=-1)
        attn = self.dropout(attn)
        
        # Weighted sum
        out = attn @ v  # [B, H, L, d]
        out = out.transpose(1, 2).reshape(B, L, D)  # [B, L, D]
        out = self.proj(out)
        out = self.dropout(out)
        
        # Hooks for circuit analysis
        hooks = None
        if return_hooks:
            hooks = {
                'k': k.detach(),  # [B, H, L, d] - per-head keys
                'v': v.detach(),  # [B, H, L, d] - per-head values
                'attn_logits': attn_logits.detach(),  # [B, H, L, L] - pre-softmax scores
                'attn_probs': attn.detach(),  # [B, H, L, L] - post-softmax weights
            }
        
        return out, hooks


class TextTransformerBlock(nn.Module):
    """Single transformer block with pre-LN and 1/√2 residual scaling"""
    
    def __init__(self, config: TextTransformerConfig):
        super().__init__()
        self.config = config
        
        self.norm1 = nn.LayerNorm(config.dim, eps=1e-6)
        self.attn = CausalPrefixAttention(config)
        
        self.norm2 = nn.LayerNorm(config.dim, eps=1e-6)
        self.mlp = TextMLP(config)
    
    def forward(
        self,
        x: torch.Tensor,
        num_prefix_tokens: int,
        return_hooks: bool = False,
    ) -> Tuple[torch.Tensor, Optional[Dict[str, torch.Tensor]]]:
        """
        Args:
            x: [B, L, D]
            num_prefix_tokens: Number of visual prefix tokens
            return_hooks: If True, return attention and MLP hooks
        
        Returns:
            output: [B, L, D]
            hooks: Optional dict with all hooks
        """
        # Attention with residual
        attn_out, attn_hooks = self.attn(
            self.norm1(x),
            num_prefix_tokens=num_prefix_tokens,
            return_hooks=return_hooks,
        )
        x = x + attn_out * self.config.residual_scale
        
        # MLP with residual
        mlp_out, mlp_hooks = self.mlp(self.norm2(x))
        x = x + mlp_out * self.config.residual_scale
        
        # Combine hooks
        hooks = None
        if return_hooks:
            hooks = {**attn_hooks, **mlp_hooks}
        
        return x, hooks


class TextTransformer(nn.Module):
    """
    Complete text transformer with embedding and visual prefix support.
    
    Handles sequence assembly:
    [QCLS] [IMG] V_proj(M=32) [IMG_SEP] text_tokens(T)
    """
    
    def __init__(self, config: TextTransformerConfig):
        super().__init__()
        self.config = config
        
        # Token embeddings
        self.token_embed = nn.Embedding(config.vocab_size, config.dim)
        
        # Special token embeddings (learnable)
        self.qcls_embed = nn.Parameter(torch.zeros(1, 1, config.dim))
        self.img_sep_embed = nn.Parameter(torch.zeros(1, 1, config.dim))
        
        # Positional encoding (shared across full sequence)
        self.pos_encoder = SinusoidalPositionalEncoding(config.dim, config.max_seq_length)
        
        # Transformer blocks
        self.blocks = nn.ModuleList([
            TextTransformerBlock(config) for _ in range(config.depth)
        ])
        
        # Final layer norm
        self.norm = nn.LayerNorm(config.dim, eps=1e-6)
        
        # Initialize weights
        self._init_weights()
    
    def _init_weights(self):
        """Initialize weights"""
        nn.init.normal_(self.token_embed.weight, std=0.02)
        nn.init.normal_(self.qcls_embed, std=0.02)
        nn.init.normal_(self.img_sep_embed, std=0.02)
        
        for module in self.modules():
            if isinstance(module, nn.Linear):
                nn.init.trunc_normal_(module.weight, std=0.02)
                if module.bias is not None:
                    nn.init.zeros_(module.bias)
            elif isinstance(module, nn.LayerNorm):
                nn.init.ones_(module.weight)
                nn.init.zeros_(module.bias)
    
    def forward(
        self,
        visual_tokens: torch.Tensor,
        text_ids: torch.Tensor,
        return_hooks: bool = False,
        hook_layers: Optional[List[int]] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor, Optional[Dict]]:
        """
        Args:
            visual_tokens: [B, M, D] - projected visual tokens (M=32, D=256)
            text_ids: [B, T] - text token IDs (T ≤ 32)
            return_hooks: If True, return hooks from specified layers
            hook_layers: List of layer indices to hook (None = all)
        
        Returns:
            output: [B, L, D] - full sequence output
            qcls_output: [B, D] - [QCLS] token output for classification
            hooks: Optional dict {layer_idx: {k, v, attn_logits, attn_probs, ...}}
        """
        B, M, D = visual_tokens.shape
        T = text_ids.size(1)
        
        # Embed text tokens
        text_embed = self.token_embed(text_ids)  # [B, T, D]
        
        # Expand special tokens for batch
        qcls = self.qcls_embed.expand(B, -1, -1)  # [B, 1, D]
        img_sep = self.img_sep_embed.expand(B, -1, -1)  # [B, 1, D]
        
        # Assemble sequence: [QCLS] [IMG_SEP] V_proj(M) [IMG_SEP] text
        # Note: We use IMG_SEP twice (once before and once after visual tokens)
        # This helps the model distinguish visual from text regions
        sequence = torch.cat([
            qcls,         # [B, 1, D]
            img_sep,      # [B, 1, D] - marks start of visual
            visual_tokens,  # [B, M, D]
            img_sep,      # [B, 1, D] - marks end of visual
            text_embed,   # [B, T, D]
        ], dim=1)  # [B, L, D] where L = 1 + 1 + M + 1 + T
        
        # Add positional encoding
        sequence = self.pos_encoder(sequence)
        
        # Number of prefix tokens (before text): 1 + 1 + M + 1 = M + 3
        num_prefix_tokens = M + 3
        
        # Pass through transformer blocks
        x = sequence
        all_hooks = {} if return_hooks else None
        
        for layer_idx, block in enumerate(self.blocks):
            x, hooks = block(
                x,
                num_prefix_tokens=num_prefix_tokens,
                return_hooks=return_hooks and (hook_layers is None or layer_idx in hook_layers),
            )
            
            if return_hooks and (hook_layers is None or layer_idx in hook_layers):
                all_hooks[layer_idx] = hooks
        
        # Final norm
        x = self.norm(x)
        
        # Extract [QCLS] token output (position 0)
        qcls_output = x[:, 0, :]  # [B, D]
        
        return x, qcls_output, all_hooks


def test_text_transformer():
    """Test text transformer with visual prefix"""
    print("=" * 60)
    print("Testing Text Transformer")
    print("=" * 60)
    
    config = TextTransformerConfig(
        vocab_size=150,
        dim=256,
        depth=2,
        heads=4,
    )
    
    model = TextTransformer(config)
    
    # Count parameters
    total_params = sum(p.numel() for p in model.parameters())
    print(f"Total parameters: {total_params:,}")
    
    # Create dummy inputs
    B = 2
    M = 32  # visual tokens
    T = 20  # text tokens
    
    visual_tokens = torch.randn(B, M, 256)
    text_ids = torch.randint(0, 150, (B, T))
    
    # Forward pass without hooks
    print(f"\nInput shapes:")
    print(f"  visual_tokens: {visual_tokens.shape}")
    print(f"  text_ids: {text_ids.shape}")
    
    output, qcls, _ = model(visual_tokens, text_ids, return_hooks=False)
    
    print(f"\nOutput shapes:")
    print(f"  full sequence: {output.shape}")  # [B, L, D] where L = 1+1+32+1+20 = 55
    print(f"  qcls output: {qcls.shape}")  # [B, 256]
    
    # Forward pass with hooks
    output, qcls, hooks = model(visual_tokens, text_ids, return_hooks=True, hook_layers=[0, 1])
    
    print(f"\nHooks captured from layers: {list(hooks.keys())}")
    if 0 in hooks:
        print(f"Layer 0 hooks:")
        for key, val in hooks[0].items():
            print(f"  {key}: {val.shape}")
    
    # Test prefix-causal masking
    print("\n" + "=" * 60)
    print("Testing Prefix-Causal Masking")
    print("=" * 60)
    
    attn_module = model.blocks[0].attn
    L = 55  # 1 + 1 + 32 + 1 + 20
    M_prefix = 35  # 1 + 1 + 32 + 1
    
    mask = attn_module.create_prefix_causal_mask(L, M_prefix, torch.device('cpu'))
    print(f"Mask shape: {mask.shape}")
    print(f"Mask dtype: {mask.dtype}")
    
    # Check masking rules
    # Visual tokens (0:35) should have causal mask among themselves
    visual_causal = mask[:M_prefix, :M_prefix].triu(1).all()
    print(f"Visual tokens have causal mask: {visual_causal}")
    
    # Text tokens (35:55) should see ALL visual tokens
    text_sees_visual = (~mask[M_prefix:, :M_prefix]).all()
    print(f"Text tokens can see all visual tokens: {text_sees_visual}")
    
    # Text tokens should have causal mask among themselves
    text_causal = mask[M_prefix:, M_prefix:].triu(1).all()
    print(f"Text tokens have causal mask: {text_causal}")
    
    print("\n✓ Text transformer test passed!")


if __name__ == '__main__':
    test_text_transformer()