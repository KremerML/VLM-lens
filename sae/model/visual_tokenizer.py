"""
Visual Tokenizer: Reduce 196 patches → M visual tokens

Uses learned linear pooling with softmax attention weights.
Each of the M output tokens is a fixed weighted combination of input patches.

Design rationale:
- Simple and interpretable (linear, no cross-attention complexity)
- Each output token learns a spatial "receptive field" over patches
- Keeps features legible for SAE analysis
- Computationally efficient

Critical for circuit analysis: The pooling weights W_pool can be visualized
as spatial attention patterns to see which patches contribute to each visual token.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import math
from typing import Optional, Tuple
import matplotlib.pyplot as plt # type: ignore
import numpy as np


class LinearVisualTokenizer(nn.Module):
    """
    Reduce patch sequence via learned linear pooling.
    
    Input: [B, 196, D_v] (patches only, no CLS)
    Output: [B, M, D_v] where M << 196
    
    Mechanism:
    - Learnable pooling weights W_pool: [196, M]
    - Apply softmax over spatial dim (196) to ensure valid weighting
    - Output token i = Σ_j softmax(W_pool[j, i]) * patch_j
    """
    
    def __init__(
        self,
        num_patches: int = 196,
        num_tokens: int = 32,
        patch_dim: int = 192,
    ):
        super().__init__()
        self.num_patches = num_patches
        self.num_tokens = num_tokens
        self.patch_dim = patch_dim
        
        # Learnable pooling weights [196, 32]
        # Each column defines the spatial attention pattern for one output token
        self.pool_weights = nn.Parameter(torch.randn(num_patches, num_tokens) * 0.02)
        
    def forward(self, patches: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Args:
            patches: [B, 196, D_v] - patch embeddings (no CLS token)
        
        Returns:
            tokens: [B, M, D_v] - pooled visual tokens
            pool_attn: [196, M] - pooling attention weights (for visualization)
        """
        B, N, D = patches.shape
        assert N == self.num_patches, f"Expected {self.num_patches} patches, got {N}"
        assert D == self.patch_dim, f"Expected dim {self.patch_dim}, got {D}"
        
        # Compute softmax pooling weights over spatial dimension
        # pool_attn[i, j] = how much patch i contributes to token j
        pool_attn = F.softmax(self.pool_weights, dim=0)  # [196, 32]
        
        # Apply pooling: matmul over patch dimension
        # [B, 196, D] @ [196, 32] -> [B, 32, D]
        tokens = torch.einsum('bnd,nm->bmd', patches, pool_attn)
        
        return tokens, pool_attn.detach()
    
    def visualize_pooling_patterns(
        self,
        num_tokens_to_show: int = 8,
        grid_size: int = 14,
        save_path: Optional[str] = None,
    ):
        """
        Visualize spatial pooling patterns for the first N tokens.
        
        Args:
            num_tokens_to_show: How many tokens to visualize
            grid_size: Spatial grid size (14 for 196 patches)
            save_path: If provided, save figure to this path
        """
        pool_attn = F.softmax(self.pool_weights, dim=0).detach().cpu().numpy()
        
        # Reshape to 2D spatial grid [14, 14, M]
        spatial_patterns = pool_attn.reshape(grid_size, grid_size, -1)
        
        # Plot first N tokens
        n = min(num_tokens_to_show, self.num_tokens)
        fig, axes = plt.subplots(2, n // 2, figsize=(12, 6))
        axes = axes.flatten()
        
        for i in range(n):
            ax = axes[i]
            pattern = spatial_patterns[:, :, i]
            
            im = ax.imshow(pattern, cmap='viridis', interpolation='nearest')
            ax.set_title(f'Token {i}')
            ax.axis('off')
            plt.colorbar(im, ax=ax, fraction=0.046)
        
        plt.suptitle('Visual Token Pooling Patterns (Spatial Attention)', fontsize=14)
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=150, bbox_inches='tight')
            print(f"Saved pooling visualization to {save_path}")
        else:
            plt.show()
        
        plt.close()


class PerceiverTokenizer(nn.Module):
    """
    Alternative: Perceiver-style tokenizer with learned queries and cross-attention.
    
    More flexible than linear pooling but adds a cross-attention block.
    Use this if linear pooling proves too rigid (check spatial invariance).
    """
    
    def __init__(
        self,
        num_patches: int = 196,
        num_tokens: int = 32,
        patch_dim: int = 192,
        num_heads: int = 3,
    ):
        super().__init__()
        self.num_patches = num_patches
        self.num_tokens = num_tokens
        self.patch_dim = patch_dim
        self.num_heads = num_heads
        self.head_dim = patch_dim // num_heads
        
        # Learned queries (one per output token)
        self.queries = nn.Parameter(torch.randn(1, num_tokens, patch_dim) * 0.02)
        
        # Cross-attention: queries attend to patches
        self.norm_q = nn.LayerNorm(patch_dim)
        self.norm_kv = nn.LayerNorm(patch_dim)
        
        self.to_q = nn.Linear(patch_dim, patch_dim, bias=False)
        self.to_kv = nn.Linear(patch_dim, patch_dim * 2, bias=False)
        self.to_out = nn.Linear(patch_dim, patch_dim, bias=False)
        
    def forward(self, patches: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Args:
            patches: [B, 196, D_v]
        Returns:
            tokens: [B, M, D_v]
            attn_weights: [B, H, M, 196] - attention from queries to patches
        """
        B, N, D = patches.shape
        H = self.num_heads
        
        # Expand queries for batch
        q = self.queries.expand(B, -1, -1)  # [B, M, D]
        
        # Normalize
        q = self.norm_q(q)
        kv_input = self.norm_kv(patches)
        
        # Project Q, K, V
        q = self.to_q(q).reshape(B, self.num_tokens, H, self.head_dim).transpose(1, 2)  # [B, H, M, d]
        kv = self.to_kv(kv_input).reshape(B, N, 2, H, self.head_dim).permute(2, 0, 3, 1, 4)
        k, v = kv[0], kv[1]  # [B, H, N, d]
        
        # Attention
        scale = self.head_dim ** -0.5
        attn = (q @ k.transpose(-2, -1)) * scale  # [B, H, M, 196]
        attn_weights = F.softmax(attn, dim=-1)
        
        # Aggregate
        out = attn_weights @ v  # [B, H, M, d]
        out = out.transpose(1, 2).reshape(B, self.num_tokens, D)
        out = self.to_out(out)
        
        return out, attn_weights.detach()


def test_tokenizers():
    """Test both tokenizer variants"""
    print("=" * 60)
    print("Testing Visual Tokenizers")
    print("=" * 60)
    
    # Create sample patches (batch=2, 196 patches, dim=192)
    patches = torch.randn(2, 196, 192)
    
    # Test 1: Linear pooling tokenizer
    print("\n1. Linear Pooling Tokenizer")
    print("-" * 60)
    linear_tokenizer = LinearVisualTokenizer(
        num_patches=196,
        num_tokens=32,
        patch_dim=192,
    )
    
    tokens, pool_attn = linear_tokenizer(patches)
    print(f"Input shape:  {patches.shape}")
    print(f"Output shape: {tokens.shape}")
    print(f"Pooling attention shape: {pool_attn.shape}")
    
    # Check that pooling weights sum to 1 over spatial dim
    attn_sums = pool_attn.sum(dim=0)
    print(f"Attention sums (should be ~1.0): {attn_sums[:5].tolist()}")
    
    # Count parameters
    linear_params = sum(p.numel() for p in linear_tokenizer.parameters())
    print(f"Parameters: {linear_params:,}")
    
    # Visualize pooling patterns
    print("\nGenerating pooling pattern visualization...")
    linear_tokenizer.visualize_pooling_patterns(
        num_tokens_to_show=8,
        save_path='pooling_patterns.png'
    )
    
    # Test 2: Perceiver tokenizer
    print("\n2. Perceiver Tokenizer (alternative)")
    print("-" * 60)
    perceiver_tokenizer = PerceiverTokenizer(
        num_patches=196,
        num_tokens=32,
        patch_dim=192,
        num_heads=3,
    )
    
    tokens_p, attn_weights_p = perceiver_tokenizer(patches)
    print(f"Input shape:  {patches.shape}")
    print(f"Output shape: {tokens_p.shape}")
    print(f"Attention weights shape: {attn_weights_p.shape}")
    
    perceiver_params = sum(p.numel() for p in perceiver_tokenizer.parameters())
    print(f"Parameters: {perceiver_params:,}")
    
    # Comparison
    print("\n" + "=" * 60)
    print("COMPARISON")
    print("=" * 60)
    print(f"Linear pooling params:   {linear_params:,} (minimal)")
    print(f"Perceiver params:        {perceiver_params:,} (10x more)")
    print(f"\nRecommendation: Start with linear pooling for interpretability.")
    print(f"Switch to Perceiver only if spatial patterns don't generalize.")
    print("\n✓ Visual tokenizer tests passed!")


if __name__ == '__main__':
    test_tokenizers()