"""
Complete VLM Wrapper: Assembles all components + heads

Architecture flow:
  Image → ViT → Visual Tokenizer → Projector → Text Transformer → Heads

Components:
- Vision: ViT-tiny (6 layers, d=192)
- Tokenizer: Linear pooling (196→32)
- Projector: GEGLU (192→256)
- Text: 2-layer transformer (d=256, causal prefix)
- Heads: QA (discriminative), Attribute auxiliaries

Hooks collected from:
- ViT layers 2, 4 (MLP pre-acts, attn K/V)
- Projector (GEGLU pre-acts)
- Text layers 0, 1 (per-head K/V, attn logits, MLP pre-acts)
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass

# Import all components from the same directory
from sae.model.vit_tiny_model import ViTTiny, ViTConfig
from sae.model.visual_tokenizer import LinearVisualTokenizer
from sae.model.geglu_projector import GEGLUProjector, ProjectorConfig
from sae.model.text_transformer import TextTransformer, TextTransformerConfig


@dataclass
class VLMConfig:
    """Complete VLM configuration"""
    # Image/Vision
    image_size: int = 224
    patch_size: int = 16
    
    # ViT
    vision_dim: int = 192
    vision_depth: int = 6
    vision_heads: int = 3
    
    # Visual tokenizer
    num_visual_tokens: int = 32
    
    # Projector (192→256)
    projector_hidden: int = 512
    
    # Text transformer
    text_dim: int = 256
    text_depth: int = 2
    text_heads: int = 4
    vocab_size: int = 150
    max_text_length: int = 32
    
    # Heads
    num_answer_classes: int = 150  # discriminative QA
    
    # Attribute heads (auxiliary)
    num_colors: int = 6
    num_shapes: int = 3
    num_sizes: int = 2
    
    # Training
    dropout: float = 0.0
    
    # Hook layers (for SAE)
    vision_hook_layers: List[int] = None  # default: [2, 4]
    text_hook_layers: List[int] = None    # default: [0, 1]
    
    def __post_init__(self):
        if self.vision_hook_layers is None:
            self.vision_hook_layers = [2, 4]
        if self.text_hook_layers is None:
            self.text_hook_layers = [0, 1]


class AttributeHeads(nn.Module):
    """Auxiliary attribute prediction heads on ViT features"""
    
    def __init__(self, config: VLMConfig):
        super().__init__()
        self.config = config
        
        # Take features from specific ViT layers
        # We'll use mean-pooled patches (not CLS)
        hidden_dim = config.vision_dim
        
        # Separate heads for each attribute type
        self.color_head = nn.Sequential(
            nn.LayerNorm(hidden_dim),
            nn.Linear(hidden_dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, config.num_colors),
        )
        
        self.shape_head = nn.Sequential(
            nn.LayerNorm(hidden_dim),
            nn.Linear(hidden_dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, config.num_shapes),
        )
        
        self.size_head = nn.Sequential(
            nn.LayerNorm(hidden_dim),
            nn.Linear(hidden_dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, config.num_sizes),
        )
    
    def forward(self, vision_features: torch.Tensor) -> Dict[str, torch.Tensor]:
        """
        Args:
            vision_features: [B, 197, D_v] from ViT (includes CLS)
        Returns:
            logits dict: {color, shape, size} each [B, num_classes]
        """
        # Mean pool over patches (exclude CLS token at position 0)
        pooled = vision_features[:, 1:, :].mean(dim=1)  # [B, D_v]
        
        return {
            'color': self.color_head(pooled),  # [B, 6]
            'shape': self.shape_head(pooled),  # [B, 3]
            'size': self.size_head(pooled),    # [B, 2]
        }


class CLEVRLiteVLM(nn.Module):
    """Complete VLM for CLEVR-Lite with SAE hooks"""
    
    def __init__(self, config: VLMConfig):
        super().__init__()
        self.config = config
        
        # 1. Vision tower (ViT-tiny)
        vit_config = ViTConfig(
            image_size=config.image_size,
            patch_size=config.patch_size,
            dim=config.vision_dim,
            depth=config.vision_depth,
            heads=config.vision_heads,
            dropout=config.dropout,
        )
        self.vision = ViTTiny(vit_config)
        
        # 2. Visual tokenizer (196→32)
        self.tokenizer = LinearVisualTokenizer(
            num_patches=vit_config.num_patches,
            num_tokens=config.num_visual_tokens,
            patch_dim=config.vision_dim,
        )
        
        # 3. Projector (192→256)
        proj_config = ProjectorConfig(
            in_dim=config.vision_dim,
            out_dim=config.text_dim,
            dropout=config.dropout,
        )
        self.projector = GEGLUProjector(proj_config)
        
        # 4. Text transformer
        text_config = TextTransformerConfig(
            dim=config.text_dim,
            depth=config.text_depth,
            heads=config.text_heads,
            vocab_size=config.vocab_size,
            max_seq_length=96,  # generous max for prefix + text
            dropout=config.dropout,
        )
        self.text_model = TextTransformer(text_config)
        
        # 5. QA head (discriminative on [QCLS])
        self.qa_head = nn.Linear(config.text_dim, config.num_answer_classes)
        
        # 6. Attribute heads (auxiliary on ViT features)
        self.attr_heads = AttributeHeads(config)
        
        # Initialize QA head
        nn.init.trunc_normal_(self.qa_head.weight, std=0.02)
        nn.init.zeros_(self.qa_head.bias)
    
    def forward(
        self,
        images: torch.Tensor,
        text_ids: torch.Tensor,
        return_hooks: bool = False,
    ) -> Dict[str, torch.Tensor]:
        """
        Args:
            images: [B, 3, 224, 224]
            text_ids: [B, T] where T ≤ max_text_length
            return_hooks: If True, collect hooks for SAE training
        
        Returns:
            outputs: {
                'qa_logits': [B, num_answer_classes],
                'attr_logits': {color, shape, size},
                'qcls_features': [B, text_dim],  # for optional contrastive loss
                'hooks': optional dict of all hooks
            }
        """
        B = images.size(0)
        
        # Initialize hooks dict
        all_hooks = {} if return_hooks else None
        
        # 1. Vision tower
        vision_out, vision_hooks = self.vision(
            images,
            return_hooks=return_hooks,
            hook_layers=self.config.vision_hook_layers if return_hooks else None,
        )  # [B, 197, 192]
        
        if return_hooks:
            all_hooks['vision'] = vision_hooks
        
        # 2. Attribute heads (on vision features)
        attr_logits = self.attr_heads(vision_out)
        
        # 3. Visual tokenizer (drop CLS, keep patches)
        patches = vision_out[:, 1:, :]  # [B, 196, 192]
        visual_tokens, pool_attn = self.tokenizer(patches)  # [B, 32, 192]
        
        if return_hooks:
            all_hooks['tokenizer'] = {'pool_attn': pool_attn}
        
        # 4. Projector (192→256)
        visual_tokens_proj, proj_hooks = self.projector(
            visual_tokens,
            return_hooks=return_hooks,
        )  # [B, 32, 256]
        
        if return_hooks:
            all_hooks['projector'] = proj_hooks
        
        # 5. Text transformer
        text_out, qcls_features, text_hooks = self.text_model(
            visual_tokens_proj,
            text_ids,
            return_hooks=return_hooks,
            hook_layers=self.config.text_hook_layers if return_hooks else None,
        )  # text_out: [B, L, 256], qcls_features: [B, 256]
        
        if return_hooks:
            all_hooks['text'] = text_hooks
        
        # 6. QA head (on [QCLS] features)
        qa_logits = self.qa_head(qcls_features)  # [B, num_answer_classes]
        
        # Prepare outputs
        outputs = {
            'qa_logits': qa_logits,
            'attr_logits': attr_logits,
            'qcls_features': qcls_features,
        }
        
        if return_hooks:
            outputs['hooks'] = all_hooks
        
        return outputs
    
    def get_num_params(self) -> Dict[str, int]:
        """Count parameters by component"""
        return {
            'vision': sum(p.numel() for p in self.vision.parameters()),
            'tokenizer': sum(p.numel() for p in self.tokenizer.parameters()),
            'projector': sum(p.numel() for p in self.projector.parameters()),
            'text': sum(p.numel() for p in self.text_model.parameters()),
            'qa_head': sum(p.numel() for p in self.qa_head.parameters()),
            'attr_heads': sum(p.numel() for p in self.attr_heads.parameters()),
            'total': sum(p.numel() for p in self.parameters()),
        }


def compute_losses(
    outputs: Dict[str, torch.Tensor],
    targets: Dict[str, torch.Tensor],
    loss_weights: Dict[str, float] = None,
) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
    """
    Compute all losses for training.
    
    Args:
        outputs: Model outputs
        targets: {
            'answer_id': [B] - QA answer class,
            'colors': [B, num_colors] - multi-hot or single label,
            'shapes': [B, num_shapes],
            'sizes': [B, num_sizes],
        }
        loss_weights: Optional custom weights
    
    Returns:
        total_loss: weighted sum
        loss_dict: individual losses for logging
    """
    if loss_weights is None:
        loss_weights = {
            'qa': 1.0,
            'attr': 0.3,
        }
    
    # QA loss (cross-entropy)
    qa_loss = F.cross_entropy(outputs['qa_logits'], targets['answer_id'])
    
    # Attribute losses
    attr_logits = outputs['attr_logits']
    
    # Color loss (assuming single label, but could be multi-hot)
    color_loss = F.cross_entropy(attr_logits['color'], targets['colors'])
    shape_loss = F.cross_entropy(attr_logits['shape'], targets['shapes'])
    size_loss = F.cross_entropy(attr_logits['size'], targets['sizes'])
    
    # Combined attribute loss
    attr_loss = (color_loss + shape_loss + size_loss) / 3.0
    
    # Total loss
    total_loss = (
        loss_weights['qa'] * qa_loss +
        loss_weights['attr'] * attr_loss
    )
    
    # Loss dict for logging
    loss_dict = {
        'total': total_loss.item(),
        'qa': qa_loss.item(),
        'attr': attr_loss.item(),
        'color': color_loss.item(),
        'shape': shape_loss.item(),
        'size': size_loss.item(),
    }
    
    return total_loss, loss_dict


def test_vlm():
    """Test complete VLM"""
    print("=" * 70)
    print("Testing Complete VLM")
    print("=" * 70)
    
    config = VLMConfig(
        vocab_size=150,
        num_answer_classes=150,
    )
    
    model = CLEVRLiteVLM(config)
    
    # Count parameters
    param_counts = model.get_num_params()
    print("\nParameter counts by component:")
    for name, count in param_counts.items():
        print(f"  {name:12s}: {count:>10,}")
    
    # Create dummy inputs
    B = 2
    images = torch.randn(B, 3, 224, 224)
    text_ids = torch.randint(0, 150, (B, 20))
    
    print(f"\nInput shapes:")
    print(f"  images: {images.shape}")
    print(f"  text_ids: {text_ids.shape}")
    
    # Forward pass without hooks
    print("\n" + "-" * 70)
    print("Forward pass (no hooks)")
    print("-" * 70)
    
    outputs = model(images, text_ids, return_hooks=False)
    
    print(f"\nOutput shapes:")
    print(f"  qa_logits: {outputs['qa_logits'].shape}")
    print(f"  qcls_features: {outputs['qcls_features'].shape}")
    print(f"  attr_logits:")
    for attr, logits in outputs['attr_logits'].items():
        print(f"    {attr}: {logits.shape}")
    
    # Forward pass with hooks
    print("\n" + "-" * 70)
    print("Forward pass (with hooks)")
    print("-" * 70)
    
    outputs = model(images, text_ids, return_hooks=True)
    
    hooks = outputs['hooks']
    print(f"\nHook categories: {list(hooks.keys())}")
    
    if 'vision' in hooks:
        print(f"Vision hooks (layers {list(hooks['vision'].keys())}):")
        for layer_idx, layer_hooks in hooks['vision'].items():
            print(f"  Layer {layer_idx}: {list(layer_hooks.keys())}")
    
    if 'projector' in hooks:
        print(f"Projector hooks: {list(hooks['projector'].keys())}")
    
    if 'text' in hooks:
        print(f"Text hooks (layers {list(hooks['text'].keys())}):")
        for layer_idx, layer_hooks in hooks['text'].items():
            print(f"  Layer {layer_idx}: {list(layer_hooks.keys())}")
    
    # Test loss computation
    print("\n" + "-" * 70)
    print("Testing loss computation")
    print("-" * 70)
    
    targets = {
        'answer_id': torch.randint(0, 150, (B,)),
        'colors': torch.randint(0, 6, (B,)),
        'shapes': torch.randint(0, 3, (B,)),
        'sizes': torch.randint(0, 2, (B,)),
    }
    
    total_loss, loss_dict = compute_losses(outputs, targets)
    
    print(f"\nLosses:")
    for name, value in loss_dict.items():
        print(f"  {name:10s}: {value:.4f}")
    
    print("\n" + "=" * 70)
    print("✓ VLM test passed!")
    print("=" * 70)


if __name__ == '__main__':
    test_vlm()