"""
Activation patching utilities for LLaVA model interpretability.

This module provides tools for:
1. Capturing intermediate activations from vision and language transformers
2. Performing activation patching to identify causally important layers
3. Collecting activation pairs for Cross-Layer Transcoder (CLT) training

Usage:
    # Capture activations
    hook_manager = ActivationHookManager(model)
    hook_manager.register_vision_hooks()
    
    # Run forward pass
    with torch.no_grad():
        output = model(**inputs)
    
    # Access activations
    vision_acts = hook_manager.get_vision_activations()
    
    # Perform patching
    patcher = ActivationPatcher(model)
    patched_output = patcher.patch_and_forward(
        inputs, clean_activations, patch_layer="vision_layer_10"
    )
"""

import torch
import torch.nn as nn
from typing import Dict, List, Optional, Tuple, Callable
from dataclasses import dataclass
from collections import defaultdict
import numpy as np


@dataclass
class ActivationCache:
    """Container for cached activations from a forward pass"""
    vision_activations: Dict[str, torch.Tensor]
    language_activations: Dict[str, torch.Tensor]
    projector_activations: Dict[str, torch.Tensor]
    
    def to_device(self, device: str):
        """Move all activations to specified device"""
        for cache in [self.vision_activations, self.language_activations, self.projector_activations]:
            for key in cache:
                cache[key] = cache[key].to(device)
        return self
    
    def detach(self):
        """Detach all tensors from computation graph"""
        for cache in [self.vision_activations, self.language_activations, self.projector_activations]:
            for key in cache:
                cache[key] = cache[key].detach()
        return self
    
    def cpu(self):
        """Move all activations to CPU"""
        return self.to_device('cpu')


class ActivationHookManager:
    """
    Manages forward hooks for capturing activations from LLaVA model.
    
    The LLaVA architecture:
        - vision_tower: CLIP ViT encoder (24 layers for base)
        - multi_modal_projector: Linear projection from vision to language space
        - language_model: LLaMA decoder (32 layers for 7B)
    """
    
    def __init__(self, model: nn.Module):
        self.model = model
        self.hooks = []
        self.activations = {
            'vision': {},
            'language': {},
            'projector': {},
        }
        
    def _make_hook(self, name: str, storage_dict: Dict[str, torch.Tensor]) -> Callable:
        """Create a hook function that stores activations"""
        def hook(module, input, output):
            # Handle different output types
            if isinstance(output, tuple):
                # For transformer layers that return (hidden_states, *extras)
                storage_dict[name] = output[0].detach()
            else:
                storage_dict[name] = output.detach()
        return hook
    
    def register_vision_hooks(self, layer_indices: Optional[List[int]] = None):
        """
        Register hooks on vision encoder layers.
        
        Args:
            layer_indices: Specific layer indices to hook. If None, hooks all layers.
        """
        vision_encoder = self.model.vision_tower.vision_model.encoder
        num_layers = len(vision_encoder.layers)
        
        if layer_indices is None:
            layer_indices = list(range(num_layers))
        
        for idx in layer_indices:
            if idx >= num_layers:
                print(f"Warning: Vision layer {idx} doesn't exist (max: {num_layers-1})")
                continue
                
            layer = vision_encoder.layers[idx]
            hook_name = f'vision_layer_{idx}'
            hook = layer.register_forward_hook(
                self._make_hook(hook_name, self.activations['vision'])
            )
            self.hooks.append(hook)
        
        print(f"Registered {len(layer_indices)} vision hooks (layers: {layer_indices})")
    
    def register_language_hooks(self, layer_indices: Optional[List[int]] = None):
        """
        Register hooks on language model layers.
        
        Args:
            layer_indices: Specific layer indices to hook. If None, hooks all layers.
        """
        language_layers = self.model.language_model.model.layers
        num_layers = len(language_layers)
        
        if layer_indices is None:
            layer_indices = list(range(num_layers))
        
        for idx in layer_indices:
            if idx >= num_layers:
                print(f"Warning: Language layer {idx} doesn't exist (max: {num_layers-1})")
                continue
                
            layer = language_layers[idx]
            hook_name = f'language_layer_{idx}'
            hook = layer.register_forward_hook(
                self._make_hook(hook_name, self.activations['language'])
            )
            self.hooks.append(hook)
        
        print(f"Registered {len(layer_indices)} language hooks (layers: {layer_indices})")
    
    def register_projector_hook(self):
        """Register hook on the multi-modal projector (vision→language bridge)"""
        projector = self.model.multi_modal_projector
        hook = projector.register_forward_hook(
            self._make_hook('projector_output', self.activations['projector'])
        )
        self.hooks.append(hook)
        print("Registered projector hook")
    
    def register_all_hooks(self):
        """Register hooks on all components: vision, projector, and language"""
        self.register_vision_hooks()
        self.register_projector_hook()
        self.register_language_hooks()
    
    def clear_activations(self):
        """Clear stored activations (useful between forward passes)"""
        self.activations = {
            'vision': {},
            'language': {},
            'projector': {},
        }
    
    def remove_hooks(self):
        """Remove all registered hooks"""
        for hook in self.hooks:
            hook.remove()
        self.hooks = []
        print(f"Removed all hooks")
    
    def get_cache(self) -> ActivationCache:
        """Get current activations as an ActivationCache object"""
        return ActivationCache(
            vision_activations=self.activations['vision'].copy(),
            language_activations=self.activations['language'].copy(),
            projector_activations=self.activations['projector'].copy(),
        )
    
    def __enter__(self):
        """Context manager entry"""
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        """Context manager exit - automatically remove hooks"""
        self.remove_hooks()


class ActivationPatcher:
    """
    Performs activation patching experiments to identify causally important layers.
    
    Typical workflow:
        1. Run model on clean input → save activations
        2. Run model on corrupted input with patching → measure output change
        3. Compare outputs to determine which layers are causally important
    """
    
    def __init__(self, model: nn.Module):
        self.model = model
        self.patch_hooks = []
        self.patch_values = {}
    
    def _make_patch_hook(self, name: str) -> Callable:
        """Create a hook that replaces activations with patched values"""
        def hook(module, input, output):
            if name in self.patch_values:
                # Replace the output with the patched activation
                patched = self.patch_values[name]
                
                # Handle tuple outputs (e.g., transformer layers)
                if isinstance(output, tuple):
                    return (patched,) + output[1:]
                else:
                    return patched
            return output
        return hook
    
    def set_patch(
        self,
        layer_type: str,  # 'vision', 'language', or 'projector'
        layer_idx: Optional[int],
        activation: torch.Tensor
    ):
        """
        Set activation to patch at specified layer.
        
        Args:
            layer_type: One of 'vision', 'language', or 'projector'
            layer_idx: Layer index (None for projector)
            activation: Tensor to patch in
        """
        if layer_type == 'projector':
            name = 'projector_output'
            layer = self.model.multi_modal_projector
        elif layer_type == 'vision':
            name = f'vision_layer_{layer_idx}'
            layer = self.model.vision_tower.vision_model.encoder.layers[layer_idx]
        elif layer_type == 'language':
            name = f'language_layer_{layer_idx}'
            layer = self.model.language_model.model.layers[layer_idx]
        else:
            raise ValueError(f"Unknown layer_type: {layer_type}")
        
        self.patch_values[name] = activation
        
        # Register hook for patching
        hook = layer.register_forward_hook(self._make_patch_hook(name))
        self.patch_hooks.append(hook)
    
    def clear_patches(self):
        """Remove all patches and hooks"""
        for hook in self.patch_hooks:
            hook.remove()
        self.patch_hooks = []
        self.patch_values = {}
    
    def patch_and_forward(
        self,
        inputs: Dict,
        clean_cache: ActivationCache,
        patch_layers: List[Tuple[str, Optional[int]]],
    ):
        """
        Run forward pass with specified layers patched from clean run.
        
        Args:
            inputs: Model inputs (already on device)
            clean_cache: Activations from clean run
            patch_layers: List of (layer_type, layer_idx) to patch
                Example: [('vision', 10), ('language', 5)]
        
        Returns:
            Model output with patched activations
        """
        # Clear any existing patches
        self.clear_patches()
        
        # Set up patches
        for layer_type, layer_idx in patch_layers:
            if layer_type == 'vision':
                activation = clean_cache.vision_activations[f'vision_layer_{layer_idx}']
            elif layer_type == 'language':
                activation = clean_cache.language_activations[f'language_layer_{layer_idx}']
            elif layer_type == 'projector':
                activation = clean_cache.projector_activations['projector_output']
            else:
                raise ValueError(f"Unknown layer_type: {layer_type}")
            
            self.set_patch(layer_type, layer_idx, activation)
        
        # Run forward pass with patches
        with torch.no_grad():
            output = self.model.generate(**inputs, max_new_tokens=50, do_sample=False)
        
        # Clean up
        self.clear_patches()
        
        return output
    
    def __enter__(self):
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        self.clear_patches()


class LayerImportanceAnalyzer:
    """
    Systematically patch each layer to measure causal importance.
    
    This performs the full activation patching experiment to identify
    which layers are most important for a specific behavior.
    """
    
    def __init__(self, model: nn.Module, processor):
        self.model = model
        self.processor = processor
        self.patcher = ActivationPatcher(model)
    
    def analyze_layer_importance(
        self,
        clean_inputs: Dict,
        corrupted_inputs: Dict,
        clean_answer: str,
        corrupted_answer: str,
        layer_type: str = 'vision',
        metric: str = 'logit_diff',
    ) -> Dict[int, float]:
        """
        Patch each layer individually and measure effect on output.
        
        Args:
            clean_inputs: Inputs that produce correct answer
            corrupted_inputs: Inputs that produce wrong answer
            clean_answer: Expected correct answer token
            corrupted_answer: Expected wrong answer token
            layer_type: Which layers to analyze ('vision', 'language', 'projector')
            metric: How to measure effect ('logit_diff' or 'prob_correct')
        
        Returns:
            Dict mapping layer_idx → importance score
        """
        # First, get clean activations
        hook_manager = ActivationHookManager(self.model)
        
        if layer_type == 'vision':
            hook_manager.register_vision_hooks()
        elif layer_type == 'language':
            hook_manager.register_language_hooks()
        elif layer_type == 'projector':
            hook_manager.register_projector_hook()
        
        # Run clean forward pass
        with torch.no_grad():
            clean_output = self.model.generate(**clean_inputs, max_new_tokens=50, do_sample=False)
        
        clean_cache = hook_manager.get_cache()
        hook_manager.remove_hooks()
        
        # Get number of layers to test
        if layer_type == 'vision':
            num_layers = len(self.model.vision_tower.vision_model.encoder.layers)
            layer_indices = range(num_layers)
        elif layer_type == 'language':
            num_layers = len(self.model.language_model.model.layers)
            layer_indices = range(num_layers)
        elif layer_type == 'projector':
            layer_indices = [None]  # Single component
        
        # Test patching each layer
        importance_scores = {}
        
        for layer_idx in layer_indices:
            # Patch this layer and run forward pass
            if layer_type == 'projector':
                patch_spec = [('projector', None)]
            else:
                patch_spec = [(layer_type, layer_idx)]
            
            patched_output = self.patcher.patch_and_forward(
                corrupted_inputs,
                clean_cache,
                patch_spec
            )
            
            # Decode output
            patched_text = self.processor.decode(patched_output[0], skip_special_tokens=True)
            
            # Compute importance score
            # Simple version: did it recover the correct answer?
            if clean_answer.lower() in patched_text.lower():
                score = 1.0
            else:
                score = 0.0
            
            importance_scores[layer_idx if layer_idx is not None else 'projector'] = score
        
        return importance_scores


def collect_activation_pairs(
    model: nn.Module,
    processor,
    data_loader,
    source_layer: int,
    target_layer: int,
    layer_type: str = 'vision',
    max_samples: int = 1000,
    device: str = 'cuda:0',
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Collect (layer_n, layer_n+1) activation pairs for CLT training.
    
    Args:
        model: LLaVA model
        processor: Processor for inputs
        data_loader: DataLoader yielding (image, question) pairs
        source_layer: Layer n (input to transcoder)
        target_layer: Layer n+1 (output to predict)
        layer_type: 'vision' or 'language'
        max_samples: Maximum number of samples to collect
        device: Device to run on
    
    Returns:
        (source_activations, target_activations) tensors of shape (N, seq_len, hidden_dim)
    """
    hook_manager = ActivationHookManager(model)
    
    # Register hooks for both layers
    if layer_type == 'vision':
        hook_manager.register_vision_hooks([source_layer, target_layer])
    elif layer_type == 'language':
        hook_manager.register_language_hooks([source_layer, target_layer])
    else:
        raise ValueError(f"layer_type must be 'vision' or 'language', got {layer_type}")
    
    source_acts = []
    target_acts = []
    
    model.eval()
    samples_collected = 0
    
    for batch in data_loader:
        if samples_collected >= max_samples:
            break
        
        # Prepare inputs (assumes batch has 'image' and 'question' keys)
        inputs = processor(
            images=batch['image'],
            text=batch['question'],
            return_tensors='pt'
        ).to(device, torch.float16)
        
        # Forward pass
        hook_manager.clear_activations()
        with torch.no_grad():
            _ = model(**inputs)
        
        # Extract activations
        cache = hook_manager.get_cache()
        
        if layer_type == 'vision':
            source_act = cache.vision_activations[f'vision_layer_{source_layer}']
            target_act = cache.vision_activations[f'vision_layer_{target_layer}']
        else:  # language
            source_act = cache.language_activations[f'language_layer_{source_layer}']
            target_act = cache.language_activations[f'language_layer_{target_layer}']
        
        source_acts.append(source_act.cpu())
        target_acts.append(target_act.cpu())
        
        samples_collected += source_act.shape[0]  # batch size
    
    hook_manager.remove_hooks()
    
    # Concatenate all batches
    source_tensor = torch.cat(source_acts, dim=0)
    target_tensor = torch.cat(target_acts, dim=0)
    
    print(f"Collected {samples_collected} activation pairs")
    print(f"Source shape: {source_tensor.shape}, Target shape: {target_tensor.shape}")
    
    return source_tensor, target_tensor
