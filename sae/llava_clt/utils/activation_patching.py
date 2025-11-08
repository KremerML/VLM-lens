"""
Simple activation patching for LLaVA 1.5-7B.

Usage:
    # Cache activations from a clean run
    cache = {}
    with save_activations(model, cache, layers=[10, 15, 20]):
        outputs = model.generate(**inputs)
    
    # Patch activations from cache into a new run
    with patch_activations(model, cache, layers=[10, 15, 20]):
        outputs = model.generate(**inputs)
"""

import torch
from contextlib import contextmanager
from typing import Dict, List, Optional


@contextmanager
def save_activations(model, cache: Dict, layers: Optional[List[int]] = None):
    """
    Context manager to save activations during forward pass.
    
    Args:
        model: LlavaForConditionalGeneration model
        cache: Dict to store activations (will be modified in-place)
        layers: List of layer indices to cache (None = all 32 layers)
    """
    if layers is None:
        layers = list(range(32))  # LLaVA 1.5-7B has 32 layers
    
    hooks = []
    
    def make_hook(layer_idx):
        def hook(module, input, output):
            # output might be a tuple or a single tensor
            if isinstance(output, tuple):
                hidden_states = output[0]
            else:
                hidden_states = output
            cache[layer_idx] = hidden_states.detach().cpu()
        return hook
    
    # Register hooks
    for layer_idx in layers:
        layer = model.model.language_model.layers[layer_idx]
        handle = layer.register_forward_hook(make_hook(layer_idx))
        hooks.append(handle)
    
    try:
        yield cache
    finally:
        # Clean up hooks
        for handle in hooks:
            handle.remove()


@contextmanager
def patch_activations(
    model, 
    cache: Dict, 
    layers: Optional[List[int]] = None,
    positions: Optional[List[int]] = None,
):
    """
    Context manager to patch activations during forward pass.
    
    Args:
        model: LlavaForConditionalGeneration model
        cache: Dict with cached activations from save_activations()
        layers: List of layer indices to patch (None = all cached layers)
        positions: List of token positions to patch (None = patch all positions)
    """
    if layers is None:
        layers = list(cache.keys())
    
    hooks = []
    
    def make_hook(layer_idx):
        def hook(module, input, output):
            # output might be a tuple or a single tensor
            if isinstance(output, tuple):
                hidden_states = output[0]
            else:
                hidden_states = output
                
            cached = cache[layer_idx].to(hidden_states.device, hidden_states.dtype)
            
            # Handle sequence length mismatch
            min_seq_len = min(hidden_states.shape[1], cached.shape[1])
            
            if positions is None:
                # Replace activation up to min sequence length
                # Keep the current sequence length to avoid breaking attention
                patched = hidden_states.clone()
                patched[:, :min_seq_len] = cached[:, :min_seq_len]
            else:
                # Replace only specific positions
                patched = hidden_states.clone()
                for pos in positions:
                    if pos < min_seq_len:
                        patched[:, pos] = cached[:, pos]
            
            # Return in same format as input
            if isinstance(output, tuple):
                return (patched,) + output[1:]
            else:
                return patched
        return hook
    
    # Register hooks
    for layer_idx in layers:
        if layer_idx not in cache:
            continue
        layer = model.model.language_model.layers[layer_idx]
        handle = layer.register_forward_hook(make_hook(layer_idx))
        hooks.append(handle)
    
    try:
        yield
    finally:
        # Clean up hooks
        for handle in hooks:
            handle.remove()


@contextmanager
def zero_ablate(model, layers: List[int]):
    """
    Context manager to zero out activations at specific layers.
    
    Args:
        model: LlavaForConditionalGeneration model
        layers: List of layer indices to ablate
    """
    hooks = []
    
    def make_hook(layer_idx):
        def hook(module, input, output):
            # output might be a tuple or a single tensor
            if isinstance(output, tuple):
                hidden_states = output[0]
            else:
                hidden_states = output
                
            zeroed = torch.zeros_like(hidden_states)
            
            # Return in same format as input
            if isinstance(output, tuple):
                return (zeroed,) + output[1:]
            else:
                return zeroed
        return hook
    
    # Register hooks
    for layer_idx in layers:
        layer = model.model.language_model.layers[layer_idx]
        handle = layer.register_forward_hook(make_hook(layer_idx))
        hooks.append(handle)
    
    try:
        yield
    finally:
        for handle in hooks:
            handle.remove()
