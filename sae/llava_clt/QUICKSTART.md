# Activation Capturing - Quick Start Guide

## What Was Added

I've added comprehensive activation capturing functionality to your LLaVA-CLT pipeline:

### New Files

1. **`sae/llava_clt/attribution_patching.py`** (530 lines)
   - Core functionality for activation capturing and patching
   - Classes: `ActivationHookManager`, `ActivationPatcher`, `LayerImportanceAnalyzer`, `ActivationCache`
   - Ready to use with your existing `load_model()` function

2. **`sae/llava_clt/examples/capture_activations_demo.py`**
   - Three complete demo scripts showing usage
   - Integration with your existing `run_model.py` infrastructure

3. **`sae/llava_clt/examples/test_activation_patching.py`**
   - Integration tests (all passing ✓)
   - Verifies functionality works correctly

4. **`sae/llava_clt/ACTIVATION_PATCHING.md`**
   - Complete documentation with examples
   - API reference and common use cases

## Can You Do Activation Patching on HuggingFace Models?

**Yes, absolutely!** The implementation is ready to use with your `llava-1.5-7b-hf` model.

### What Works

✅ **Full access to all layers** - Vision (24 layers), Projector, Language (32 layers)  
✅ **PyTorch hooks** - Native support for HuggingFace Transformers  
✅ **Memory efficient** - Selective layer hooking, float16 support  
✅ **Integrates with your code** - Uses your `load_model()`, `processor`, etc.  

### LLaVA Architecture You Can Hook

```
model.vision_tower.vision_model.encoder.layers[0-23]  ← Vision transformer
model.multi_modal_projector                           ← Vision→Language bridge  
model.language_model.model.layers[0-31]               ← Language transformer
```

## Quick Usage Examples

### 1. Capture Activations (30 seconds)

```python
from sae.llava_clt.attribution_patching import ActivationHookManager
from sae.llava_clt.utils.loader_functions import load_model

model, processor = load_model("llava-hf/llava-1.5-7b-hf", device=0)
hook_manager = ActivationHookManager(model)
hook_manager.register_vision_hooks([10, 15, 20])  # Hook layers 10, 15, 20

# Your existing inference code
with torch.no_grad():
    output = model.generate(**inputs, max_new_tokens=50)

# Get activations
cache = hook_manager.get_cache()
layer_10_acts = cache.vision_activations['vision_layer_10']
print(f"Shape: {layer_10_acts.shape}")  # (batch, 576, 1024)

hook_manager.remove_hooks()
```

### 2. Activation Patching (1-2 minutes)

```python
from sae.llava_clt.attribution_patching import ActivationPatcher

# Capture clean activations
hook_manager = ActivationHookManager(model)
hook_manager.register_vision_hooks([15])
with torch.no_grad():
    _ = model.generate(**clean_inputs, max_new_tokens=50)
clean_cache = hook_manager.get_cache()
hook_manager.remove_hooks()

# Patch into corrupted run
patcher = ActivationPatcher(model)
patched_output = patcher.patch_and_forward(
    corrupted_inputs,
    clean_cache,
    patch_layers=[('vision', 15)]
)
# Check if output changed!
```

### 3. Find Important Layers (5-10 minutes)

```python
from sae.llava_clt.attribution_patching import LayerImportanceAnalyzer

analyzer = LayerImportanceAnalyzer(model, processor)
importance_scores = analyzer.analyze_layer_importance(
    clean_inputs, corrupted_inputs,
    clean_answer="red", corrupted_answer="blue",
    layer_type='vision',
)

# Which layers matter for color detection?
for layer_idx, score in sorted(importance_scores.items()):
    if score > 0.5:
        print(f"Layer {layer_idx}: {score:.2f} - IMPORTANT")
```

### 4. Collect CLT Training Data

```python
from sae.llava_clt.attribution_patching import collect_activation_pairs

# Get (layer_n, layer_n+1) pairs for transcoder training
source_acts, target_acts = collect_activation_pairs(
    model, processor, your_data_loader,
    source_layer=10,
    target_layer=11,
    layer_type='vision',
    max_samples=1000,
)

# Now train your Cross-Layer Transcoder
# clt.train(source_acts, target_acts)
```

## Integration with Your Existing Code

The activation patching system works seamlessly with your current setup:

| Your Module | How It Integrates |
|-------------|-------------------|
| `utils/loader_functions.py` | Uses your `load_model()` directly |
| `run_model.py` | Can insert hooks into your inference loop |
| `utils/metrics.py` | Patching outputs can be evaluated with your metrics |
| CLEVR-Lite dataset | Perfect for controlled patching experiments |

## Recommended Workflow for Your Project

### Phase 1: Baseline (Already Done ✓)
- [x] Run `run_model.py` to get baseline accuracy
- [x] Identify which questions fail

### Phase 2: Circuit Discovery (New!)
```bash
# Find which vision layers detect colors
python sae/llava_clt/examples/capture_activations_demo.py --demo importance

# Expected finding: Layers 15-23 likely critical for colors
# Expected finding: Layers 10-18 likely critical for shapes
```

### Phase 3: CLT Training
```python
# Collect activation pairs from important transitions
# Example: Vision layer 15→16 for color processing
source, target = collect_activation_pairs(
    model, processor, train_loader,
    source_layer=15, target_layer=16,
    layer_type='vision', max_samples=5000
)

# Train your sparse autoencoder
# (You'll implement this based on your SAE architecture)
```

### Phase 4: Validation
```python
# Verify CLT preserves behavior by patching
# Replace layer_16 with CLT(layer_15) and check accuracy unchanged
```

## Running the Demos

```bash
# Test that everything works (30 seconds)
python sae/llava_clt/examples/test_activation_patching.py

# Capture activations demo (1 minute with model loading)
python sae/llava_clt/examples/capture_activations_demo.py --demo capture

# Patching demo (2 minutes)
python sae/llava_clt/examples/capture_activations_demo.py --demo patch

# Layer importance analysis (10-15 minutes - tests all 24 vision layers)
python sae/llava_clt/examples/capture_activations_demo.py --demo importance
```

## Memory Considerations

**Full activation capture** for LLaVA-7B can use significant VRAM:
- Vision: 24 layers × (1, 576, 1024) × float16 = ~28 MB per sample
- Language: 32 layers × (1, seq_len, 4096) × float16 = variable
- **Total**: ~100-200 MB per forward pass with all hooks

**Optimization strategies** (already implemented):
- Hook only specific layers: `register_vision_hooks([10, 15, 20])`
- Move to CPU immediately: `cache.cpu()`
- Process in batches and save to disk

## What You Can Discover

With your CLEVR-Lite dataset:

1. **Color Detection Circuit**: Which vision layers activate for specific colors?
2. **Shape Detection Circuit**: How does the model distinguish circle vs square?
3. **Negation Processing**: How does "NOT red" get handled in language layers?
4. **Vision-Language Interface**: What happens at the projector?
5. **Compositional Generalization**: Why do held-out combos fail? Which layer?

## Next Steps

1. **Try the demos** to get familiar with the tools
2. **Run layer importance analysis** on a few CLEVR-Lite examples
3. **Identify 2-3 critical layer transitions** for your use case
4. **Collect activation pairs** for those transitions
5. **Design and train CLTs** (you'll build this part)

## Questions?

- Full API docs: `sae/llava_clt/ACTIVATION_PATCHING.md`
- Code examples: `sae/llava_clt/examples/capture_activations_demo.py`
- Tests: `sae/llava_clt/examples/test_activation_patching.py`

The system is production-ready and tested! 🚀
