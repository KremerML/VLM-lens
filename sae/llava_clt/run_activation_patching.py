"""
Simple activation patching experiments for LLaVA.

Example 1: Patch activations from one example to another
Example 2: Zero-ablate each layer to measure importance
"""

import torch
import argparse
from pathlib import Path
from PIL import Image
from tqdm import tqdm

from sae.llava_clt.utils.loader_functions import load_model, load_dataset
from sae.llava_clt.utils.llm_output_parser import extract_answer
from sae.llava_clt.utils.activation_patching import save_activations, patch_activations, zero_ablate


def run_model(model, processor, image, question, device=0):
    """Run model and return generated text."""
    # Format prompt
    conversation = [{
        "role": "user",
        "content": [
            {"type": "text", "text": question},
            {"type": "image"},
        ],
    }]
    prompt = processor.apply_chat_template(conversation, add_generation_prompt=True)
    
    # Prepare inputs
    inputs = processor(images=image, text=prompt, return_tensors='pt')
    inputs = {k: v.to(device, torch.float16 if v.dtype == torch.float32 else v.dtype) 
              for k, v in inputs.items()}
    
    # Generate
    with torch.no_grad():
        output = model.generate(**inputs, max_new_tokens=50, do_sample=False)
    
    # Decode
    generated_text = processor.decode(output[0], skip_special_tokens=True)
    if "ASSISTANT:" in generated_text:
        generated_text = generated_text.split("ASSISTANT:")[-1].strip()
    
    return generated_text


def example_patching(args):
    """Patch activations from source example into target example."""
    
    print("Loading model...")
    model, processor = load_model(args.model_id, args.device)
    
    print("Loading dataset...")
    questions = load_dataset(args.data_dir, args.split)
    
    # Pick two examples
    source_q = questions[0]
    target_q = questions[1]
    
    # Load images
    data_dir = Path(args.data_dir)
    source_image = Image.open(data_dir / source_q['image_path'])
    target_image = Image.open(data_dir / target_q['image_path'])
    
    print("\n" + "="*60)
    print("SOURCE EXAMPLE")
    print("="*60)
    print(f"Question: {source_q['question']}")
    print(f"Answer: {source_q['answer']}")
    
    # Run source and cache activations
    cache = {}
    with save_activations(model, cache, layers=[10, 15, 20]):
        source_output = run_model(model, processor, source_image, 
                                  source_q['question'], args.device)
    
    print(f"Model output: {source_output}")
    print(f"Cached {len(cache)} layers")
    
    print("\n" + "="*60)
    print("TARGET EXAMPLE (clean)")
    print("="*60)
    print(f"Question: {target_q['question']}")
    print(f"Answer: {target_q['answer']}")
    
    # Run target clean
    target_clean = run_model(model, processor, target_image,
                            target_q['question'], args.device)
    print(f"Model output: {target_clean}")
    
    print("\n" + "="*60)
    print("TARGET EXAMPLE (patched)")
    print("="*60)
    
    # Run target with patching
    with patch_activations(model, cache, layers=[10, 15, 20]):
        target_patched = run_model(model, processor, target_image,
                                   target_q['question'], args.device)
    
    print(f"Model output: {target_patched}")
    print(f"\nChanged: {target_clean != target_patched}")


def ablation_experiment(args):
    """Zero-ablate each layer and measure accuracy impact."""
    
    print("Loading model...")
    model, processor = load_model(args.model_id, args.device)
    
    print("Loading dataset...")
    questions = load_dataset(args.data_dir, args.split)[:args.max_samples]
    
    data_dir = Path(args.data_dir)
    
    # Measure clean accuracy
    print("\n" + "="*60)
    print("CLEAN RUN")
    print("="*60)
    
    clean_correct = 0
    for q in tqdm(questions, desc="Clean run"):
        image = Image.open(data_dir / q['image_path'])
        output = run_model(model, processor, image, q['question'], args.device)
        pred = extract_answer(output, q['question_type'])
        if pred == q['answer'].lower():
            clean_correct += 1
    
    clean_acc = clean_correct / len(questions)
    print(f"Clean accuracy: {clean_acc:.2%} ({clean_correct}/{len(questions)})")
    
    # Ablate each layer
    print("\n" + "="*60)
    print("LAYER ABLATION")
    print("="*60)
    
    for layer_idx in range(32):
        layer_correct = 0
        
        for q in questions:
            image = Image.open(data_dir / q['image_path'])
            
            with zero_ablate(model, layers=[layer_idx]):
                output = run_model(model, processor, image, q['question'], args.device)
            
            pred = extract_answer(output, q['question_type'])
            if pred == q['answer'].lower():
                layer_correct += 1
        
        layer_acc = layer_correct / len(questions)
        impact = clean_acc - layer_acc
        
        print(f"Layer {layer_idx:2d}: {layer_acc:.2%} (impact: {impact:+.2%})")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--mode', choices=['example', 'ablation'], default='example')
    parser.add_argument('--data_dir', default='/home/ron/Documents/Github/VLM-lens/data')
    parser.add_argument('--split', default='val')
    parser.add_argument('--model_id', default='llava-hf/llava-1.5-7b-hf')
    parser.add_argument('--device', type=int, default=0)
    parser.add_argument('--max_samples', type=int, default=50)
    
    args = parser.parse_args()
    
    if args.mode == 'example':
        example_patching(args)
    elif args.mode == 'ablation':
        ablation_experiment(args)


if __name__ == '__main__':
    main()
