"""
Run LLaVA on CLEVR-Lite dataset and collect baseline accuracy.

This script:
1. Loads the model and dataset
2. Runs inference on all questions
3. Computes accuracy metrics (overall, by question type, held-out vs seen)
4. Saves predictions and metrics for later analysis
"""

import json
import torch
from pathlib import Path
from PIL import Image
from transformers import AutoProcessor, LlavaForConditionalGeneration
from tqdm import tqdm
from typing import Dict, List
import argparse
from collections import defaultdict
import re
from sae.custom_clevr_lite.src.clevr_lite_config import CLEVRLiteConfig


def load_model(model_id: str = "llava-hf/llava-1.5-7b-hf", device: int = 0):
    """Load LLaVA model and processor"""
    print(f"Loading model: {model_id}")
    model = LlavaForConditionalGeneration.from_pretrained(
        model_id,
        torch_dtype=torch.float16,
        low_cpu_mem_usage=True,
    ).to(device)
    
    processor = AutoProcessor.from_pretrained(model_id)
    
    return model, processor


def load_dataset(data_dir: str, split: str = 'val') -> List[Dict]:
    """Load questions from JSON file"""
    questions_file = Path(data_dir) / f"{split}_questions.json"
    
    with open(questions_file, 'r') as f:
        questions = json.load(f)
    
    print(f"Loaded {len(questions)} questions from {split} split")
    return questions


def format_prompt(question: str, processor) -> str:
    """Format question as LLaVA chat prompt"""
    conversation = [
        {
            "role": "user",
            "content": [
                {"type": "text", "text": question},
                {"type": "image"},
            ],
        },
    ]
    prompt = processor.apply_chat_template(conversation, add_generation_prompt=True)
    return prompt


def extract_answer(generated_text: str, question_type: str) -> str:
    """
    Extract the actual answer from the model's generated text by matching against
    the dataset's closed-set vocabulary, scoped by the question type.
    """

    text = generated_text.strip().lower()

    # Normalize punctuation to spaces and collapse whitespace
    text = re.sub(r"[^\w\s]", " ", text)
    text = re.sub(r"\s+", " ", text).strip()

    # Allowed vocab from the data generator
    COLORS = set(CLEVRLiteConfig.COLORS)   # ['red','blue','green','yellow','purple','cyan']
    SHAPES = set(CLEVRLiteConfig.SHAPES)   # ['cube','sphere','triangle']

    # A few common natural synonyms the VLM might produce
    synonym_map = {
        # shapes
        "circle": "sphere",
        "round": "sphere",
        "ball": "sphere",
        "spherical": "sphere",
        "triangular": "triangle",
        "pyramid": "triangle",     # sometimes VLMs say "pyramid" for 2D triangle-like
        "box": "cube",
        "cubic": "cube",
        "block": "cube",

        # colors
        "violet": "purple",
        "magenta": "purple",       # occasionally appears for saturated purple
        "turquoise": "cyan",
        "teal": "cyan",            # close enough for this toy dataset
        "gold": "yellow",
        "golden": "yellow",
        "navy": "blue",
        "azure": "blue",
        "lime": "green",
    }

    # Build search target list based on question type
    if "color" in question_type:
        target_set = COLORS
    elif "shape" in question_type:
        target_set = SHAPES
    else:
        # Fallback: consider both if somehow a new template slips in
        target_set = COLORS.union(SHAPES)

    # Tokenize while preserving order for first-match selection
    tokens = text.split()

    # First pass: direct match to allowed vocab
    for tok in tokens:
        if tok in target_set:
            return tok

    # Second pass: map synonyms → canonical and match
    for tok in tokens:
        mapped = synonym_map.get(tok)
        if mapped and mapped in target_set:
            return mapped

    # Third pass: light pattern-based hints like "... is a/an <word>" or "... is <word>"
    # Then apply the same vocab filtering.
    m = re.search(r"\bis\b(?:\s+a|n)?\s+(\w+)", text)  # captures the word after "is", "is a", "is an"
    if m:
        w = m.group(1)
        if w in target_set:
            return w
        mapped = synonym_map.get(w)
        if mapped and mapped in target_set:
            return mapped

    # If nothing valid found, return empty string to mark as incorrect downstream
    return ""



def run_inference(
    model,
    processor,
    questions: List[Dict],
    data_dir: str,
    device: int = 0,
    max_samples: int = None,
) -> List[Dict]:
    """
    Run model inference on all questions.
    
    Returns:
        List of dicts with keys: question_idx, question, ground_truth, prediction, correct
    """
    results = []
    
    # Optionally limit samples for quick testing
    if max_samples:
        questions = questions[:max_samples]
    
    model.eval()
    
    for idx, q_data in enumerate(tqdm(questions, desc="Running inference")):
        # Load image
        image_path = Path(data_dir) / q_data['image_path']
        raw_image = Image.open(image_path)
        
        # Format prompt
        prompt = format_prompt(q_data['question'], processor)
        
        # Prepare inputs
        inputs = processor(
            images=raw_image,
            text=prompt,
            return_tensors='pt'
        ).to(device, torch.float16)
        
        # Generate
        with torch.no_grad():
            output = model.generate(
                **inputs,
                max_new_tokens=50,
                do_sample=False,
            )
        
        # Decode (skip prompt tokens)
        generated_text = processor.decode(output[0], skip_special_tokens=True)
        
        # Extract answer from generated text
        # LLaVA returns the full conversation, so we need to extract just the assistant's response
        if "ASSISTANT:" in generated_text:
            generated_text = generated_text.split("ASSISTANT:")[-1].strip()

        predicted_answer = extract_answer(generated_text, q_data['question_type'])
        ground_truth = q_data['answer'].lower()
        
        # Check correctness
        correct = predicted_answer == ground_truth
        
        results.append({
            'question_idx': idx,
            'scene_id': q_data['scene_id'],
            'question': q_data['question'],
            'question_type': q_data['question_type'],
            'ground_truth': ground_truth,
            'prediction': predicted_answer,
            'generated_text': generated_text,
            'correct': correct,
            'is_held_out_combo': q_data['is_held_out_combo'],
            'image_path': q_data['image_path'],
        })
    
    return results


def compute_metrics(results: List[Dict]) -> Dict:
    """Compute accuracy metrics from results"""
    
    total = len(results)
    correct = sum(1 for r in results if r['correct'])
    
    metrics = {
        'overall_accuracy': correct / total if total > 0 else 0.0,
        'total_questions': total,
        'correct_predictions': correct,
    }
    
    # Accuracy by question type
    by_type = defaultdict(lambda: {'correct': 0, 'total': 0})
    for r in results:
        qtype = r['question_type']
        by_type[qtype]['total'] += 1
        if r['correct']:
            by_type[qtype]['correct'] += 1
    
    metrics['by_question_type'] = {
        qtype: {
            'accuracy': stats['correct'] / stats['total'] if stats['total'] > 0 else 0.0,
            'correct': stats['correct'],
            'total': stats['total'],
        }
        for qtype, stats in by_type.items()
    }
    
    # Accuracy on held-out vs seen combinations
    held_out_results = [r for r in results if r['is_held_out_combo']]
    seen_results = [r for r in results if not r['is_held_out_combo']]
    
    metrics['held_out_combo'] = {
        'accuracy': sum(1 for r in held_out_results if r['correct']) / len(held_out_results) if held_out_results else 0.0,
        'total': len(held_out_results),
    }
    
    metrics['seen_combo'] = {
        'accuracy': sum(1 for r in seen_results if r['correct']) / len(seen_results) if seen_results else 0.0,
        'total': len(seen_results),
    }
    
    return metrics


def print_metrics(metrics: Dict):
    """Pretty print metrics"""
    print("\n" + "="*60)
    print("BASELINE ACCURACY METRICS")
    print("="*60)
    
    print(f"\nOverall Accuracy: {metrics['overall_accuracy']:.2%}")
    print(f"Correct: {metrics['correct_predictions']} / {metrics['total_questions']}")
    
    print("\n--- By Question Type ---")
    for qtype, stats in metrics['by_question_type'].items():
        print(f"  {qtype:30s}: {stats['accuracy']:.2%} ({stats['correct']}/{stats['total']})")
    
    print("\n--- Compositional Generalization ---")
    print(f"  Seen combinations:     {metrics['seen_combo']['accuracy']:.2%} ({metrics['seen_combo']['total']} questions)")
    print(f"  Held-out combinations: {metrics['held_out_combo']['accuracy']:.2%} ({metrics['held_out_combo']['total']} questions)")
    
    print("="*60 + "\n")


def save_results(results: List[Dict], metrics: Dict, output_dir: str):
    """Save predictions and metrics to files"""
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    
    # Save detailed predictions
    predictions_file = output_path / 'predictions.json'
    with open(predictions_file, 'w') as f:
        json.dump(results, f, indent=2)
    print(f"Saved predictions to: {predictions_file}")
    
    # Save metrics
    metrics_file = output_path / 'metrics.json'
    with open(metrics_file, 'w') as f:
        json.dump(metrics, f, indent=2)
    print(f"Saved metrics to: {metrics_file}")
    
    # Save error analysis (incorrect predictions)
    errors = [r for r in results if not r['correct']]
    errors_file = output_path / 'errors.json'
    with open(errors_file, 'w') as f:
        json.dump(errors, f, indent=2)
    print(f"Saved {len(errors)} errors to: {errors_file}")


def main():
    parser = argparse.ArgumentParser(description="Run baseline evaluation on CLEVR-Lite")
    parser.add_argument('--data_dir', type=str, default='/home/ron/Documents/Github/VLM-lens/data',
                        help='Path to CLEVR-Lite dataset directory')
    parser.add_argument('--split', type=str, default='val', choices=['train', 'val'],
                        help='Dataset split to evaluate')
    parser.add_argument('--model_id', type=str, default='llava-hf/llava-1.5-7b-hf',
                        help='HuggingFace model ID')
    parser.add_argument('--device', type=int, default=0,
                        help='GPU device ID')
    parser.add_argument('--max_samples', type=int, default=None,
                        help='Maximum number of samples to evaluate (for quick testing)')
    parser.add_argument('--output_dir', type=str, default='./results/baseline',
                        help='Directory to save results')
    
    args = parser.parse_args()
    
    # Load model
    model, processor = load_model(args.model_id, args.device)
    
    # Load dataset
    questions = load_dataset(args.data_dir, args.split)
    
    # Run inference
    results = run_inference(
        model, processor, questions, args.data_dir,
        device=args.device, max_samples=args.max_samples
    )
    
    # Compute metrics
    metrics = compute_metrics(results)
    
    # Print metrics
    print_metrics(metrics)
    
    # Save results
    save_results(results, metrics, args.output_dir)
    
    print("✓ Baseline evaluation complete!")


if __name__ == '__main__':
    main()