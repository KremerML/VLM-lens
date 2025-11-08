"""
Run LLaVA on CLEVR-Lite dataset and collect baseline accuracy.

This script:
1. Loads the model and dataset
2. Runs inference on all questions
3. Computes accuracy metrics (overall, by question type, held-out vs seen)
4. Saves predictions and metrics for later analysis
"""

import torch
from pathlib import Path
from PIL import Image
from tqdm import tqdm
from typing import Dict, List
import argparse
from transformers import LlavaProcessor, LlavaForConditionalGeneration

from sae.llava_clt.utils.metrics import compute_metrics, print_metrics
from sae.llava_clt.utils.loader_functions import load_dataset, load_model, save_results
from sae.llava_clt.utils.llm_output_parser import extract_answer


def format_prompt(question: str, processor: LlavaProcessor, system_prompt: str = None) -> str:
    """Format question as LLaVA chat prompt with optional system message"""
    conversation = []
    
    # Add system message if provided
    if system_prompt:
        conversation.append({
            "role": "system",
            "content": [{"type": "text", "text": system_prompt}],
        })
    
    # Add user message with question and image
    conversation.append({
        "role": "user",
        "content": [
            {"type": "text", "text": question},
            {"type": "image"},
        ],
    })
    
    prompt = processor.apply_chat_template(conversation, add_generation_prompt=True)
    return prompt


def run_inference(
    model: LlavaForConditionalGeneration,
    processor: LlavaProcessor,
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
        
        # system_prompt = "Only answer with one word - the specific attribute asked about (colors: red, blue, green, yellow, purple, black; shapes: square, circle, triangle)."
        prompt = format_prompt(q_data['question'], processor) #, system_prompt=system_prompt)

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