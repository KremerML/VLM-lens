import json
import torch
from pathlib import Path
from transformers import AutoProcessor, LlavaForConditionalGeneration
from typing import Dict, List


def load_model(model_id: str = "llava-hf/llava-1.5-7b-hf", device: int = 0):
    """Load LLaVA model and processor"""
    print(f"Loading model: {model_id}")
    model = LlavaForConditionalGeneration.from_pretrained(
        model_id,
        dtype=torch.float16,
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