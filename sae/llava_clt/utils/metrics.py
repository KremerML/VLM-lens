from typing import Dict, List
import json
from pathlib import Path
from collections import defaultdict


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


