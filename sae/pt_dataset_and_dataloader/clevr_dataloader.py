"""
PyTorch Dataset and Vocabulary for CLEVR-Lite

Handles:
- Image loading and normalization
- Text tokenization with minimal vocabulary
- Batching with padding
- Held-out combination filtering
"""

from torch.utils.data import DataLoader
from pathlib import Path
from typing import Tuple
from sae.pt_dataset_and_dataloader.clevr_vocabulary import CLEVRVocabulary
from sae.pt_dataset_and_dataloader.clevr_dataset import CLEVRLiteDataset


def create_dataloaders(
    data_dir: str,
    batch_size: int = 64,
    num_workers: int = 4,
    image_size: int = 224,
    max_question_length: int = 32,
) -> Tuple[DataLoader, DataLoader, CLEVRVocabulary]:
    """
    Create train and validation dataloaders with shared vocabulary.
    
    Returns:
        train_loader, val_loader, vocab
    """
    # Build vocabulary from training data
    train_questions = Path(data_dir) / "train_questions.json"
    vocab = CLEVRVocabulary(str(train_questions))
    
    # Create datasets
    train_dataset = CLEVRLiteDataset(
        data_dir=data_dir,
        split='train',
        vocab=vocab,
        max_question_length=max_question_length,
        image_size=image_size,
        filter_held_out=False,  # training uses only in-domain
    )
    
    val_dataset = CLEVRLiteDataset(
        data_dir=data_dir,
        split='val',
        vocab=vocab,
        max_question_length=max_question_length,
        image_size=image_size,
        filter_held_out=None,  # validation includes all
    )
    
    # Create dataloaders
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=True,
        drop_last=True,  # for stable batch stats
    )
    
    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True,
    )
    
    print(f"\n✓ DataLoaders created:")
    print(f"  Train: {len(train_dataset)} samples, {len(train_loader)} batches")
    print(f"  Val: {len(val_dataset)} samples, {len(val_loader)} batches")
    print(f"  Vocabulary: {len(vocab)} tokens")
    
    return train_loader, val_loader, vocab


def test_dataloader():
    """Quick test of the dataloader"""
    print("Testing dataloader...")
    
    train_loader, val_loader, vocab = create_dataloaders(
        data_dir='sae/custom_clevr_lite/data/clevr_lite_data',
        batch_size=4,
        num_workers=0,  # single-threaded for testing
    )
    
    # Get one batch
    batch = next(iter(train_loader))
    
    print(f"\nBatch shapes:")
    print(f"  image: {batch['image'].shape}")
    print(f"  question_ids: {batch['question_ids'].shape}")
    print(f"  question_mask: {batch['question_mask'].shape}")
    print(f"  answer_id: {batch['answer_id'].shape}")
    
    # Decode first sample
    print(f"\nFirst sample:")
    print(f"  Question: {vocab.decode(batch['question_ids'][0].tolist())}")
    print(f"  Answer: {batch['answer_text'][0]}")
    print(f"  Held-out: {batch['is_held_out'][0].item()}")
    
    val_held_out_count = sum(batch['is_held_out'].sum().item() for batch in val_loader)
    print(f"\nValidation held-out samples: {val_held_out_count}")

if __name__ == '__main__':
    test_dataloader()