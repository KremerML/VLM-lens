import torch
from torch.utils.data import Dataset
from torchvision import transforms
from PIL import Image
import json
from pathlib import Path
from typing import Dict, Optional
from clevr_vocabulary import CLEVRVocabulary

class CLEVRLiteDataset(Dataset):
    """PyTorch Dataset for CLEVR-Lite"""
    
    def __init__(
        self,
        data_dir: str,
        split: str = 'train',
        vocab: Optional[CLEVRVocabulary] = None,
        max_question_length: int = 32,
        image_size: int = 224,
        filter_held_out: Optional[bool] = None,  # None=all, True=only held-out, False=only in-domain
    ):
        """
        Args:
            data_dir: Root directory containing train/val folders
            split: 'train' or 'val'
            vocab: Vocabulary object (if None, creates from questions)
            max_question_length: Maximum tokens in question
            image_size: Image size (already 224 from generator)
            filter_held_out: Filter by held-out combo status
        """
        self.data_dir = Path(data_dir)
        self.split = split
        self.max_question_length = max_question_length
        
        # Load questions
        questions_file = self.data_dir / f"{split}_questions.json"
        with open(questions_file, 'r') as f:
            self.data = json.load(f)
        
        # Build or use vocabulary
        if vocab is None:
            print(f"Building vocabulary from {split} split...")
            self.vocab = CLEVRVocabulary(str(questions_file))
        else:
            self.vocab = vocab
        
        print(f"Vocabulary size: {len(self.vocab)}")
        
        # Filter by held-out status if requested
        if filter_held_out is not None:
            original_len = len(self.data)
            self.data = [d for d in self.data if d['is_held_out_combo'] == filter_held_out]
            print(f"Filtered {split}: {original_len} -> {len(self.data)} samples (held_out={filter_held_out})")
        
        # Image transforms (minimal - already clean from generator)
        self.transform = transforms.Compose([
            transforms.Resize((image_size, image_size)),
            transforms.ToTensor(),  # [0, 1]
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),  # ImageNet stats
        ])
    
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        """
        Returns:
            {
                'image': Tensor [3, 224, 224]
                'question_ids': Tensor [max_len] (padded)
                'question_mask': Tensor [max_len] (1=real token, 0=padding)
                'answer_id': Tensor [] (single token for discriminative QA)
                'answer_text': str (for reference)
                'is_held_out': bool
                'scene_id': int
            }
        """
        item = self.data[idx]
        
        # Load image
        img_path = self.data_dir / item['image_path']
        image = Image.open(img_path).convert('RGB')
        image = self.transform(image)
        
        # Encode question
        question_ids = self.vocab.encode(item['question'])
        
        # Pad/truncate question
        if len(question_ids) > self.max_question_length:
            question_ids = question_ids[:self.max_question_length]
        
        # Create mask (1 = real token, 0 = padding)
        question_mask = [1] * len(question_ids)
        
        # Pad to max length
        padding_length = self.max_question_length - len(question_ids)
        question_ids.extend([self.vocab.pad_id] * padding_length)
        question_mask.extend([0] * padding_length)
        
        # Encode answer (single token for now - discriminative)
        answer_ids = self.vocab.encode(item['answer'])
        answer_id = answer_ids[0] if answer_ids else self.vocab.pad_id
        
        return {
            'image': image,
            'question_ids': torch.tensor(question_ids, dtype=torch.long),
            'question_mask': torch.tensor(question_mask, dtype=torch.float),
            'answer_id': torch.tensor(answer_id, dtype=torch.long),
            'answer_text': item['answer'],
            'is_held_out': item['is_held_out_combo'],
            'scene_id': item['scene_id'],
        }