# sae/pt_dataset_and_dataloader/clevr_collate.py
from typing import List, Dict, Any
import torch

def collate_clevr(batch: List[Dict[str, Any]]) -> Dict[str, Any]:
    out: Dict[str, Any] = {}
    # Fixed-size tensors → stack
    out["image"]         = torch.stack([b["image"] for b in batch], dim=0)                 # [B, 3, 224, 224]
    out["question_ids"]  = torch.stack([b["question_ids"] for b in batch], dim=0)         # [B, T]
    out["question_mask"] = torch.stack([b["question_mask"] for b in batch], dim=0)        # [B, T]
    out["answer_id"]     = torch.stack([b["answer_id"] for b in batch], dim=0)            # [B]

    # Scalars / flags → tensors
    out["is_held_out"]   = torch.tensor([bool(b["is_held_out"]) for b in batch], dtype=torch.bool)  # [B]
    out["scene_id"]      = torch.tensor([int(b["scene_id"]) for b in batch], dtype=torch.long)      # [B]

    # Ragged / non-tensor fields → keep as Python lists (do NOT try to stack)
    out["answer_text"]   = [b["answer_text"] for b in batch]                 # List[str]
    out["scene_objects"] = [b["scene_objects"] for b in batch]               # List[List[dict]]

    return out
