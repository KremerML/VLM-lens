"""
Training Script for CLEVR-Lite VLM with Staged Training

Stages:
1. Attribute pretrain: Train only attribute heads on frozen vision
2. Joint training: Train full model end-to-end
3. Head-sharpen: Fine-tune QA head with frozen backbone

Features:
- AdamW optimizer with cosine LR schedule
- Gradient clipping
- Checkpointing (save/load)
- Held-out combo accuracy tracking
- Minimal logging
"""

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from pathlib import Path
from tqdm import tqdm # type: ignore
from typing import Dict

from sae.model.vlm_wrapper import CLEVRLiteVLM, VLMConfig, compute_losses
from sae.pt_dataset_and_dataloader.clevr_dataloader import create_dataloaders


class VLMTrainer:
    """Trainer for CLEVR-Lite VLM with staged training"""
    
    def __init__(
        self,
        model: CLEVRLiteVLM,
        train_loader: DataLoader,
        val_loader: DataLoader,
        config: Dict,
        device: str = 'cuda',
    ):
        self.model = model.to(device)
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.config = config
        self.device = device
        
        # Optimizer and scheduler (will be reset per stage)
        self.optimizer = None
        self.scheduler = None
        
        # Tracking
        self.global_step = 0
        self.best_val_acc = 0.0
        
        # Create checkpoint dir
        self.checkpoint_dir = Path(config['checkpoint_dir'])
        self.checkpoint_dir.mkdir(parents=True, exist_ok=True)
    
    def _setup_optimizer(self, params, lr: float, num_steps: int):
        """Setup optimizer and cosine scheduler"""
        self.optimizer = torch.optim.AdamW(
            params,
            lr=lr,
            betas=(0.9, 0.999),
            weight_decay=self.config['weight_decay'],
        )
        
        self.scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            self.optimizer,
            T_max=num_steps,
            eta_min=lr * 0.1,
        )
    
    def _get_targets(self, batch: Dict) -> Dict[str, torch.Tensor]:
        """Extract targets from batch"""
        # For attributes, we need to create dummy targets from scene objects
        # In a real scenario, you'd extract these from scene_objects
        # For now, use answer_id as a proxy (you should precompute these)
        B = batch['answer_id'].size(0)
        
        return {
            'answer_id': batch['answer_id'].to(self.device),
            'colors': torch.randint(0, 6, (B,)).to(self.device),  # TODO: extract from scene
            'shapes': torch.randint(0, 3, (B,)).to(self.device),  # TODO: extract from scene
            'sizes': torch.randint(0, 2, (B,)).to(self.device),   # TODO: extract from scene
        }
    
    def train_step(self, batch: Dict, loss_weights: Dict[str, float]) -> Dict[str, float]:
        """Single training step"""
        self.model.train()
        
        # Move to device
        images = batch['image'].to(self.device)
        text_ids = batch['question_ids'].to(self.device)
        targets = self._get_targets(batch)
        
        # Forward pass
        outputs = self.model(images, text_ids, return_hooks=False)
        
        # Compute loss
        loss, loss_dict = compute_losses(outputs, targets, loss_weights)
        
        # Backward pass
        self.optimizer.zero_grad()
        loss.backward()
        
        # Gradient clipping
        torch.nn.utils.clip_grad_norm_(
            self.model.parameters(),
            self.config['grad_clip'],
        )
        
        self.optimizer.step()
        self.scheduler.step()
        
        # Compute accuracy
        qa_preds = outputs['qa_logits'].argmax(dim=-1)
        qa_acc = (qa_preds == targets['answer_id']).float().mean().item()
        
        loss_dict['qa_acc'] = qa_acc
        loss_dict['lr'] = self.scheduler.get_last_lr()[0]
        
        self.global_step += 1
        
        return loss_dict
    
    @torch.no_grad()
    def evaluate(self) -> Dict[str, float]:
        """Evaluate on validation set"""
        self.model.eval()
        
        total_loss = 0.0
        total_correct = 0
        total_samples = 0
        
        # Track held-out combo accuracy separately
        held_out_correct = 0
        held_out_samples = 0
        
        for batch in tqdm(self.val_loader, desc='Evaluating', leave=False):
            images = batch['image'].to(self.device)
            text_ids = batch['question_ids'].to(self.device)
            targets = self._get_targets(batch)
            
            # Forward pass
            outputs = self.model(images, text_ids, return_hooks=False)
            
            # Compute loss
            loss, _ = compute_losses(outputs, targets)
            total_loss += loss.item() * images.size(0)
            
            # Accuracy
            qa_preds = outputs['qa_logits'].argmax(dim=-1)
            correct = (qa_preds == targets['answer_id'])
            
            total_correct += correct.sum().item()
            total_samples += images.size(0)
            
            # Held-out accuracy
            is_held_out = batch['is_held_out'].bool()
            if is_held_out.any():
                held_out_correct += correct[is_held_out].sum().item()
                held_out_samples += is_held_out.sum().item()
        
        metrics = {
            'val_loss': total_loss / total_samples,
            'val_acc': total_correct / total_samples,
            'val_held_out_acc': held_out_correct / held_out_samples if held_out_samples > 0 else 0.0,
        }
        
        return metrics
    
    def save_checkpoint(self, stage: str, epoch: int, metrics: Dict):
        """Save model checkpoint"""
        checkpoint = {
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'scheduler_state_dict': self.scheduler.state_dict(),
            'global_step': self.global_step,
            'epoch': epoch,
            'stage': stage,
            'metrics': metrics,
            'config': self.config,
        }
        
        # Save latest
        checkpoint_path = self.checkpoint_dir / f'{stage}_latest.pt'
        torch.save(checkpoint, checkpoint_path)
        
        # Save best (based on val accuracy)
        if metrics.get('val_acc', 0.0) > self.best_val_acc:
            self.best_val_acc = metrics['val_acc']
            best_path = self.checkpoint_dir / f'{stage}_best.pt'
            torch.save(checkpoint, best_path)
            print(f"  → Saved best checkpoint: {best_path}")
    
    def load_checkpoint(self, checkpoint_path: str):
        """Load model checkpoint"""
        checkpoint = torch.load(checkpoint_path, map_location=self.device)
        
        self.model.load_state_dict(checkpoint['model_state_dict'])
        
        if 'optimizer_state_dict' in checkpoint and self.optimizer is not None:
            self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        
        if 'scheduler_state_dict' in checkpoint and self.scheduler is not None:
            self.scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
        
        self.global_step = checkpoint.get('global_step', 0)
        
        print(f"Loaded checkpoint from {checkpoint_path}")
        print(f"  Stage: {checkpoint.get('stage', 'unknown')}")
        print(f"  Epoch: {checkpoint.get('epoch', 'unknown')}")
        print(f"  Metrics: {checkpoint.get('metrics', {})}")
    
    def train_stage_1_attribute_pretrain(self):
        """Stage 1: Pretrain attribute heads on frozen vision"""
        print("\n" + "=" * 70)
        print("STAGE 1: Attribute Pretrain")
        print("=" * 70)
        
        # Freeze everything except attribute heads
        for param in self.model.parameters():
            param.requires_grad = False
        for param in self.model.attr_heads.parameters():
            param.requires_grad = True
        
        # Setup optimizer
        num_epochs = self.config['stage1_epochs']
        num_steps = len(self.train_loader) * num_epochs
        
        self._setup_optimizer(
            self.model.attr_heads.parameters(),
            lr=self.config['stage1_lr'],
            num_steps=num_steps,
        )
        
        # Training loop
        loss_weights = {'qa': 0.0, 'attr': 1.0}  # Only attribute loss
        
        for epoch in range(num_epochs):
            print(f"\nEpoch {epoch + 1}/{num_epochs}")
            
            pbar = tqdm(self.train_loader, desc='Training')
            for batch in pbar:
                metrics = self.train_step(batch, loss_weights)
                pbar.set_postfix({
                    'loss': f"{metrics['total']:.4f}",
                    'attr': f"{metrics['attr']:.4f}",
                    'lr': f"{metrics['lr']:.2e}",
                })
            
            # Evaluate
            val_metrics = self.evaluate()
            print(f"  Val Loss: {val_metrics['val_loss']:.4f}")
            print(f"  Val Acc: {val_metrics['val_acc']:.4f}")
            print(f"  Val Held-out Acc: {val_metrics['val_held_out_acc']:.4f}")
            
            # Save checkpoint
            self.save_checkpoint('stage1', epoch, val_metrics)
        
        # Unfreeze for next stage
        for param in self.model.parameters():
            param.requires_grad = True
    
    def train_stage_2_joint(self):
        """Stage 2: Joint end-to-end training"""
        print("\n" + "=" * 70)
        print("STAGE 2: Joint Training")
        print("=" * 70)
        
        # All parameters trainable
        for param in self.model.parameters():
            param.requires_grad = True
        
        # Setup optimizer
        num_epochs = self.config['stage2_epochs']
        num_steps = len(self.train_loader) * num_epochs
        
        self._setup_optimizer(
            self.model.parameters(),
            lr=self.config['stage2_lr'],
            num_steps=num_steps,
        )
        
        # Training loop
        loss_weights = {'qa': 1.0, 'attr': 0.3}  # Balanced
        
        for epoch in range(num_epochs):
            print(f"\nEpoch {epoch + 1}/{num_epochs}")
            
            pbar = tqdm(self.train_loader, desc='Training')
            for batch in pbar:
                metrics = self.train_step(batch, loss_weights)
                pbar.set_postfix({
                    'loss': f"{metrics['total']:.4f}",
                    'qa': f"{metrics['qa']:.4f}",
                    'acc': f"{metrics['qa_acc']:.4f}",
                    'lr': f"{metrics['lr']:.2e}",
                })
            
            # Evaluate
            val_metrics = self.evaluate()
            print(f"  Val Loss: {val_metrics['val_loss']:.4f}")
            print(f"  Val Acc: {val_metrics['val_acc']:.4f}")
            print(f"  Val Held-out Acc: {val_metrics['val_held_out_acc']:.4f}")
            
            # Save checkpoint
            self.save_checkpoint('stage2', epoch, val_metrics)
    
    def train_stage_3_head_sharpen(self):
        """Stage 3: Fine-tune QA head with frozen backbone"""
        print("\n" + "=" * 70)
        print("STAGE 3: Head Sharpen")
        print("=" * 70)
        
        # Freeze backbone, train only QA head
        for param in self.model.parameters():
            param.requires_grad = False
        for param in self.model.qa_head.parameters():
            param.requires_grad = True
        
        # Setup optimizer
        num_epochs = self.config['stage3_epochs']
        num_steps = len(self.train_loader) * num_epochs
        
        self._setup_optimizer(
            self.model.qa_head.parameters(),
            lr=self.config['stage3_lr'],
            num_steps=num_steps,
        )
        
        # Training loop
        loss_weights = {'qa': 1.0, 'attr': 0.0}  # Only QA loss
        
        for epoch in range(num_epochs):
            print(f"\nEpoch {epoch + 1}/{num_epochs}")
            
            pbar = tqdm(self.train_loader, desc='Training')
            for batch in pbar:
                metrics = self.train_step(batch, loss_weights)
                pbar.set_postfix({
                    'loss': f"{metrics['total']:.4f}",
                    'acc': f"{metrics['qa_acc']:.4f}",
                    'lr': f"{metrics['lr']:.2e}",
                })
            
            # Evaluate
            val_metrics = self.evaluate()
            print(f"  Val Loss: {val_metrics['val_loss']:.4f}")
            print(f"  Val Acc: {val_metrics['val_acc']:.4f}")
            print(f"  Val Held-out Acc: {val_metrics['val_held_out_acc']:.4f}")
            
            # Save checkpoint
            self.save_checkpoint('stage3', epoch, val_metrics)


def main():
    """Main training script"""
    
    # Configuration
    config = {
        # Data
        'data_dir': 'sae/custom_clevr_lite/data/clevr_lite_data',
        'batch_size': 64,
        'num_workers': 4,
        
        # Model
        'vocab_size': 150,
        'num_answer_classes': 150,
        
        # Training stages
        'stage1_epochs': 2,
        'stage1_lr': 1e-3,
        
        'stage2_epochs': 10,
        'stage2_lr': 3e-4,
        
        'stage3_epochs': 3,
        'stage3_lr': 1e-4,
        
        # Optimization
        'weight_decay': 0.01,
        'grad_clip': 1.0,
        
        # Checkpointing
        'checkpoint_dir': './checkpoints',
        
        # Device
        'device': 'cuda' if torch.cuda.is_available() else 'cpu',
    }
    
    print("=" * 70)
    print("CLEVR-Lite VLM Training")
    print("=" * 70)
    print(f"Device: {config['device']}")
    
    # Create dataloaders
    print("\nCreating dataloaders...")
    train_loader, val_loader, vocab = create_dataloaders(
        data_dir=config['data_dir'],
        batch_size=config['batch_size'],
        num_workers=config['num_workers'],
    )
    
    # Create model
    print("\nCreating model...")
    model_config = VLMConfig(
        vocab_size=len(vocab),
        num_answer_classes=len(vocab),
    )
    model = CLEVRLiteVLM(model_config)
    
    param_counts = model.get_num_params()
    print(f"Total parameters: {param_counts['total']:,}")
    
    # Create trainer
    trainer = VLMTrainer(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        config=config,
        device=config['device'],
    )
    
    # Run staged training
    trainer.train_stage_1_attribute_pretrain()
    trainer.train_stage_2_joint()
    trainer.train_stage_3_head_sharpen()
    
    print("\n" + "=" * 70)
    print("Training complete!")
    print("=" * 70)
    print(f"Checkpoints saved to: {config['checkpoint_dir']}")


if __name__ == '__main__':
    main()