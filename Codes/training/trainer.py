"""
Training pipeline for baseline U-Net
"""

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm
import numpy as np
from pathlib import Path
import json
from typing import Dict, Optional

from utils.metrics import compute_metrics
from utils.visualization import save_comparison_plot


class Trainer:
    """Trainer for baseline U-Net model"""
    
    def __init__(
        self,
        model: nn.Module,
        train_loader: DataLoader,
        val_loader: DataLoader,
        criterion: nn.Module,
        optimizer: torch.optim.Optimizer,
        scheduler: Optional[torch.optim.lr_scheduler._LRScheduler] = None,
        device: str = 'cuda',
        save_dir: str = './experiments/baseline',
        log_interval: int = 10,
        save_interval: int = 5,
        visualize_interval: int = 5
    ):
        self.model = model.to(device)
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.criterion = criterion
        self.optimizer = optimizer
        self.scheduler = scheduler
        self.device = device
        self.save_dir = Path(save_dir)
        self.log_interval = log_interval
        self.save_interval = save_interval
        self.visualize_interval = visualize_interval
        
        # Create directories
        self.save_dir.mkdir(parents=True, exist_ok=True)
        (self.save_dir / 'checkpoints').mkdir(exist_ok=True)
        (self.save_dir / 'visualizations').mkdir(exist_ok=True)
        
        # TensorBoard
        self.writer = SummaryWriter(self.save_dir / 'logs')
        
        # Training state
        self.epoch = 0
        self.best_val_loss = float('inf')
        self.train_losses = []
        self.val_losses = []
    
    def train_epoch(self):
        """Train for one epoch"""
        
        self.model.train()
        epoch_losses = []
        
        pbar = tqdm(self.train_loader, desc=f'Epoch {self.epoch+1} [Train]')
        
        for batch_idx, batch in enumerate(pbar):
            # Move to device
            t1 = batch['t1'].to(self.device)
            t2 = batch['t2'].to(self.device)
            flair = batch['flair'].to(self.device)
            t1ce = batch['t1ce'].to(self.device)
            
            # Concatenate input modalities
            inputs = torch.cat([t1, t2, flair], dim=1)  # (B, 3, D, H, W)
            
            # Forward pass
            outputs = self.model(inputs)  # (B, 1, D, H, W)
            
            # Compute loss
            if isinstance(self.criterion, nn.MSELoss):
                loss = self.criterion(outputs, t1ce)
                loss_dict = {'mse': loss.item()}
            else:
                loss, loss_dict = self.criterion(outputs, t1ce)
            
            # Backward pass
            self.optimizer.zero_grad()
            loss.backward()
            
            # Gradient clipping
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
            
            self.optimizer.step()
            
            # Record
            epoch_losses.append(loss.item())
            
            # Update progress bar
            pbar.set_postfix({'loss': loss.item()})
            
            # Log to TensorBoard
            if batch_idx % self.log_interval == 0:
                global_step = self.epoch * len(self.train_loader) + batch_idx
                self.writer.add_scalar('train/loss', loss.item(), global_step)
                
                for key, value in loss_dict.items():
                    self.writer.add_scalar(f'train/{key}', value, global_step)
        
        return np.mean(epoch_losses)
    
    def validate(self):
        """Validate on validation set"""
        
        self.model.eval()
        epoch_losses = []
        all_metrics = []
        
        pbar = tqdm(self.val_loader, desc=f'Epoch {self.epoch+1} [Val]')
        
        with torch.no_grad():
            for batch_idx, batch in enumerate(pbar):
                # Move to device
                t1 = batch['t1'].to(self.device)
                t2 = batch['t2'].to(self.device)
                flair = batch['flair'].to(self.device)
                t1ce = batch['t1ce'].to(self.device)
                
                # Concatenate inputs
                inputs = torch.cat([t1, t2, flair], dim=1)
                
                # Forward pass
                outputs = self.model(inputs)
                
                # Compute loss
                if isinstance(self.criterion, nn.MSELoss):
                    loss = self.criterion(outputs, t1ce)
                else:
                    loss, _ = self.criterion(outputs, t1ce)
                
                epoch_losses.append(loss.item())
                
                # Compute metrics
                metrics = compute_metrics(outputs, t1ce)
                all_metrics.append(metrics)
                
                # Update progress bar
                pbar.set_postfix({
                    'loss': loss.item(),
                    'ssim': metrics['ssim']
                })
                
                # Visualize first batch
                if batch_idx == 0 and self.epoch % self.visualize_interval == 0:
                    self.visualize_predictions(batch, outputs)
        
        # Average metrics
        avg_metrics = {
            key: np.mean([m[key] for m in all_metrics])
            for key in all_metrics[0].keys()
        }
        
        return np.mean(epoch_losses), avg_metrics
    
    def visualize_predictions(self, batch, outputs):
        """Save visualization of predictions"""
        
        save_path = self.save_dir / 'visualizations' / f'epoch_{self.epoch+1:03d}.png'
        
        save_comparison_plot(
            t1=batch['t1'][0, 0].cpu().numpy(),
            t2=batch['t2'][0, 0].cpu().numpy(),
            flair=batch['flair'][0, 0].cpu().numpy(),
            real=batch['t1ce'][0, 0].cpu().numpy(),
            synthetic=outputs[0, 0].cpu().numpy(),
            save_path=save_path,
            slice_idx=77
        )
    
    def save_checkpoint(self, is_best=False):
        """Save model checkpoint"""
        
        checkpoint = {
            'epoch': self.epoch,
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'best_val_loss': self.best_val_loss,
            'train_losses': self.train_losses,
            'val_losses': self.val_losses,
        }
        
        if self.scheduler is not None:
            checkpoint['scheduler_state_dict'] = self.scheduler.state_dict()
        
        # Save latest
        torch.save(checkpoint, self.save_dir / 'checkpoints' / 'latest.pth')
        
        # Save best
        if is_best:
            torch.save(checkpoint, self.save_dir / 'checkpoints' / 'best.pth')
        
        # Save periodic
        if self.epoch % self.save_interval == 0:
            torch.save(
                checkpoint,
                self.save_dir / 'checkpoints' / f'epoch_{self.epoch:03d}.pth'
            )
    
    def train(self, num_epochs):
        """Main training loop"""
        
        print(f"Starting training for {num_epochs} epochs...")
        print(f"Save directory: {self.save_dir}")
        
        for epoch in range(num_epochs):
            self.epoch = epoch
            
            # Train
            train_loss = self.train_epoch()
            self.train_losses.append(train_loss)
            
            # Validate
            val_loss, val_metrics = self.validate()
            self.val_losses.append(val_loss)
            
            # Learning rate scheduling
            if self.scheduler is not None:
                self.scheduler.step()
            
            # Log to TensorBoard
            self.writer.add_scalar('epoch/train_loss', train_loss, epoch)
            self.writer.add_scalar('epoch/val_loss', val_loss, epoch)
            
            for key, value in val_metrics.items():
                self.writer.add_scalar(f'epoch/val_{key}', value, epoch)
            
            # Print epoch summary
            print(f"\nEpoch {epoch+1}/{num_epochs}")
            print(f"Train Loss: {train_loss:.6f}")
            print(f"Val Loss: {val_loss:.6f}")
            print(f"Val SSIM: {val_metrics['ssim']:.4f}")
            print(f"Val PSNR: {val_metrics['psnr']:.2f} dB")
            
            # Save checkpoint
            is_best = val_loss < self.best_val_loss
            if is_best:
                self.best_val_loss = val_loss
                print("✓ New best model!")
            
            self.save_checkpoint(is_best=is_best)
        
        # Save final training history
        history = {
            'train_losses': self.train_losses,
            'val_losses': self.val_losses,
        }
        
        with open(self.save_dir / 'history.json', 'w') as f:
            json.dump(history, f, indent=2)
        
        print(f"\n✅ Training completed!")
        print(f"Best validation loss: {self.best_val_loss:.6f}")
        
        self.writer.close()