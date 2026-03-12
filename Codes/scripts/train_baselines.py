"""
Main training script for baseline U-Net
"""

import sys
sys.path.append('..')

import torch
import yaml
from pathlib import Path
from torch.utils.data import DataLoader

from data.dataset import BraTSDataset, collate_fn
from data.transforms import get_transforms
from models.unet_3d import UNet3D
from training.trainer import Trainer
from training.losses import get_loss_function


def load_config(config_path='../configs/config.yaml'):
    """Load configuration from YAML file"""
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    return config


def main():
    # Load config
    config = load_config()
    
    print("="*60)
    print("BraTS Synthesis - Baseline U-Net Training")
    print("="*60)
    
    # Set device
    device = torch.device(config['hardware']['device'] if torch.cuda.is_available() else 'cpu')
    print(f"\nDevice: {device}")
    
    # Create datasets
    print("\nLoading datasets...")
    
    train_dataset = BraTSDataset(
        root_dir=config['data']['root_dir'],
        split='train',
        train_ratio=config['data']['train_split'],
        seed=config['data']['seed'],
        transforms=get_transforms('train'),
        load_seg=False  # Don't need segmentation for synthesis
    )
    
    val_dataset = BraTSDataset(
        root_dir=config['data']['root_dir'],
        split='val',
        train_ratio=config['data']['train_split'],
        seed=config['data']['seed'],
        transforms=None,
        load_seg=False
    )
    
    print(f"Train samples: {len(train_dataset)}")
    print(f"Val samples: {len(val_dataset)}")
    
    # Create dataloaders
    train_loader = DataLoader(
        train_dataset,
        batch_size=config['training']['batch_size'],
        shuffle=True,
        num_workers=config['hardware']['num_workers'],
        pin_memory=config['hardware']['pin_memory'],
        collate_fn=collate_fn
    )
    
    val_loader = DataLoader(
        val_dataset,
        batch_size=config['training']['batch_size'],
        shuffle=False,
        num_workers=config['hardware']['num_workers'],
        pin_memory=config['hardware']['pin_memory'],
        collate_fn=collate_fn
    )
    
    # Create model
    print("\nInitializing model...")
    
    model = UNet3D(
        in_channels=config['model']['in_channels'],
        out_channels=config['model']['out_channels'],
        base_channels=config['model']['base_channels'],
        num_levels=config['model']['num_levels'],
        dropout=config['model']['dropout']
    )
    
    print(f"Model parameters: {model.count_parameters():,}")
    
    # Loss function
    criterion = get_loss_function('mse')
    
    # Optimizer
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=config['training']['learning_rate'],
        weight_decay=config['training']['weight_decay']
    )
    
    # Learning rate scheduler
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer,
        T_max=config['training']['num_epochs']
    )
    
    # Create save directory
    save_dir = Path(config['logging']['save_dir']) / config['logging']['experiment_name']
    save_dir.mkdir(parents=True, exist_ok=True)
    
    # Save config
    with open(save_dir / 'config.yaml', 'w') as f:
        yaml.dump(config, f)
    
    # Create trainer
    trainer = Trainer(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        criterion=criterion,
        optimizer=optimizer,
        scheduler=scheduler,
        device=device,
        save_dir=str(save_dir),
        log_interval=config['logging']['log_interval'],
        save_interval=config['logging']['save_interval'],
        visualize_interval=config['logging']['visualize_interval']
    )
    
    # Train
    print("\nStarting training...\n")
    
    trainer.train(num_epochs=config['training']['num_epochs'])
    
    print("\n" + "="*60)
    print("Training completed!")
    print("="*60)


if __name__ == '__main__':
    main()