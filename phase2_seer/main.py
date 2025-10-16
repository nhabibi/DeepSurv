"""
Phase 2: SEER Application - Main Training Script
DeepSurv on SEER comorbid cancer data
"""

import os
import sys
import torch
import numpy as np
import json
import argparse

# Add phase2 src to path
phase2_dir = os.path.dirname(__file__)
sys.path.insert(0, os.path.join(phase2_dir, 'src'))

from config import MODEL_CONFIG, TRAINING_CONFIG, DATA_CONFIG, PATHS, LOSS_CONFIG
from model import DeepSurv
from data_loader import load_seer_data, prepare_data_loaders
from train import Trainer
from evaluation import evaluate_model, plot_training_curves, plot_risk_distribution


def parse_args():
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(description='Phase 2: Train DeepSurv on SEER Data')
    parser.add_argument('--generate-data', action='store_true',
                        help='Show instructions to generate SEER comorbid data')
    parser.add_argument('--train-data', type=str,
                        default='data/train_comorbid.csv',
                        help='Path to training CSV file')
    parser.add_argument('--val-data', type=str,
                        default='data/val_comorbid.csv',
                        help='Path to validation CSV file')
    parser.add_argument('--cpu', action='store_true',
                        help='Force CPU (disable GPU)')
    return parser.parse_args()


def set_seed(seed: int):
    """Set random seed for reproducibility."""
    torch.manual_seed(seed)
    np.random.seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def select_device(force_cpu: bool = False) -> str:
    """Select the best available device."""
    if force_cpu:
        return 'cpu'
    
    if torch.cuda.is_available():
        return 'cuda'
    elif hasattr(torch.backends, 'mps') and torch.backends.mps.is_available():
        return 'mps'
    else:
        return 'cpu'


def main():
    # Parse arguments
    args = parse_args()
    
    # If generate-data flag, show instructions
    if args.generate_data:
        print("\n" + "="*70)
        print("GENERATE SEER COMORBID DATA")
        print("="*70)
        print("\nRun this code to generate data/train_comorbid.csv, val_comorbid.csv, test_comorbid.csv:")
        print("\nSee: docs/PHASE2_SEER_GUIDE.md for data generation code")
        print("\nOr use the backup generation script if available.")
        print("="*70)
        return
    
    # Set seed
    set_seed(DATA_CONFIG['random_seed'])
    
    # Select device
    device = select_device(force_cpu=args.cpu)
    print(f"\n{'='*70}")
    print(f"PHASE 2: SEER APPLICATION")
    print(f"{'='*70}")
    print(f"Device: {device}")
    print(f"PyTorch version: {torch.__version__}")
    
    # =========================================================================
    # Load SEER Data
    # =========================================================================
    
    print(f"\nLoading SEER data...")
    
    # Load training data
    train_path = os.path.join(phase2_dir, args.train_data)
    features_train, times_train, events_train, scaler, feature_names = load_seer_data(
        train_path,
        normalize=DATA_CONFIG['normalize']
    )
    
    # Load validation data
    val_path = os.path.join(phase2_dir, args.val_data)
    features_val, times_val, events_val, _, _ = load_seer_data(
        val_path,
        normalize=DATA_CONFIG['normalize']
    )
    
    # Apply same scaler to validation data
    if scaler is not None:
        n_numerical = len([f for f in feature_names if not any(cat in f for cat in ['race_', 'marital_status_', 'first_cancer_site_'])])
        features_val[:, :n_numerical] = scaler.transform(features_val[:, :n_numerical])
    
    print(f"\nTraining data:")
    print(f"  Features: {features_train.shape}")
    print(f"  Events: {events_train.sum()} deaths ({events_train.mean()*100:.1f}%), "
          f"{(1-events_train).sum()} censored ({(1-events_train.mean())*100:.1f}%)")
    print(f"  Time range: [{times_train.min():.2f}, {times_train.max():.2f}]")
    
    print(f"\nValidation data:")
    print(f"  Features: {features_val.shape}")
    print(f"  Events: {events_val.sum()} deaths ({events_val.mean()*100:.1f}%), "
          f"{(1-events_val).sum()} censored ({(1-events_val.mean())*100:.1f}%)")
    
    # =========================================================================
    # Prepare Data Loaders
    # =========================================================================
    
    from torch.utils.data import DataLoader
    from data_loader import SurvivalDataset
    
    train_dataset = SurvivalDataset(features_train, times_train, events_train)
    val_dataset = SurvivalDataset(features_val, times_val, events_val)
    
    train_loader = DataLoader(
        train_dataset,
        batch_size=TRAINING_CONFIG['batch_size'],
        shuffle=True
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=TRAINING_CONFIG['batch_size'],
        shuffle=False
    )
    
    print(f"\nData loaders:")
    print(f"  Training: {len(train_loader.dataset)} samples, {len(train_loader)} batches")
    print(f"  Validation: {len(val_loader.dataset)} samples, {len(val_loader)} batches")
    
    # =========================================================================
    # Initialize Model
    # =========================================================================
    
    n_features = features_train.shape[1]
    model = DeepSurv(
        input_dim=n_features,
        hidden_layers=MODEL_CONFIG['hidden_layers'],
        activation=MODEL_CONFIG['activation'],
        dropout=MODEL_CONFIG['dropout'],
        use_batch_norm=MODEL_CONFIG['batch_norm']
    ).to(device)
    
    print(f"\nModel architecture:")
    print(f"  Input: {n_features} features (SEER comorbid data)")
    print(f"  Hidden layers: {MODEL_CONFIG['hidden_layers']}")
    print(f"  Activation: {MODEL_CONFIG['activation']}")
    print(f"  Dropout: {MODEL_CONFIG['dropout']}")
    print(f"  Batch norm: {MODEL_CONFIG['batch_norm']}")
    print(f"  Parameters: {sum(p.numel() for p in model.parameters())}")
    
    # =========================================================================
    # Train Model
    # =========================================================================
    
    trainer = Trainer(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        device=device,
        learning_rate=TRAINING_CONFIG['learning_rate'],
        l2_reg=TRAINING_CONFIG['l2_reg'],
        optimizer_name=TRAINING_CONFIG['optimizer'],
        momentum=TRAINING_CONFIG['momentum'],
        tie_method=LOSS_CONFIG['tie_method']
    )
    
    print(f"\nTraining configuration:")
    print(f"  Optimizer: {TRAINING_CONFIG['optimizer']}")
    print(f"  Learning rate: {TRAINING_CONFIG['learning_rate']}")
    print(f"  L2 regularization: {TRAINING_CONFIG['l2_reg']}")
    print(f"  Momentum: {TRAINING_CONFIG['momentum']}")
    print(f"  Batch size: {TRAINING_CONFIG['batch_size']}")
    print(f"  Max epochs: {TRAINING_CONFIG['max_epochs']}")
    print(f"  Early stopping: {TRAINING_CONFIG['early_stopping_patience']} epochs")
    print(f"  Tie handling: {LOSS_CONFIG['tie_method']}")
    
    # Train
    history = trainer.train(
        n_epochs=TRAINING_CONFIG['max_epochs'],
        patience=TRAINING_CONFIG['early_stopping_patience']
    )
    
    # =========================================================================
    # Final Evaluation
    # =========================================================================
    
    print(f"\n{'='*70}")
    print(f"FINAL EVALUATION")
    print(f"{'='*70}")
    
    train_metrics = evaluate_model(
        model, train_loader, device,
        tie_method=LOSS_CONFIG['tie_method']
    )
    val_metrics = evaluate_model(
        model, val_loader, device,
        tie_method=LOSS_CONFIG['tie_method']
    )
    
    print(f"\nTraining set:")
    print(f"  C-index: {train_metrics['c_index']:.4f}")
    print(f"  Loss: {train_metrics['loss']:.4f}")
    
    print(f"\nValidation set:")
    print(f"  C-index: {val_metrics['c_index']:.4f}")
    print(f"  Loss: {val_metrics['loss']:.4f}")
    
    # =========================================================================
    # Save Results
    # =========================================================================
    
    # Save model checkpoint
    checkpoint_dir = os.path.join(phase2_dir, 'results', 'checkpoints')
    os.makedirs(checkpoint_dir, exist_ok=True)
    checkpoint_path = os.path.join(checkpoint_dir, 'best_model.pt')
    
    torch.save({
        'model_state_dict': model.state_dict(),
        'model_config': MODEL_CONFIG,
        'training_config': TRAINING_CONFIG,
        'train_metrics': train_metrics,
        'val_metrics': val_metrics,
        'history': history,
        'feature_names': feature_names
    }, checkpoint_path)
    
    print(f"\n✅ Model saved to: {checkpoint_path}")
    
    # Plot training curves
    figures_dir = os.path.join(phase2_dir, 'results', 'figures')
    os.makedirs(figures_dir, exist_ok=True)
    
    plot_training_curves(
        history,
        save_path=os.path.join(figures_dir, 'training_curves.png')
    )
    
    print(f"✅ Training curves saved to: {figures_dir}/training_curves.png")
    
    print(f"\n{'='*70}")
    print(f"PHASE 2 COMPLETE!")
    print(f"{'='*70}\n")


if __name__ == '__main__':
    main()
