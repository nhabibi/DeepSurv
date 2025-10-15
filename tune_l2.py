"""
Quick L2 regularization tuning script.
Tests different L2 values to find optimal for PyTorch.
"""

import os
import torch
import numpy as np
from config import MODEL_CONFIG, TRAINING_CONFIG, DATA_CONFIG, PATHS, LOSS_CONFIG
from model import DeepSurv
from data_loader import create_synthetic_data, prepare_data_loaders
from train import Trainer
from evaluation import concordance_index

# Suppress warnings
import warnings
warnings.filterwarnings('ignore')


def quick_test(l2_value, n_samples=2000, max_epochs=100, patience=20):
    """
    Quick test with specific L2 value.
    
    Args:
        l2_value: L2 regularization value to test
        n_samples: Number of samples
        max_epochs: Max epochs
        patience: Early stopping patience
        
    Returns:
        best_val_ci, final_val_ci, epochs_trained
    """
    # Set seed for reproducibility
    torch.manual_seed(42)
    np.random.seed(42)
    
    # Device
    if torch.backends.mps.is_available():
        device = 'mps'
    elif torch.cuda.is_available():
        device = 'cuda'
    else:
        device = 'cpu'
    
    # Generate data
    features, times, events = create_synthetic_data(
        n_samples=n_samples,
        n_features=10,
        data_type='linear',
        random_seed=42
    )
    
    # Prepare data
    train_loader, val_loader = prepare_data_loaders(
        features, times, events,
        batch_size=64,
        validation_split=0.2,
        normalize=True,
        random_seed=42
    )
    
    # Build model
    input_dim = features.shape[1]
    model = DeepSurv(
        input_dim=input_dim,
        hidden_layers=MODEL_CONFIG['hidden_layers'],
        activation=MODEL_CONFIG['activation'],
        dropout=MODEL_CONFIG['dropout'],
        use_batch_norm=MODEL_CONFIG['use_batch_norm']
    )
    
    # Trainer
    trainer = Trainer(
        model=model,
        device=device,
        learning_rate=TRAINING_CONFIG['learning_rate'],
        lr_decay=TRAINING_CONFIG['lr_decay'],
        momentum=TRAINING_CONFIG['momentum'],
        optimizer_name=TRAINING_CONFIG['optimizer'],
        l2_reg=l2_value,  # Use test value
        l1_reg=TRAINING_CONFIG['l1_reg'],
        loss_method=LOSS_CONFIG['method']
    )
    
    # Train (quiet mode)
    history = trainer.fit(
        train_loader, val_loader,
        num_epochs=max_epochs,
        early_stopping_patience=patience,
        save_path=None,  # Don't save
        verbose=False
    )
    
    # Get best and final C-index
    best_val_ci = max(history['val_ci'])
    final_val_ci = history['val_ci'][-1]
    epochs_trained = len(history['val_ci'])
    
    return best_val_ci, final_val_ci, epochs_trained


def main():
    """Test different L2 values."""
    print("=" * 60)
    print("L2 Regularization Grid Search (PyTorch)")
    print("=" * 60)
    print(f"Fixed: LR={TRAINING_CONFIG['learning_rate']}, 100 epochs max, 2000 samples")
    print(f"Testing: L2 values")
    print("=" * 60)
    print()
    
    # L2 values to test
    l2_values = [0.0, 0.001, 0.005, 0.01, 0.05, 0.1, 0.5, 1.0]
    
    results = []
    
    for l2 in l2_values:
        print(f"Testing L2 = {l2:6.3f}...", end=" ", flush=True)
        
        try:
            best_ci, final_ci, epochs = quick_test(l2)
            results.append({
                'l2': l2,
                'best_ci': best_ci,
                'final_ci': final_ci,
                'epochs': epochs
            })
            print(f"Best CI: {best_ci:.4f}, Final CI: {final_ci:.4f}, Epochs: {epochs}")
        except Exception as e:
            print(f"ERROR: {e}")
            results.append({
                'l2': l2,
                'best_ci': 0.0,
                'final_ci': 0.0,
                'epochs': 0
            })
    
    print()
    print("=" * 60)
    print("Summary Results")
    print("=" * 60)
    print(f"{'L2':>8} | {'Best C-Index':>12} | {'Final C-Index':>12} | {'Epochs':>6}")
    print("-" * 60)
    
    for r in results:
        print(f"{r['l2']:8.3f} | {r['best_ci']:12.4f} | {r['final_ci']:12.4f} | {r['epochs']:6d}")
    
    print("=" * 60)
    
    # Find best
    best_result = max(results, key=lambda x: x['best_ci'])
    print(f"\n🏆 Best L2: {best_result['l2']:.3f} with C-index: {best_result['best_ci']:.4f}")
    print()


if __name__ == '__main__':
    main()
