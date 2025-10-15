"""
Find minimal L2 adjustment required for PyTorch.
Goal: Stay as close to vanilla L2=10.0 as possible while enabling learning.
"""

import os
import torch
import numpy as np
from config import MODEL_CONFIG, TRAINING_CONFIG, DATA_CONFIG, PATHS, LOSS_CONFIG
from model import DeepSurv
from data_loader import create_synthetic_data, prepare_data_loaders
from train import Trainer

import warnings
warnings.filterwarnings('ignore')


def test_l2(l2_value, device='mps'):
    """Quick test with specific L2."""
    torch.manual_seed(42)
    np.random.seed(42)
    
    # Data
    features, times, events = create_synthetic_data(
        n_samples=2000, n_features=10, data_type='linear', random_seed=42
    )
    
    # Normalize features
    from sklearn.preprocessing import StandardScaler
    scaler = StandardScaler()
    features = scaler.fit_transform(features)
    
    train_loader, val_loader = prepare_data_loaders(
        features, times, events,
        batch_size=64, validation_split=0.2, random_seed=42
    )
    
    # Model
    model = DeepSurv(
        input_dim=10,
        hidden_layers=[25, 25],
        activation='relu',
        dropout=0.0,
        use_batch_norm=False
    )
    
    # Trainer
    trainer = Trainer(
        model=model, device=device,
        learning_rate=1e-3,  # Fixed
        lr_decay=0.001,
        momentum=0.9,
        optimizer_name='sgd',
        l2_reg=l2_value,  # Test this
        l1_reg=0.0,
        loss_method='efron'
    )
    
    # Train (50 epochs, patience 15)
    history = trainer.fit(
        train_loader, val_loader,
        num_epochs=50,
        early_stopping_patience=15,
        save_path=None,
        verbose=False
    )
    
    best_ci = max(history['val_ci'])
    epochs = len(history['val_ci'])
    
    return best_ci, epochs


if __name__ == '__main__':
    device = 'mps' if torch.backends.mps.is_available() else 'cpu'
    
    print("=" * 70)
    print("Finding Minimal L2 Adjustment Required for PyTorch")
    print("=" * 70)
    print(f"Goal: Stay as close to vanilla L2=10.0 as possible")
    print(f"Test: Fixed LR=1e-3, varying L2, 50 epochs max")
    print(f"Device: {device}")
    print("=" * 70)
    print()
    
    # Test from high to low
    test_values = [
        10.0,   # Vanilla (expected to fail)
        5.0,    # Half vanilla
        1.0,    # 10% vanilla  
        0.5,    # Further reduction
        0.1,    # Conservative
        0.01,   # Known working
    ]
    
    print(f"{'L2 Value':>10} | {'Best C-Index':>12} | {'Epochs':>6} | {'Status'}")
    print("-" * 70)
    
    results = []
    for l2 in test_values:
        try:
            best_ci, epochs = test_l2(l2, device)
            status = "✅ Learning" if best_ci > 0.60 else "❌ No learning"
            results.append((l2, best_ci, epochs, status))
            print(f"{l2:10.3f} | {best_ci:12.4f} | {epochs:6d} | {status}")
        except Exception as e:
            print(f"{l2:10.3f} | ERROR: {str(e)[:30]}")
    
    print("=" * 70)
    
    # Find highest L2 that works
    working = [(l2, ci) for l2, ci, _, status in results if "Learning" in status]
    if working:
        best_l2, best_ci = max(working, key=lambda x: x[0])  # Highest L2 that works
        print(f"\n🎯 RECOMMENDATION:")
        print(f"   Highest L2 that enables learning: {best_l2:.3f} (C-index: {best_ci:.4f})")
        print(f"   This is {best_l2/10.0*100:.1f}% of vanilla L2=10.0")
        print(f"\n📝 This represents the MINIMAL adjustment required for PyTorch")
    else:
        print("\n❌ No working L2 value found in range tested")
