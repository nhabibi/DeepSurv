"""
Main script to train vanilla DeepSurv
"""

import os
import torch
import numpy as np
import json
import argparse

from config import MODEL_CONFIG, TRAINING_CONFIG, DATA_CONFIG, PATHS, LOSS_CONFIG
from model import DeepSurv
from data_loader import load_data, prepare_data_loaders, create_synthetic_data
from train import Trainer
from evaluation import evaluate_model, plot_training_curves, plot_risk_distribution


def parse_args():
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(description='Train vanilla DeepSurv')
    parser.add_argument('--data', type=str, default=None,
                        help='Path to CSV file (default: use synthetic data)')
    parser.add_argument('--data-type', type=str, default='linear',
                        choices=['linear', 'gaussian'],
                        help='Synthetic data type: linear or gaussian')
    parser.add_argument('--n-samples', type=int, default=5000,
                        help='Number of synthetic samples')
    parser.add_argument('--n-features', type=int, default=10,
                        help='Number of features')
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
    """Select device: MPS > CUDA > CPU."""
    if force_cpu:
        return 'cpu'
    elif torch.cuda.is_available():
        return 'cuda'
    elif torch.backends.mps.is_available():
        return 'mps'
    else:
        return 'cpu'


def main():
    """Main training pipeline."""
    
    # Parse arguments
    args = parse_args()
    set_seed(DATA_CONFIG['random_seed'])
    
    # Create output directories
    for path in PATHS.values():
        os.makedirs(path, exist_ok=True)
    
    # Select device
    device = select_device(args.cpu)
    print(f"Using device: {device}")
    if device == 'mps':
        print("🚀 Using Apple Silicon GPU (MPS)")
    elif device == 'cuda':
        print("🚀 Using NVIDIA GPU (CUDA)")
    
    # ------------------------------------------------------------------------
    # Load Data
    # ------------------------------------------------------------------------
    print("\n" + "="*50)
    print("Loading data...")
    print("="*50)
    
    if args.data:
        # Load from CSV
        features, times, events, _ = load_data(
            args.data,
            normalize=DATA_CONFIG['normalize']
        )
    else:
        # Generate synthetic data
        print(f"Generating synthetic {args.data_type} data...")
        features, times, events = create_synthetic_data(
            n_samples=args.n_samples,
            n_features=args.n_features,
            data_type=args.data_type,
            random_seed=DATA_CONFIG['random_seed']
        )
    
    print(f"Data: {features.shape[0]} samples, {features.shape[1]} features")
    print(f"Events: {int(events.sum())}, Censored: {int((1-events).sum())}")
    print(f"Event rate: {events.mean():.1%}")
    
    # ------------------------------------------------------------------------
    # Prepare Data Loaders
    # ------------------------------------------------------------------------
    train_loader, val_loader = prepare_data_loaders(
        features, times, events,
        batch_size=TRAINING_CONFIG['batch_size'],
        validation_split=TRAINING_CONFIG['validation_split'],
        random_seed=DATA_CONFIG['random_seed']
    )
    
    # ------------------------------------------------------------------------
    # Create Model
    # ------------------------------------------------------------------------
    print("\n" + "="*50)
    print("Building model...")
    print("="*50)
    
    MODEL_CONFIG['input_dim'] = features.shape[1]
    model = DeepSurv(**MODEL_CONFIG)
    
    print(model)
    print(f"Parameters: {sum(p.numel() for p in model.parameters()):,}")
    
    # ------------------------------------------------------------------------
    # Train Model
    # ------------------------------------------------------------------------
    print("\n" + "="*50)
    print("Training...")
    print("="*50)
    
    trainer = Trainer(
        model=model,
        device=device,
        learning_rate=TRAINING_CONFIG['learning_rate'],
        lr_decay=TRAINING_CONFIG['lr_decay'],
        momentum=TRAINING_CONFIG['momentum'],
        optimizer_name=TRAINING_CONFIG['optimizer'],
        l2_reg=TRAINING_CONFIG['l2_reg'],
        l1_reg=TRAINING_CONFIG['l1_reg'],
        loss_method=LOSS_CONFIG['method']
    )
    
    history = trainer.fit(
        train_loader, val_loader,
        num_epochs=TRAINING_CONFIG['num_epochs'],
        early_stopping_patience=TRAINING_CONFIG['early_stopping_patience'],
        save_path=os.path.join(PATHS['model_dir'], 'best_model.pt'),
        verbose=True
    )
    
    # ------------------------------------------------------------------------
    # Results
    # ------------------------------------------------------------------------
    print("\n" + "="*50)
    print("Results")
    print("="*50)
    
    # Plot training curves
    plot_training_curves(
        history,
        save_path=os.path.join(PATHS['results_dir'], 'training_curves.png')
    )
    
    # Evaluate on validation set
    val_ci, val_risks, val_times, val_events = evaluate_model(model, val_loader, device)
    
    # Plot risk distribution
    plot_risk_distribution(
        val_risks, val_events,
        save_path=os.path.join(PATHS['results_dir'], 'risk_distribution.png')
    )
    
    # Save results
    results = {
        'final_train_loss': history['train_loss'][-1],
        'final_val_loss': history['val_loss'][-1],
        'final_train_ci': history['train_ci'][-1],
        'final_val_ci': history['val_ci'][-1],
        'best_val_ci': max(history['val_ci']),
    }
    
    with open(os.path.join(PATHS['results_dir'], 'results.json'), 'w') as f:
        json.dump(results, f, indent=4)
    
    print(f"\n✅ Final C-Index: {val_ci:.4f}")
    print(f"✅ Best C-Index: {max(history['val_ci']):.4f}")
    print("\nTraining complete!")


if __name__ == "__main__":
    main()
