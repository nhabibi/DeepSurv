"""
Phase 2: SEER Data Application
Vanilla DeepSurv applied to SEER-like comorbidity data (25 features)
"""

import os
import sys
import torch
import numpy as np

# Add src to path
phase2_dir = os.path.dirname(__file__)
sys.path.insert(0, os.path.join(phase2_dir, 'src'))

from config import MODEL_CONFIG, TRAINING_CONFIG, DATA_CONFIG, LOSS_CONFIG
from model import DeepSurv
from data_loader import create_synthetic_data, prepare_data_loaders
from train import Trainer
from evaluation import evaluate_model, plot_training_curves


def main():
    # Set seed for reproducibility
    torch.manual_seed(DATA_CONFIG['random_seed'])
    np.random.seed(DATA_CONFIG['random_seed'])
    
    # Select device
    if torch.cuda.is_available():
        device = 'cuda'
    elif hasattr(torch.backends, 'mps') and torch.backends.mps.is_available():
        device = 'mps'
    else:
        device = 'cpu'
    
    print(f"\n{'='*70}")
    print(f"PHASE 2: SEER DATA APPLICATION (VANILLA)")
    print(f"{'='*70}")
    print(f"Device: {device}")
    
    # =========================================================================
    # Generate SEER-like Synthetic Data (25 features: demographics + comorbidities)
    # =========================================================================
    
    print(f"\nGenerating SEER-like synthetic data...")
    
    # Generate 25 features: 5 demographics + 20 comorbidities
    np.random.seed(DATA_CONFIG['random_seed'])
    n_samples = 5000
    
    # Demographics (5 features)
    age = np.random.normal(65, 15, n_samples).clip(18, 95)
    race_white = np.random.binomial(1, 0.7, n_samples)
    race_black = np.random.binomial(1, 0.15, n_samples)
    race_asian = np.random.binomial(1, 0.1, n_samples)
    race_other = 1 - (race_white + race_black + race_asian).clip(0, 1)
    
    # Comorbidities (20 binary features)
    comorbidities = np.random.binomial(1, 0.3, (n_samples, 20))
    
    # Combine all 25 features
    features_raw = np.column_stack([
        age, race_white, race_black, race_asian, race_other,
        comorbidities
    ])
    
    # Normalize
    features = (features_raw - features_raw.mean(axis=0)) / (features_raw.std(axis=0) + 1e-8)
    
    # Generate hazard (age + race + comorbidities)
    weights = np.concatenate([
        [1.0],  # age
        [0.3, 0.5, 0.2, 0.25],  # race
        np.random.uniform(0.2, 0.8, 20)  # comorbidities (stronger signal)
    ])
    
    log_hazard = np.dot(features, weights)
    hazard = np.exp(log_hazard)
    hazard = hazard / np.mean(hazard)
    
    # Generate survival times (months)
    survival_times = np.random.exponential(36.0 / (hazard + 1e-8))  # ~3 years mean
    censoring_times = np.random.exponential(72.0, size=n_samples)  # ~6 years mean
    
    times = np.minimum(survival_times, censoring_times)
    events = (survival_times <= censoring_times).astype(float)
    
    print(f"  Samples: {len(features)}")
    print(f"  Features: {features.shape[1]} (5 demographics + 20 comorbidities)")
    print(f"  Events: {events.sum()} ({events.mean()*100:.1f}%)")
    print(f"  Censored: {(1-events).sum()} ({(1-events.mean())*100:.1f}%)")
    print(f"  Survival: mean={times.mean():.1f} months, median={np.median(times):.1f} months")
    
    # Save SEER data to CSV
    import pandas as pd
    data_dir = os.path.join(phase2_dir, 'data')
    os.makedirs(data_dir, exist_ok=True)
    
    feature_names = ['age', 'race_white', 'race_black', 'race_asian', 'race_other'] + \
                    [f'comorbid_{i}' for i in range(20)]
    df = pd.DataFrame(features, columns=feature_names)
    df['survival_months'] = times
    df['vital_status'] = events
    
    data_path = os.path.join(data_dir, 'seer_synthetic_5000_comorbid.csv')
    df.to_csv(data_path, index=False)
    print(f"\n✅ Data saved: {data_path}")
    
    # =========================================================================
    # Prepare Data Loaders
    # =========================================================================
    
    train_loader, val_loader = prepare_data_loaders(
        features, times, events,
        batch_size=TRAINING_CONFIG['batch_size'],
        validation_split=TRAINING_CONFIG['validation_split'],
        random_seed=DATA_CONFIG['random_seed']
    )
    
    print(f"\nData splits:")
    print(f"  Training: {len(train_loader.dataset)} samples")
    print(f"  Validation: {len(val_loader.dataset)} samples")
    
    # =========================================================================
    # Initialize Model (25 input features for SEER)
    # =========================================================================
    
    model = DeepSurv(
        input_dim=25,  # SEER: 5 demographics + 20 comorbidities
        hidden_layers=MODEL_CONFIG['hidden_layers'],
        activation=MODEL_CONFIG['activation'],
        dropout=MODEL_CONFIG['dropout'],
        use_batch_norm=MODEL_CONFIG['use_batch_norm']
    ).to(device)
    
    print(f"\nModel:")
    print(f"  Input: 25 features (SEER comorbidity)")
    print(f"  Architecture: {MODEL_CONFIG['hidden_layers']}")
    print(f"  Activation: {MODEL_CONFIG['activation']}")
    print(f"  Parameters: {sum(p.numel() for p in model.parameters())}")
    
    # =========================================================================
    # Train Model
    # =========================================================================
    
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
    
    print(f"\nTraining (Vanilla - same as Phase 1):")
    print(f"  LR: {TRAINING_CONFIG['learning_rate']}")
    print(f"  L2: {TRAINING_CONFIG['l2_reg']}")
    print(f"  Batch size: {TRAINING_CONFIG['batch_size']}")
    print(f"  Max epochs: {TRAINING_CONFIG['max_epochs']}")
    print(f"  Early stop: {TRAINING_CONFIG['early_stopping_patience']}")
    
    history = trainer.fit(
        train_loader=train_loader,
        val_loader=val_loader,
        num_epochs=TRAINING_CONFIG['max_epochs'],
        early_stopping_patience=TRAINING_CONFIG['early_stopping_patience']
    )
    
    # =========================================================================
    # Final Evaluation
    # =========================================================================
    
    print(f"\n{'='*70}")
    print(f"FINAL RESULTS")
    print(f"{'='*70}")
    
    train_ci, _, _, _ = evaluate_model(model, train_loader, device)
    val_ci, _, _, _ = evaluate_model(model, val_loader, device)
    
    print(f"\nTraining:   C-index = {train_ci:.4f}")
    print(f"Validation: C-index = {val_ci:.4f}")
    
    # =========================================================================
    # Save Results
    # =========================================================================
    
    # Save model
    checkpoint_dir = os.path.join(phase2_dir, 'results', 'checkpoints')
    os.makedirs(checkpoint_dir, exist_ok=True)
    checkpoint_path = os.path.join(checkpoint_dir, 'best_model.pt')
    
    torch.save({
        'model_state_dict': model.state_dict(),
        'model_config': MODEL_CONFIG,
        'training_config': TRAINING_CONFIG,
        'train_c_index': train_ci,
        'val_c_index': val_ci,
        'history': history
    }, checkpoint_path)
    
    print(f"\n✅ Model saved: {checkpoint_path}")
    
    # Save training curves
    figures_dir = os.path.join(phase2_dir, 'results', 'figures')
    os.makedirs(figures_dir, exist_ok=True)
    plot_training_curves(history, save_path=os.path.join(figures_dir, 'training.png'))
    
    print(f"✅ Figures saved: {figures_dir}/training.png")
    print(f"\n{'='*70}")
    print(f"PHASE 2 COMPLETE!")
    print(f"{'='*70}\n")


if __name__ == '__main__':
    main()
