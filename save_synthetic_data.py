"""
Save synthetic data for analysis and inspection.
This helps understand the data structure before applying to real SEER data.
"""

import numpy as np
import pandas as pd
from data_loader import generate_synthetic_data
from pathlib import Path

def save_synthetic_data(n_samples=5000, n_features=10, data_type='linear'):
    """Generate and save synthetic survival data."""
    
    print(f"Generating synthetic {data_type} data...")
    print(f"  Samples: {n_samples}")
    print(f"  Features: {n_features}")
    
    # Generate data
    X_train, X_val, X_test, y_train, y_val, y_test, feature_names = generate_synthetic_data(
        n_samples=n_samples,
        n_features=n_features,
        data_type=data_type
    )
    
    # Create DataFrames for better inspection
    train_df = pd.DataFrame(X_train, columns=feature_names)
    train_df['time'] = y_train[:, 0]
    train_df['event'] = y_train[:, 1]
    
    val_df = pd.DataFrame(X_val, columns=feature_names)
    val_df['time'] = y_val[:, 0]
    val_df['event'] = y_val[:, 1]
    
    test_df = pd.DataFrame(X_test, columns=feature_names)
    test_df['time'] = y_test[:, 0]
    test_df['event'] = y_test[:, 1]
    
    # Save to CSV
    output_dir = Path('data/synthetic')
    output_dir.mkdir(parents=True, exist_ok=True)
    
    train_path = output_dir / f'train_{data_type}_{n_samples}samples.csv'
    val_path = output_dir / f'val_{data_type}_{n_samples}samples.csv'
    test_path = output_dir / f'test_{data_type}_{n_samples}samples.csv'
    
    train_df.to_csv(train_path, index=False)
    val_df.to_csv(val_path, index=False)
    test_df.to_csv(test_path, index=False)
    
    print(f"\n✅ Data saved:")
    print(f"  Train: {train_path} ({len(train_df)} samples)")
    print(f"  Val:   {val_path} ({len(val_df)} samples)")
    print(f"  Test:  {test_path} ({len(test_df)} samples)")
    
    # Print summary statistics
    print(f"\n📊 Data Summary:")
    print(f"  Features shape: {X_train.shape}")
    print(f"  Time range: [{train_df['time'].min():.2f}, {train_df['time'].max():.2f}]")
    print(f"  Event rate: {train_df['event'].mean():.2%}")
    print(f"  Censoring rate: {(1 - train_df['event'].mean()):.2%}")
    
    print(f"\n📈 Feature Statistics (Train):")
    print(train_df[feature_names].describe())
    
    print(f"\n⏰ Survival Time Statistics (Train):")
    print(train_df[['time', 'event']].describe())
    
    return train_df, val_df, test_df

if __name__ == '__main__':
    import argparse
    
    parser = argparse.ArgumentParser(description='Save synthetic survival data')
    parser.add_argument('--n-samples', type=int, default=5000, help='Number of samples')
    parser.add_argument('--n-features', type=int, default=10, help='Number of features')
    parser.add_argument('--data-type', type=str, default='linear', 
                        choices=['linear', 'nonlinear'], help='Data generation type')
    
    args = parser.parse_args()
    
    save_synthetic_data(
        n_samples=args.n_samples,
        n_features=args.n_features,
        data_type=args.data_type
    )
