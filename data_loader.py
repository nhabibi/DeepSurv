"""
Data loading and preprocessing utilities.
"""

import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset, DataLoader
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from typing import Tuple, Optional


# ============================================================================
# Dataset Class
# ============================================================================

class SurvivalDataset(Dataset):
    """PyTorch Dataset for survival data."""
    
    def __init__(
        self,
        features: np.ndarray,
        times: np.ndarray,
        events: np.ndarray
    ):
        self.features = torch.FloatTensor(features)
        self.times = torch.FloatTensor(times)
        self.events = torch.FloatTensor(events)
    
    def __len__(self) -> int:
        return len(self.features)
    
    def __getitem__(self, idx: int):
        return self.features[idx], self.times[idx], self.events[idx]


# ============================================================================
# Data Loading
# ============================================================================

def load_data(
    file_path: str,
    feature_cols: Optional[list] = None,
    time_col: str = 'time',
    event_col: str = 'event',
    normalize: bool = True
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, Optional[StandardScaler]]:
    """
    Load survival data from CSV file.
    
    Args:
        file_path: Path to CSV file
        feature_cols: Feature columns (if None, use all except time/event)
        time_col: Survival time column
        event_col: Event indicator column (1=event, 0=censored)
        normalize: Normalize features to zero mean and unit variance
    
    Returns:
        features, times, events, scaler (or None if not normalized)
    """
    df = pd.read_csv(file_path)
    
    # Auto-detect feature columns
    if feature_cols is None:
        feature_cols = [col for col in df.columns if col not in [time_col, event_col]]
    
    # Extract data
    features = df[feature_cols].values
    times = df[time_col].values
    events = df[event_col].values
    
    # Normalize if requested
    scaler = None
    if normalize:
        scaler = StandardScaler()
        features = scaler.fit_transform(features)
    
    return features, times, events, scaler


# ============================================================================
# Data Loaders
# ============================================================================

def prepare_data_loaders(
    features: np.ndarray,
    times: np.ndarray,
    events: np.ndarray,
    batch_size: int = 64,
    validation_split: float = 0.2,
    random_seed: int = 42
) -> Tuple[DataLoader, DataLoader]:
    """
    Split data and create PyTorch DataLoaders.
    
    Args:
        features: Feature matrix [n_samples, n_features]
        times: Survival times [n_samples]
        events: Event indicators [n_samples]
        batch_size: Number of samples per batch
        validation_split: Fraction of data for validation (0-1)
        random_seed: Random seed for reproducibility
    
    Returns:
        train_loader, val_loader
    """
    # Split data
    train_idx, val_idx = train_test_split(
        np.arange(len(features)),
        test_size=validation_split,
        random_state=random_seed
    )
    
    # Create datasets
    train_dataset = SurvivalDataset(features[train_idx], times[train_idx], events[train_idx])
    val_dataset = SurvivalDataset(features[val_idx], times[val_idx], events[val_idx])
    
    # Create loaders
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
    
    return train_loader, val_loader


# ============================================================================
# Synthetic Data Generation
# ============================================================================

def create_synthetic_data(
    n_samples: int = 1000,
    n_features: int = 10,
    data_type: str = 'linear',
    random_seed: int = 42
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Generate synthetic survival data for testing.
    
    Args:
        n_samples: Number of samples
        n_features: Number of features
        data_type: 'linear' (simple) or 'gaussian' (non-linear)
        random_seed: Random seed for reproducibility
    
    Returns:
        features, times, events
    """
    np.random.seed(random_seed)
    
    # Generate features and compute hazard
    if data_type == 'linear':
        features = np.random.randn(n_samples, n_features)
        hazard = np.exp(0.5 * features[:, 0] - 0.3 * features[:, 1])
    elif data_type == 'gaussian':
        features = np.random.uniform(-1, 1, size=(n_samples, n_features))
        z = np.sum(features[:, :2] ** 2, axis=1)
        hazard = np.exp(3.0 * np.exp(-z / (2 * 0.5 ** 2)))
    else:
        raise ValueError(f"data_type must be 'linear' or 'gaussian', got: {data_type}")
    
    # Generate survival and censoring times
    survival_times = np.random.exponential(1.0 / hazard)
    censoring_times = np.random.exponential(2.0, size=n_samples)
    
    # Observed times and event indicators
    times = np.minimum(survival_times, censoring_times)
    events = (survival_times <= censoring_times).astype(float)
    
    return features, times, events
