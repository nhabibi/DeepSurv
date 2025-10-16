"""
Evaluation metrics for survival analysis.
"""

import numpy as np
import matplotlib.pyplot as plt
from typing import Optional


# ============================================================================
# Concordance Index
# ============================================================================

def concordance_index(risk_scores: np.ndarray, times: np.ndarray, 
                     events: np.ndarray) -> float:
    """
    Calculate concordance index (C-index) - survival model performance metric.
    
    C-index measures the fraction of all pairs where the model correctly orders
    patients by survival time. Higher risk should predict shorter survival.
    
    Args:
        risk_scores: Predicted risk scores (higher = higher risk)
        times: Observed survival times
        events: Event indicators (1=event, 0=censored)
    
    Returns:
        C-index from 0 to 1 (0.5=random, 1.0=perfect, >0.7=good)
    """
    concordant = 0.0
    permissible = 0
    
    # Compare all pairs
    for i in range(len(risk_scores)):
        # Skip censored cases
        if events[i] == 0:
            continue
        
        for j in range(len(risk_scores)):
            if i == j:
                continue
            
            # Only consider pairs where patient i had event before patient j
            if times[i] < times[j]:
                permissible += 1
                # Concordant if higher risk predicts shorter survival
                if risk_scores[i] > risk_scores[j]:
                    concordant += 1.0
                elif risk_scores[i] == risk_scores[j]:
                    concordant += 0.5
    
    return concordant / permissible if permissible > 0 else 0.5


# ============================================================================
# Visualization
# ============================================================================

def plot_training_curves(history: dict, save_path: Optional[str] = None):
    """Plot training history (loss and C-index over epochs)."""
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4))
    
    # Loss curve
    ax1.plot(history['train_loss'], label='Train', linewidth=2)
    ax1.plot(history['val_loss'], label='Validation', linewidth=2)
    ax1.set_xlabel('Epoch')
    ax1.set_ylabel('Loss')
    ax1.set_title('Training and Validation Loss')
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    
    # C-index curve
    ax2.plot(history['train_ci'], label='Train', linewidth=2)
    ax2.plot(history['val_ci'], label='Validation', linewidth=2)
    ax2.set_xlabel('Epoch')
    ax2.set_ylabel('C-Index')
    ax2.set_title('Training and Validation C-Index')
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.close()
    else:
        plt.show()

# ----------------------------------------------------------------------------

def plot_risk_distribution(risk_scores: np.ndarray, events: np.ndarray, 
                          save_path: Optional[str] = None):
    """Plot risk score distributions for events vs censored cases."""
    fig, ax = plt.subplots(figsize=(8, 6))
    
    # Separate by event status
    event_risks = risk_scores[events == 1]
    censored_risks = risk_scores[events == 0]
    
    # Plot histograms
    ax.hist(event_risks, bins=30, alpha=0.6, label=f'Events (n={len(event_risks)})', 
            density=True, color='red')
    ax.hist(censored_risks, bins=30, alpha=0.6, label=f'Censored (n={len(censored_risks)})', 
            density=True, color='blue')
    
    ax.set_xlabel('Risk Score')
    ax.set_ylabel('Density')
    ax.set_title('Risk Score Distribution by Event Status')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.close()
    else:
        plt.show()


# ============================================================================
# Model Evaluation
# ============================================================================

def evaluate_model(model, data_loader, device: str = 'cpu'):
    """
    Evaluate model on dataset.
    
    Args:
        model: Trained model
        data_loader: DataLoader
        device: Device
    
    Returns:
        c_index, risk_scores, times, events
    """
    import torch
    
    model.eval()
    all_risks = []
    all_times = []
    all_events = []
    
    with torch.no_grad():
        for features, times, events in data_loader:
            features = features.to(device)
            risk_scores = model(features).squeeze()
            
            all_risks.append(risk_scores.cpu().numpy())
            all_times.append(times.numpy())
            all_events.append(events.numpy())
    
    all_risks = np.concatenate(all_risks)
    all_times = np.concatenate(all_times)
    all_events = np.concatenate(all_events)
    
    ci = concordance_index(all_risks, all_times, all_events)
    
    return ci, all_risks, all_times, all_events
