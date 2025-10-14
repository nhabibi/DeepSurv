"""
Training utilities for DeepSurv.
"""

import torch
import torch.optim as optim
import numpy as np
from typing import Dict, Optional, Tuple
import os
from tqdm import tqdm

from model import DeepSurv
from loss import CoxPHLoss
from evaluation import concordance_index


# ============================================================================
# Trainer Class
# ============================================================================

class Trainer:
    """
    Trainer class for DeepSurv.
    
    Args:
        model: DeepSurv model
        device: Device ('cpu' or 'cuda')
        learning_rate: Learning rate
        l2_reg: L2 regularization
        optimizer_name: Optimizer ('adam' or 'sgd')
        loss_method: Loss method ('breslow' or 'efron')
    """
    
    def __init__(
        self,
        model: DeepSurv,
        device: str = 'cpu',
        learning_rate: float = 0.001,
        l2_reg: float = 0.001,
        optimizer_name: str = 'adam',
        loss_method: str = 'efron'
    ):
        self.model = model.to(device)
        self.device = device
        self.l2_reg = l2_reg
        
        # --------------------------------------------------------------------
        # Loss
        # --------------------------------------------------------------------
        self.criterion = CoxPHLoss(method=loss_method)
        
        # --------------------------------------------------------------------
        # Optimizer
        # --------------------------------------------------------------------
        if optimizer_name == 'adam':
            self.optimizer = optim.Adam(
                model.parameters(),
                lr=learning_rate,
                weight_decay=l2_reg
            )
        elif optimizer_name == 'sgd':
            self.optimizer = optim.SGD(
                model.parameters(),
                lr=learning_rate,
                momentum=0.9,
                weight_decay=l2_reg
            )
        else:
            raise ValueError(f"Unknown optimizer: {optimizer_name}")
        
        # --------------------------------------------------------------------
        # History
        # --------------------------------------------------------------------
        self.history = {
            'train_loss': [],
            'val_loss': [],
            'train_ci': [],
            'val_ci': []
        }
    
    # ------------------------------------------------------------------------
    # Training Loop
    # ------------------------------------------------------------------------
    
    def train_epoch(self, train_loader) -> Tuple[float, float]:
        """Train model for one epoch and return average loss and C-index."""
        self.model.train()
        total_loss = 0.0
        all_risks, all_times, all_events = [], [], []
        
        for features, times, events in train_loader:
            # Move to device
            features = features.to(self.device)
            times = times.to(self.device)
            events = events.to(self.device)
            
            # Forward pass
            risk_scores = self.model(features).squeeze()
            loss = self.criterion(risk_scores, times, events)
            
            # Backward
            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()
            
            # Collect predictions for C-index
            total_loss += loss.item() * len(features)
            all_risks.append(risk_scores.detach().cpu().numpy())
            all_times.append(times.cpu().numpy())
            all_events.append(events.cpu().numpy())
        
        # Compute metrics
        avg_loss = total_loss / len(train_loader.dataset)
        all_risks = np.concatenate(all_risks)
        all_times = np.concatenate(all_times)
        all_events = np.concatenate(all_events)
        ci = concordance_index(all_risks, all_times, all_events)
        
        return avg_loss, ci
    
    # ------------------------------------------------------------------------
    # Validation
    # ------------------------------------------------------------------------
    
    def validate(self, val_loader) -> Tuple[float, float]:
        """Evaluate model on validation set."""
        self.model.eval()
        total_loss = 0.0
        all_risks, all_times, all_events = [], [], []
        all_events = []
        
        with torch.no_grad():
            for features, times, events in val_loader:
                # Move to device
                features = features.to(self.device)
                times = times.to(self.device)
                events = events.to(self.device)
                
                # Forward pass only
                risk_scores = self.model(features).squeeze()
                loss = self.criterion(risk_scores, times, events)
                
                # Collect predictions
                total_loss += loss.item() * len(features)
                all_risks.append(risk_scores.cpu().numpy())
                all_times.append(times.cpu().numpy())
                all_events.append(events.cpu().numpy())
        
        # Compute metrics
        avg_loss = total_loss / len(val_loader.dataset)
        all_risks = np.concatenate(all_risks)
        all_times = np.concatenate(all_times)
        all_events = np.concatenate(all_events)
        ci = concordance_index(all_risks, all_times, all_events)
        
        return avg_loss, ci
    
    # ------------------------------------------------------------------------
    # Full Training Pipeline
    # ------------------------------------------------------------------------
    
    def fit(
        self,
        train_loader,
        val_loader,
        num_epochs: int = 100,
        early_stopping_patience: int = 10,
        save_path: Optional[str] = None,
        verbose: bool = True
    ) -> Dict:
        """
        Train model with early stopping and model checkpointing.
        
        Args:
            train_loader: Training DataLoader
            val_loader: Validation DataLoader
            num_epochs: Maximum training epochs
            early_stopping_patience: Stop if no improvement for N epochs
            save_path: Path to save best model checkpoint
            verbose: Show progress bar
        
        Returns:
            Training history dict with losses and C-indices
        """
        best_val_loss = float('inf')
        patience_counter = 0
        
        # Setup progress bar
        iterator = tqdm(range(num_epochs), desc="Training") if verbose else range(num_epochs)
        
        for epoch in iterator:
            # Train and validate
            train_loss, train_ci = self.train_epoch(train_loader)
            val_loss, val_ci = self.validate(val_loader)
            
            # Save history
            self.history['train_loss'].append(train_loss)
            self.history['val_loss'].append(val_loss)
            self.history['train_ci'].append(train_ci)
            self.history['val_ci'].append(val_ci)
            
            # Update progress bar
            if verbose:
                iterator.set_postfix({
                    'train_loss': f'{train_loss:.4f}',
                    'val_loss': f'{val_loss:.4f}',
                    'train_ci': f'{train_ci:.4f}',
                    'val_ci': f'{val_ci:.4f}'
                })
            
            # Save best model and early stopping
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                patience_counter = 0
                
                # Save checkpoint
                if save_path:
                    os.makedirs(os.path.dirname(save_path), exist_ok=True)
                    torch.save({
                        'epoch': epoch,
                        'model_state_dict': self.model.state_dict(),
                        'optimizer_state_dict': self.optimizer.state_dict(),
                        'val_loss': val_loss,
                        'val_ci': val_ci,
                    }, save_path)
            else:
                patience_counter += 1
                if patience_counter >= early_stopping_patience:
                    if verbose:
                        print(f"\nEarly stopping at epoch {epoch + 1}")
                    break
        
        return self.history
