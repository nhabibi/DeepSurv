"""
Training utilities for DeepSurv
"""

import torch
import torch.optim as optim
import numpy as np
from typing import Dict, Tuple
import os
from tqdm import tqdm

from loss import CoxPHLoss
from evaluation import concordance_index


class Trainer:
    """
    Trainer for vanilla DeepSurv with SGD + Nesterov momentum.
    
    All hyperparameters must be provided explicitly from config.py.
    No defaults to ensure all values come from centralized configuration.
    
    Args:
        model: DeepSurv model
        device: Device ('cpu', 'cuda', or 'mps')
        learning_rate: Initial learning rate (from config)
        lr_decay: Learning rate power decay coefficient (from config)
        momentum: Momentum coefficient for SGD (from config)
        optimizer_name: 'sgd' or 'adam' (from config)
        l2_reg: L2 regularization weight decay (from config)
        l1_reg: L1 regularization (from config, currently unused)
        loss_method: 'efron' or 'breslow' (from config)
    """
    
    def __init__(
        self,
        model,
        device: str,
        learning_rate: float,
        lr_decay: float,
        momentum: float,
        optimizer_name: str,
        l2_reg: float,
        l1_reg: float,
        loss_method: str
    ):
        self.model = model.to(device)
        self.device = device
        self.initial_lr = learning_rate
        self.lr_decay = lr_decay
        
        # Loss function
        self.criterion = CoxPHLoss(method=loss_method)
        
        # Optimizer (vanilla uses SGD with Nesterov momentum)
        if optimizer_name == 'sgd':
            self.optimizer = optim.SGD(
                model.parameters(),
                lr=learning_rate,
                momentum=momentum,
                nesterov=True,
                weight_decay=l2_reg
            )
        elif optimizer_name == 'adam':
            self.optimizer = optim.Adam(
                model.parameters(),
                lr=learning_rate,
                weight_decay=l2_reg
            )
        else:
            raise ValueError(f"Unknown optimizer: {optimizer_name}")
        
        # Training history
        self.history = {
            'train_loss': [],
            'val_loss': [],
            'train_ci': [],
            'val_ci': []
        }
    
    def train_epoch(self, train_loader) -> Tuple[float, float]:
        """Train for one epoch."""
        self.model.train()
        total_loss = 0.0
        all_risks, all_times, all_events = [], [], []
        
        for features, times, events in train_loader:
            features = features.to(self.device)
            times = times.to(self.device)
            events = events.to(self.device)
            
            # Forward pass
            risk_scores = self.model(features).squeeze()
            loss = self.criterion(risk_scores, times, events)
            
            # Backward pass
            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()
            
            # Collect for metrics
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
    
    def validate(self, val_loader) -> Tuple[float, float]:
        """Validate on validation set."""
        self.model.eval()
        total_loss = 0.0
        all_risks, all_times, all_events = [], [], []
        
        with torch.no_grad():
            for features, times, events in val_loader:
                features = features.to(self.device)
                times = times.to(self.device)
                events = events.to(self.device)
                
                # Forward pass only
                risk_scores = self.model(features).squeeze()
                loss = self.criterion(risk_scores, times, events)
                
                # Collect for metrics
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
    
    def fit(
        self,
        train_loader,
        val_loader,
        num_epochs: int = 500,
        early_stopping_patience: int = 50,
        save_path: str = None,
        verbose: bool = True
    ) -> Dict:
        """
        Train model with early stopping.
        
        Args:
            train_loader: Training DataLoader
            val_loader: Validation DataLoader
            num_epochs: Maximum epochs
            early_stopping_patience: Early stopping patience
            save_path: Path to save best model
            verbose: Show progress bar
            
        Returns:
            Training history dict
        """
        best_val_loss = float('inf')
        patience_counter = 0
        
        iterator = tqdm(range(num_epochs), desc="Training") if verbose else range(num_epochs)
        
        for epoch in iterator:
            # Power learning rate decay (vanilla DeepSurv)
            if self.lr_decay > 0:
                current_lr = self.initial_lr / (1 + epoch * self.lr_decay)
                for param_group in self.optimizer.param_groups:
                    param_group['lr'] = current_lr
            
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
            
            # Save best model
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                patience_counter = 0
                
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
