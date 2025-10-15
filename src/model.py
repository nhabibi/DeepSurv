"""
DeepSurv: Cox Proportional Hazards Deep Neural Network
Vanilla implementation based on: https://arxiv.org/abs/1606.00931
"""

import torch
import torch.nn as nn
from typing import List


class DeepSurv(nn.Module):
    """
    Vanilla DeepSurv model for survival analysis.
    
    Architecture:
    - Fully connected neural network
    - Optional batch normalization and dropout (disabled in vanilla)
    - Single output node (log hazard ratio)
    
    Args:
        input_dim: Number of input features
        hidden_layers: List of hidden layer sizes [default: [25, 25]]
        activation: Activation function [default: 'relu']
        dropout: Dropout rate [default: 0.0 (disabled)]
        use_batch_norm: Use batch normalization [default: False]
    """
    
    def __init__(
        self,
        input_dim: int,
        hidden_layers: List[int] = [25, 25],
        activation: str = 'relu',
        dropout: float = 0.0,
        use_batch_norm: bool = False
    ):
        super(DeepSurv, self).__init__()
        
        # Activation function
        activations = {
            'relu': nn.ReLU(),
            'selu': nn.SELU(),
            'tanh': nn.Tanh()
        }
        if activation not in activations:
            raise ValueError(f"Unknown activation: {activation}")
        activation_fn = activations[activation]
        
        # Build network layers
        layers = []
        layer_sizes = [input_dim] + hidden_layers
        
        for i in range(len(hidden_layers)):
            # Linear layer
            layers.append(nn.Linear(layer_sizes[i], layer_sizes[i+1]))
            
            # Optional batch normalization (vanilla: False)
            if use_batch_norm:
                layers.append(nn.BatchNorm1d(layer_sizes[i+1]))
            
            # Activation
            layers.append(activation_fn)
            
            # Optional dropout (vanilla: 0.0)
            if dropout > 0:
                layers.append(nn.Dropout(dropout))
        
        # Output layer: single node for log hazard ratio
        layers.append(nn.Linear(hidden_layers[-1], 1))
        
        self.network = nn.Sequential(*layers)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass: compute log hazard ratio.
        
        Args:
            x: Input features [batch_size, input_dim]
            
        Returns:
            Log hazard ratio [batch_size, 1]
        """
        return self.network(x)
