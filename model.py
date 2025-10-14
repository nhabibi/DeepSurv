"""
DeepSurv: Cox Proportional Hazards Deep Neural Network
Based on: https://arxiv.org/abs/1606.00931
"""

import torch
import torch.nn as nn
from typing import List


# ============================================================================
# DeepSurv Model
# ============================================================================

class DeepSurv(nn.Module):
    """
    DeepSurv model for survival analysis.
    
    Args:
        input_dim: Number of input features
        hidden_layers: List of hidden layer sizes
        activation: Activation function
        dropout: Dropout probability
        use_batch_norm: Whether to use batch normalization
    """
    
    def __init__(
        self,
        input_dim: int,
        hidden_layers: List[int] = [64, 32, 16],
        activation: str = 'relu',
        dropout: float = 0.3,
        use_batch_norm: bool = True
    ):
        super(DeepSurv, self).__init__()
        # --------------------------------------------------------------------
        # Configuration
        # --------------------------------------------------------------------
        self.input_dim = input_dim
        self.hidden_layers = hidden_layers
        self.use_batch_norm = use_batch_norm
        
        # --------------------------------------------------------------------
        # Activation function
        # --------------------------------------------------------------------
        activations = {
            'relu': nn.ReLU(),
            'selu': nn.SELU(),
            'tanh': nn.Tanh()
        }
        if activation not in activations:
            raise ValueError(f"Unknown activation: {activation}")
        self.activation = activations[activation]
        
        # --------------------------------------------------------------------
        # Build network
        # --------------------------------------------------------------------
        layers = []
        layer_sizes = [input_dim] + hidden_layers
        
        for i in range(len(hidden_layers)):
            # Linear layer
            layers.append(nn.Linear(layer_sizes[i], layer_sizes[i+1]))
            
            # Batch normalization (optional)
            if use_batch_norm:
                layers.append(nn.BatchNorm1d(layer_sizes[i+1]))
            
            # Activation
            layers.append(self.activation)
            
            # Dropout (optional)
            if dropout > 0:
                layers.append(nn.Dropout(dropout))
        
        # --------------------------------------------------------------------
        # Output layer
        # --------------------------------------------------------------------
        layers.append(nn.Linear(hidden_layers[-1], 1))
        
        self.network = nn.Sequential(*layers)
    
    # ------------------------------------------------------------------------
    # Forward Pass
    # ------------------------------------------------------------------------
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass."""
        return self.network(x)
    
    # ------------------------------------------------------------------------
    # Prediction
    # ------------------------------------------------------------------------
    
    def predict_risk(self, x: torch.Tensor) -> torch.Tensor:
        """Predict risk scores."""
        self.eval()
        with torch.no_grad():
            return self.forward(x).squeeze()
