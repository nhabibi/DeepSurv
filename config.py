"""
Vanilla DeepSurv Configuration
All hyperparameters match the original paper (Katzman et al., 2018)
"""

# Model Architecture (Vanilla)
MODEL_CONFIG = {
    'input_dim': None,              # Set automatically from data
    'hidden_layers': [25, 25],      # Original: [25, 25]
    'activation': 'relu',           # Original: rectify (ReLU)
    'dropout': 0.0,                 # Original: None (disabled)
    'use_batch_norm': False,        # Original: False (disabled)
}

# Training Parameters (Vanilla)
TRAINING_CONFIG = {
    'learning_rate': 1e-4,          # Original: 1e-4 or 1e-5
    'lr_decay': 0.001,              # Original: 0.001 (power decay)
    'momentum': 0.9,                # Original: 0.9 (Nesterov)
    'optimizer': 'sgd',             # Original: SGD with Nesterov momentum
    'l2_reg': 10.0,                 # Original: 10.0
    'l1_reg': 0.0,                  # Original: 0.0
    'batch_size': 64,               # Reasonable default
    'num_epochs': 2000,             # Original: 500-2000
    'early_stopping_patience': 2000,# High patience for convergence
    'validation_split': 0.2,        # Standard 80/20 split
}

# Data Parameters
DATA_CONFIG = {
    'normalize': True,              # Standardize features
    'random_seed': 42,
}

# Loss Function
LOSS_CONFIG = {
    'method': 'efron',              # Efron approximation for ties
}

# Output Directories
PATHS = {
    'model_dir': 'models/',
    'results_dir': 'results/',
}
