"""
Configuration file for DeepSurv model.
All hyperparameters centralized here for easy experimentation.
"""

# ============================================================================
# Model Architecture
# ============================================================================
MODEL_CONFIG = {
    'input_dim': None,  # Set automatically based on data
    'hidden_layers': [64, 32, 16],
    'activation': 'relu',  # 'relu', 'selu', 'tanh'
    'dropout': 0.3,
    'use_batch_norm': True,
}

# ============================================================================
# Training Parameters
# ============================================================================
TRAINING_CONFIG = {
    'learning_rate': 0.001,
    'batch_size': 64,
    'num_epochs': 100,
    'optimizer': 'adam',  # 'adam', 'sgd'
    'l2_reg': 0.001,
    'early_stopping_patience': 10,
    'validation_split': 0.2,
}

# ============================================================================
# Data Parameters
# ============================================================================
DATA_CONFIG = {
    'normalize': True,
    'random_seed': 42,
}

# ============================================================================
# Loss Function
# ============================================================================
LOSS_CONFIG = {
    'method': 'efron',  # 'efron' or 'breslow'
}

# ============================================================================
# Paths
# ============================================================================
PATHS = {
    'data_dir': 'data/',
    'model_dir': 'models/',
    'results_dir': 'results/',
}
