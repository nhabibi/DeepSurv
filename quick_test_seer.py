"""Quick test on SEER data (500 samples, 10 epochs)"""

import sys
sys.path.insert(0, 'src')

import torch
from src.config import MODEL_CONFIG, TRAINING_CONFIG
from src.model import DeepSurv
from src.data_loader import load_seer_data, prepare_data_loaders
from src.train import Trainer

# Load small dataset
print("Loading small SEER test data (500 samples)...")
features, times, events, _, _ = load_seer_data(
    'data/seer/train_seer_test_500.csv',
    normalize=True
)

# Create data loaders
train_loader, val_loader = prepare_data_loaders(
    features, times, events,
    batch_size=32,  # Smaller batch for quick test
    validation_split=0.2,
    random_seed=42
)

# Build model
MODEL_CONFIG['input_dim'] = features.shape[1]
model = DeepSurv(**MODEL_CONFIG)

print(f"\nModel: {MODEL_CONFIG['hidden_layers']}")
print(f"Input features: {features.shape[1]}")
print(f"Parameters: {sum(p.numel() for p in model.parameters()):,}")

# Train for 10 epochs
device = 'mps' if torch.backends.mps.is_available() else 'cpu'
print(f"\nDevice: {device}")
print("\nStarting quick training (10 epochs)...\n")

trainer = Trainer(
    model=model,
    device=device,
    learning_rate=TRAINING_CONFIG['learning_rate'],
    lr_decay=TRAINING_CONFIG['lr_decay'],
    momentum=TRAINING_CONFIG['momentum'],
    optimizer_name=TRAINING_CONFIG['optimizer'],
    l2_reg=TRAINING_CONFIG['l2_reg'],
    l1_reg=TRAINING_CONFIG['l1_reg'],
    loss_method='efron'
)

history = trainer.fit(
    train_loader, val_loader,
    num_epochs=10,  # Just 10 epochs for quick test
    early_stopping_patience=50,
    save_path=None,  # Don't save
    verbose=True
)

print("\n✅ Quick test complete!")
print(f"Final validation C-index: {history['val_ci'][-1]:.4f}")
