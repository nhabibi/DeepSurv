"""
Quick test to verify synthetic data is learnable
"""
import numpy as np
import sys
sys.path.append('.')
from evaluation import concordance_index

# Generate synthetic data
np.random.seed(42)
n_samples = 5000
n_features = 10

features = np.random.randn(n_samples, n_features)
weights = np.array([1.0, -0.8, 0.6, -0.4, 0.3, -0.2, 0.15, -0.1, 0.05, -0.03])[:n_features]
log_hazard = np.dot(features, weights)
hazard = np.exp(log_hazard)

# Normalize hazard
hazard = hazard / np.mean(hazard)
survival_times = np.random.exponential(1.0 / (hazard + 1e-8))
censoring_times = np.random.exponential(2.0, size=n_samples)

times = np.minimum(survival_times, censoring_times)
events = (survival_times <= censoring_times).astype(float)

# Check if TRUE log hazard achieves high C-index
c_index = concordance_index(log_hazard, times, events)
print(f"✅ C-index with TRUE log hazard: {c_index:.4f}")
print(f"Event rate: {events.mean():.2%}")
print()

if c_index > 0.70:
    print("✅ Data is LEARNABLE - synthetic data generation is correct!")
    print("   The issue is likely with model training (hyperparameters or loss).")
else:
    print("❌ Data has WEAK SIGNAL - data generation might need adjustment.")
    print("   Even with perfect knowledge, C-index is low.")
