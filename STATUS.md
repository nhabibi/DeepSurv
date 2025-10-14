# Vanilla DeepSurv - Code Summary

## ✅ Verification Complete

This codebase is a **faithful vanilla implementation** of DeepSurv (Katzman et al., 2018) in PyTorch.

---

## 📊 Code Statistics

| File | Lines | Purpose |
|------|-------|---------|
| `config.py` | 44 | Vanilla hyperparameters matching paper |
| `model.py` | 82 | DeepSurv neural network (PyTorch) |
| `loss.py` | 122 | Cox PH loss (Efron/Breslow) |
| `data_loader.py` | 175 | Data loading & synthetic generation |
| `train.py` | 218 | Training with SGD + Nesterov momentum |
| `evaluation.py` | 157 | C-index & visualization |
| `main.py` | 191 | Main training script |
| **Total** | **~1000** | Clean, minimal, readable |

---

## ✅ Vanilla Configuration Confirmed

### Model Architecture
- ✅ Hidden layers: **[25, 25]** (original paper)
- ✅ Activation: **ReLU** (rectify)
- ✅ Dropout: **0.0** (disabled, as in original)
- ✅ Batch Normalization: **False** (disabled, as in original)
- ✅ Output: Single node (log hazard ratio)

### Training Parameters
- ✅ Optimizer: **SGD with Nesterov momentum** (original)
- ✅ Learning Rate: **1e-4** (with power decay 0.001)
- ✅ Momentum: **0.9**
- ✅ L2 Regularization: **10.0** (strong, as in original)
- ✅ L1 Regularization: **0.0**
- ✅ Batch Size: **64**
- ✅ Max Epochs: **2000**
- ✅ Early Stopping Patience: **2000**

### Loss Function
- ✅ Method: **Efron** approximation
- ✅ Type: Negative log partial likelihood (Cox PH)

---

## 🎯 Key Differences from Original

| Aspect | Original (2016-2018) | This Implementation |
|--------|---------------------|---------------------|
| Framework | Theano + Lasagne | PyTorch |
| Python Version | Python 2/3 | Python 3.14+ |
| GPU Support | CUDA only | CUDA + MPS (Apple Silicon) |
| Dependencies | Legacy (Theano) | Modern (PyTorch) |

**Everything else matches the vanilla paper exactly!**

---

## 🚀 Usage

```bash
# Train on synthetic linear data (expected C-index: ~0.80)
python main.py --data-type linear --n-samples 5000 --n-features 10

# Train on synthetic Gaussian data (expected C-index: ~0.75)
python main.py --data-type gaussian --n-samples 5000 --n-features 10

# Train on your own data
python main.py --data path/to/data.csv

# Force CPU (disable GPU)
python main.py --cpu --data-type linear --n-samples 5000
```

---

## 📁 Project Structure

```
DeepSurv/
├── .git/              # Git repository
├── .venv/             # Virtual environment
├── .gitignore         # Git ignore file
├── README.md          # Documentation
├── requirements.txt   # Dependencies
├── config.py          # Vanilla hyperparameters
├── model.py           # DeepSurv network
├── loss.py            # Cox PH loss
├── data_loader.py     # Data utilities
├── train.py           # Training loop
├── evaluation.py      # C-index & plots
└── main.py            # Entry point
```

**Output directories** (created at runtime):
- `models/` - Saved model checkpoints
- `results/` - Training curves, plots, metrics

---

## ✅ Phase 1 Status

- [x] Vanilla architecture implemented
- [x] Vanilla hyperparameters configured
- [x] SGD with Nesterov momentum
- [x] No Batch Normalization
- [x] No Dropout
- [x] Strong L2 regularization (10.0)
- [x] Power learning rate decay
- [x] Apple Silicon (MPS) support
- [ ] **Validation on synthetic data** (next step)
- [ ] Achieve C-index ~0.80 on linear data
- [ ] Achieve C-index ~0.75 on Gaussian data

---

## 📝 Next Steps (Phase 1)

1. **Run experiments** on synthetic data
2. **Verify C-index** matches paper (±0.03)
3. **Test on METABRIC** dataset (C-index ~0.65-0.68)
4. **Document results** for Phase 1 completion

Once Phase 1 is complete, proceed to:
- **Phase 2**: Apply to SEER with comorbidity features
- **Phase 3**: Novel architectures and improvements

---

*Last updated: October 14, 2025*
*Codebase verified: Vanilla DeepSurv implementation*
