# DeepSurv: Cox Proportional Hazards Deep Neural Network

PyTorch implementation of **[DeepSurv](https://arxiv.org/abs/1606.00931)** for survival analysis with comorbidity extension.

**Paper**: [Katzman et al., 2018](https://arxiv.org/abs/1606.00931) | [Original Code](https://github.com/jaredleekatzman/DeepSurv)

---

## 🎯 3-Phase Roadmap

### **Phase 1: Learn** 📚
Implement and validate DeepSurv on benchmark datasets
- Train on simulated + METABRIC
- Verify C-index matches paper (±0.03)
- **Time**: 1-2 weeks

### **Phase 2: Extend** 🔬
Apply to SEER with comorbidity features
- Use METABRIC hyperparameters (same domain)
- Add comorbidity features
- Establish baseline
- **Time**: 2-3 months

### **Phase 3: Improve** 🎯
Novel architecture + hyperparameter tuning
- Multi-task learning for dual cancers
- Competing risks modeling
- Systematic hyperparameter search
- **Time**: 3-6 months

---

## 📊 Phase 1: Essential Datasets

| Dataset | Type | Size | Expected C-Index |
|---------|------|------|------------------|
| Simulated (Linear) | Synthetic | 5K | ~0.80 |
| Simulated (Gaussian) | Synthetic | 5K | ~0.75 |
| METABRIC | Cancer | 2K | ~0.65-0.68 |

**Note**: Other datasets (WHAS, SUPPORT, GBSG, Rotterdam) are optional. See `data/README.md` for details.

---

## 🚀 Quick Start

```bash
# 1. Install
pip install torch numpy pandas scikit-learn matplotlib scipy tqdm

# 2. Train on simulated data
python main.py --data data/simulated/linear.csv

# 3. Train on METABRIC
python main.py --data data/metabric/metabric.csv

# ✅ Phase 1 complete when all C-indices match (±0.03)
```

---

## 📁 Structure

```
DeepSurv/
├── config.py          # Hyperparameters
├── model.py           # Network architecture
├── loss.py            # Cox PH loss
├── data_loader.py     # Data preprocessing
├── train.py           # Training loop
├── evaluation.py      # C-index & visualization
├── main.py            # Entry point
└── data/              # Datasets + README
```

---

## 🎓 Research Goal

PhD thesis on survival analysis with **comorbidity** (multiple cancers) using SEER data.

---

## 📚 Citation

```bibtex
@article{katzman2018deepsurv,
  title={DeepSurv: personalized treatment recommender system using a Cox proportional hazards deep neural network},
  author={Katzman, Jared L and Shaham, Uri and Cloninger, Alexander and Bates, Jonathan and Jiang, Tingting and Kluger, Yuval},
  journal={BMC medical research methodology},
  volume={18},
  pages={24},
  year={2018}
}
```
features, times, events, _ = load_data('data/whas/whas.csv')

# Prepare loaders
train_loader, val_loader = prepare_data_loaders(features, times, events)

# Create and train model
model = DeepSurv(input_dim=features.shape[1])
trainer = Trainer(model)
history = trainer.fit(train_loader, val_loader, num_epochs=100)

# Evaluate
print(f"C-Index: {history['val_ci'][-1]:.4f}")
```

---
