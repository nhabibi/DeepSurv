# DeepSurv: Cox Proportional Hazards Deep Neural Network

**Vanilla PyTorch implementation** of **[DeepSurv](https://arxiv.org/abs/1606.00931)** for survival analysis.

**Paper**: [Katzman et al., 2018](https://arxiv.org/abs/1606.00931) | [Original Code (Theano/Lasagne)](https://github.com/jaredleekatzman/DeepSurv)

This is a faithful PyTorch reimplementation of the original DeepSurv architecture and hyperparameters.

---

## 🎯 3-Phase Roadmap

### **Phase 1: Learn** 📚 (Current)
Implement and validate vanilla DeepSurv on benchmark datasets
- ✅ Vanilla architecture: [25, 25] hidden layers
- ✅ SGD with Nesterov momentum (no Adam)
- ✅ No Batch Normalization, No Dropout
- ✅ Strong L2 regularization (10.0)
- ✅ Power learning rate decay
- 🔄 Train on simulated data
- ⏳ Verify C-index matches paper (±0.03)
- **Time**: 1-2 weeks

### **Phase 2: Extend** 🔬
Apply to SEER with comorbidity features
- Use validated hyperparameters
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

## 📊 Phase 1: Expected Benchmarks

| Dataset | Type | Size | Expected C-Index |
|---------|------|------|------------------|
| Simulated (Linear) | Synthetic | 5K | ~0.80 |
| Simulated (Gaussian) | Synthetic | 5K | ~0.75 |
| METABRIC | Cancer | 2K | ~0.65-0.68 |

**Phase 1 Goal**: Achieve C-indices within ±0.03 of paper results

---

## 🚀 Quick Start

### 1. Install Dependencies

```bash
pip install torch numpy pandas scikit-learn matplotlib scipy tqdm
```

Or using requirements.txt:
```bash
pip install -r requirements.txt
```

### 2. Train on Synthetic Data

```bash
# Linear synthetic data (should achieve C-index ~0.80)
python main.py --data-type linear --n-samples 5000 --n-features 10

# Gaussian synthetic data (should achieve C-index ~0.75)
python main.py --data-type gaussian --n-samples 5000 --n-features 10
```

### 3. Train on Your Own Data

```bash
python main.py --data path/to/your/data.csv
```

Your CSV should have columns: features, `time`, `event`

---

## 📁 Project Structure

```
DeepSurv/
├── config.py          # Vanilla hyperparameters from paper
├── model.py           # DeepSurv neural network (PyTorch)
├── loss.py            # Cox PH loss (Efron/Breslow)
├── data_loader.py     # Data preprocessing & synthetic generation
├── train.py           # Training loop with SGD+Nesterov
├── evaluation.py      # C-index & visualization
├── main.py            # Entry point
└── requirements.txt   # Dependencies
```

---

## ⚙️ Vanilla Configuration (Phase 1)

### Model Architecture
- **Hidden Layers**: [25, 25]
- **Activation**: ReLU (rectify)
- **Dropout**: None (disabled)
- **Batch Normalization**: None (disabled)
- **Standardization**: Enabled

### Training Parameters
- **Optimizer**: SGD with Nesterov momentum
- **Learning Rate**: 1e-4 (with power decay)
- **Momentum**: 0.9
- **L2 Regularization**: 10.0
- **L1 Regularization**: 0.0
- **Batch Size**: 64
- **Max Epochs**: 2000
- **Early Stopping**: Patience of 2000

### Loss Function
- **Method**: Efron (for tied event times)
- **Type**: Negative log partial likelihood (Cox PH)

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

---

## 🎓 Research Goal

PhD thesis on survival analysis with **comorbidity** (multiple cancers) using SEER data.

**Current Status**: Phase 1 - Validating vanilla implementation

---

## 📝 Notes

- This implementation uses **PyTorch** instead of the original **Theano/Lasagne**
- All hyperparameters match the original paper's defaults
- No modifications or improvements yet (that comes in Phase 3)
- Apple Silicon (M1/M2/M3) GPU support via MPS backend
