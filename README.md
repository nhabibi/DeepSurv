# DeepSurv: Cox Proportional Hazards Deep Neural Network

**PyTorch implementation** of **[DeepSurv](https://arxiv.org/abs/1606.00931)** for survival analysis research.

**Paper**: [Katzman et al., 2018](https://arxiv.org/abs/1606.00931) | [Original Code](https://github.com/jaredleekatzman/DeepSurv)

---

## 📂 Project Structure

```
DeepSurv/
├── README.md                      # This file
├── main.py                        # Training entry point
├── requirements.txt               # Dependencies
│
├── src/                           # Core implementation
│   ├── config.py                 # Hyperparameters
│   ├── model.py                  # DeepSurv architecture
│   ├── loss.py                   # Cox proportional hazards loss
│   ├── train.py                  # Training loop
│   ├── evaluation.py             # C-index evaluation
│   └── data_loader.py            # Data loading utilities
│
├── phase1_vanilla/                # Phase 1: Vanilla DeepSurv baseline
│   ├── README.md                 # Phase 1 overview
│   ├── docs/                     # Documentation
│   │   ├── VANILLA_BASELINE.md   # Technical implementation details
│   │   ├── PHASE1_SUMMARY.md     # Research summary for thesis
│   │   └── FINAL_RESULTS.md      # Validated results (C-index=0.7111)
│   └── scripts/                  # Testing scripts
│       ├── find_minimal_l2.py    # L2 regularization testing
│       └── save_synthetic_data.py # Data saving utility
│
├── phase2_seer/                   # Phase 2: SEER data application
│   ├── README.md                 # Phase 2 overview
│   ├── docs/                     # Documentation
│   │   ├── SEER_GUIDE.md         # SEER integration guide
│   │   └── SEER_QUESTIONS.md     # Data requirements
│   └── scripts/                  # SEER-specific scripts (to be added)
│
├── data/                          # Data directory
│   ├── synthetic/                # Synthetic survival data
│   └── seer/                     # SEER data (when available)
│
└── results/                       # Training outputs
    ├── checkpoints/              # Model weights (.pt files)
    ├── figures/                  # Training curves and plots
    └── logs/                     # Training metrics (.json)
```

---

## 🎯 Quick Start

### Installation
```bash
# Create virtual environment
python -m venv .venv
source .venv/bin/activate  # On Windows: .venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt
```

### Run Training
```bash
# Train on synthetic data
python main.py --data-type linear --n-samples 5000 --n-features 10

# Options
python main.py --help
```

---

## 🔍 What Does DeepSurv Predict?

**DeepSurv predicts RISK SCORE (log hazard ratio), NOT time or event!**

- **Input**: Patient features [feature_1, ..., feature_10]
- **Output**: Single risk score (e.g., 0.85, -0.30)
- **Interpretation**: 
  - Higher score → Higher risk of death (sooner)
  - Lower score → Lower risk of death (later)
- **Training**: Uses (time, event) to learn relative risk ranking
- **Evaluation**: C-index measures ranking accuracy (0.71 = 71% pairs correctly ordered)

**NOT predicted**: Exact survival time, probability of death  
**Purpose**: Rank patients by relative risk for clinical decisions

---

## 📚 Documentation

### Phase 1: Vanilla Baseline (✅ Complete)
- **[Overview](phase1_vanilla/README.md)** - Phase 1 summary
- **[Results](phase1_vanilla/docs/FINAL_RESULTS.md)** - C-index=0.7111 on synthetic data
- **[Technical Details](phase1_vanilla/docs/VANILLA_BASELINE.md)** - Implementation specifics
- **[Research Summary](phase1_vanilla/docs/PHASE1_SUMMARY.md)** - For thesis writing

**Key Achievement**: Validated vanilla DeepSurv baseline with 2 framework-required adaptations (LR×10, L2÷1000)

### Phase 2: SEER Data (🔄 Ready)
- **[Overview](phase2_seer/README.md)** - Phase 2 plan
- **[SEER Guide](phase2_seer/docs/PHASE2_SEER_GUIDE.md)** - Integration steps
- **[Data Questions](phase2_seer/docs/SEER_QUESTIONS.md)** - What data is needed

**Next Step**: Answer SEER questions to begin Phase 2

---

## 🎓 Research Summary

### Objective
Establish valid vanilla DeepSurv baseline in PyTorch for survival analysis research.

### Approach
- Implement vanilla architecture: [25, 25] ReLU network
- Cox proportional hazards loss (Efron approximation)
- Minimal framework-required adaptations
- Extensive empirical validation

### Results
- **Best C-Index**: 0.7111 on synthetic linear data
- **Framework Adaptations**: 2 hyperparameters (LR, L2)
- **Validation**: Multiple test runs confirming reproducibility

### For Thesis
> "We established a vanilla DeepSurv baseline with the original architecture from Katzman et al. (2018). Due to mathematical differences between Theano/Lasagne and PyTorch regularization, we made two empirically-validated framework adjustments. This provides a valid baseline for comparing comorbidity-aware survival models."

---

## 🔧 Configuration

Key hyperparameters in `src/config.py`:

| Parameter | Value | Notes |
|-----------|-------|-------|
| Architecture | [25, 25] | Vanilla hidden layers |
| Activation | ReLU | No changes |
| Learning Rate | 1e-3 | Adapted (Theano→PyTorch) |
| L2 Regularization | 0.01 | Adapted (weight_decay semantics) |
| Optimizer | SGD + Nesterov | Vanilla (momentum=0.9) |
| Batch Size | 64 | Vanilla |
| Early Stopping | 50 epochs | Vanilla |

See [VANILLA_BASELINE.md](phase1_vanilla/docs/VANILLA_BASELINE.md) for complete details.

---

## 📊 Results

### Phase 1: Synthetic Data
- **Dataset**: 5000 samples, 10 features, linear hazard
- **Best C-Index**: 0.7111
- **Training**: 164 epochs (early stopped)
- **Device**: Apple M1 (MPS)

### Validation
- ✅ Model learns successfully
- ✅ Achieves C-index > 0.70
- ✅ Reproducible across runs
- ✅ Framework adaptations validated

---

## 🚀 Next Steps

1. **Phase 2**: Apply to SEER cancer survival data
2. **Phase 3**: Add comorbidity features
3. **Phase 4**: Novel architectures and comparisons

---

## 📖 References

- Katzman, J. L., et al. (2018). "DeepSurv: personalized treatment recommender system using a Cox proportional hazards deep neural network." *BMC Medical Research Methodology*, 18(1), 24.
- Original implementation: https://github.com/jaredleekatzman/DeepSurv

---

## 📝 License

This is research code for PhD thesis work. See institution policies for usage rights.

---

**Status**: Phase 1 ✅ Complete | Phase 2 🔄 Ready to Start
