# DeepSurv: Cox Proportional Hazards Deep Neural Network

**PyTorch implementation** of **[DeepSurv](https://arxiv.org/abs/1606.00931)** for survival analysis research.

**Paper**: [Katzman et al., 2018](https://arxiv.org/abs/1606.00931) | [Original Code](https://github.com/jaredleekatzman/DeepSurv)

---

## 🗺️ Research Roadmap

**Phase 1: Vanilla Baseline** (✅ Complete)
- Implement vanilla DeepSurv [25, 25] architecture
- Validate on synthetic linear data
- **Result**: C-index = 0.7111

**Phase 2: SEER Application** (🔄 In Progress)
- Apply to realistic cancer registry data (breast + vaginal)
- Minimal changes to baseline
- **Goal**: Establish performance on clinical data

**Phase 3: Comorbidity Analysis** (📋 Planned)
- Identify patients with dual-cancer comorbidity
- Analyze survival patterns
- **Goal**: Understand comorbidity effects

**Phase 4: Comorbidity-Aware Architectures** (🔮 Future)
- Multi-input networks (separate pathways per cancer)
- Attention mechanisms (cancer interaction modeling)
- Cross-stitch networks (shared + cancer-specific features)
- **Goal**: Improve prediction for comorbid cases

**Phase 5: Advanced Methods** (🔮 Optional)
- DeepHit: Time prediction via discrete distributions
- Random Survival Forests: Non-parametric baselines
- Transformers: Attention-based survival modeling
- **Goal**: Comprehensive method comparison

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
│   ├── vanilla_synthetic_linear/ # Phase 1: Vanilla baseline data
│   └── seer/                     # Phase 2+: SEER cancer data
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

**Model Output**: Risk score (log hazard ratio), not survival time or event probability.

### Cox Proportional Hazards Framework

DeepSurv implements the Cox proportional hazards model (Cox, 1972) using deep neural networks. The model learns a risk function h(x) = exp(f(x)), where f(x) is the neural network output.

**Key properties:**
- **Input**: Patient covariates X ∈ ℝⁿ
- **Output**: Risk score f(x) ∈ ℝ (higher = higher mortality risk)
- **Training**: Partial likelihood maximization using (time, event) pairs
- **Inference**: Relative risk ranking among patients

### Why Ranking Instead of Direct Time Prediction?

**Censoring challenge**: In survival data, event times for censored patients are unknown. For patient i with (tᵢ, δᵢ):
- If δᵢ = 1 (event occurred): tᵢ is the true survival time
- If δᵢ = 0 (censored): tᵢ is a lower bound; true survival time > tᵢ is unknown

This makes regression-based time prediction infeasible (undefined target for censored cases).

**Cox solution**: Instead of predicting absolute survival times, the model learns to rank patients by relative risk. The Cox partial likelihood only requires ordering information, making it robust to censoring.

### Alternative Approaches

For applications requiring time predictions:
- **Parametric models**: Weibull AFT, Log-normal (assume distribution)
- **Random Survival Forests** (Ishwaran et al., 2008): Non-parametric survival curves
- **DeepHit** (Lee et al., 2018): Discrete-time survival distributions via deep learning
- **SurvTRACE** (2022): Transformer-based survival analysis

**Current implementation**: Cox model as established baseline for comparison.

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
