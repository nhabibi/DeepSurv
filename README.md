# DeepSurv: Cox Proportional Hazards Deep Neural Network

**PyTorch implementation** of **[DeepSurv](https://arxiv.org/abs/1606.00931)** for survival analysis research.

**Paper**: [Katzman et al., 2018](https://arxiv.org/abs/1606.00931) | [Original Code](https://github.com/jaredleekatzman/DeepSurv)

---

## 🗺️ Research Roadmap

| Phase | Status | Goal |
|-------|--------|------|
| **1. Vanilla Baseline** | ✅ Complete | Establish baseline (C-index=0.7662) |
| **2. SEER Application** | 🔄 In Progress | Apply to clinical cancer data |
| **3. Comorbidity Analysis** | 📋 Planned | Study dual-cancer survival patterns |
| **4. Comorbidity-Aware Arch** | 🔮 Future | Multi-input, attention, cross-stitch |
| **5. Advanced Methods** | 🔮 Optional | DeepHit, RSF, Transformers |

---

## 📂 Project Structure

```
DeepSurv/
├── README.md                      # All documentation (this file)
├── requirements.txt
├── phase1_vanilla/
│   ├── main.py                    # --generate-data to save, then train
│   ├── src/                       # Vanilla codebase
│   ├── data/                      # → train/val/test_synthetic.csv
│   └── results/                   # → checkpoints/, figures/
└── phase2_seer/
    ├── main.py                    # Training script
    ├── src/                       # SEER codebase (data_loader modified)
    ├── data/                      # → train/val/test_comorbid.csv
    └── results/                   # → checkpoints/, figures/
```

**Design**: Maximally minimal. All docs in README. Each phase self-contained.

---

## 🎯 Quick Start

```bash
# Install
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt

# Phase 1
cd phase1_vanilla
python main.py --generate-data  # Generate & save data
python main.py                  # Train

# Phase 2  
cd ../phase2_seer
# Generate data first (see SEER Data Generation below)
python main.py                  # Train
```

---

## 📊 SEER Data Generation

Phase 2 requires synthetic SEER comorbid data. Run this once to generate `data/train|val|test_comorbid.csv`:

```python
# Save as generate_seer.py in phase2_seer/ and run once
# See phase2_seer/src/data_loader.py for schema details
# Or use real SEER data if available
```

---

## 🔍 What Does DeepSurv Predict?

**Output**: Risk score f(x) ∈ ℝ, not survival time or event probability.

**Cox Framework**: Learns relative risk ranking using Cox partial likelihood (Cox, 1972). For patient i with (tᵢ, δᵢ):
- δᵢ = 1 (event): tᵢ is true survival time
- δᵢ = 0 (censored): tᵢ is lower bound, true time unknown

**Why ranking?** Censoring makes regression infeasible (undefined target). Cox partial likelihood requires only ordering information.

**Time prediction alternatives**: Parametric (Weibull AFT), Random Survival Forests (2008), DeepHit (2018), SurvTRACE (2022).

**Input flexibility**: Architecture `[25, 25]` adapts to any input size. For `input_dim=d`, first layer weight matrix is `W₁ ∈ ℝ^(25×d)`, automatically allocated by `nn.Linear(d, 25)`. Example:
- Vanilla synthetic (10 features): `10 → [25, 25] → 1`
- SEER comorbid (25 features): `25 → [25, 25] → 1`

**Output targets**: Both use `(survival_months, vital_status)` for Cox loss supervision, not direct prediction.

---

## 📊 Phase 1 Results

| Metric | Value |
|--------|-------|
| **C-Index** | **0.7662** (validation) / **0.7778** (training) |
| Dataset | Synthetic linear, 5000 samples, 10 features |
| Architecture | [25, 25] ReLU (vanilla) |
| Hyperparameters | LR=1e-3, L2=0.01, SGD+Nesterov |
| Training | 165 epochs (early stopped, patience=100) |
| Validation Split | 15% (750 samples) |
| Event Rate | 20.3% (stronger signal, less censoring) |
| Device | Apple M1 (MPS) |

**Framework Adaptations**: 
- LR: 1e-4 → 1e-3 (10×) - PyTorch SGD dynamics
- L2: 10.0 → 0.01 (÷1000) - PyTorch weight_decay convention
- Synthetic signal: 2× hazard weights for reproducible learning

---

## 📊 Phase 2: SEER Application

**Status**: 🔄 In Progress

**Data**: Breast + Vaginal comorbid cancer (5000 patients)
- Mean survival: 37.8 months
- Death rate: 53.1% | Censoring: 46.9%
- Features: 25 (demographics, tumor, treatment, comorbidity)

**Model Changes**: None (same architecture, just more input features)

**Data Changes**: 
- `load_seer_data()` added to handle categorical encoding
- Column names updated for SEER schema

**Next Steps**:
1. ✅ Data generated with realistic survival distribution
2. ✅ Data loader updated for 25 features
3. ⏳ Full training run (waiting to execute)

---

## 📋 Phase Comparison

| Aspect | Phase 1 (Vanilla) | Phase 2 (SEER) |
|--------|-------------------|----------------|
| **Data** | Synthetic linear (10 features) | SEER comorbid (25 features) |
| **Samples** | 5000 | 5000 (3500 train / 750 val / 750 test) |
| **Model** | [25, 25] ReLU | [25, 25] ReLU (same) |
| **Hyperparams** | LR=1e-3, L2=0.01 | LR=1e-3, L2=0.01 (same) |
| **C-Index** | 0.7662 | ⏳ Pending |
| **Code Changes** | N/A | Only `data_loader.py` |

---

## 🔧 Configuration

Key hyperparameters in `src/config.py`:

| Parameter | Vanilla | Adapted | Reason |
|-----------|---------|---------|--------|
| Architecture | [25, 25] ReLU | ✓ | - |
| Optimizer | SGD+Nesterov (momentum=0.9) | ✓ | - |
| Batch Size | 64 | ✓ | - |
| Early Stop | 50 epochs | ✓ | - |
| **Learning Rate** | 1e-4 | **1e-3** | PyTorch optimization dynamics |
| **L2 Reg** | 10.0 | **0.01** | PyTorch weight_decay semantics |

---

## � References

- Cox, D. R. (1972). "Regression models and life-tables." *JRSS*.
- Katzman, J. L., et al. (2018). "DeepSurv." *BMC Medical Research Methodology*, 18(1), 24.
- Ishwaran, H., et al. (2008). "Random survival forests." *Ann. Appl. Stat*.
- Lee, C., et al. (2018). "DeepHit." *AAAI*.

---

**Status**: Phase 1 ✅ Complete | Phase 2 🔄 In Progress | Research code for PhD thesis
