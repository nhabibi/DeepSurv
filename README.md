# DeepSurv: Cox Proportional Hazards Deep Neural Network

**PyTorch implementation** of **[DeepSurv](https://arxiv.org/abs/1606.00931)** for survival analysis research.

**Paper**: [Katzman et al., 2018](https://arxiv.org/abs/1606.00931) | [Original Code](https://github.com/jaredleekatzman/DeepSurv)

---

## 🗺️ Research Roadmap

| Phase | Status | Goal |
|-------|--------|------|
| **1. Vanilla Baseline** | ✅ Complete | Establish baseline (C-index=0.7111) |
| **2. SEER Application** | 🔄 In Progress | Apply to clinical cancer data |
| **3. Comorbidity Analysis** | 📋 Planned | Study dual-cancer survival patterns |
| **4. Comorbidity-Aware Arch** | 🔮 Future | Multi-input, attention, cross-stitch |
| **5. Advanced Methods** | 🔮 Optional | DeepHit, RSF, Transformers |

---

## 📂 Project Structure

```
DeepSurv/
├── main.py, requirements.txt
├── src/                           # model, loss, train, eval, data_loader
├── phase1_vanilla/                # Docs + scripts
├── phase2_seer/                   # Docs + scripts  
├── data/
│   ├── vanilla_synthetic_linear/  # Phase 1
│   └── seer/                      # Phase 2+
└── results/                       # checkpoints, figures, logs
```

---

## 🎯 Quick Start

```bash
# Install
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt

# Train vanilla baseline
python main.py --data-type linear --n-samples 5000 --n-features 10

# Train on SEER data  
python main.py --data-source seer
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
| **C-Index** | **0.7111** |
| Dataset | Synthetic linear, 5000 samples, 10 features |
| Architecture | [25, 25] ReLU (vanilla) |
| Hyperparameters | LR=1e-3, L2=0.01, SGD+Nesterov |
| Training | 164 epochs (early stopped) |
| Device | Apple M1 (MPS) |

**Framework Adaptations**: LR×10 and L2÷1000 due to PyTorch/Theano differences. Empirically validated.

**Docs**: See `phase1_vanilla/docs/` for technical details and thesis summary.

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
