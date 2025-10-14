# DeepSurv Benchmark Results Comparison

## 📊 Current Results vs. Paper Expectations

### Simulated Linear Data

| Metric | Paper Expected | Current Result | Status |
|--------|---------------|----------------|---------|
| **C-Index** | **~0.80** | **0.46** | ❌ **FAILED** |
| Dataset Size | 5,000 samples | 5,000 samples | ✅ |
| Features | 10 | 10 | ✅ |
| Event Rate | ~66% | 66.7% | ✅ |

---

## 🔴 Problem Identified

The C-index of **0.46** is **below random chance** (0.5), indicating:
- Model is predicting in the **wrong direction** (lower risk → shorter survival)
- Possible issue with risk score interpretation
- Data generation might not match paper's approach

---

## 🔍 Potential Issues to Investigate

### 1. **Risk Score Direction** ⚠️
- Current model outputs a single value (log hazard)
- Need to verify: **Higher output = Higher risk = Shorter survival**
- C-index < 0.5 suggests we might be measuring it backwards

### 2. **Data Generation**
Current implementation:
```python
hazard = np.exp(0.5 * features[:, 0] - 0.3 * features[:, 1])
survival_times = np.random.exponential(1.0 / hazard)
```

Possible issues:
- Only using 2 features out of 10 for hazard
- May not match original paper's data generation
- Need to check original DeepSurv paper/code

### 3. **Model Architecture**
Current: `[64, 32, 16]` with BatchNorm + Dropout 0.3

Original paper might use different architecture.

### 4. **Training Issues**
- Early stopping at epoch 11-13
- Loss not decreasing much
- C-index not improving during training

---

## 📋 Action Items

1. [ ] **Check risk score direction** - Verify concordance index calculation
2. [ ] **Review original paper** - Check exact data generation procedure  
3. [ ] **Compare with original code** - https://github.com/jaredleekatzman/DeepSurv
4. [ ] **Test with all features** - Use all 10 features in hazard calculation
5. [ ] **Increase training** - Remove/increase early stopping patience
6. [ ] **Simplify model** - Try simpler architecture first

---

## 🎯 Expected Benchmarks (Phase 1)

| Dataset | Size | Expected C-Index | Target |
|---------|------|------------------|--------|
| Simulated Linear | 5K | ~0.80 | Must achieve ±0.03 |
| Simulated Gaussian | 5K | ~0.75 | Must achieve ±0.03 |
| METABRIC | 2K | ~0.65-0.68 | Must achieve ±0.03 |

**Phase 1 Status**: ❌ NOT COMPLETE - Need to fix fundamental issues

---

## 📝 Notes

- Current implementation has all basic components (model, loss, evaluation)
- Structure is correct but results suggest fundamental issue
- Need to validate against original paper's implementation
- May need to debug data generation and risk calculation

---

*Last updated: October 14, 2025*
