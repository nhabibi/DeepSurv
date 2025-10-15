# Phase 2: SEER Data Integration

**Goal**: Apply validated vanilla baseline to real SEER data with minimal changes

**Status**: 🔄 **READY TO START** - Waiting for SEER data details

---

## 🎯 Objective

Apply the **exact same validated model** from Phase 1 to real SEER cancer survival data:
- **Same architecture**: [25, 25] ReLU
- **Same loss function**: Cox proportional hazards
- **Same training**: SGD + Nesterov
- **Same hyperparameters**: LR=1e-3, L2=0.01 (validated)

**Only changes**: Data loading and preprocessing

---

## 📁 Contents

### Documentation (`docs/`)
- **[SEER_GUIDE.md](docs/PHASE2_SEER_GUIDE.md)** - Complete SEER integration guide
- **[SEER_QUESTIONS.md](docs/SEER_QUESTIONS.md)** - Data requirements checklist

### Scripts (`scripts/`)
- (SEER-specific scripts will be added here)

---

## ❓ What We Need

**Please answer** in `docs/SEER_QUESTIONS.md`:

1. **Do you have SEER data?**
   - YES → Provide file path and format
   - NO → We'll create synthetic SEER-like data

2. **Cancer site?** (breast, lung, colorectal, etc.)

3. **Comorbidities?** (diabetes, hypertension, etc.)

4. **Time period?** (e.g., 2000-2018)

---

## 🔧 Minimal Changes Strategy

### What STAYS THE SAME ✅
- Model architecture: [25, 25] ReLU
- Loss function: Cox proportional hazards
- Training procedure: Same as Phase 1
- Hyperparameters: LR=1e-3, L2=0.01
- Evaluation: C-index

### What CHANGES 🔧
1. **Data loading**: Read SEER CSV instead of synthetic
2. **Preprocessing**: 
   - Handle categorical variables (one-hot encoding)
   - Normalize continuous features
   - Handle missing values
3. **Configuration**: Update feature count (10 → N)

### Files That DON'T Change
- ❌ `src/model.py` - Architecture unchanged
- ❌ `src/loss.py` - Cox loss unchanged
- ❌ `src/train.py` - Training loop unchanged
- ❌ `src/evaluation.py` - C-index unchanged

### Files That Change Minimally
- ✏️ `src/data_loader.py` - Add `load_seer_data()`
- ✏️ `src/config.py` - Update feature count
- ✏️ `main.py` - Add `--data-source seer` option

### New Files
- ➕ `phase2_seer/scripts/seer_preprocessing.py` - SEER-specific preprocessing

---

## 📋 Implementation Plan

### Step 1: Data Preparation
- [ ] Answer SEER questions
- [ ] Obtain/generate SEER data
- [ ] Inspect data format

### Step 2: Small Sample Testing
- [ ] Create small sample (100-1000 patients)
- [ ] Quick test run (50 epochs)
- [ ] Verify pipeline works

### Step 3: Apply Minimal Changes
- [ ] Update `src/data_loader.py` with SEER loader
- [ ] Update `src/config.py` with feature count
- [ ] Add preprocessing for categorical variables
- [ ] Document all changes

### Step 4: Full Training
- [ ] Run with full SEER data
- [ ] Same hyperparameters (LR=1e-3, L2=0.01)
- [ ] Compare to Phase 1 baseline

### Step 5: Documentation
- [ ] Document changes made
- [ ] Show before/after code
- [ ] Present results to supervisor

---

## 🎓 For Your Supervisor

**Key Message:**
> "We're applying the exact same validated model to real clinical data. Only the data source changed—everything else (model, training, hyperparameters) remains identical. This proves our baseline is robust and applicable to real-world scenarios."

**What to Show:**
1. Phase 1: C-index=0.71 on synthetic data
2. Phase 2: Minimal code changes (data loading only)
3. Phase 2: C-index on SEER data
4. Comparison: Baseline establishes valid reference

---

## 📊 Expected Workflow

```
Phase 1 Baseline (✅ Complete)
         ↓
   Answer SEER Questions
         ↓
   Load/Generate SEER Data
         ↓
   Small Sample Test
         ↓
   Full SEER Training
         ↓
   Phase 2 Complete ✅
```

---

**Previous**: [Phase 1: Vanilla Baseline](../phase1_vanilla/)

**Next**: Start by answering `docs/SEER_QUESTIONS.md`
