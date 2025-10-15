# Phase 2: SEER Data Integration Guide

**Goal**: Minimal changes to apply validated DeepSurv baseline to real SEER data

---

## Step 1: Understanding SEER Data Access

### ❓ Do You Have SEER Data Access?

**SEER (Surveillance, Epidemiology, and End Results) Program**
- Maintained by: National Cancer Institute (NCI)
- Access: Requires signed Data Use Agreement
- Website: https://seer.cancer.gov/

**To Access SEER Data:**
1. Visit: https://seer.cancer.gov/data/access.html
2. Sign Data Use Agreement
3. Download SEER*Stat software OR ASCII data files
4. Typical turnaround: Few days to weeks

### 📊 SEER Data Format

**Typical SEER Variables:**
- **Demographics**: Age, sex, race, marital status
- **Tumor**: Site, histology, grade, stage, size
- **Treatment**: Surgery, radiation, chemotherapy flags
- **Outcome**: Survival months, vital status (alive/dead)
- **Comorbidities**: (if linked to Medicare data)

**File Format:**
- Fixed-width text files OR
- CSV exports from SEER*Stat

---

## Step 2: What We Need from You

**Please confirm:**

1. ✅ **Do you already have SEER data downloaded?**
   - If YES: Where is it located? What format?
   - If NO: I'll create a realistic synthetic SEER-like dataset

2. ✅ **Which cancer site?** 
   - Breast cancer? Lung? Colorectal? All sites?

3. ✅ **Time period?**
   - e.g., 2000-2018?

4. ✅ **Which variables/comorbidities?**
   - Basic demographics?
   - Charlson Comorbidity Index?
   - Specific conditions (diabetes, hypertension, etc.)?

---

## Step 3: Minimal Changes Strategy

### What STAYS THE SAME ✅
- **Model architecture**: [25, 25] ReLU
- **Loss function**: Cox proportional hazards (Efron)
- **Training loop**: Same as Phase 1
- **Hyperparameters**: LR=1e-3, L2=0.01 (validated)
- **Optimizer**: SGD + Nesterov
- **Evaluation**: C-index

### What CHANGES 🔧
1. **Data loading**: Read SEER CSV instead of synthetic generation
2. **Feature preprocessing**: 
   - Handle categorical variables (one-hot encoding)
   - Normalize continuous features
   - Handle missing values
3. **Configuration**: Update feature names and counts

---

## Step 4: File Changes Overview

### Files That DON'T Change
- ❌ `model.py` - Architecture stays vanilla
- ❌ `loss.py` - Cox loss stays the same
- ❌ `train.py` - Training loop unchanged
- ❌ `evaluation.py` - C-index calculation unchanged

### Files That Change Minimally
- ✏️ `data_loader.py` - Add `load_seer_data()` function
- ✏️ `config.py` - Update feature counts (10 → actual SEER features)
- ✏️ `main.py` - Add `--data-source seer` option

### New Files
- ➕ `seer_preprocessing.py` - SEER-specific preprocessing
- ➕ `data/seer/README.md` - Data documentation

---

## Step 5: Teaching Your Supervisor

**Narrative for Supervisor:**

> "We established a vanilla DeepSurv baseline on synthetic data (Phase 1). Now we're applying the EXACT SAME model to real SEER data. The only changes are:
> 
> 1. **Data loading**: Reading SEER CSV instead of generating synthetic data
> 2. **Preprocessing**: Handling categorical variables and missing values (standard ML practice)
> 3. **Configuration**: Updating feature count (10 → N SEER features)
> 
> The model, loss function, training procedure, and validated hyperparameters remain identical. This demonstrates that our baseline is robust and applicable to real-world clinical data."

---

## Next Steps

**Option A: You Have SEER Data**
1. Tell me the file path
2. I'll inspect the format
3. Create `load_seer_data()` function
4. Test on small sample (100-1000 patients)
5. Run full training

**Option B: Create Synthetic SEER-Like Data**
1. I'll create realistic synthetic SEER data with:
   - Demographics (age, sex, race)
   - Tumor characteristics (stage, grade, size)
   - Comorbidity indicators (diabetes, hypertension, etc.)
   - Survival outcome (months, event)
2. Test the pipeline end-to-end
3. You can swap in real SEER data later

**Which option?**

---

## 📚 References for SEER

- SEER Program: https://seer.cancer.gov/
- Data Dictionary: https://seer.cancer.gov/data-software/documentation/
- SEER*Stat: https://seer.cancer.gov/seerstat/

---

**Status**: Waiting for your input on SEER data availability
