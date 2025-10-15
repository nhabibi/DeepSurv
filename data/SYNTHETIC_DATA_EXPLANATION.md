# Synthetic Data Explanation

## 📊 Data Structure Overview

The saved synthetic data (`data/synthetic/*.csv`) is **synthetic survival data** for testing the DeepSurv model.

---

## 📁 Columns Breakdown

### Input Features (X)
**Columns**: `feature_1` through `feature_10` (10 columns)

**Meaning**:
- These are **synthetic patient characteristics** or covariates
- In real SEER data, these would be things like: age, tumor size, tumor stage, comorbidities, etc.
- Here: Random standardized values (mean=0, std=1) from normal distribution
- **These are what the model learns from to predict survival**

**Example interpretation (if this were real data)**:
- `feature_1`: Could be age (standardized)
- `feature_2`: Could be tumor size (standardized)
- `feature_3`: Could be tumor stage (standardized)
- etc.

### Output/Target Variables (y)
**Columns**: `time` and `event` (2 columns)

#### 1. `time` (Survival Time)
**Meaning**: 
- **How long the patient survived** (or was followed) in time units
- This is the **observed time** to either death or last follow-up
- Range: 0 to ~17 (arbitrary time units - could be months, years, etc.)

**Examples**:
- `time = 0.0055`: Patient died/censored very early (0.0055 time units)
- `time = 4.365`: Patient died/censored after 4.365 time units
- `time = 1.228`: Patient died/censored after 1.228 time units

#### 2. `event` (Event Indicator / Censoring)
**Meaning**:
- **Did the event (death) occur?**
- `event = 1.0` → **Event occurred** (patient died) - we observed the actual death
- `event = 0.0` → **Censored** (patient still alive or lost to follow-up) - we don't know if/when they died

**Examples**:
- `time = 1.228, event = 1.0` → Patient died at time 1.228
- `time = 2.840, event = 0.0` → Patient was still alive at time 2.840 (censored)

---

## 🎯 What the Model Predicts

### DeepSurv Prediction: **Risk Score (Log Hazard Ratio)**

**NOT predicted directly**:
- ❌ Exact time of death
- ❌ Probability of death at specific time

**WHAT IS predicted**:
- ✅ **Risk score** (log hazard ratio) - a relative measure of risk
- Higher risk score → Higher risk of event (death)
- Lower risk score → Lower risk of event (death)

### How It Works:

1. **Input**: `[feature_1, feature_2, ..., feature_10]`
   - Example: `[0.497, -0.138, 0.648, 1.523, -0.234, -0.234, 1.579, 0.767, -0.469, 0.543]`

2. **Model computes**: Risk score (scalar value)
   - Example: `risk_score = 0.85`

3. **Interpretation**:
   - Patient A: risk_score = 0.85
   - Patient B: risk_score = -0.30
   - → Patient A has **higher risk** than Patient B
   - → Patient A is more likely to die sooner (relative to B)

---

## 📈 How Data Is Generated

### For Linear Data:

```
1. Generate features: feature_1, ..., feature_10 ~ Normal(0, 1)

2. Compute true hazard:
   log_hazard = 1.0*feature_1 - 0.8*feature_2 + 0.6*feature_3 - 0.4*feature_4 + ...
   hazard = exp(log_hazard)

3. Generate survival time:
   survival_time ~ Exponential(1/hazard)
   - High hazard → Die sooner
   - Low hazard → Die later

4. Generate censoring time:
   censoring_time ~ Exponential(0.5)
   - Random time when patient might be lost to follow-up

5. Observed data:
   time = min(survival_time, censoring_time)
   event = 1 if survival_time <= censoring_time else 0
   - We observe whichever happens first: death or censoring
```

### Feature Weights (True Relationship):
```
feature_1:   +1.0  (strong positive effect - increases risk)
feature_2:   -0.8  (strong negative effect - decreases risk)
feature_3:   +0.6  (moderate positive effect)
feature_4:   -0.4  (moderate negative effect)
feature_5:   +0.3  (weak positive effect)
feature_6:   -0.2  (weak negative effect)
feature_7:   +0.15
feature_8:   -0.1
feature_9:   +0.05
feature_10:  -0.03 (very weak negative effect)
```

---

## 🔬 Example Row Interpretation

```csv
feature_1=0.497, feature_2=-0.138, feature_3=0.648, ..., feature_10=0.543, time=0.0055, event=1.0
```

**Interpretation**:
- **Patient characteristics** (features): Random values representing patient/tumor characteristics
- **Outcome**: Patient died (`event=1.0`) very early (`time=0.0055`)
- **Model's job**: Learn that patients with these feature values tend to die early

---

## 🎯 Model Training Goal

**Objective**: Learn to predict risk scores that correctly rank patients by survival time

**Loss function**: Cox proportional hazards loss
- Ensures patients who died earlier have higher predicted risk than those who died later
- Handles censored data correctly (patients still alive)

**Evaluation metric**: C-index (concordance index)
- Measures how well the model ranks patients
- C-index = 0.71 means the model correctly orders 71% of patient pairs by survival time

---

## 📊 Real-World Translation

### In SEER Data (Phase 2):

**Input features** would be:
```
age, sex, race, tumor_stage, tumor_grade, tumor_size, 
surgery, radiation, chemotherapy,
diabetes, hypertension, heart_disease, etc.
```

**Output** would be:
```
time = survival_months (e.g., 36 months = 3 years)
event = vital_status (1 = dead, 0 = alive)
```

**Model prediction**:
```
risk_score = Model([age, sex, race, ...])
→ Higher risk = More likely to die sooner
→ Lower risk = More likely to survive longer
```

---

## Summary

| Component | Description | Example |
|-----------|-------------|---------|
| **Input (X)** | `feature_1` to `feature_10` | Patient/tumor characteristics |
| **Output (y)** | `time` (survival time) | 0.0055, 1.228, 4.365 (time units) |
|  | `event` (death occurred?) | 1.0 (died) or 0.0 (censored) |
| **Prediction** | Risk score (log hazard) | 0.85 (high risk), -0.30 (low risk) |
| **Goal** | Rank patients correctly | C-index = 0.71 (71% correct) |

**The model learns**: Which feature patterns → High/low risk of death
