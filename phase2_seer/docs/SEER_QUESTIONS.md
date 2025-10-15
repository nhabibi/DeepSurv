# SEER Data Questions - Please Answer

**Before proceeding with SEER integration, I need to know:**

---

## 1. 📁 Do you have SEER data already?

- [ ] **YES** - I have downloaded SEER data
  - File location: `____________________________`
  - File format: [ ] CSV [ ] TXT (fixed-width) [ ] Other: `________`
  - File size: `__________ MB/GB`
  
- [ ] **NO** - I don't have SEER data yet
  - → I'll create synthetic SEER-like data for testing

---

## 2. 🎯 Cancer Site

Which cancer are you studying?
- [ ] Breast cancer
- [ ] Lung cancer  
- [ ] Colorectal cancer
- [ ] Prostate cancer
- [ ] All cancer sites
- [ ] Other: `____________________________`

---

## 3. 📅 Time Period

- Years: `__________ to __________` (e.g., 2000 to 2018)

---

## 4. 📊 Variables / Features

Which variables do you need? (Check all that apply)

### Demographics:
- [ ] Age at diagnosis
- [ ] Sex/Gender
- [ ] Race
- [ ] Marital status
- [ ] Other: `____________________________`

### Tumor Characteristics:
- [ ] Tumor stage (AJCC/SEER)
- [ ] Tumor grade
- [ ] Tumor size
- [ ] Histology type
- [ ] Number of positive lymph nodes
- [ ] Other: `____________________________`

### Treatment:
- [ ] Surgery (yes/no)
- [ ] Radiation (yes/no)
- [ ] Chemotherapy (yes/no)
- [ ] Other: `____________________________`

### Comorbidities:
- [ ] Charlson Comorbidity Index (CCI)
- [ ] Diabetes
- [ ] Hypertension
- [ ] Heart disease
- [ ] COPD
- [ ] Other conditions: `____________________________`

### Outcome:
- [ ] Survival months (time)
- [ ] Vital status (event: dead/alive)
- [ ] Cause of death

---

## 5. 🔍 Sample Size

- Approximate number of patients: `__________`
- Any specific inclusion/exclusion criteria?
  - Age range: `____________________________`
  - Stage restrictions: `____________________________`
  - Other: `____________________________`

---

## 6. 🎓 For Your Supervisor

**What aspect of comorbidity are you focusing on?**

- [ ] Effect of comorbidity count on survival
- [ ] Specific comorbidity combinations
- [ ] Comorbidity vs tumor characteristics interaction
- [ ] Other: `____________________________`

---

## Current Status

**What I can do RIGHT NOW:**

### Option A: You Have SEER Data
✅ You tell me file location
✅ I inspect the format
✅ Create `load_seer_data()` function
✅ Generate small sample (100-1000 patients) for testing
✅ Test on small sample first
✅ Run full training

### Option B: Create Synthetic SEER-Like Data
✅ I create realistic SEER-like data with:
   - Demographics (age, sex, race)
   - Tumor features (stage, grade, size)
   - Comorbidity indicators (5-10 conditions)
   - Survival outcome (months, event)
✅ Test the complete pipeline
✅ You swap in real data later (minimal code change)

**Which option do you prefer?**

---

**Please fill this out or just tell me:**
1. Do you have SEER data? (yes/no)
2. If no, should I create synthetic SEER-like data?
3. What cancer site?
4. What comorbidities are most important?
