"""
Generate Synthetic SEER-like Data for Breast and Vaginal Cancer COMORBIDITY in Women

This script creates realistic synthetic data that mimics SEER cancer registry data
for studying CANCER-CANCER COMORBIDITY (women with BOTH breast and vaginal cancer).

Groups:
1. Breast cancer ONLY
2. Vaginal cancer ONLY  
3. BOTH cancers (comorbidity) ← Key focus!

Features:
- Demographics: age, race, marital status
- Tumor characteristics: stage, grade, size (for each cancer)
- Treatment: surgery, radiation, chemotherapy
- Outcomes: survival time, vital status
"""

import numpy as np
import pandas as pd
from pathlib import Path

np.random.seed(42)


def generate_seer_data(n_samples=5000):
    """
    Generate synthetic SEER-like data for breast and vaginal cancer COMORBIDITY.
    
    Three groups:
    1. Breast only (~70%)
    2. Vaginal only (~20%)
    3. BOTH cancers - COMORBIDITY (~10%)
    
    Args:
        n_samples: Number of patients to generate
        
    Returns:
        pd.DataFrame with all features and outcomes
    """
    
    print(f"Generating {n_samples} synthetic SEER patients...")
    print("Focus: Cancer-cancer COMORBIDITY (breast + vaginal)")
    
    # ==========================================================================
    # DEMOGRAPHICS
    # ==========================================================================
    
    # Age at diagnosis (women with breast/vaginal cancer)
    # Comorbidity patients tend to be older
    age = np.random.gamma(shape=8, scale=7, size=n_samples) + 35
    age = np.clip(age, 25, 95).astype(int)
    
    # Race (SEER categories)
    race_probs = [0.70, 0.15, 0.10, 0.05]  # White, Black, Asian, Other
    race = np.random.choice(['White', 'Black', 'Asian', 'Other'], 
                           size=n_samples, p=race_probs)
    
    # Marital status
    marital_probs = [0.55, 0.20, 0.15, 0.10]  # Married, Single, Divorced, Widowed
    marital_status = np.random.choice(['Married', 'Single', 'Divorced', 'Widowed'],
                                     size=n_samples, p=marital_probs)
    
    # ==========================================================================
    # CANCER COMORBIDITY STATUS
    # ==========================================================================
    
    # Three groups: breast only, vaginal only, BOTH (comorbidity)
    cancer_group_probs = [0.70, 0.20, 0.10]  # Breast, Vaginal, BOTH
    cancer_group = np.random.choice(['Breast_only', 'Vaginal_only', 'Both'], 
                                   size=n_samples, p=cancer_group_probs)
    
    has_breast = (cancer_group == 'Breast_only') | (cancer_group == 'Both')
    has_vaginal = (cancer_group == 'Vaginal_only') | (cancer_group == 'Both')
    has_both = (cancer_group == 'Both')
    
    # ==========================================================================
    # TUMOR CHARACTERISTICS
    # ==========================================================================
    
    # AJCC Stage (I, II, III, IV)
    # More patients at earlier stages (screening effect)
    stage_probs = [0.35, 0.30, 0.20, 0.15]  # I, II, III, IV
    stage = np.random.choice([1, 2, 3, 4], size=n_samples, p=stage_probs)
    
    # Grade (1=well diff, 2=moderate, 3=poor, 4=undifferentiated)
    grade_probs = [0.20, 0.35, 0.30, 0.15]
    grade = np.random.choice([1, 2, 3, 4], size=n_samples, p=grade_probs)
    
    # Tumor size in cm (influenced by stage)
    tumor_size_base = np.random.gamma(shape=2, scale=1.5, size=n_samples)
    tumor_size = tumor_size_base * (0.5 + 0.3 * stage)  # Larger tumors in later stages
    tumor_size = np.clip(tumor_size, 0.1, 15.0)
    
    # Number of positive lymph nodes (more in later stages)
    nodes_prob = 0.1 + 0.2 * (stage - 1)  # Higher stages → more nodes
    has_positive_nodes = np.random.binomial(1, nodes_prob, size=n_samples)
    n_positive_nodes = np.zeros(n_samples)
    n_positive_nodes[has_positive_nodes == 1] = np.random.poisson(
        lam=2 * stage[has_positive_nodes == 1], 
        size=has_positive_nodes.sum()
    )
    n_positive_nodes = np.clip(n_positive_nodes, 0, 30).astype(int)
    
    # ==========================================================================
    # TREATMENT
    # ==========================================================================
    
    # Surgery (most patients get surgery, especially early stage)
    surgery_prob = 0.95 - 0.15 * (stage - 1)  # Lower for advanced stage
    surgery = np.random.binomial(1, surgery_prob, size=n_samples)
    
    # Radiation (common for breast, depends on stage/surgery)
    radiation_prob = 0.50 + 0.10 * (stage - 1) + 0.20 * surgery
    radiation_prob = np.clip(radiation_prob, 0, 1)
    radiation = np.random.binomial(1, radiation_prob, size=n_samples)
    
    # Chemotherapy (more common in advanced stages)
    chemo_prob = 0.20 + 0.20 * (stage - 1)
    chemo_prob = np.clip(chemo_prob, 0, 1)
    chemotherapy = np.random.binomial(1, chemo_prob, size=n_samples)
    
    # ==========================================================================
    # COMORBIDITIES
    # ==========================================================================
    
    # Age increases comorbidity risk
    age_factor = (age - 40) / 50  # Normalize age effect
    age_factor = np.clip(age_factor, 0, 1)
    
    # Diabetes (increases with age)
    diabetes_prob = 0.10 + 0.15 * age_factor
    diabetes = np.random.binomial(1, diabetes_prob, size=n_samples)
    
    # Hypertension (very common, increases with age)
    hypertension_prob = 0.20 + 0.30 * age_factor
    hypertension = np.random.binomial(1, hypertension_prob, size=n_samples)
    
    # Heart disease (increases with age, related to diabetes/hypertension)
    heart_prob = 0.05 + 0.15 * age_factor + 0.10 * diabetes + 0.10 * hypertension
    heart_prob = np.clip(heart_prob, 0, 1)
    heart_disease = np.random.binomial(1, heart_prob, size=n_samples)
    
    # COPD (smoking-related, increases with age)
    copd_prob = 0.05 + 0.10 * age_factor
    copd = np.random.binomial(1, copd_prob, size=n_samples)
    
    # Kidney disease (related to diabetes/hypertension)
    kidney_prob = 0.03 + 0.08 * diabetes + 0.05 * hypertension
    kidney_disease = np.random.binomial(1, kidney_prob, size=n_samples)
    
    # Liver disease
    liver_prob = 0.02 + 0.03 * age_factor
    liver_disease = np.random.binomial(1, liver_prob, size=n_samples)
    
    # Charlson Comorbidity Index (CCI) - weighted sum
    cci = (diabetes * 1 + 
           heart_disease * 1 + 
           copd * 1 + 
           kidney_disease * 2 + 
           liver_disease * 3)
    
    # ==========================================================================
    # SURVIVAL OUTCOMES
    # ==========================================================================
    
    # Calculate hazard based on risk factors
    # More risk factors → higher hazard → shorter survival
    
    log_hazard = (
        # Tumor characteristics (biggest effect)
        0.30 * (stage - 1) +           # Stage is major predictor
        0.15 * (grade - 1) +           # Grade matters
        0.05 * tumor_size +            # Size matters
        0.02 * n_positive_nodes +      # Nodes matter
        
        # Demographics
        0.01 * (age - 55) +            # Older age → worse
        0.10 * (cancer_site == 'Vaginal') +  # Vaginal worse prognosis
        
        # Treatment (protective)
        -0.30 * surgery +              # Surgery helps
        -0.15 * radiation +            # Radiation helps
        -0.10 * chemotherapy +         # Chemo helps (but given to worse cases)
        
        # Comorbidities (harmful)
        0.20 * diabetes +              # Diabetes worsens outcome
        0.15 * hypertension +          # Hypertension worsens
        0.30 * heart_disease +         # Heart disease significant
        0.20 * copd +                  # COPD worsens
        0.25 * kidney_disease +        # Kidney disease significant
        0.35 * liver_disease +         # Liver disease very significant
        
        # Random variation
        np.random.normal(0, 0.3, size=n_samples)
    )
    
    hazard = np.exp(log_hazard)
    
    # Generate survival times (exponential distribution)
    survival_times = np.random.exponential(scale=1.0 / (hazard + 1e-8))
    
    # Scale to months (0-120 months = 10 years)
    survival_months = survival_times * 30  # Scale to realistic months
    survival_months = np.clip(survival_months, 0.1, 120)
    
    # Censoring (some patients are still alive at end of study)
    # More censoring for early stage, less for late stage
    censoring_rate = 0.60 - 0.10 * (stage - 1)
    censoring_rate = np.clip(censoring_rate, 0.2, 0.7)
    censoring_times = np.random.exponential(scale=60, size=n_samples)
    
    # Observed time is minimum of death and censoring
    observed_time = np.minimum(survival_months, censoring_times)
    event = (survival_months <= censoring_times).astype(int)  # 1=died, 0=censored
    
    # ==========================================================================
    # CREATE DATAFRAME
    # ==========================================================================
    
    df = pd.DataFrame({
        # Demographics
        'age': age,
        'race': race,
        'marital_status': marital_status,
        
        # Cancer
        'cancer_site': cancer_site,
        'stage': stage,
        'grade': grade,
        'tumor_size_cm': tumor_size.round(2),
        'n_positive_nodes': n_positive_nodes,
        
        # Treatment
        'surgery': surgery,
        'radiation': radiation,
        'chemotherapy': chemotherapy,
        
        # Comorbidities
        'diabetes': diabetes,
        'hypertension': hypertension,
        'heart_disease': heart_disease,
        'copd': copd,
        'kidney_disease': kidney_disease,
        'liver_disease': liver_disease,
        'charlson_cci': cci,
        
        # Outcomes
        'survival_months': observed_time.round(2),
        'vital_status': event  # 1=dead, 0=alive/censored
    })
    
    # Print summary statistics
    print("\n" + "="*60)
    print("SYNTHETIC SEER DATA SUMMARY")
    print("="*60)
    print(f"\nTotal patients: {len(df)}")
    print(f"\nCancer sites:")
    print(df['cancer_site'].value_counts())
    print(f"\nStage distribution:")
    print(df['stage'].value_counts().sort_index())
    print(f"\nAge: mean={df['age'].mean():.1f}, range=[{df['age'].min()}-{df['age'].max()}]")
    print(f"\nSurvival months: mean={df['survival_months'].mean():.1f}, median={df['survival_months'].median():.1f}")
    print(f"\nVital status:")
    print(f"  Dead: {df['vital_status'].sum()} ({df['vital_status'].mean()*100:.1f}%)")
    print(f"  Alive/Censored: {(1-df['vital_status']).sum()} ({(1-df['vital_status'].mean())*100:.1f}%)")
    print(f"\nComorbidities prevalence:")
    print(f"  Diabetes: {df['diabetes'].mean()*100:.1f}%")
    print(f"  Hypertension: {df['hypertension'].mean()*100:.1f}%")
    print(f"  Heart disease: {df['heart_disease'].mean()*100:.1f}%")
    print(f"  COPD: {df['copd'].mean()*100:.1f}%")
    print(f"  Kidney disease: {df['kidney_disease'].mean()*100:.1f}%")
    print(f"  Liver disease: {df['liver_disease'].mean()*100:.1f}%")
    print(f"\nCharlson CCI: mean={df['charlson_cci'].mean():.2f}, max={df['charlson_cci'].max()}")
    print(f"\nTreatment rates:")
    print(f"  Surgery: {df['surgery'].mean()*100:.1f}%")
    print(f"  Radiation: {df['radiation'].mean()*100:.1f}%")
    print(f"  Chemotherapy: {df['chemotherapy'].mean()*100:.1f}%")
    print("="*60)
    
    return df


def split_data(df, train_ratio=0.70, val_ratio=0.15, test_ratio=0.15):
    """Split data into train/val/test sets."""
    n = len(df)
    n_train = int(n * train_ratio)
    n_val = int(n * val_ratio)
    
    # Shuffle
    df_shuffled = df.sample(frac=1, random_state=42).reset_index(drop=True)
    
    train_df = df_shuffled[:n_train]
    val_df = df_shuffled[n_train:n_train+n_val]
    test_df = df_shuffled[n_train+n_val:]
    
    print(f"\nSplit sizes:")
    print(f"  Train: {len(train_df)} ({len(train_df)/n*100:.1f}%)")
    print(f"  Val:   {len(val_df)} ({len(val_df)/n*100:.1f}%)")
    print(f"  Test:  {len(test_df)} ({len(test_df)/n*100:.1f}%)")
    
    return train_df, val_df, test_df


if __name__ == '__main__':
    # Create data directory
    data_dir = Path(__file__).parent.parent.parent / 'data' / 'seer'
    data_dir.mkdir(parents=True, exist_ok=True)
    
    # Generate data
    print("🏥 Generating synthetic SEER-like data...")
    print("   Cancer sites: Breast + Vaginal")
    print("   Population: Women")
    print("   Features: Demographics, Tumor, Treatment, Comorbidities")
    print()
    
    df = generate_seer_data(n_samples=5000)
    
    # Split into train/val/test
    train_df, val_df, test_df = split_data(df)
    
    # Save to CSV
    train_path = data_dir / 'train_seer_breast_vaginal_5000.csv'
    val_path = data_dir / 'val_seer_breast_vaginal_5000.csv'
    test_path = data_dir / 'test_seer_breast_vaginal_5000.csv'
    
    train_df.to_csv(train_path, index=False)
    val_df.to_csv(val_path, index=False)
    test_df.to_csv(test_path, index=False)
    
    print(f"\n✅ Data saved to:")
    print(f"   {train_path}")
    print(f"   {val_path}")
    print(f"   {test_path}")
    print("\n🚀 Ready for Phase 2 training!")
