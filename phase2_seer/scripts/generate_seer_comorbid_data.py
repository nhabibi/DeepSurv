"""
Generate Synthetic SEER-like Data for Breast-Vaginal Cancer COMORBIDITY

This simulates the RESULT of the real SEER workflow:
1. Find patients with BOTH breast AND vaginal cancer diagnoses
2. Merge/join their records into one combined record
3. Create a dataset ready for DeepSurv

Each row represents ONE patient who has BOTH cancers, with features from both diagnoses.
"""

import numpy as np
import pandas as pd
from pathlib import Path

np.random.seed(42)


def generate_seer_comorbid_data(n_samples=5000):
    """
    Generate synthetic data for patients with BOTH breast and vaginal cancer.
    
    This simulates what you'll get after joining SEER records for patients
    with multiple primary cancers.
    
    Args:
        n_samples: Number of comorbid cancer patients to generate
        
    Returns:
        pd.DataFrame with combined features from both cancers
    """
    
    print(f"Generating {n_samples} patients with BREAST + VAGINAL comorbidity...")
    print("(Simulating joined SEER records)")
    
    # ==========================================================================
    # DEMOGRAPHICS (shared across both cancers)
    # ==========================================================================
    
    # Age at FIRST cancer diagnosis
    # Comorbid cancer patients tend to be older
    age_first_dx = np.random.gamma(shape=9, scale=6, size=n_samples) + 40
    age_first_dx = np.clip(age_first_dx, 35, 90).astype(int)
    
    # Time between first and second cancer (months)
    # Typically 6 months to 10 years
    time_between_cancers = np.random.gamma(shape=3, scale=12, size=n_samples)
    time_between_cancers = np.clip(time_between_cancers, 3, 120).astype(int)
    
    # Age at second diagnosis
    age_second_dx = age_first_dx + (time_between_cancers / 12).astype(int)
    age_second_dx = np.clip(age_second_dx, age_first_dx, 95).astype(int)
    
    # Race (SEER categories)
    race_probs = [0.72, 0.14, 0.09, 0.05]  # White, Black, Asian, Other
    race = np.random.choice(['White', 'Black', 'Asian', 'Other'], 
                           size=n_samples, p=race_probs)
    
    # Marital status at first diagnosis
    marital_probs = [0.52, 0.18, 0.20, 0.10]  # Married, Single, Divorced, Widowed
    marital_status = np.random.choice(['Married', 'Single', 'Divorced', 'Widowed'],
                                     size=n_samples, p=marital_probs)
    
    # ==========================================================================
    # WHICH CANCER CAME FIRST?
    # ==========================================================================
    
    # 75% breast first (more common), 25% vaginal first
    first_cancer = np.random.choice(['Breast', 'Vaginal'], 
                                    size=n_samples, p=[0.75, 0.25])
    
    # ==========================================================================
    # BREAST CANCER CHARACTERISTICS
    # ==========================================================================
    
    # Breast cancer stage (AJCC)
    breast_stage_probs = [0.38, 0.32, 0.20, 0.10]  # I, II, III, IV
    breast_stage = np.random.choice([1, 2, 3, 4], size=n_samples, p=breast_stage_probs)
    
    # Breast cancer grade
    breast_grade_probs = [0.22, 0.38, 0.28, 0.12]
    breast_grade = np.random.choice([1, 2, 3, 4], size=n_samples, p=breast_grade_probs)
    
    # Breast tumor size (cm)
    breast_tumor_size = np.random.gamma(shape=2.5, scale=1.2, size=n_samples) * (0.6 + 0.25 * breast_stage)
    breast_tumor_size = np.clip(breast_tumor_size, 0.3, 12.0)
    
    # Breast positive nodes
    breast_nodes_prob = 0.12 + 0.18 * (breast_stage - 1)
    breast_has_nodes = np.random.binomial(1, breast_nodes_prob, size=n_samples)
    breast_positive_nodes = np.zeros(n_samples)
    breast_positive_nodes[breast_has_nodes == 1] = np.random.poisson(
        lam=1.8 * breast_stage[breast_has_nodes == 1], 
        size=breast_has_nodes.sum()
    )
    breast_positive_nodes = np.clip(breast_positive_nodes, 0, 25).astype(int)
    
    # Breast treatment
    breast_surgery_prob = 0.92 - 0.12 * (breast_stage - 1)
    breast_surgery = np.random.binomial(1, breast_surgery_prob, size=n_samples)
    
    breast_radiation_prob = 0.55 + 0.08 * (breast_stage - 1)
    breast_radiation = np.random.binomial(1, np.clip(breast_radiation_prob, 0, 1), size=n_samples)
    
    breast_chemo_prob = 0.25 + 0.20 * (breast_stage - 1)
    breast_chemo = np.random.binomial(1, np.clip(breast_chemo_prob, 0, 1), size=n_samples)
    
    # ==========================================================================
    # VAGINAL CANCER CHARACTERISTICS
    # ==========================================================================
    
    # Vaginal cancer stage (tends to be more advanced at diagnosis)
    vaginal_stage_probs = [0.25, 0.30, 0.28, 0.17]  # I, II, III, IV
    vaginal_stage = np.random.choice([1, 2, 3, 4], size=n_samples, p=vaginal_stage_probs)
    
    # Vaginal cancer grade
    vaginal_grade_probs = [0.18, 0.35, 0.32, 0.15]
    vaginal_grade = np.random.choice([1, 2, 3, 4], size=n_samples, p=vaginal_grade_probs)
    
    # Vaginal tumor size (cm) - typically smaller
    vaginal_tumor_size = np.random.gamma(shape=2.0, scale=0.8, size=n_samples) * (0.5 + 0.3 * vaginal_stage)
    vaginal_tumor_size = np.clip(vaginal_tumor_size, 0.2, 8.0)
    
    # Vaginal treatment
    vaginal_surgery_prob = 0.75 - 0.15 * (vaginal_stage - 1)
    vaginal_surgery = np.random.binomial(1, vaginal_surgery_prob, size=n_samples)
    
    vaginal_radiation_prob = 0.70 + 0.10 * (vaginal_stage - 1)  # Radiation very common
    vaginal_radiation = np.random.binomial(1, np.clip(vaginal_radiation_prob, 0, 1), size=n_samples)
    
    vaginal_chemo_prob = 0.30 + 0.18 * (vaginal_stage - 1)
    vaginal_chemo = np.random.binomial(1, np.clip(vaginal_chemo_prob, 0, 1), size=n_samples)
    
    # ==========================================================================
    # COMBINED TREATMENT (some overlap possible)
    # ==========================================================================
    
    # Any surgery (from either cancer)
    any_surgery = (breast_surgery | vaginal_surgery).astype(int)
    
    # Any radiation (from either cancer)
    any_radiation = (breast_radiation | vaginal_radiation).astype(int)
    
    # Any chemotherapy (from either cancer)
    any_chemotherapy = (breast_chemo | vaginal_chemo).astype(int)
    
    # ==========================================================================
    # SURVIVAL OUTCOMES
    # ==========================================================================
    
    # Having TWO cancers significantly worsens prognosis
    # Survival measured from FIRST cancer diagnosis
    
    # Calculate combined hazard
    log_hazard = (
        # Breast cancer effects
        0.25 * (breast_stage - 1) +
        0.12 * (breast_grade - 1) +
        0.04 * breast_tumor_size +
        0.015 * breast_positive_nodes +
        
        # Vaginal cancer effects  
        0.30 * (vaginal_stage - 1) +      # Vaginal cancer often worse
        0.15 * (vaginal_grade - 1) +
        0.06 * vaginal_tumor_size +
        
        # COMORBIDITY PENALTY (having BOTH cancers)
        0.50 +                             # Base penalty for dual cancer
        
        # Age effect
        0.015 * (age_first_dx - 55) +
        
        # Time between cancers (shorter interval = worse)
        -0.008 * (time_between_cancers / 12) +  # Longer gap slightly better
        
        # Treatment (protective)
        -0.25 * any_surgery +
        -0.18 * any_radiation +
        -0.12 * any_chemotherapy +
        
        # Random variation
        np.random.normal(0, 0.35, size=n_samples)
    )
    
    hazard = np.exp(log_hazard)
    
    # Generate survival times from FIRST diagnosis
    survival_times = np.random.exponential(scale=1.0 / (hazard + 1e-8))
    survival_months = survival_times * 25  # Scale to realistic months
    survival_months = np.clip(survival_months, 0.5, 100)  # Shorter survival due to dual cancer
    
    # Censoring (less than single cancer due to worse prognosis)
    censoring_rate = 0.35  # Lower censoring rate (more deaths observed)
    censoring_times = np.random.exponential(scale=50, size=n_samples)
    
    # Observed time and event
    observed_time = np.minimum(survival_months, censoring_times)
    event = (survival_months <= censoring_times).astype(int)
    
    # ==========================================================================
    # CREATE DATAFRAME
    # ==========================================================================
    
    df = pd.DataFrame({
        # Demographics
        'age_first_diagnosis': age_first_dx,
        'age_second_diagnosis': age_second_dx,
        'months_between_cancers': time_between_cancers,
        'first_cancer_site': first_cancer,
        'race': race,
        'marital_status': marital_status,
        
        # Breast cancer features
        'breast_stage': breast_stage,
        'breast_grade': breast_grade,
        'breast_tumor_size_cm': breast_tumor_size.round(2),
        'breast_positive_nodes': breast_positive_nodes,
        'breast_surgery': breast_surgery,
        'breast_radiation': breast_radiation,
        'breast_chemotherapy': breast_chemo,
        
        # Vaginal cancer features
        'vaginal_stage': vaginal_stage,
        'vaginal_grade': vaginal_grade,
        'vaginal_tumor_size_cm': vaginal_tumor_size.round(2),
        'vaginal_surgery': vaginal_surgery,
        'vaginal_radiation': vaginal_radiation,
        'vaginal_chemotherapy': vaginal_chemo,
        
        # Combined treatment
        'any_surgery': any_surgery,
        'any_radiation': any_radiation,
        'any_chemotherapy': any_chemotherapy,
        
        # Outcomes (from first diagnosis)
        'survival_months': observed_time.round(2),
        'vital_status': event  # 1=dead, 0=alive/censored
    })
    
    # Print summary statistics
    print("\n" + "="*70)
    print("BREAST-VAGINAL COMORBIDITY DATA SUMMARY")
    print("="*70)
    print(f"\nTotal patients: {len(df)} (all have BOTH cancers)")
    
    print(f"\n📅 TIMING:")
    print(f"  Age at first diagnosis: mean={df['age_first_diagnosis'].mean():.1f}, range=[{df['age_first_diagnosis'].min()}-{df['age_first_diagnosis'].max()}]")
    print(f"  Months between cancers: mean={df['months_between_cancers'].mean():.1f}, median={df['months_between_cancers'].median():.1f}")
    
    print(f"\n🎯 WHICH CAME FIRST:")
    print(df['first_cancer_site'].value_counts())
    
    print(f"\n🔬 BREAST CANCER:")
    print(f"  Stage distribution: {dict(df['breast_stage'].value_counts().sort_index())}")
    print(f"  Mean tumor size: {df['breast_tumor_size_cm'].mean():.2f} cm")
    print(f"  Positive nodes: {df['breast_positive_nodes'].mean():.2f} average")
    
    print(f"\n🔬 VAGINAL CANCER:")
    print(f"  Stage distribution: {dict(df['vaginal_stage'].value_counts().sort_index())}")
    print(f"  Mean tumor size: {df['vaginal_tumor_size_cm'].mean():.2f} cm")
    
    print(f"\n💊 TREATMENT:")
    print(f"  Any surgery: {df['any_surgery'].mean()*100:.1f}%")
    print(f"  Any radiation: {df['any_radiation'].mean()*100:.1f}%")
    print(f"  Any chemotherapy: {df['any_chemotherapy'].mean()*100:.1f}%")
    
    print(f"\n⏱️ SURVIVAL:")
    print(f"  Survival months: mean={df['survival_months'].mean():.1f}, median={df['survival_months'].median():.1f}")
    print(f"  Deaths: {df['vital_status'].sum()} ({df['vital_status'].mean()*100:.1f}%)")
    print(f"  Censored: {(1-df['vital_status']).sum()} ({(1-df['vital_status'].mean())*100:.1f}%)")
    
    print(f"\n💡 NOTE: Higher death rate expected due to dual cancer burden")
    print("="*70)
    
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
    
    print(f"\n📊 Split sizes:")
    print(f"  Train: {len(train_df)} ({len(train_df)/n*100:.1f}%)")
    print(f"  Val:   {len(val_df)} ({len(val_df)/n*100:.1f}%)")
    print(f"  Test:  {len(test_df)} ({len(test_df)/n*100:.1f}%)")
    
    return train_df, val_df, test_df


if __name__ == '__main__':
    # Create data directory
    data_dir = Path(__file__).parent.parent.parent / 'data' / 'seer'
    data_dir.mkdir(parents=True, exist_ok=True)
    
    # Generate data
    print("🏥 Generating synthetic SEER-like COMORBID CANCER data...")
    print("   Cancer combination: Breast + Vaginal")
    print("   Population: Women with BOTH cancers")
    print("   Simulates: Joined/merged SEER records")
    print()
    
    df = generate_seer_comorbid_data(n_samples=5000)
    
    # Split into train/val/test
    train_df, val_df, test_df = split_data(df)
    
    # Save to CSV
    train_path = data_dir / 'train_seer_comorbid_5000.csv'
    val_path = data_dir / 'val_seer_comorbid_5000.csv'
    test_path = data_dir / 'test_seer_comorbid_5000.csv'
    
    train_df.to_csv(train_path, index=False)
    val_df.to_csv(val_path, index=False)
    test_df.to_csv(test_path, index=False)
    
    print(f"\n✅ Data saved to:")
    print(f"   {train_path}")
    print(f"   {val_path}")
    print(f"   {test_path}")
    print("\n🎯 This simulates the result of joining real SEER records!")
    print("   Each row = 1 patient with features from BOTH cancers")
    print("\n🚀 Ready for Phase 2 training!")
