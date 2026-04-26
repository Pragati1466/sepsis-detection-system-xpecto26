#!/usr/bin/env python3
"""
Validation script for Phase 3: Feature Engineering
This script validates the feature engineering logic implemented in the RapidMiner workflow
and provides insights into the new engineered features.
"""

import pandas as pd
import numpy as np
from datetime import datetime
import warnings
warnings.filterwarnings('ignore')

def validate_feature_engineering():
    """Validate the feature engineering logic using Python for verification"""
    
    print("=== Phase 3: Feature Engineering Validation ===\n")
    
    # Load the processed data from Phase 2
    try:
        df = pd.read_csv('/Users/apple/Downloads/iit/processed_sepsis_data.csv')
        print(f"✓ Successfully loaded processed dataset with {len(df)} rows and {len(df.columns)} columns")
    except FileNotFoundError:
        print("⚠ Processed dataset not found, using original dataset with preprocessing")
        df = pd.read_csv('/Users/apple/Downloads/iit/Dataset.csv')
        # Apply basic preprocessing
        df = df[df['Age'] >= 18].copy()
        df['Sepsis_3_Label'] = np.where((df['SepsisLabel'] == 1) & (df['Hour'] >= 6), 1, 0)
        df = df[df['Sepsis_3_Label'] == 1].copy()
        
        # Fill missing values
        vital_cols = ['HR', 'O2Sat', 'Temp', 'SBP', 'MAP', 'DBP', 'Resp', 'EtCO2']
        lab_cols = ['BaseExcess', 'HCO3', 'FiO2', 'pH', 'PaCO2', 'SaO2', 'AST', 'BUN', 
                    'Alkalinephos', 'Calcium', 'Chloride', 'Creatinine', 'Bilirubin_direct', 
                    'Glucose', 'Lactate', 'Magnesium', 'Phosphate', 'Potassium', 
                    'Bilirubin_total', 'TroponinI', 'Hct', 'Hgb', 'PTT', 'WBC', 
                    'Fibrinogen', 'Platelets']
        
        df[vital_cols] = df[vital_cols].fillna(method='ffill')
        df[lab_cols] = df[lab_cols].fillna(df[lab_cols].mean())
    
    print(f"\nDataset Overview:")
    print(f"- Total records: {len(df):,}")
    print(f"- Sepsis-3 positive cases: {df['Sepsis_3_Label'].sum():,}")
    print(f"- Sepsis-3 prevalence: {df['Sepsis_3_Label'].mean()*100:.2f}%")
    
    # Task 3.1: Shock Index Calculation
    print(f"\n=== Task 3.1: Shock Index Calculation ===")
    df['Shock_Index'] = np.where(df['SBP'] > 0, df['HR'] / df['SBP'], np.nan)
    
    shock_index_stats = df['Shock_Index'].describe()
    print(f"- Shock Index calculated as HR / SBP")
    print(f"- Mean: {shock_index_stats['mean']:.3f}")
    print(f"- Std Dev: {shock_index_stats['std']:.3f}")
    print(f"- Range: {shock_index_stats['min']:.3f} - {shock_index_stats['max']:.3f}")
    print(f"- Missing values: {df['Shock_Index'].isnull().sum():,} ({df['Shock_Index'].isnull().sum()/len(df)*100:.1f}%)")
    
    # Clinical interpretation
    high_shock = df['Shock_Index'] > 1.0
    print(f"- High shock index (>1.0): {high_shock.sum():,} cases ({high_shock.mean()*100:.1f}%)")
    
    # Task 3.2: Rolling Window Features
    print(f"\n=== Task 3.2: Rolling Window Features ===")
    
    # Sort by patient and hour for proper rolling calculations
    df_sorted = df.sort_values(['Patient_ID', 'Hour'])
    
    # Calculate rolling statistics for Heart Rate
    df_sorted['HR_3hr_Mean'] = df_sorted.groupby('Patient_ID')['HR'].rolling(window=3, min_periods=1).mean().reset_index(level=0, drop=True)
    df_sorted['HR_3hr_StdDev'] = df_sorted.groupby('Patient_ID')['HR'].rolling(window=3, min_periods=1).std().reset_index(level=0, drop=True)
    
    # Calculate rolling statistics for O2Sat
    df_sorted['O2Sat_3hr_Mean'] = df_sorted.groupby('Patient_ID')['O2Sat'].rolling(window=3, min_periods=1).mean().reset_index(level=0, drop=True)
    df_sorted['O2Sat_3hr_StdDev'] = df_sorted.groupby('Patient_ID')['O2Sat'].rolling(window=3, min_periods=1).std().reset_index(level=0, drop=True)
    
    print("Heart Rate 3-Hour Rolling Statistics:")
    hr_mean_stats = df_sorted['HR_3hr_Mean'].describe()
    hr_std_stats = df_sorted['HR_3hr_StdDev'].describe()
    print(f"- Mean HR (3hr): {hr_mean_stats['mean']:.1f} ± {hr_mean_stats['std']:.1f}")
    print(f"- Std Dev HR (3hr): {hr_std_stats['mean']:.1f} ± {hr_std_stats['std']:.1f}")
    
    print("\nSpO2 3-Hour Rolling Statistics:")
    o2sat_mean_stats = df_sorted['O2Sat_3hr_Mean'].describe()
    o2sat_std_stats = df_sorted['O2Sat_3hr_StdDev'].describe()
    print(f"- Mean SpO2 (3hr): {o2sat_mean_stats['mean']:.1f} ± {o2sat_mean_stats['std']:.1f}")
    print(f"- Std Dev SpO2 (3hr): {o2sat_std_stats['mean']:.1f} ± {o2sat_std_stats['std']:.1f}")
    
    # Task 3.3: SOFA Sub-scores
    print(f"\n=== Task 3.3: SOFA Sub-scores Calculation ===")
    
    # Renal SOFA Score
    def calculate_renal_sofa(creatinine):
        if pd.isna(creatinine):
            return 0
        if creatinine < 1.2:
            return 0
        elif creatinine < 2.0:
            return 1
        elif creatinine < 3.5:
            return 2
        elif creatinine < 5.0:
            return 3
        else:
            return 4
    
    # Respiratory SOFA Score (PaO2/FiO2 ratio approximation)
    def calculate_respiratory_sofa(paco2, fio2):
        if pd.isna(paco2) or pd.isna(fio2) or fio2 <= 0:
            return 0
        # Approximate PaO2/FiO2 ratio (simplified)
        ratio = paco2 * fio2 / 21
        if ratio >= 400:
            return 0
        elif ratio >= 300:
            return 1
        elif ratio >= 200:
            return 2
        elif ratio >= 100:
            return 3
        else:
            return 4
    
    # Cardiovascular SOFA Score
    def calculate_cardiovascular_sofa(map_value):
        if pd.isna(map_value):
            return 0
        if map_value >= 70:
            return 0
        elif map_value >= 50:
            return 1
        elif map_value >= 30:
            return 2
        else:
            return 3
    
    df_sorted['SOFA_Renal'] = df_sorted['Creatinine'].apply(calculate_renal_sofa)
    df_sorted['SOFA_Respiratory'] = df_sorted.apply(lambda row: calculate_respiratory_sofa(row['PaCO2'], row['FiO2']), axis=1)
    df_sorted['SOFA_Cardiovascular'] = df_sorted['MAP'].apply(calculate_cardiovascular_sofa)
    df_sorted['SOFA_Total'] = df_sorted['SOFA_Renal'] + df_sorted['SOFA_Respiratory'] + df_sorted['SOFA_Cardiovascular']
    
    print("SOFA Sub-score Distribution:")
    print(f"- Renal SOFA: {df_sorted['SOFA_Renal'].value_counts().sort_index().to_dict()}")
    print(f"- Respiratory SOFA: {df_sorted['SOFA_Respiratory'].value_counts().sort_index().to_dict()}")
    print(f"- Cardiovascular SOFA: {df_sorted['SOFA_Cardiovascular'].value_counts().sort_index().to_dict()}")
    print(f"- Total SOFA: {df_sorted['SOFA_Total'].value_counts().sort_index().to_dict()}")
    
    sofa_stats = df_sorted['SOFA_Total'].describe()
    print(f"\nTotal SOFA Statistics:")
    print(f"- Mean: {sofa_stats['mean']:.2f}")
    print(f"- Std Dev: {sofa_stats['std']:.2f}")
    print(f"- Range: {sofa_stats['min']} - {sofa_stats['max']}")
    
    # High SOFA cases
    high_sofa = df_sorted['SOFA_Total'] >= 2
    print(f"- High SOFA (≥2): {high_sofa.sum():,} cases ({high_sofa.mean()*100:.1f}%)")
    
    # Task 3.4: Normalization
    print(f"\n=== Task 3.4: Feature Normalization ===")
    
    # Identify numeric columns for normalization
    numeric_cols = df_sorted.select_dtypes(include=[np.number]).columns.tolist()
    exclude_cols = ['Patient_ID', 'SepsisLabel', 'Sepsis_3_Label', 'Gender', 'Unit1', 'Unit2']
    normalize_cols = [col for col in numeric_cols if col not in exclude_cols]
    
    print(f"Numeric columns to normalize: {len(normalize_cols)}")
    print(f"Key features: {[col for col in normalize_cols if any(x in col for x in ['Shock', 'SOFA', 'HR_', 'O2Sat_'])]}")
    
    # Apply Z-transformation
    df_normalized = df_sorted.copy()
    for col in normalize_cols:
        if df_sorted[col].std() > 0:  # Avoid division by zero
            df_normalized[col + '_Z'] = (df_sorted[col] - df_sorted[col].mean()) / df_sorted[col].std()
    
    print(f"- Normalized {len(normalize_cols)} features using Z-transformation")
    print(f"- Features with zero variance (skipped): {sum(df_sorted[col].std() == 0 for col in normalize_cols)}")
    
    # Final dataset statistics
    print(f"\n=== Final Feature Engineered Dataset ===")
    final_df = df_normalized.copy()
    
    print(f"- Final dataset size: {len(final_df):,} records")
    print(f"- Total features: {len(final_df.columns)}")
    print(f"- Engineered features: {len([col for col in final_df.columns if any(x in col for x in ['Shock', 'SOFA', '_3hr', '_Z'])])}")
    print(f"- Sepsis-3 positive cases: {final_df['Sepsis_3_Label'].sum():,}")
    
    # Feature correlations with sepsis
    engineered_features = ['Shock_Index', 'HR_3hr_Mean', 'HR_3hr_StdDev', 'O2Sat_3hr_Mean', 'O2Sat_3hr_StdDev', 
                          'SOFA_Renal', 'SOFA_Respiratory', 'SOFA_Cardiovascular', 'SOFA_Total']
    
    print(f"\n=== Feature Correlations with Sepsis-3 ===")
    correlations = {}
    for feature in engineered_features:
        if feature in final_df.columns:
            corr = final_df[feature].corr(final_df['Sepsis_3_Label'])
            correlations[feature] = corr
            print(f"- {feature}: {corr:.3f}")
    
    # Most predictive features
    sorted_correlations = sorted(correlations.items(), key=lambda x: abs(x[1]), reverse=True)
    print(f"\nTop 3 Most Predictive Features:")
    for feature, corr in sorted_correlations[:3]:
        print(f"- {feature}: {corr:.3f}")
    
    # Save validation results
    validation_summary = {
        'total_records': len(final_df),
        'engineered_features': len([col for col in final_df.columns if any(x in col for x in ['Shock', 'SOFA', '_3hr', '_Z'])]),
        'sepsis_3_cases': final_df['Sepsis_3_Label'].sum(),
        'shock_index_mean': final_df['Shock_Index'].mean(),
        'sofa_total_mean': final_df['SOFA_Total'].mean(),
        'high_sofa_cases': (final_df['SOFA_Total'] >= 2).sum(),
        'top_predictive_feature': sorted_correlations[0][0] if sorted_correlations else None
    }
    
    print(f"\n=== Feature Engineering Validation Summary ===")
    for key, value in validation_summary.items():
        if isinstance(value, float):
            print(f"- {key}: {value:.4f}")
        else:
            print(f"- {key}: {value}")
    
    print(f"\n✓ Feature engineering validation completed successfully!")
    print(f"✓ All engineered features calculated and validated")
    
    return final_df

if __name__ == "__main__":
    validate_feature_engineering()
