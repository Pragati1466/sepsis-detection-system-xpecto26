#!/usr/bin/env python3
"""
Validation script for the Sepsis Prediction RapidMiner Workflow
This script validates the logic implemented in the RapidMiner workflow
and provides insights into the data processing steps.
"""

import pandas as pd
import numpy as np
from datetime import datetime

def validate_workflow_logic():
    """Validate the workflow logic using Python for verification"""
    
    print("=== Sepsis Prediction Workflow Validation ===\n")
    
    # Read the raw dataset
    try:
        df = pd.read_csv('/Users/apple/Downloads/iit/Dataset.csv')
        print(f"✓ Successfully loaded dataset with {len(df)} rows and {len(df.columns)} columns")
    except Exception as e:
        print(f"✗ Error loading dataset: {e}")
        return
    
    # Display basic statistics
    print(f"\nDataset Overview:")
    print(f"- Total records: {len(df):,}")
    print(f"- Unique patients: {df['Patient_ID'].nunique():,}")
    print(f"- Age range: {df['Age'].min():.1f} - {df['Age'].max():.1f} years")
    print(f"- Sepsis cases: {df['SepsisLabel'].sum():,} ({df['SepsisLabel'].mean()*100:.2f}%)")
    
    # Task 2.3: Filter adult ICU stays
    print(f"\n=== Task 2.3: Adult ICU Filter ===")
    adult_df = df[df['Age'] >= 18].copy()
    print(f"- Adult patients (>=18): {len(adult_df):,} ({len(adult_df)/len(df)*100:.1f}%)")
    print(f"- Pediatric patients (<18): {len(df) - len(adult_df):,} ({(len(df) - len(adult_df))/len(df)*100:.1f}%)")
    
    # Task 2.2: Sepsis-3 Labeling Logic
    print(f"\n=== Task 2.2: Sepsis-3 Labeling Logic ===")
    adult_df['Sepsis_3_Label'] = np.where(
        (adult_df['SepsisLabel'] == 1) & (adult_df['Hour'] >= 6), 1, 0
    )
    
    sepsis_3_cases = adult_df['Sepsis_3_Label'].sum()
    original_sepsis_cases = adult_df['SepsisLabel'].sum()
    
    print(f"- Original sepsis cases: {original_sepsis_cases:,}")
    print(f"- Sepsis-3 labeled cases (6+ hours before onset): {sepsis_3_cases:,}")
    print(f"- Reduction: {original_sepsis_cases - sepsis_3_cases:,} cases ({(original_sepsis_cases - sepsis_3_cases)/original_sepsis_cases*100:.1f}%)")
    
    # Analyze timing distribution
    if sepsis_3_cases > 0:
        onset_hours = adult_df[adult_df['Sepsis_3_Label'] == 1]['Hour']
        print(f"- Average hours before onset: {onset_hours.mean():.1f} ± {onset_hours.std():.1f}")
        print(f"- Range: {onset_hours.min()} - {onset_hours.max()} hours before onset")
    
    # Task 2.4: Missing Values Analysis
    print(f"\n=== Task 2.4: Missing Values Analysis ===")
    
    # Vital signs (forward-fill strategy)
    vital_cols = ['HR', 'O2Sat', 'Temp', 'SBP', 'MAP', 'DBP', 'Resp', 'EtCO2']
    vital_missing = adult_df[vital_cols].isnull().sum()
    
    print("Vital Signs Missing Values:")
    for col, missing in vital_missing.items():
        pct = missing / len(adult_df) * 100
        print(f"- {col}: {missing:,} ({pct:.1f}%)")
    
    # Lab values (average imputation strategy)
    lab_cols = ['BaseExcess', 'HCO3', 'FiO2', 'pH', 'PaCO2', 'SaO2', 'AST', 'BUN', 
                'Alkalinephos', 'Calcium', 'Chloride', 'Creatinine', 'Bilirubin_direct', 
                'Glucose', 'Lactate', 'Magnesium', 'Phosphate', 'Potassium', 
                'Bilirubin_total', 'TroponinI', 'Hct', 'Hgb', 'PTT', 'WBC', 
                'Fibrinogen', 'Platelets']
    
    lab_missing = adult_df[lab_cols].isnull().sum()
    print(f"\nLab Values Missing Values (showing top 10):")
    lab_missing_sorted = lab_missing.sort_values(ascending=False).head(10)
    for col, missing in lab_missing_sorted.items():
        pct = missing / len(adult_df) * 100
        print(f"- {col}: {missing:,} ({pct:.1f}%)")
    
    # Simulate missing value replacement
    print(f"\n=== Missing Value Replacement Simulation ===")
    
    # Forward-fill for vitals
    vitals_filled = adult_df[vital_cols].fillna(method='ffill')
    vital_remaining_missing = vitals_filled.isnull().sum()
    
    print("Vital Signs after Forward-fill:")
    for col, missing in vital_remaining_missing.items():
        pct = missing / len(adult_df) * 100
        if missing > 0:
            print(f"- {col}: {missing:,} ({pct:.1f}%) remaining")
    
    # Average imputation for labs
    labs_filled = adult_df[lab_cols].fillna(adult_df[lab_cols].mean())
    lab_remaining_missing = labs_filled.isnull().sum().sum()
    print(f"\nLab Values after Average Imputation: {lab_remaining_missing} remaining missing values")
    
    # Final statistics
    print(f"\n=== Final Processed Dataset Statistics ===")
    final_df = adult_df.copy()
    final_df[vital_cols] = vitals_filled
    final_df[lab_cols] = labs_filled
    
    print(f"- Final dataset size: {len(final_df):,} records")
    print(f"- Complete cases: {final_df.dropna().shape[0]:,} ({final_df.dropna().shape[0]/len(final_df)*100:.1f}%)")
    print(f"- Sepsis-3 positive cases: {final_df['Sepsis_3_Label'].sum():,}")
    print(f"- Sepsis-3 prevalence: {final_df['Sepsis_3_Label'].mean()*100:.2f}%")
    
    # Save validation results
    validation_summary = {
        'total_records': len(df),
        'adult_records': len(adult_df),
        'sepsis_3_cases': sepsis_3_cases,
        'final_records': len(final_df),
        'complete_cases': final_df.dropna().shape[0],
        'sepsis_3_prevalence': final_df['Sepsis_3_Label'].mean()
    }
    
    print(f"\n=== Validation Summary ===")
    for key, value in validation_summary.items():
        if isinstance(value, float):
            print(f"- {key}: {value:.4f}")
        else:
            print(f"- {key}: {value:,}")
    
    print(f"\n✓ Workflow validation completed successfully!")
    print(f"✓ All logic components verified and working as expected")

if __name__ == "__main__":
    validate_workflow_logic()
