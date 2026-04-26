#!/usr/bin/env python3
"""
Final Success Metrics Validation for Xpecto'26 Hackathon
This script validates that our model meets or exceeds all target metrics
"""

import pandas as pd
import numpy as np
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.metrics import roc_auc_score, classification_report, confusion_matrix
from sklearn.preprocessing import StandardScaler
from imblearn.over_sampling import SMOTE
import warnings
warnings.filterwarnings('ignore')

def validate_success_metrics():
    """Validate all success metrics against hackathon targets"""
    
    print("=== Xpecto'26 Success Metrics Validation ===\n")
    
    # Load and preprocess data
    df = pd.read_csv('/Users/apple/Downloads/iit/Dataset.csv')
    print(f"✓ Dataset loaded: {len(df):,} records")
    
    # Preprocessing pipeline
    df = df[df['Age'] >= 18].copy()
    df['Sepsis_3_Label'] = np.where((df['SepsisLabel'] == 1) & (df['Hour'] >= 6), 1, 0)
    
    # Handle missing values
    vital_cols = ['HR', 'O2Sat', 'Temp', 'SBP', 'MAP', 'DBP', 'Resp', 'EtCO2']
    lab_cols = ['BaseExcess', 'HCO3', 'FiO2', 'pH', 'PaCO2', 'SaO2', 'AST', 'BUN', 
                'Alkalinephos', 'Calcium', 'Chloride', 'Creatinine', 'Bilirubin_direct', 
                'Glucose', 'Lactate', 'Magnesium', 'Phosphate', 'Potassium', 
                'Bilirubin_total', 'TroponinI', 'Hct', 'Hgb', 'PTT', 'WBC', 
                'Fibrinogen', 'Platelets']
    
    df[vital_cols] = df[vital_cols].fillna(method='ffill')
    df[lab_cols] = df[lab_cols].fillna(df[lab_cols].mean())
    
    # Feature engineering
    df['Shock_Index'] = np.where(df['SBP'] > 0, df['HR'] / df['SBP'], np.nan)
    
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
    
    df['SOFA_Renal'] = df['Creatinine'].apply(calculate_renal_sofa)
    df['SOFA_Cardiovascular'] = df['MAP'].apply(calculate_cardiovascular_sofa)
    df['SOFA_Total'] = df['SOFA_Renal'] + df['SOFA_Cardiovascular']
    
    # Select features
    feature_cols = ['Age', 'HR', 'O2Sat', 'Temp', 'SBP', 'MAP', 'DBP', 'Resp',
                   'BaseExcess', 'HCO3', 'FiO2', 'pH', 'PaCO2', 'Creatinine', 
                   'Glucose', 'Lactate', 'Platelets', 'Shock_Index', 'SOFA_Total']
    
    df_clean = df[feature_cols + ['Sepsis_3_Label']].dropna()
    print(f"✓ Clean dataset: {len(df_clean):,} records")
    
    # Prepare data
    X = df_clean[feature_cols]
    y = df_clean['Sepsis_3_Label']
    
    # Apply SMOTE balancing
    smote = SMOTE(sampling_strategy=8/9, random_state=2001)
    X_resampled, y_resampled = smote.fit_resample(X, y)
    
    # Train-test split
    X_train, X_test, y_train, y_test = train_test_split(
        X_resampled, y_resampled, test_size=0.2, stratify=y_resampled, random_state=2001
    )
    
    # Normalize features
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    
    # Train final model
    model = GradientBoostingClassifier(
        n_estimators=100, 
        max_depth=5, 
        learning_rate=0.1, 
        random_state=2001
    )
    model.fit(X_train_scaled, y_train)
    
    # Make predictions
    y_pred = model.predict(X_test_scaled)
    y_proba = model.predict_proba(X_test_scaled)[:, 1]
    
    # Calculate metrics
    auc = roc_auc_score(y_test, y_proba)
    
    # Confusion matrix
    tn, fp, fn, tp = confusion_matrix(y_test, y_pred).ravel()
    
    # Calculate sensitivity and specificity
    sensitivity = tp / (tp + fn) if (tp + fn) > 0 else 0
    specificity = tn / (tn + fp) if (tn + fp) > 0 else 0
    precision = tp / (tp + fp) if (tp + fp) > 0 else 0
    f1_score = 2 * (precision * sensitivity) / (precision + sensitivity) if (precision + sensitivity) > 0 else 0
    
    # Cross-validation for robustness
    cv_scores = cross_val_score(model, X_train_scaled, y_train, cv=10, scoring='roc_auc')
    cv_mean = cv_scores.mean()
    cv_std = cv_scores.std()
    
    # Success metrics targets
    targets = {
        'AUC': 0.87,
        'Sensitivity': 0.85,
        'F1_Score': 0.80,
        'Specificity': 0.80
    }
    
    achieved = {
        'AUC': auc,
        'Sensitivity': sensitivity,
        'F1_Score': f1_score,
        'Specificity': specificity
    }
    
    print("=== SUCCESS METRICS VALIDATION ===")
    print(f"{'Metric':<15} {'Target':<10} {'Achieved':<10} {'Status':<8}")
    print("-" * 50)
    
    all_passed = True
    for metric, target in targets.items():
        achieved_value = achieved[metric]
        status = "✅ PASS" if achieved_value >= target else "❌ FAIL"
        if achieved_value < target:
            all_passed = False
        print(f"{metric:<15} {target:<10.3f} {achieved_value:<10.3f} {status}")
    
    print("\n=== DETAILED PERFORMANCE METRICS ===")
    print(f"Primary Metrics:")
    print(f"- AUC: {auc:.4f} (Target: ≥0.87)")
    print(f"- Sensitivity: {sensitivity:.4f} ({sensitivity*100:.1f}%) (Target: ≥85%)")
    print(f"- Specificity: {specificity:.4f} ({specificity*100:.1f}%)")
    print(f"- Precision: {precision:.4f} ({precision*100:.1f}%)")
    print(f"- F1-Score: {f1_score:.4f} (Target: ≥0.80)")
    
    print(f"\nCross-Validation Results:")
    print(f"- 10-fold CV AUC: {cv_mean:.4f} ± {cv_std:.4f}")
    print(f"- CV Range: {cv_scores.min():.4f} - {cv_scores.max():.4f}")
    
    print(f"\nConfusion Matrix:")
    print(f"- True Positives: {tp}")
    print(f"- False Positives: {fp}")
    print(f"- True Negatives: {tn}")
    print(f"- False Negatives: {fn}")
    
    print(f"\nClinical Impact:")
    print(f"- Sepsis Cases Correctly Identified: {tp} ({sensitivity*100:.1f}%)")
    print(f"- False Alarms: {fp} ({(fp/(tn+fp))*100:.1f}% of negative cases)")
    print(f"- Missed Cases: {fn} ({(fn/(tp+fn))*100:.1f}% of positive cases)")
    
    # Additional validation metrics
    print(f"\n=== ADDITIONAL VALIDATION METRICS ===")
    
    # Calibration
    from sklearn.calibration import calibration_curve
    prob_true, prob_pred = calibration_curve(y_test, y_proba, n_bins=10)
    calibration_error = np.mean(np.abs(prob_true - prob_pred))
    print(f"- Calibration Error: {calibration_error:.4f}")
    
    # Precision-Recall AUC
    from sklearn.metrics import precision_recall_curve, auc as pr_auc
    precision_curve, recall_curve, _ = precision_recall_curve(y_test, y_proba)
    pr_auc_score = pr_auc(recall_curve, precision_curve)
    print(f"- PR-AUC: {pr_auc_score:.4f}")
    
    # Feature importance
    feature_importance = pd.DataFrame({
        'Feature': feature_cols,
        'Importance': model.feature_importances_
    }).sort_values('Importance', ascending=False)
    
    print(f"\n=== TOP 10 FEATURE IMPORTANCE ===")
    for i, row in feature_importance.head(10).iterrows():
        print(f"{i+1:2d}. {row['Feature']:<20}: {row['Importance']:.4f}")
    
    # Final assessment
    print(f"\n=== FINAL ASSESSMENT ===")
    if all_passed:
        print("🎉 ALL SUCCESS METRICS ACHIEVED!")
        print("✅ Ready for Xpecto'26 submission")
    else:
        print("⚠️  Some targets not met - need optimization")
    
    # Generate validation report
    validation_report = {
        'success_metrics_passed': all_passed,
        'auc': auc,
        'sensitivity': sensitivity,
        'specificity': specificity,
        'precision': precision,
        'f1_score': f1_score,
        'cv_auc_mean': cv_mean,
        'cv_auc_std': cv_std,
        'true_positives': tp,
        'false_positives': fp,
        'true_negatives': tn,
        'false_negatives': fn,
        'calibration_error': calibration_error,
        'pr_auc': pr_auc_score
    }
    
    # Save validation results
    pd.DataFrame([validation_report]).to_csv('/Users/apple/Downloads/iit/success_metrics_validation.csv', index=False)
    feature_importance.to_csv('/Users/apple/Downloads/iit/final_feature_importance.csv', index=False)
    
    print(f"\n✓ Validation results saved to success_metrics_validation.csv")
    print(f"✓ Feature importance saved to final_feature_importance.csv")
    
    return validation_report, feature_importance

if __name__ == "__main__":
    results, importance = validate_success_metrics()
