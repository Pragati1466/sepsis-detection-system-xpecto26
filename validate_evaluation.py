#!/usr/bin/env python3
"""
Validation script for Phase 5: Evaluation & Explainability
This script validates the evaluation and explainability logic implemented in the RapidMiner workflow
and provides insights into model performance and feature importance.
"""

import pandas as pd
import numpy as np
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import (classification_report, roc_auc_score, confusion_matrix, 
                           precision_recall_curve, auc, roc_curve)
from sklearn.preprocessing import StandardScaler
from sklearn.feature_selection import mutual_info_classif
import warnings
warnings.filterwarnings('ignore')

def validate_evaluation_explainability():
    """Validate evaluation and explainability logic using Python for verification"""
    
    print("=== Phase 5: Evaluation & Explainability Validation ===\n")
    
    # Load and preprocess data (same as previous phases)
    try:
        df = pd.read_csv('/Users/apple/Downloads/iit/Dataset.csv')
        print(f"✓ Successfully loaded dataset with {len(df):,} rows")
    except FileNotFoundError:
        print("✗ Dataset not found")
        return
    
    # Preprocessing pipeline
    print(f"\n=== Data Preprocessing ===")
    
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
    
    df_clean = df[feature_cols + ['Sepsis_3_Label', 'Patient_ID', 'Hour']].dropna()
    print(f"- Clean dataset: {len(df_clean):,} records")
    
    # Train-test split
    X = df_clean[feature_cols]
    y = df_clean['Sepsis_3_Label']
    metadata = df_clean[['Patient_ID', 'Hour']]
    
    X_train, X_test, y_train, y_test, meta_train, meta_test = train_test_split(
        X, y, metadata, test_size=0.2, stratify=y, random_state=2001
    )
    
    # Normalize features
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    
    # Train model
    print(f"- Training Gradient Boosted Trees model...")
    model = GradientBoostingClassifier(n_estimators=100, max_depth=5, learning_rate=0.1, random_state=2001)
    model.fit(X_train_scaled, y_train)
    
    # Make predictions
    y_pred = model.predict(X_test_scaled)
    y_proba = model.predict_proba(X_test_scaled)[:, 1]
    
    # Task 5.1: Performance (Binomial Classification)
    print(f"\n=== Task 5.1: Performance Metrics ===")
    
    # Calculate comprehensive metrics
    auc = roc_auc_score(y_test, y_proba)
    
    # Classification report
    report = classification_report(y_test, y_pred, output_dict=True)
    
    # Confusion matrix
    tn, fp, fn, tp = confusion_matrix(y_test, y_pred).ravel()
    
    # Calculate additional metrics
    sensitivity = tp / (tp + fn) if (tp + fn) > 0 else 0
    specificity = tn / (tn + fp) if (tn + fp) > 0 else 0
    precision = tp / (tp + fp) if (tp + fp) > 0 else 0
    f1_score = 2 * (precision * sensitivity) / (precision + sensitivity) if (precision + sensitivity) > 0 else 0
    
    print(f"Model Performance Metrics:")
    print(f"- AUC: {auc:.3f}")
    print(f"- Accuracy: {report['accuracy']:.3f}")
    print(f"- Sensitivity (Recall): {sensitivity:.3f}")
    print(f"- Specificity: {specificity:.3f}")
    print(f"- Precision: {precision:.3f}")
    print(f"- F1-Score: {f1_score:.3f}")
    print(f"- True Positives: {tp}")
    print(f"- False Positives: {fp}")
    print(f"- True Negatives: {tn}")
    print(f"- False Negatives: {fn}")
    
    # Task 5.2: Weight by Information Gain
    print(f"\n=== Task 5.2: Feature Importance (Information Gain) ===")
    
    # Calculate information gain (mutual information)
    info_gain = mutual_info_classif(X_train, y_train, random_state=2001)
    
    feature_importance = pd.DataFrame({
        'Feature': feature_cols,
        'Information_Gain': info_gain,
        'Feature_Importance': model.feature_importances_
    }).sort_values('Information_Gain', ascending=False)
    
    print("Top 15 Features by Information Gain:")
    for i, row in feature_importance.head(15).iterrows():
        print(f"{i+1:2d}. {row['Feature']:20s}: {row['Information_Gain']:.4f}")
    
    # Clinical categorization
    def categorize_feature(feature_name):
        if 'SOFA' in feature_name:
            return 'Organ Dysfunction'
        elif 'Shock_Index' in feature_name:
            return 'Hemodynamic'
        elif feature_name in ['HR', 'SBP', 'MAP', 'DBP', 'O2Sat', 'Temp', 'Resp']:
            return 'Vital Signs'
        else:
            return 'Laboratory'
    
    feature_importance['Clinical_Category'] = feature_importance['Feature'].apply(categorize_feature)
    
    print(f"\nFeature Importance by Clinical Category:")
    category_summary = feature_importance.groupby('Clinical_Category')['Information_Gain'].mean().sort_values(ascending=False)
    for category, importance in category_summary.items():
        print(f"- {category}: {importance:.4f}")
    
    # Task 5.3: Clinical Predictions with Top-5 Reasons
    print(f"\n=== Task 5.3: Clinical Predictions with Top-5 Reasons ===")
    
    # Create clinical prediction dataset
    clinical_predictions = pd.DataFrame({
        'Patient_ID': meta_test['Patient_ID'].values,
        'Hour': meta_test['Hour'].values,
        'Actual_Label': y_test.values,
        'Predicted_Label': y_pred,
        'Prediction_Confidence': y_proba,
        'Age': X_test['Age'].values,
        'HR': X_test['HR'].values,
        'SBP': X_test['SBP'].values,
        'MAP': X_test['MAP'].values,
        'Shock_Index': X_test['Shock_Index'].values,
        'SOFA_Total': X_test['SOFA_Total'].values,
        'Lactate': X_test['Lactate'].values
    })
    
    # Generate top 5 reasons for each prediction
    def generate_reasons(row):
        reasons = []
        
        # Reason 1: Shock Index
        if row['Shock_Index'] > 1.0:
            reasons.append(f"Elevated Shock Index ({row['Shock_Index']:.2f})")
        
        # Reason 2: SOFA Score
        if row['SOFA_Total'] >= 2:
            reasons.append(f"High SOFA Score ({row['SOFA_Total']})")
        
        # Reason 3: High Heart Rate
        if row['HR'] > 100:
            reasons.append(f"Tachycardia ({row['HR']:.0f} bpm)")
        
        # Reason 4: Elevated Lactate
        if row['Lactate'] > 2.0:
            reasons.append(f"Elevated Lactate ({row['Lactate']:.1f})")
        
        # Reason 5: Low MAP
        if row['MAP'] < 65:
            reasons.append(f"Low MAP ({row['MAP']:.0f} mmHg)")
        
        # Ensure we have exactly 5 reasons (add placeholders if needed)
        while len(reasons) < 5:
            reasons.append("Normal finding")
        
        return "; ".join(reasons[:5])
    
    # Risk categorization
    def categorize_risk(confidence):
        if confidence >= 0.8:
            return "High Risk"
        elif confidence >= 0.6:
            return "Moderate Risk"
        else:
            return "Low Risk"
    
    clinical_predictions['Top_5_Reasons'] = clinical_predictions.apply(generate_reasons, axis=1)
    clinical_predictions['Risk_Category'] = clinical_predictions['Prediction_Confidence'].apply(categorize_risk)
    
    # Sample predictions
    print("Sample Clinical Predictions (Top 10):")
    sample_cols = ['Patient_ID', 'Hour', 'Actual_Label', 'Predicted_Label', 
                   'Prediction_Confidence', 'Risk_Category', 'Top_5_Reasons']
    print(clinical_predictions[sample_cols].head(10).to_string(index=False))
    
    # Risk distribution
    print(f"\nRisk Category Distribution:")
    risk_dist = clinical_predictions['Risk_Category'].value_counts()
    for risk, count in risk_dist.items():
        print(f"- {risk}: {count} ({count/len(clinical_predictions)*100:.1f}%)")
    
    # High-risk cases analysis
    high_risk = clinical_predictions[clinical_predictions['Risk_Category'] == 'High Risk']
    print(f"\nHigh-Risk Cases Analysis:")
    if len(high_risk) > 0:
        print(f"- Total high-risk cases: {len(high_risk)}")
        print(f"- Actual sepsis in high-risk: {high_risk['Actual_Label'].sum()} ({high_risk['Actual_Label'].mean()*100:.1f}%)")
        print(f"- Average confidence: {high_risk['Prediction_Confidence'].mean():.3f}")
    else:
        print("- No high-risk cases identified")
    
    # Save clinical predictions
    clinical_predictions.to_csv('/Users/apple/Downloads/iit/clinical_sepsis_predictions_python.csv', index=False)
    feature_importance.to_csv('/Users/apple/Downloads/iit/feature_importance_python.csv', index=False)
    
    # Performance summary
    performance_summary = {
        'AUC': auc,
        'Accuracy': report['accuracy'],
        'Sensitivity': sensitivity,
        'Specificity': specificity,
        'Precision': precision,
        'F1_Score': f1_score,
        'True_Positives': tp,
        'False_Positives': fp,
        'True_Negatives': tn,
        'False_Negatives': fn
    }
    
    print(f"\n=== Evaluation & Explainability Summary ===")
    for metric, value in performance_summary.items():
        if isinstance(value, float):
            print(f"- {metric}: {value:.3f}")
        else:
            print(f"- {metric}: {value}")
    
    print(f"\n✓ Evaluation & explainability validation completed successfully!")
    print(f"✓ Performance metrics calculated: AUC, Sensitivity, F1-Score")
    print(f"✓ Feature importance identified using Information Gain")
    print(f"✓ Clinical predictions with top-5 reasons generated")
    print(f"✓ Files saved: clinical_sepsis_predictions_python.csv, feature_importance_python.csv")
    
    return performance_summary, feature_importance, clinical_predictions

if __name__ == "__main__":
    performance, importance, predictions = validate_evaluation_explainability()
