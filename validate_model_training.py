#!/usr/bin/env python3
"""
Validation script for Phase 4: Model Training & Balancing
This script validates the model training logic implemented in the RapidMiner workflow
and provides insights into the class balancing and model performance expectations.
"""

import pandas as pd
import numpy as np
from sklearn.ensemble import GradientBoostingClassifier, RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split, StratifiedKFold, cross_val_score
from sklearn.metrics import classification_report, roc_auc_score, confusion_matrix
from sklearn.preprocessing import StandardScaler
from imblearn.over_sampling import SMOTE
import warnings
warnings.filterwarnings('ignore')

def validate_model_training():
    """Validate the model training logic using Python for verification"""
    
    print("=== Phase 4: Model Training & Balancing Validation ===\n")
    
    # Load and preprocess data
    try:
        df = pd.read_csv('/Users/apple/Downloads/iit/Dataset.csv')
        print(f"✓ Successfully loaded dataset with {len(df):,} rows")
    except FileNotFoundError:
        print("✗ Dataset not found")
        return
    
    # Apply preprocessing pipeline
    print(f"\n=== Data Preprocessing ===")
    
    # Filter adults
    df = df[df['Age'] >= 18].copy()
    print(f"- Adult patients: {len(df):,}")
    
    # Generate Sepsis-3 labels (keep both classes for training)
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
    print(f"- Feature engineering...")
    df['Shock_Index'] = np.where(df['SBP'] > 0, df['HR'] / df['SBP'], np.nan)
    
    # SOFA scores
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
    df['SOFA_Total'] = df['SOFA_Renal'] + df['SOFA_Cardiovascular']  # Simplified
    
    # Select features for modeling
    feature_cols = ['Age', 'HR', 'O2Sat', 'Temp', 'SBP', 'MAP', 'DBP', 'Resp',
                   'BaseExcess', 'HCO3', 'FiO2', 'pH', 'PaCO2', 'Creatinine', 
                   'Glucose', 'Lactate', 'Platelets', 'Shock_Index', 'SOFA_Total']
    
    # Remove rows with missing target or features
    df_clean = df[feature_cols + ['Sepsis_3_Label']].dropna()
    print(f"- Clean dataset: {len(df_clean):,} records")
    
    # Task 4.1: Class Balance Analysis
    print(f"\n=== Task 4.1: Class Balance Analysis ===")
    
    class_counts = df_clean['Sepsis_3_Label'].value_counts()
    print(f"Original class distribution:")
    print(f"- Non-sepsis (0): {class_counts[0]:,} ({class_counts[0]/len(df_clean)*100:.1f}%)")
    print(f"- Sepsis (1): {class_counts[1]:,} ({class_counts[1]/len(df_clean)*100:.1f}%)")
    
    imbalance_ratio = class_counts[0] / class_counts[1]
    print(f"- Imbalance ratio: {imbalance_ratio:.1f}:1")
    
    # Task 4.1: SMOTE Balancing
    print(f"\n=== SMOTE Balancing (8:1 ratio) ===")
    
    X = df_clean[feature_cols]
    y = df_clean['Sepsis_3_Label']
    
    # Apply SMOTE with 8:1 ratio
    smote = SMOTE(sampling_strategy=8/9, random_state=2001)  # 8:1 ratio means 8 non-sepsis for each sepsis
    X_resampled, y_resampled = smote.fit_resample(X, y)
    
    balanced_counts = pd.Series(y_resampled).value_counts()
    print(f"After SMOTE balancing:")
    print(f"- Non-sepsis (0): {balanced_counts[0]:,}")
    print(f"- Sepsis (1): {balanced_counts[1]:,}")
    print(f"- New ratio: {balanced_counts[0]/balanced_counts[1]:.1f}:1")
    print(f"- Total samples: {len(X_resampled):,}")
    
    # Task 4.2: Train-Test Split
    print(f"\n=== Task 4.2: Train-Test Split (80/20) ===")
    
    # Use the balanced dataset
    X_train, X_test, y_train, y_test = train_test_split(
        X_resampled, y_resampled, test_size=0.2, stratify=y_resampled, random_state=2001
    )
    
    print(f"Training set: {len(X_train):,} samples")
    print(f"Test set: {len(X_test):,} samples")
    print(f"Training class distribution: {pd.Series(y_train).value_counts().to_dict()}")
    print(f"Test class distribution: {pd.Series(y_test).value_counts().to_dict()}")
    
    # Normalize features
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    
    # Task 4.3: 10-fold Stratified Cross-Validation
    print(f"\n=== Task 4.3: 10-fold Stratified Cross-Validation ===")
    
    cv = StratifiedKFold(n_splits=10, shuffle=True, random_state=2001)
    
    models = {
        'Gradient Boosted Trees': GradientBoostingClassifier(n_estimators=100, max_depth=5, learning_rate=0.1, random_state=2001),
        'Random Forest': RandomForestClassifier(n_estimators=100, max_depth=10, random_state=2001),
        'Logistic Regression': LogisticRegression(max_iter=100, random_state=2001)
    }
    
    cv_results = {}
    for name, model in models.items():
        scores = cross_val_score(model, X_train_scaled, y_train, cv=cv, scoring='roc_auc')
        cv_results[name] = {
            'mean_auc': scores.mean(),
            'std_auc': scores.std(),
            'scores': scores
        }
        print(f"- {name}: AUC = {scores.mean():.3f} ± {scores.std():.3f}")
    
    # Task 4.4: Model Training and Evaluation
    print(f"\n=== Task 4.4: Model Training & Evaluation ===")
    
    final_results = {}
    
    for name, model in models.items():
        # Train model
        model.fit(X_train_scaled, y_train)
        
        # Predictions
        y_pred = model.predict(X_test_scaled)
        y_proba = model.predict_proba(X_test_scaled)[:, 1]
        
        # Metrics
        auc = roc_auc_score(y_test, y_proba)
        
        # Detailed classification report
        report = classification_report(y_test, y_pred, output_dict=True)
        
        final_results[name] = {
            'AUC': auc,
            'Accuracy': report['accuracy'],
            'Precision_0': report['0']['precision'],
            'Recall_0': report['0']['recall'],
            'F1_0': report['0']['f1-score'],
            'Precision_1': report['1']['precision'],
            'Recall_1': report['1']['recall'],
            'F1_1': report['1']['f1-score']
        }
        
        print(f"\n{name} Results:")
        print(f"- AUC: {auc:.3f}")
        print(f"- Accuracy: {report['accuracy']:.3f}")
        print(f"- Non-sepsis: Precision={report['0']['precision']:.3f}, Recall={report['0']['recall']:.3f}, F1={report['0']['f1-score']:.3f}")
        print(f"- Sepsis: Precision={report['1']['precision']:.3f}, Recall={report['1']['recall']:.3f}, F1={report['1']['f1-score']:.3f}")
    
    # Model Comparison
    print(f"\n=== Model Comparison ===")
    
    comparison_df = pd.DataFrame(final_results).T
    print(comparison_df.round(3))
    
    # Best model by AUC
    best_model = comparison_df['AUC'].idxmax()
    print(f"\nBest performing model: {best_model} (AUC: {comparison_df.loc[best_model, 'AUC']:.3f})")
    
    # Feature importance (for tree-based models)
    print(f"\n=== Feature Importance Analysis ===")
    
    # Get feature importance from Random Forest
    rf_model = models['Random Forest']
    rf_model.fit(X_train_scaled, y_train)
    
    feature_importance = pd.DataFrame({
        'feature': feature_cols,
        'importance': rf_model.feature_importances_
    }).sort_values('importance', ascending=False)
    
    print("Top 10 Most Important Features:")
    for i, row in feature_importance.head(10).iterrows():
        print(f"{i+1}. {row['feature']}: {row['importance']:.3f}")
    
    # Validation Summary
    print(f"\n=== Model Training Validation Summary ===")
    
    validation_summary = {
        'total_samples': len(df_clean),
        'original_imbalance_ratio': imbalance_ratio,
        'balanced_samples': len(X_resampled),
        'training_samples': len(X_train),
        'test_samples': len(X_test),
        'best_model': best_model,
        'best_auc': comparison_df.loc[best_model, 'AUC'],
        'cv_folds': 10,
        'models_tested': len(models)
    }
    
    for key, value in validation_summary.items():
        if isinstance(value, float):
            print(f"- {key}: {value:.3f}")
        else:
            print(f"- {key}: {value}")
    
    print(f"\n✓ Model training validation completed successfully!")
    print(f"✓ All components validated: SMOTE balancing, train-test split, cross-validation, and model training")
    print(f"✓ Best model identified: {best_model}")
    
    return final_results, feature_importance

if __name__ == "__main__":
    results, importance = validate_model_training()
