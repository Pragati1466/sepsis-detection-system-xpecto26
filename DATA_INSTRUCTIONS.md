# Dataset Instructions for SepsisNet

## 📊 Dataset Information

The **MIMIC-IV Dataset** used in this project is too large for GitHub (146.69 MB). Here's how to obtain and set up the data:

## 🔗 Data Source

### Official MIMIC-IV Database
- **Repository**: https://physionet.org/content/mimiciv/2.2/
- **License**: PhysioNet Credited User License
- **Citation**: Johnson AEW, et al. MIMIC-IV, a freely accessible electronic health record dataset. Sci Data. 2023.

### Required Files
1. **Download**: `mimic-iv-2.2.0.zip` from PhysioNet
2. **Extract**: ICU patient records
3. **Process**: Use the preprocessing pipeline in our workflows

## 🚀 Quick Setup

### Option 1: Download Full Dataset
```bash
# 1. Create PhysioNet account (free)
# 2. Download MIMIC-IV v2.2.0
# 3. Extract to your local directory
# 4. Run our preprocessing workflow
```

### Option 2: Sample Dataset (For Testing)
```python
# Use our sample clinical predictions
python generate_sample_data.py
```

## 📁 File Structure Expected
```
/Users/apple/Downloads/iit/
├── Dataset.csv                    # MIMIC-IV processed data (146.69 MB)
├── sepsis_evaluation_workflow.rmp # Main RapidMiner workflow
├── final_metrics_validation.py    # Validation script
└── sample_clinical_predictions.csv # Sample outputs (1.6 KB)
```

## ⚙️ Data Processing Pipeline

Our RapidMiner workflow handles:
- **Adult filtering** (Age ≥ 18)
- **Sepsis-3 labeling** (6-hour pre-onset)
- **Missing value imputation** (vitals: forward-fill, labs: KNN)
- **Feature engineering** (Shock Index, SOFA scores, temporal windows)

## 🎯 Data Specifications

- **Total Records**: 1,552,210 ICU patient-hours
- **Unique Patients**: 40,336
- **Clinical Variables**: 44 per observation
- **Time Period**: 2008-2019
- **Sepsis Prevalence**: 1.57%

## 📞 Support

For dataset issues:
- **PhysioNet**: https://physionet.org/support/
- **Project Issues**: GitHub repository issues

---

*Note: The dataset is excluded from GitHub due to size limits. Follow the instructions above to obtain the data.*
