# 🚨 Churn Alert — E-commerce Customer Churn Prediction

> A complete ML pipeline that predicts customer churn in e-commerce, built as a Streamlit application with end-to-end data science workflow.

---

## 📋 Project Structure

```
Project-User-Churn/
├── data/
│   └── ecommerce_churn.csv        ← 📥 Place Kaggle dataset here
├── src/
│   ├── data_cleaning_01.py        # Step 1: Type fixes & typo correction
│   ├── missing_analysis_02.py     # Step 2: MCAR/MAR/MNAR analysis
│   ├── eda_03.py                  # Step 3: Deep EDA
│   ├── rfm_segmentation_04.py    # Step 4: SQLite RFM scoring
│   ├── preprocessing_05.py        # Step 5: OHE + KNN + SMOTE
│   ├── modelling_06.py            # Step 6: XGBoost + Optuna + SHAP
│   ├── clustering_07.py           # Step 7: K-Means clustering
│   ├── risk_tiers_08.py           # Step 8: Risk tier assignment
│   └── survival_analysis_09.py   # Step 9: KM + Cox PH
├── models/                        # Saved .pkl files
├── outputs/                       # Generated plots (.png)
├── app/
│   └── streamlit_app.py           # 7-page Streamlit dashboard
├── run_pipeline.py                # ▶ Master runner
└── requirements.txt
```

---

## 🚀 Quick Start

### 1. Download the Dataset
Go to: https://www.kaggle.com/datasets/ankitverma2010/ecommerce-customer-churn-analysis-and-prediction

Download and place the CSV in the `data/` folder as:
```
data/ecommerce_churn.csv
```

### 2. Install Dependencies
```bash
pip install -r requirements.txt
```

### 3. Run the Full Pipeline
```bash
python run_pipeline.py
```
This runs all 9 analysis steps (~20–30 min depending on hardware, mostly Optuna tuning).

### 4. Launch the Dashboard
```bash
streamlit run app/streamlit_app.py
```

---

## 🏗️ Pipeline Steps

| # | Script | Description | Key Output |
|---|--------|-------------|------------|
| 1 | `data_cleaning_01.py` | Type casting, typo fixes | `ecommerce_churn_clean.csv` |
| 2 | `missing_analysis_02.py` | Missingness matrix, MCAR/MAR/MNAR | Plots in `outputs/` |
| 3 | `eda_03.py` | Pearson, Cramér's V, profiling | Plots in `outputs/` |
| 4 | `rfm_segmentation_04.py` | SQLite NTILE RFM scoring | `ecommerce_churn_rfm.csv` |
| 5 | `preprocessing_05.py` | OHE + KNN Imputer + SMOTE | `X_train.csv`, `X_test.csv` |
| 6 | `modelling_06.py` | XGBoost + Optuna (40 trials) + SHAP | `xgb_model.pkl` |
| 7 | `clustering_07.py` | K-Means (Elbow + Silhouette) | `cluster_profiles.csv` |
| 8 | `risk_tiers_08.py` | High/Medium/Low risk tiers | `risk_table.csv` |
| 9 | `survival_analysis_09.py` | Kaplan-Meier + Cox PH | `cox_summary.csv` |

---

## 📊 Dashboard Pages

| Page | Content |
|------|---------|
| 📊 **Overview** | KPI cards, churn distribution, pipeline summary |
| 🔍 **EDA** | Interactive tenure/satisfaction plots, correlations, customer profiles |
| 💎 **RFM Segmentation** | Segment distribution, churn rates, RFM score analysis |
| 🤖 **Model Results** | ROC curve, confusion matrix, SHAP feature importance |
| ⚠️ **Risk Dashboard** | Tier distribution, action matrix, top at-risk customers |
| 📈 **Survival Analysis** | Kaplan-Meier curves, Cox PH hazard ratios, individual curves |
| 🔎 **Customer Lookup** | Real-time prediction for any customer profile |

---

## 🎯 Key Results (Blog Benchmarks)

| Metric | Result |
|--------|--------|
| ROC-AUC | ~98.88% |
| F1-Score | ~92.23% |
| Precision | ~93.99% |
| Recall | ~90.53% |
| Cox PH Concordance | ~0.77 |

---

## 🛠️ Tech Stack

`pandas` · `numpy` · `matplotlib` · `seaborn` · `plotly` · `missingno`  
`scikit-learn` · `imbalanced-learn` · `xgboost` · `optuna` · `shap`  
`lifelines` · `streamlit` · `sqlite3` · `joblib`

---

## 💡 Key Insights from the Blog

1. **"Death Valley"**: 0–3 month customers churn at 50%+ → Fix onboarding
2. **5-star Paradox**: Highest satisfaction score → highest churn (one-time buyers)
3. **Complaint = 🚨**: 31.7% churn vs ~10% for non-complainers (Cox HR = 2.17)
4. **Recent Customers paradox**: Newest segment has highest RFM churn rate
5. **CashbackAmount**: The #1 retention anchor (negative SHAP impact on churn)

---

*Based on the project: [Churn Alert — AIO Conquer Blog](https://aioconquer.aivietnam.edu.vn/posts/churn-alert-xay-dung-mo-hinh-du-doan-churn-khach-hang-trong-e-commerce)*
