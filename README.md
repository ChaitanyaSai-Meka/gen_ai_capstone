# Loan Default Prediction System

> A machine learning dashboard for predicting the probability of loan default, built with Streamlit and scikit-learn.

**Live App:** [https://genaicapstone-m8ezlrzqqq8ctmmfgsh6ur.streamlit.app](https://genaicapstone-m8ezlrzqqq8ctmmfgsh6ur.streamlit.app/)

**Repository:** [https://github.com/ChaitanyaSai-Meka/gen_ai_capstone](https://github.com/ChaitanyaSai-Meka/gen_ai_capstone)

---

## Problem Statement

Financial institutions face significant losses when borrowers fail to repay loans. This project builds a predictive system that classifies applicants into **Low**, **Medium**, and **High** default risk categories based on their demographic and financial profiles — enabling faster, data-driven lending decisions.

---

## Key Sub-Features

1. **Custom Data Preprocessing Pipeline** — Automated outlier capping (99th percentile), ordinal encoding for education levels, one-hot encoding for categorical features, and binary encoding for gender/defaults. All steps are modularised in `preprocessing.py`.
2. **Real-Time Inference via Streamlit Dashboard** — Users input applicant details through a form and get instant default probability predictions with risk classification (Low/Medium/High). The app also displays the top 5 factors influencing each prediction.
3. **Multi-Model Comparison Dashboard** — Side-by-side accuracy bar charts, overlaid ROC curves, and a consolidated metrics table (Accuracy, F1-Score, ROC-AUC) to compare all three models in one view.

---

## Project Structure

```
gen_ai_capstone/
│
├── app.py                  # Streamlit frontend — UI, forms, predictions, charts
├── preprocessing.py        # Data loading, cleaning, outlier handling, encoding
├── loan_data.csv           # Dataset (45,000 records, 14 features)
├── README.md
│
├── models/
│   ├── logistic_regression.py  # Logistic Regression model training
│   ├── decision_tree.py        # Decision Tree model training
│   └── xgboost_model.py        # XGBoost model training
│
└── notebooks/
    ├── preprocessing.ipynb         # EDA — charts, distributions, correlation matrix
    ├── logistic_regression.ipynb   # LR training and evaluation
    ├── decision_tree.ipynb         # DT training and evaluation
    └── xgboost.ipynb               # XGBoost training and evaluation
```

---

## Setup & Installation

### Prerequisites
- Python 3.10+

### 1. Clone the Repository
```bash
git clone https://github.com/ChaitanyaSai-Meka/gen_ai_capstone.git
cd gen_ai_capstone
```

### 2. Install Dependencies
```bash
pip install -r requirements.txt
```

### 3. Run the App
```bash
streamlit run app.py
```

---

## Models Implemented

| Model | Description | Accuracy | F1-Score | ROC-AUC |
|---|---|---|---|---|
| Logistic Regression | Linear baseline model using sigmoid function | 85.25% | 0.7389 | 0.9523 |
| Decision Tree | Non-linear tree-based model with depth control | 88.03% | 0.7714 | 0.9627 |
| **XGBoost (Best)** | **Gradient boosted ensemble — best performer** | **91.27%** | **0.8239** | **0.9763** |

All models handle class imbalance (78% No Default vs 22% Default) using `class_weight='balanced'` or `scale_pos_weight`.

---

## Dataset

- **Source:** [Loan Approval Classification Data — Kaggle](https://www.kaggle.com/datasets/taweilo/loan-approval-classification-data)
- **Records:** ~45,000 (43,691 after cleaning)
- **Features:** 14 (demographic + financial + credit history)
- **Target:** `loan_status` (1 = Default, 0 = No Default)
- **Class Split:** 78% No Default, 22% Default (imbalanced)

---

## Team

| Name | Role |
|---|---|
| Dhanvin Vadlamudi | Team Lead |
| Meka Chaitanya Sai | Team Member |
| Killi Akshith Kumar | Team Member |
| Akhil Nath Reddy | Team Member |

---

## Academic Integrity Declaration

> We, the above-named team members, hereby affirm that the core logic, model architecture, preprocessing pipeline, and Streamlit application code in this repository are our own original work. No Generative AI tool was used to directly produce the core implementation. All use of AI tools was limited to research, understanding concepts, and debugging, in accordance with course guidelines.

---

This project was developed as part of the **Intro to GenAI Capstone Project** at **NST Sonipat**.
