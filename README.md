# 🏦 Loan Default Prediction System

> A machine learning dashboard for predicting the probability of loan default, built with Streamlit and scikit-learn.

---

## 📌 Problem Statement

Financial institutions face significant losses when borrowers fail to repay loans. This project builds a predictive system that classifies applicants into **Low**, **Medium**, and **High** default risk categories based on their demographic and financial profiles — enabling faster, data-driven lending decisions.

---

## 🚀 Live Demo

> 🔗 **[Deployed App Link]** — *(Replace with your Streamlit Cloud / Hugging Face URL)*

---

## 🗂️ Project Structure

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
│   └── random_forest.py        # Random Forest model training
│
└── notebooks/
    ├── preprocessing.ipynb         # EDA — charts, distributions, correlation matrix
    ├── logistic_regression.ipynb   # LR training and evaluation
    ├── decision_tree.ipynb         # DT training and evaluation
    └── random_forest.ipynb         # RF training and evaluation
```

---

## ⚙️ Setup & Installation

### Prerequisites
- Python 3.10+

### 1. Clone the Repository
```bash
git clone https://github.com/<your-username>/gen_ai_capstone.git
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

## 🧠 Models Implemented

| Model | Description |
|---|---|
| Logistic Regression | Linear model that predicts probability of default based on weighted features |
| Decision Tree | Tree-based model that splits data using feature thresholds to make decisions |
| Random Forest | Ensemble of 100 Decision Trees for more stable and accurate predictions |

All models use `class_weight='balanced'` to handle the class imbalance (78% No Default vs 22% Default).

---

## 📊 Dataset

- **Source:** `loan_data.csv`
- **Records:** 45,000
- **Features:** 14 (demographic + financial + credit history)
- **Target:** `loan_status` (1 = Default, 0 = No Default)

---

## 👥 Team

| Name | Roll No | Batch |
|---|---|---|
| [Team Member 1] | [Roll No] | [Batch] |
| [Team Member 2] | [Roll No] | [Batch] |
| [Team Member 3] | [Roll No] | [Batch] |
| [Team Member 4] | [Roll No] | [Batch] |

---

## ✅ Academic Integrity Declaration

> We, the above-named team members, hereby affirm that the core logic, model architecture, preprocessing pipeline, and Streamlit application code in this repository are our own original work. No Generative AI tool was used to directly produce the core implementation. All use of AI tools was limited to research, understanding concepts, and debugging, in accordance with course guidelines.

---

## 📄 License

This project was developed as part of the **Intro to GenAI Capstone Project** at **NST Sonipat**.
