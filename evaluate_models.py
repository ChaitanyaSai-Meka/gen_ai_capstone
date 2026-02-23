import pandas as pd
import numpy as np
from app import load_data, train_models

df = load_data()
lr_model, dt_model, rf_model, X_train, X_test, y_train, y_test = train_models(df)

input_data = pd.DataFrame({
    'person_age': [30],
    'person_gender': [1],
    'person_education': ['High School'],
    'person_income': [60000],
    'person_emp_exp': [5],
    'person_home_ownership': ['RENT'],
    'loan_amnt': [10000],
    'loan_intent': ['PERSONAL'],
    'loan_int_rate': [10.0],
    'loan_percent_income': [10000 / 60000],
    'cb_person_cred_hist_length': [5],
    'credit_score': [650],
    'previous_loan_defaults_on_file': [0]
})

print("Logistic Regression Prob:", lr_model.predict_proba(input_data)[0][1])
print("Decision Tree Prob:", dt_model.predict_proba(input_data)[0][1])
print("Random Forest Prob:", rf_model.predict_proba(input_data)[0][1])

lr_preprocessor = lr_model.named_steps['preprocessor']

numeric_features = [
    'person_age', 'person_income', 'person_emp_exp',
    'loan_amnt', 'loan_int_rate', 'loan_percent_income',
    'cb_person_cred_hist_length', 'credit_score',
    'previous_loan_defaults_on_file', 'person_gender'
]
ordinal_features = ['person_education']
nominal_features = ['person_home_ownership', 'loan_intent']

ohe_feature_names = lr_preprocessor.named_transformers_['nominal'].get_feature_names_out(nominal_features).tolist()
all_feature_names = numeric_features + ordinal_features + ohe_feature_names

lr_coef = lr_model.named_steps['classifier'].coef_[0]
X_processed = lr_preprocessor.transform(input_data)[0]

print("\nContributing features for LR (Score * Coef):")
contributions = [(all_feature_names[i], X_processed[i] * lr_coef[i], X_processed[i], lr_coef[i]) for i in range(len(all_feature_names))]
contributions.sort(key=lambda x: x[1], reverse=True)
for feat, cont, val, coef in contributions:
    print(f"{feat:35} contribution: {cont:8.4f} (val: {val:8.4f}, coef: {coef:8.4f})")
    
dt_clf = dt_model.named_steps['classifier']
dt_preprocessor = dt_model.named_steps['preprocessor']
X_dt_processed = dt_preprocessor.transform(input_data)[0]

print("\nDecision Tree Path:")
node_indicator = dt_clf.decision_path(X_dt_processed.reshape(1, -1))
leaf_id = dt_clf.apply(X_dt_processed.reshape(1, -1))[0]
feature = dt_clf.tree_.feature
threshold = dt_clf.tree_.threshold

node_index = node_indicator.indices[node_indicator.indptr[0]:node_indicator.indptr[1]]

for node_id in node_index:
    if leaf_id == node_id:
        classes = dt_clf.classes_
        value = dt_clf.tree_.value[node_id][0]
        prob = value[1] / sum(value)
        print(f"Leaf node {node_id}, class distribution: {value}, probability of default: {prob:.4f}")
        continue
    
    if (X_dt_processed[feature[node_id]] <= threshold[node_id]):
        threshold_sign = "<="
    else:
        threshold_sign = ">"
        
    print(f"Node {node_id}: {all_feature_names[feature[node_id]]} = {X_dt_processed[feature[node_id]]:.2f} {threshold_sign} {threshold[node_id]:.2f}")

print("\nRandom Forest Path (Estimator 0):")
rf_clf = rf_model.named_steps['classifier'].estimators_[0]
rf_preprocessor = rf_model.named_steps['preprocessor']
X_rf_processed = rf_preprocessor.transform(input_data)[0]

node_indicator_rf = rf_clf.decision_path(X_rf_processed.reshape(1, -1))
leaf_id_rf = rf_clf.apply(X_rf_processed.reshape(1, -1))[0]
feature_rf = rf_clf.tree_.feature
threshold_rf = rf_clf.tree_.threshold

node_index_rf = node_indicator_rf.indices[node_indicator_rf.indptr[0]:node_indicator_rf.indptr[1]]

for node_id in node_index_rf:
    if leaf_id_rf == node_id:
        value = rf_clf.tree_.value[node_id][0]
        prob = value[1] / sum(value)
        print(f"Leaf node {node_id}, class distribution: {value}, probability of default: {prob:.4f}")
        continue
    
    if (X_rf_processed[feature_rf[node_id]] <= threshold_rf[node_id]):
        threshold_sign = "<="
    else:
        threshold_sign = ">"
        
    print(f"Node {node_id}: {all_feature_names[feature_rf[node_id]]} = {X_rf_processed[feature_rf[node_id]]:.2f} {threshold_sign} {threshold_rf[node_id]:.2f}")
