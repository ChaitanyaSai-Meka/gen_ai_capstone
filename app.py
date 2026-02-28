import streamlit as st
import pandas as pd
import numpy as np
from sklearn.metrics import accuracy_score, f1_score, roc_auc_score, roc_curve, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns

st.set_page_config(
    page_title="Loan Default Prediction",
    page_icon="bar-chart",
    layout="wide",
    initial_sidebar_state="expanded"
)

from preprocessing import load_and_clean_data, preprocess_features
import models.logistic_regression as lr_module
import models.decision_tree as dt_module
import models.xgboost_model as xgb_module


@st.cache_data
def load_data():
    return load_and_clean_data('loan_data.csv')


@st.cache_resource
def train_all_models(_df):
    lr_result = lr_module.train(_df)
    dt_result = dt_module.train(_df)
    xgb_result = xgb_module.train(_df)
    return lr_result, dt_result, xgb_result


df = load_data()

with st.spinner("Training models..."):
    lr_result, dt_result, xgb_result = train_all_models(df)

lr_model, lr_scaler, lr_features, lr_X_train, lr_X_test, lr_y_train, lr_y_test = lr_result
dt_model, dt_features, dt_X_train, dt_X_test, dt_y_train, dt_y_test = dt_result
xgb_model, xgb_features, xgb_X_train, xgb_X_test, xgb_y_train, xgb_y_test = xgb_result


def get_accuracy(model, X_test, y_test, scaler=None):
    if scaler is not None:
        X_test = scaler.transform(X_test)
    y_pred = model.predict(X_test)
    return accuracy_score(y_test, y_pred)


def get_proba(model, X_test, scaler=None):
    if scaler is not None:
        X_test = scaler.transform(X_test)
    return model.predict_proba(X_test)[:, 1]


acc_lr = get_accuracy(lr_model, lr_X_test, lr_y_test, lr_scaler)
acc_dt = get_accuracy(dt_model, dt_X_test, dt_y_test)
acc_xgb = get_accuracy(xgb_model, xgb_X_test, xgb_y_test)

model_names = ["Logistic Regression", "Decision Tree", "XGBoost"]
accuracies = [acc_lr * 100, acc_dt * 100, acc_xgb * 100]
best_idx = accuracies.index(max(accuracies))


def main():

    st.sidebar.header("Configuration")
    selected_model_name = st.sidebar.selectbox(
        "Select Prediction Model",
        options=model_names,
        index=best_idx
    )

    if selected_model_name == "Logistic Regression":
        model, scaler, feature_names = lr_model, lr_scaler, lr_features
        X_test, y_test = lr_X_test, lr_y_test
    elif selected_model_name == "Decision Tree":
        model, scaler, feature_names = dt_model, None, dt_features
        X_test, y_test = dt_X_test, dt_y_test
    else:
        model, scaler, feature_names = xgb_model, None, xgb_features
        X_test, y_test = xgb_X_test, xgb_y_test

    sel_acc = get_accuracy(model, X_test, y_test, scaler)
    sel_prob = get_proba(model, X_test, scaler)
    sel_auc = roc_auc_score(y_test, sel_prob)

    if scaler is not None:
        y_pred_sel = model.predict(scaler.transform(X_test))
    else:
        y_pred_sel = model.predict(X_test)
    sel_f1 = f1_score(y_test, y_pred_sel)

    st.sidebar.markdown("---")
    st.sidebar.header("Model Performance")
    st.sidebar.info("Active Model: " + selected_model_name)
    st.sidebar.metric("Test Accuracy", str(round(sel_acc * 100, 2)) + "%")
    st.sidebar.metric("F1-Score", str(round(sel_f1, 4)))
    st.sidebar.metric("ROC AUC Score", str(round(sel_auc, 4)))

    st.sidebar.markdown("---")
    st.sidebar.header("About This Model")
    if selected_model_name == "Logistic Regression":
        st.sidebar.write("A linear model that predicts default probability based on weighted features like income and credit score.")
    elif selected_model_name == "Decision Tree":
        st.sidebar.write("Splits data step by step based on feature thresholds to classify borrowers into default or no-default.")
    else:
        st.sidebar.write("Gradient boosting algorithm that builds trees sequentially, each correcting errors of the previous ones.")

    st.title("Loan Default Prediction System")

    st.write(
        "Banks lose money when borrowers fail to repay loans. This system uses machine learning "
        "to predict default probability based on applicant profiles, classifying them into "
        "Low, Medium, and High risk categories."
    )

    st.markdown("---")

    st.subheader("Model Comparison")

    col_bar, col_roc = st.columns(2)

    with col_bar:
        fig1, ax1 = plt.subplots(figsize=(5, 3))
        colors = ['#5b9bd5', '#ed7d31', '#ffc000']
        bars = ax1.bar(model_names, accuracies, color=colors, edgecolor='white')
        for bar, acc in zip(bars, accuracies):
            ax1.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.2,
                     f'{acc:.1f}%', ha='center', fontsize=9, fontweight='bold')
        ax1.set_ylabel("Accuracy (%)", fontsize=9)
        ax1.set_ylim(min(accuracies) - 5, max(accuracies) + 3)
        ax1.set_title("Test Accuracy", fontsize=10)
        ax1.tick_params(axis='x', labelsize=7.5)
        ax1.tick_params(axis='y', labelsize=8)
        ax1.grid(axis='y', linestyle='--', alpha=0.4)
        plt.tight_layout()
        st.pyplot(fig1)
        st.caption("📊 Metric: **Accuracy** — percentage of correct predictions on test data.")

    with col_roc:
        prob_lr = get_proba(lr_model, lr_X_test, lr_scaler)
        prob_dt = get_proba(dt_model, dt_X_test)
        prob_xgb = get_proba(xgb_model, xgb_X_test)

        fig2, ax2 = plt.subplots(figsize=(5, 3))
        roc_colors = ['#5b9bd5', '#ed7d31', '#ffc000']
        for name, yt, yp, c in [
            ("LR", lr_y_test, prob_lr, roc_colors[0]),
            ("DT", dt_y_test, prob_dt, roc_colors[1]),
            ("XGB", xgb_y_test, prob_xgb, roc_colors[2]),
        ]:
            fpr, tpr, _ = roc_curve(yt, yp)
            auc_val = roc_auc_score(yt, yp)
            ax2.plot(fpr, tpr, color=c, linewidth=1.5, label=f'{name} ({auc_val:.3f})')

        ax2.plot([0, 1], [0, 1], color='#ccc', linestyle='--', linewidth=1)
        ax2.set_xlabel("FPR", fontsize=9)
        ax2.set_ylabel("TPR", fontsize=9)
        ax2.set_title("ROC Curves", fontsize=10)
        ax2.legend(fontsize=8)
        ax2.tick_params(labelsize=8)
        ax2.grid(True, linestyle='--', alpha=0.4)
        plt.tight_layout()
        st.pyplot(fig2)
        st.caption("📊 Metric: **ROC AUC** — measures how well the model separates defaulters from non-defaulters.")

    st.markdown("")
    st.markdown("**Metrics**")

    f1_lr = f1_score(lr_y_test, lr_model.predict(lr_scaler.transform(lr_X_test)))
    f1_dt = f1_score(dt_y_test, dt_model.predict(dt_X_test))
    f1_xgb = f1_score(xgb_y_test, xgb_model.predict(xgb_X_test))

    auc_lr = roc_auc_score(lr_y_test, prob_lr)
    auc_dt = roc_auc_score(dt_y_test, prob_dt)
    auc_xgb = roc_auc_score(xgb_y_test, prob_xgb)

    metrics_df = pd.DataFrame({
        "Model": model_names,
        "Accuracy (%)": [f"{a:.2f}" for a in accuracies],
        "F1-Score": [f"{f:.4f}" for f in [f1_lr, f1_dt, f1_xgb]],
        "ROC AUC": [f"{a:.4f}" for a in [auc_lr, auc_dt, auc_xgb]],
    })
    metrics_df.index = metrics_df.index + 1
    st.table(metrics_df)
    st.caption("📊 Metrics: **Accuracy** — correct predictions; **F1-Score** — balance of precision & recall; **ROC AUC** — separability of classes.")

    st.markdown("---")

    st.subheader("Predict Default Risk")

    st.markdown('<style> div[data-testid="stRadio"] label { font-size: 1.1rem !important; font-weight: 600; } div[data-testid="stRadio"] div[role="radiogroup"] label p { font-size: 1.05rem !important; } </style>', unsafe_allow_html=True)

    st.markdown("**Select Model:**")
    predict_model_name = st.radio(
        "Select Model",
        model_names,
        index=best_idx,
        horizontal=True,
        label_visibility="collapsed"
    )

    if predict_model_name == "Logistic Regression":
        model, scaler, feature_names = lr_model, lr_scaler, lr_features
        X_test, y_test = lr_X_test, lr_y_test
    elif predict_model_name == "Decision Tree":
        model, scaler, feature_names = dt_model, None, dt_features
        X_test, y_test = dt_X_test, dt_y_test
    else:
        model, scaler, feature_names = xgb_model, None, xgb_features
        X_test, y_test = xgb_X_test, xgb_y_test

    col_form, col_result = st.columns([2, 1])

    with col_form:
        with st.form("prediction_form"):
            col_a, col_b = st.columns(2)

            with col_a:
                age = st.number_input("Age", min_value=18, max_value=100, value=30)
                gender = st.selectbox("Gender", ["male", "female"])
                income = st.number_input("Annual Income ($)", min_value=0, value=60000, step=1000)
                emp_len = st.number_input("Employment Experience (Years)", min_value=0, max_value=50, value=5)
                education = st.selectbox("Education Level", ['High School', 'Associate', 'Bachelor', 'Master', 'Doctorate'])
                home_ownership = st.selectbox("Home Ownership", df['person_home_ownership'].unique())

            with col_b:
                loan_amnt = st.number_input("Loan Amount ($)", min_value=100, value=10000, step=500)
                loan_intent = st.selectbox("Loan Intent", df['loan_intent'].unique())
                loan_int_rate = st.number_input("Interest Rate (%)", min_value=0.0, max_value=30.0, value=10.0, step=0.1)
                cred_hist_length = st.number_input("Credit History Length (Years)", min_value=0, max_value=50, value=5)
                credit_score = st.number_input("Credit Score", min_value=300, max_value=850, value=650)
                prev_defaults = st.selectbox("Previous Defaults", ["No", "Yes"])

            submit = st.form_submit_button("Predict", use_container_width=True, type="primary")

    with col_result:
        st.markdown("**Result**")

        with st.expander("What is Default Probability?"):
            st.write("""
            **Default Probability** is the chance that a borrower will not repay their loan.

            Our models analyze the applicant's profile and estimate this probability:
            - **Low Risk** (below 40%): Applicant is likely to repay.
            - **Medium Risk** (40% to 60%): Needs closer review.
            - **High Risk** (above 60%): Applicant may not repay.
            """)

        if submit:
            loan_to_income = loan_amnt / max(1, income)
            gender_value = 1 if gender == "male" else 0
            default_value = 1 if prev_defaults == "Yes" else 0

            input_data = pd.DataFrame({
                'person_age': [age],
                'person_gender': [gender_value],
                'person_education': [education],
                'person_income': [income],
                'person_emp_exp': [emp_len],
                'person_home_ownership': [home_ownership],
                'loan_amnt': [loan_amnt],
                'loan_intent': [loan_intent],
                'loan_int_rate': [loan_int_rate],
                'loan_percent_income': [loan_to_income],
                'cb_person_cred_hist_length': [cred_hist_length],
                'credit_score': [credit_score],
                'previous_loan_defaults_on_file': [default_value]
            })

            input_processed = preprocess_features(input_data)

            for col in feature_names:
                if col not in input_processed.columns:
                    input_processed[col] = 0
            input_processed = input_processed[feature_names]

            if scaler is not None:
                input_for_pred = scaler.transform(input_processed)
            else:
                input_for_pred = input_processed

            prob = model.predict_proba(input_for_pred)[0][1]

            st.metric("Default Probability", f'{prob * 100:.1f}%')

            if prob >= 0.60:
                st.error("HIGH RISK - Review carefully or decline.")
            elif prob >= 0.40:
                st.warning("MEDIUM RISK - Needs manual review.")
            else:
                st.success("LOW RISK - Can be approved.")

            st.markdown("---")
            st.markdown("**Top Factors Affecting Prediction:**")

            if predict_model_name == "Logistic Regression":
                coefficients = model.coef_[0]
                feature_importance = []
                for i in range(len(feature_names)):
                    feature_importance.append((feature_names[i], coefficients[i]))
                feature_importance.sort(key=lambda x: abs(x[1]), reverse=True)
                top_5 = feature_importance[:5]
                for i in range(len(top_5)):
                    name = top_5[i][0]
                    coef = top_5[i][1]
                    clean_name = name.replace("person_", "").replace("loan_", "").replace("_", " ").title()
                    direction = "Increases Risk" if coef > 0 else "Decreases Risk"
                    st.write(f"{i+1}. **{clean_name}** — {direction}")
            else:
                importances = model.feature_importances_
                feature_importance = []
                for i in range(len(feature_names)):
                    feature_importance.append((feature_names[i], importances[i]))
                feature_importance.sort(key=lambda x: x[1], reverse=True)
                top_5 = feature_importance[:5]
                for i in range(len(top_5)):
                    name = top_5[i][0]
                    imp = top_5[i][1]
                    clean_name = name.replace("person_", "").replace("loan_", "").replace("_", " ").title()
                    st.write(f"{i+1}. **{clean_name}** — Importance: {imp:.3f}")
        else:
            st.info("Fill in the form and click Predict.")


if __name__ == "__main__":
    main()
