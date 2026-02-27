import streamlit as st
import pandas as pd
import numpy as np
from sklearn.metrics import accuracy_score, roc_auc_score, roc_curve, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns

st.set_page_config(
    page_title="Loan Default Prediction Dashboard",
    page_icon="bar-chart",
    layout="wide",
    initial_sidebar_state="expanded"
)



from preprocessing import load_and_clean_data, preprocess_features
import models.logistic_regression as lr_module
import models.decision_tree as dt_module
import models.random_forest as rf_module


df = load_and_clean_data('loan_data.csv')

with st.spinner("Training models... please wait..."):
    lr_model, lr_scaler, lr_features, lr_X_train, lr_X_test, lr_y_train, lr_y_test = lr_module.train(df)
    dt_model, dt_features, dt_X_train, dt_X_test, dt_y_train, dt_y_test = dt_module.train(df)
    rf_model, rf_features, rf_X_train, rf_X_test, rf_y_train, rf_y_test = rf_module.train(df)


def main():
    """Main function that runs the Streamlit dashboard"""

    st.title("Loan Default Evaluation System")

    with st.expander("What is Default Risk?"):
        st.write("""
        **Default Risk** is the chance that a borrower will not repay their loan.
        
        Our models analyze the applicant's profile and estimate this probability:
        - **Low Risk** (below 40%): Applicant is likely to repay.
        - **Medium Risk** (40% to 60%): Needs closer review.
        - **High Risk** (above 60%): Applicant may not repay.
        """)

    st.markdown("Enter applicant details below to predict their loan default risk.")

    st.sidebar.header("Configuration")
    selected_model_name = st.sidebar.selectbox(
        "Select Prediction Model",
        options=["Logistic Regression", "Decision Tree", "Random Forest"]
    )

    if selected_model_name == "Logistic Regression":
        model = lr_model
        scaler = lr_scaler
        feature_names = lr_features
        X_test = lr_X_test
        y_test = lr_y_test
    elif selected_model_name == "Decision Tree":
        model = dt_model
        scaler = None
        feature_names = dt_features
        X_test = dt_X_test
        y_test = dt_y_test
    else:
        model = rf_model
        scaler = None
        feature_names = rf_features
        X_test = rf_X_test
        y_test = rf_y_test

    if scaler is not None:
        X_test_scaled = scaler.transform(X_test)
    else:
        X_test_scaled = X_test

    y_prob = model.predict_proba(X_test_scaled)[:, 1]

    y_pred = []
    for p in y_prob:
        if p >= 0.5:
            y_pred.append(1)
        else:
            y_pred.append(0)

    acc = accuracy_score(y_test, y_pred)
    auc = roc_auc_score(y_test, y_prob)

    st.sidebar.markdown("---")
    st.sidebar.header("Model Performance")
    st.sidebar.info("Active Model: " + selected_model_name)
    st.sidebar.metric("Test Accuracy", str(round(acc * 100, 2)) + "%")
    st.sidebar.metric("ROC AUC Score", str(round(auc, 4)))

    st.sidebar.markdown("---")
    st.sidebar.header("About This Model")
    if selected_model_name == "Logistic Regression":
        st.sidebar.write("Logistic Regression is a simple linear model. It finds which factors like income and credit score are most related to default risk.")
    elif selected_model_name == "Decision Tree":
        st.sidebar.write("Decision Tree splits the data step by step based on feature values to make predictions. It can find non-linear patterns.")
    else:
        st.sidebar.write("Random Forest uses 100 Decision Trees together. Combining many trees gives more stable and accurate results.")

    tab1, tab2 = st.tabs(["Prediction Dashboard", "Model Analytics"])

    with tab1:
        col_inputs, col_results = st.columns([1.5, 1])

        with col_inputs:
            st.header("Applicant Information")

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

                submit = st.form_submit_button("Predict Default Risk", type="primary", use_container_width=True)

        with col_results:
            st.header("Results")

            if submit:
                loan_to_income = loan_amnt / max(1, income)

                if gender == "male":
                    gender_value = 1
                else:
                    gender_value = 0

                if prev_defaults == "Yes":
                    default_value = 1
                else:
                    default_value = 0

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

                st.metric("Probability of Default", str(round(prob * 100, 1)) + "%")

                if prob >= 0.60:
                    st.error("HIGH RISK OF DEFAULT")
                    st.write("**Action:** Look at this application more carefully or decline it.")
                elif prob >= 0.40:
                    st.warning("MEDIUM RISK OF DEFAULT")
                    st.write("**Action:** Needs manual review. Consider adding conditions.")
                else:
                    st.success("LOW RISK OF DEFAULT")
                    st.write("**Action:** Approve based on standard policy.")

                st.markdown("---")
                st.subheader("Top Factors")

                if selected_model_name == "Logistic Regression":
                    coefficients = model.coef_[0]
                    feature_importance = []
                    for i in range(len(feature_names)):
                        name = feature_names[i]
                        coef = coefficients[i]
                        feature_importance.append((name, coef))

                    feature_importance.sort(key=lambda x: abs(x[1]), reverse=True)
                    top_5 = feature_importance[:5]

                    st.write("Top 5 factors for " + selected_model_name + ":")
                    for i in range(len(top_5)):
                        name = top_5[i][0]
                        coef = top_5[i][1]
                        clean_name = name.replace("person_", "").replace("loan_", "").replace("_", " ").title()
                        if coef > 0:
                            direction = "Increases Risk"
                        else:
                            direction = "Decreases Risk"
                        st.write(str(i + 1) + ". **" + clean_name + "** (" + direction + ")")

                else:
                    importances = model.feature_importances_
                    feature_importance = []
                    for i in range(len(feature_names)):
                        name = feature_names[i]
                        imp = importances[i]
                        feature_importance.append((name, imp))

                    feature_importance.sort(key=lambda x: x[1], reverse=True)
                    top_5 = feature_importance[:5]

                    st.write("Top 5 factors for " + selected_model_name + ":")
                    for i in range(len(top_5)):
                        name = top_5[i][0]
                        imp = top_5[i][1]
                        clean_name = name.replace("person_", "").replace("loan_", "").replace("_", " ").title()
                        st.write(str(i + 1) + ". **" + clean_name + "** (Importance: " + str(round(imp, 2)) + ")")

    with tab2:
        st.header("Model Training Analytics: " + selected_model_name)
        st.write("Performance charts for the selected model.")

        col_chart1, col_chart2 = st.columns(2)

        with col_chart1:
            st.subheader("ROC Curve")

            fpr, tpr, thresholds = roc_curve(y_test, y_prob)

            fig_roc, ax_roc = plt.subplots(figsize=(6, 4))
            ax_roc.plot(fpr, tpr, color='blue', linewidth=2, label='ROC Curve (AUC = ' + str(round(auc, 4)) + ')')
            ax_roc.plot([0, 1], [0, 1], color='gray', linestyle='--', linewidth=1, label='Random Baseline')
            ax_roc.set_xlabel('False Positive Rate')
            ax_roc.set_ylabel('True Positive Rate')
            ax_roc.set_title('ROC Curve')
            ax_roc.legend()
            ax_roc.grid(True, linestyle='--', alpha=0.7)
            st.pyplot(fig_roc)

        with col_chart2:
            st.subheader("Confusion Matrix")

            cm = confusion_matrix(y_test, y_pred)

            fig_cm, ax_cm = plt.subplots(figsize=(6, 4))
            sns.heatmap(
                cm,
                annot=True,
                fmt='d',
                ax=ax_cm,
                cmap='Blues',
                xticklabels=['No Default', 'Default'],
                yticklabels=['No Default', 'Default']
            )
            ax_cm.set_xlabel('Predicted Label')
            ax_cm.set_ylabel('True Label')
            ax_cm.set_title('Confusion Matrix')
            st.pyplot(fig_cm)


if __name__ == "__main__":
    main()
