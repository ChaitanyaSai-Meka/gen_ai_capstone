import streamlit as st
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, OrdinalEncoder, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score

st.set_page_config(
    page_title="Loan Default Prediction Dashboard",
    page_icon="bar-chart",
    layout="wide",
    initial_sidebar_state="expanded"
)

st.markdown("""
<style>
    .stApp {
        background-color: #f8f9fa;
        color: #212529;
    }
    .css-1d391kg, .css-1lcbmhc {
        background-color: #ffffff;
    }
    .metric-card {
        background-color: #ffffff;
        padding: 20px;
        border-radius: 10px;
        box-shadow: 0 4px 6px rgba(0,0,0,0.1);
        text-align: center;
        border: 1px solid #e9ecef;
    }
    h1, h2, h3 {
        color: #343a40;
    }
</style>
""", unsafe_allow_html=True)

@st.cache_data
def load_data():
    df = pd.read_csv('loan_data.csv')
    df = df.dropna()
    
    for col in ['person_age', 'person_income', 'person_emp_exp']:
        cap = df[col].quantile(0.99)
        df = df[df[col] <= cap]
        
    df['previous_loan_defaults_on_file'] = df['previous_loan_defaults_on_file'].map({'Yes': 1, 'No': 0})
    df['person_gender'] = df['person_gender'].map({'male': 1, 'female': 0})
    
    return df

@st.cache_resource
def train_models(df):
    X = df.drop(columns=['loan_status'])
    y = df['loan_status']
    
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)
    
    education_order = [['High School', 'Associate', 'Bachelor', 'Master', 'Doctorate']]
    ordinal_features = ['person_education']
    nominal_features = ['person_home_ownership', 'loan_intent']
    numeric_features = [
        'person_age', 'person_income', 'person_emp_exp',
        'loan_amnt', 'loan_int_rate', 'loan_percent_income',
        'cb_person_cred_hist_length', 'credit_score',
        'previous_loan_defaults_on_file', 'person_gender'
    ]
    
    lr_preprocessor = ColumnTransformer(
        transformers=[
            ('num', StandardScaler(), numeric_features),
            ('ordinal', OrdinalEncoder(categories=education_order), ordinal_features),
            ('nominal', OneHotEncoder(drop='first', sparse_output=False), nominal_features),
        ],
        remainder='drop'
    )
    
    lr_pipeline = Pipeline(steps=[
        ('preprocessor', lr_preprocessor),
        ('classifier', LogisticRegression(
            solver='saga',
            max_iter=2000,
            class_weight='balanced',
            C=1.0,
            random_state=42,
            n_jobs=-1
        ))
    ])
    
    dt_preprocessor = ColumnTransformer(
        transformers=[
            ('num', 'passthrough', numeric_features),
            ('ordinal', OrdinalEncoder(categories=education_order), ordinal_features),
            ('nominal', OneHotEncoder(drop='first', sparse_output=False), nominal_features),
        ],
        remainder='drop'
    )
    
    dt_pipeline = Pipeline(steps=[
        ('preprocessor', dt_preprocessor),
        ('classifier', DecisionTreeClassifier(
            max_depth=8,
            min_samples_leaf=10,
            criterion='gini',
            class_weight='balanced',
            random_state=42
        ))
    ])
    
    rf_pipeline = Pipeline(steps=[
        ('preprocessor', dt_preprocessor),
        ('classifier', RandomForestClassifier(
            n_estimators=100,
            max_depth=8,
            min_samples_leaf=10,
            class_weight='balanced',
            random_state=42,
            n_jobs=-1
        ))
    ])
    
    lr_pipeline.fit(X_train, y_train)
    dt_pipeline.fit(X_train, y_train)
    rf_pipeline.fit(X_train, y_train)
    
    return lr_pipeline, dt_pipeline, rf_pipeline, X_train, X_test, y_train, y_test

def get_feature_contributions(model, X_input, is_lr=True):
    """Calculate feature contributions for the given model and input."""
    if is_lr:
        preprocessor = model.named_steps['preprocessor']
        clf = model.named_steps['classifier']
        
        numeric_features = [
            'person_age', 'person_income', 'person_emp_exp',
            'loan_amnt', 'loan_int_rate', 'loan_percent_income',
            'cb_person_cred_hist_length', 'credit_score',
            'previous_loan_defaults_on_file', 'person_gender'
        ]
        ordinal_features = ['person_education']
        nominal_features = ['person_home_ownership', 'loan_intent']
        
        ohe_feature_names = preprocessor.named_transformers_['nominal'].get_feature_names_out(nominal_features).tolist()
        feature_names = numeric_features + ordinal_features + ohe_feature_names
        
        coefs = clf.coef_[0]
        X_processed = preprocessor.transform(X_input)[0]
        
        contributions = [(feature_names[i], X_processed[i] * coefs[i]) for i in range(len(feature_names))]
        contributions.sort(key=lambda x: abs(x[1]), reverse=True)
        return contributions[:5]
    else:
        clf = model.named_steps['classifier']
        
        numeric_features = [
            'person_age', 'person_income', 'person_emp_exp',
            'loan_amnt', 'loan_int_rate', 'loan_percent_income',
            'cb_person_cred_hist_length', 'credit_score',
            'previous_loan_defaults_on_file', 'person_gender'
        ]
        ordinal_features = ['person_education']
        nominal_features = ['person_home_ownership', 'loan_intent']
        
        preprocessor = model.named_steps['preprocessor']
        ohe_feature_names = preprocessor.named_transformers_['nominal'].get_feature_names_out(nominal_features).tolist()
        feature_names = numeric_features + ordinal_features + ohe_feature_names
        
        importances = clf.feature_importances_
        important_features = [(feature_names[i], importances[i]) for i in range(len(feature_names))]
        important_features.sort(key=lambda x: x[1], reverse=True)
        return important_features[:5]

def main():
    st.title("Loan Default Evaluation System")
    
    with st.expander("What is Default Risk?"):
        st.write("""
        **Default Risk** is the probability that an applicant will fail to repay their loan according to the agreed-upon terms. 
        
        Our machine learning models analyze historical data and the applicant's current financial profile to estimate this probability.
        - **Low Risk** (< 40%): The applicant has a strong profile and is highly likely to repay the loan.
        - **Medium Risk** (40% - 60%): The applicant presents some risk factors that require closer human review or conditional approval terms.
        - **High Risk** (> 60%): The applicant's profile closely matches historical defaults and should likely be declined.
        """)
        
    st.markdown("Enter applicant details to predict likelihood of loan default using advanced machine learning models.")
    
    with st.spinner("Initializing models and processing data (this may take a moment)..."):
        df = load_data()
        lr_model, dt_model, rf_model, X_train, X_test, y_train, y_test = train_models(df)
        
    st.sidebar.header("Configuration")
    selected_model = st.sidebar.selectbox(
        "Select Prediction Model",
        options=["Logistic Regression", "Decision Tree", "Random Forest"]
    )
    
    if selected_model == "Logistic Regression":
        model = lr_model
    elif selected_model == "Decision Tree":
        model = dt_model
    else:
        model = rf_model
    is_lr = selected_model == "Logistic Regression"
    
    st.sidebar.markdown("---")
    st.sidebar.header("System Metrics")
    st.sidebar.info(f"Active Model: {selected_model}")
    
    threshold_low_med = 0.40
    threshold_med_high = 0.60
    
    y_prob = model.predict_proba(X_test)[:, 1]
    y_pred = (y_prob >= threshold_med_high).astype(int) # Standard acc usually at 0.5, but we show custom metrics
    
    acc = accuracy_score(y_test, (y_prob >= 0.5).astype(int))
    auc = roc_auc_score(y_test, y_prob)
    
    st.sidebar.metric("Test Accuracy", f"{acc:.2%}")
    st.sidebar.metric("ROC AUC", f"{auc:.4f}")
    
    st.sidebar.markdown("---")
    st.sidebar.header("Model Insights")
    if is_lr:
        st.sidebar.write("Logistic Regression relies heavily on linear combinations. It tightly correlates specific factors like Renting and Loan-to-Income ratios directly to risk.")
    elif selected_model == "Decision Tree":
        st.sidebar.write("Decision Trees look for non-linear patterns. They can identify that a Renter with high income and no prior defaults is actually very safe.")
    else:
        st.sidebar.write("Random Forests average the predictions of many Decision Trees. This reduces overfitting and typically provides the most robust predictions on unfamiliar data.")
    
    tab1, tab2 = st.tabs(["Prediction Dashboard", "Model Analytics"])
    
    with tab1:
        col_inputs, col_analysis = st.columns([1.5, 1])
        
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

        with col_analysis:
            st.header("Analysis & Results")
            
            if submit:
                input_data = pd.DataFrame({
                    'person_age': [age],
                    'person_gender': [1 if gender == "male" else 0],
                    'person_education': [education],
                    'person_income': [income],
                    'person_emp_exp': [emp_len],
                    'person_home_ownership': [home_ownership],
                    'loan_amnt': [loan_amnt],
                    'loan_intent': [loan_intent],
                    'loan_int_rate': [loan_int_rate],
                    'loan_percent_income': [loan_amnt / max(1, income)],
                    'cb_person_cred_hist_length': [cred_hist_length],
                    'credit_score': [credit_score],
                    'previous_loan_defaults_on_file': [1 if prev_defaults == "Yes" else 0]
                })
                
                prob = model.predict_proba(input_data)[0][1]
                
                st.metric("Probability of Default", f"{prob:.1%}")
                
                if prob >= threshold_med_high:
                    st.error("HIGH RISK OF DEFAULT")
                    st.write("**Action:** Scrutinize application further or decline.")
                elif prob >= threshold_low_med:
                    st.warning("MEDIUM RISK OF DEFAULT")
                    st.write("**Action:** Requires manual review. Consider conditional terms or higher interest rates to offset risk.")
                else:
                    st.success("LOW RISK OF DEFAULT")
                    st.write("**Action:** Approve conditionally based on general underwriting policy.")

                st.markdown("---")
                st.subheader("Model Decision Drivers")
                
                contributions = get_feature_contributions(model, input_data, is_lr)
                
                st.write(f"Top 5 key factors influencing the **{selected_model}** model for this specific applicant:")
                for i, (feat, val) in enumerate(contributions):
                    clean_feat = feat.replace("person_", "").replace("loan_", "").replace("_", " ").title()
                    if is_lr:
                        direction = "Increased Risk" if val > 0 else "Decreased Risk"
                        st.markdown(f"{i+1}. **{clean_feat}** ({direction})")
                    else:
                        st.markdown(f"{i+1}. **{clean_feat}** (Relative Importance: {val:.2f})")
                        
    with tab2:
        st.header(f"Model Training Analytics: {selected_model}")
        st.write("Below are the historical performance charts generated during the model training phase from our data science notebooks.")
        
        
        col_c1, col_c2 = st.columns(2)
        
        with col_c1:
            st.subheader("ROC Curve")
            from sklearn.metrics import roc_curve
            import matplotlib.pyplot as plt
            
            fpr, tpr, _ = roc_curve(y_test, y_prob)
            
            fig_roc, ax_roc = plt.subplots(figsize=(6, 4))
            ax_roc.plot(fpr, tpr, color='#7b5ea7', lw=2.5, label=f'ROC Curve (AUC = {auc:.4f})')
            ax_roc.plot([0, 1], [0, 1], color='#666680', linestyle='--', lw=1.2, label='Random')
            ax_roc.fill_between(fpr, tpr, alpha=0.1, color='#7b5ea7')
            ax_roc.set_xlabel('False Positive Rate')
            ax_roc.set_ylabel('True Positive Rate')
            ax_roc.legend()
            ax_roc.grid(True, linestyle='--', alpha=0.7)
            st.pyplot(fig_roc)
            
        with col_c2:
            st.subheader("Confusion Matrix")
            from sklearn.metrics import confusion_matrix
            import seaborn as sns
            
            cm = confusion_matrix(y_test, (y_prob >= 0.5).astype(int))
            fig_cm, ax_cm = plt.subplots(figsize=(6, 4))
            sns.heatmap(cm, annot=True, fmt='d', ax=ax_cm, cmap='Purples',
                        xticklabels=['No Default', 'Default'],
                        yticklabels=['No Default', 'Default'])
            ax_cm.set_xlabel('Predicted Label')
            ax_cm.set_ylabel('True Label')
            st.pyplot(fig_cm)
            
            
if __name__ == "__main__":
    main()
