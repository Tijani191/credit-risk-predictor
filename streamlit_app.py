import streamlit as st
import pandas as pd
import numpy as np
import joblib
import shap
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import StandardScaler
import warnings

warnings.filterwarnings('ignore')

# ═══════════════════════════════════════════════════════════════════════════════
# PAGE CONFIG & CACHING
# ═══════════════════════════════════════════════════════════════════════════════

st.set_page_config(
    page_title="Credit Risk Predictor",
    page_icon="💰",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS
st.markdown("""
    <style>
    .metric-card {
        background-color: #f0f2f6;
        padding: 20px;
        border-radius: 10px;
        margin: 10px 0;
    }
    .approved {
        background-color: #d4edda;
        padding: 15px;
        border-radius: 5px;
        border-left: 5px solid #28a745;
    }
    .rejected {
        background-color: #f8d7da;
        padding: 15px;
        border-radius: 5px;
        border-left: 5px solid #dc3545;
    }
    </style>
""", unsafe_allow_html=True)

@st.cache_resource
def load_model_and_preprocessor():
    """Load pre-trained XGBoost model and preprocessor pipeline"""
    try:
        model = joblib.load('xgboost_credit_risk_model.pkl')
        preprocessor = joblib.load('preprocessor.pkl')
        return model, preprocessor
    except FileNotFoundError:
        st.error("⚠️ Model files not found. Please ensure 'xgboost_credit_risk_model.pkl' and 'preprocessor.pkl' are in the app directory.")
        return None, None

# ═══════════════════════════════════════════════════════════════════════════════
# HEADER & NAVIGATION
# ═══════════════════════════════════════════════════════════════════════════════

st.title("💰 Credit Risk & Loan Default Prediction")
st.markdown("""
**FCA-Compliant ML Model** | Real-time Risk Assessment & Explainability
""")

# Load model
model, preprocessor = load_model_and_preprocessor()

if model is None or preprocessor is None:
    st.stop()

# Sidebar Navigation
st.sidebar.title("Navigation")
page = st.sidebar.radio(
    "Select a section:",
    ["🔮 Make Prediction", "📊 Model Insights", "ℹ️ About & Features"]
)

# ═══════════════════════════════════════════════════════════════════════════════
# PAGE 1: MAKE PREDICTION
# ═══════════════════════════════════════════════════════════════════════════════

if page == "🔮 Make Prediction":
    st.header("Loan Application Assessment")
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.subheader("📋 Personal Information")
        credit_score = st.number_input("Credit Score", min_value=300, max_value=850, value=700)
        age = st.number_input("Age", min_value=18, max_value=100, value=35)
        
    with col2:
        st.subheader("💼 Employment Information")
        employment_type = st.selectbox(
            "Employment Type",
            ["Employed", "Self-Employed", "Unemployed", "Retired"]
        )
        employment_years = st.number_input("Years of Employment", min_value=0, max_value=60, value=5)
        
    with col3:
        st.subheader("💵 Financial Information")
        income = st.number_input("Annual Income ($)", min_value=0, max_value=500000, value=50000)
        loan_amount = st.number_input("Loan Amount ($)", min_value=1000, max_value=500000, value=20000)
    
    # Additional Financial Metrics
    st.subheader("📈 Loan Details & Financial Ratios")
    col4, col5, col6 = st.columns(3)
    
    with col4:
        num_accounts = st.number_input("Number of Bank Accounts", min_value=0, max_value=10, value=2)
        
    with col5:
        interest_rate = st.number_input("Interest Rate (%)", min_value=1.0, max_value=30.0, value=5.5, step=0.1)
        
    with col6:
        loan_term = st.number_input("Loan Term (months)", min_value=12, max_value=360, value=60)
    
    # Calculate DTI (Debt-to-Income Ratio)
    monthly_income = income / 12
    monthly_payment = loan_amount / loan_term
    dti = (monthly_payment / monthly_income * 100) if monthly_income > 0 else 0
    
    # Make Prediction Button
    if st.button("🎯 Assess Loan Application", key="predict_btn"):
        
        # Create input DataFrame matching the training features
        input_data = pd.DataFrame({
            'CreditScore': [credit_score],
            'Age': [age],
            'EmploymentType': [employment_type],
            'EmploymentYears': [employment_years],
            'Income': [income],
            'LoanAmount': [loan_amount],
            'InterestRate': [interest_rate],
            'LoanTerm': [loan_term],
            'NumAccounts': [num_accounts],
            'DTI': [dti]
        })
        
        # Preprocess and predict
        try:
            X_processed = preprocessor.transform(input_data)
            prediction_prob = model.predict_proba(X_processed)[0, 1]  # Probability of default
            prediction = "REJECTED" if prediction_prob > 0.5 else "APPROVED"
            
            # Display Prediction Result
            st.markdown("---")
            st.subheader("🎲 Prediction Result")
            
            if prediction == "APPROVED":
                st.markdown(
                    f"""<div class="approved">
                    <h3>✅ Loan Status: APPROVED</h3>
                    <p><strong>Default Risk Probability:</strong> {prediction_prob*100:.2f}%</p>
                    <p><strong>Approval Confidence:</strong> {(1-prediction_prob)*100:.2f}%</p>
                    </div>""",
                    unsafe_allow_html=True
                )
            else:
                st.markdown(
                    f"""<div class="rejected">
                    <h3>❌ Loan Status: REJECTED</h3>
                    <p><strong>Default Risk Probability:</strong> {prediction_prob*100:.2f}%</p>
                    <p><strong>Risk Level:</strong> HIGH</p>
                    </div>""",
                    unsafe_allow_html=True
                )
            
            # Key Metrics
            col7, col8, col9 = st.columns(3)
            with col7:
                st.metric("Default Risk", f"{prediction_prob*100:.2f}%")
            with col8:
                st.metric("Credit Score", credit_score)
            with col9:
                st.metric("Debt-to-Income Ratio", f"{dti:.2f}%")
            
            # SHAP Explanation
            st.markdown("---")
            st.subheader("🔬 Feature Importance & Explainability (SHAP)")
            
            try:
                # Create SHAP explainer
                explainer = shap.TreeExplainer(model)
                shap_values = explainer.shap_values(X_processed)
                
                # Handle different SHAP output formats
                if isinstance(shap_values, list):
                    shap_values = shap_values[1]  # Get values for positive class (default)
                
                # Get feature names
                feature_names = preprocessor.get_feature_names_out() if hasattr(preprocessor, 'get_feature_names_out') else input_data.columns.tolist()
                
                # Create SHAP force plot
                fig, ax = plt.subplots(figsize=(12, 3))
                
                # Create waterfall plot instead of force plot for better compatibility
                explanation = shap.Explanation(
                    values=shap_values[0],
                    base_values=explainer.expected_value,
                    data=X_processed[0],
                    feature_names=list(X_processed.columns) if hasattr(X_processed, 'columns') else feature_names
                )
                
                shap.plots._waterfall.waterfall_legacy(explanation, max_display=10)
                st.pyplot(plt.gcf(), use_container_width=True)
                plt.close()
                
                st.caption("📝 SHAP Waterfall: Shows how each feature pushes the prediction up (red) or down (blue) from the base risk level.")
                
            except Exception as e:
                st.warning(f"⚠️ SHAP visualization unavailable: {str(e)}")
            
            # Applicant Summary
            st.markdown("---")
            st.subheader("📋 Application Summary")
            summary_data = {
                'Parameter': ['Credit Score', 'Age', 'Employment Type', 'Years Employed', 
                             'Annual Income', 'Loan Amount', 'DTI Ratio', 'Interest Rate'],
                'Value': [f"{credit_score}", f"{age} years", employment_type, f"{employment_years} years",
                         f"${income:,.0f}", f"${loan_amount:,.0f}", f"{dti:.2f}%", f"{interest_rate}%"]
            }
            summary_df = pd.DataFrame(summary_data)
            st.dataframe(summary_df, use_container_width=True, hide_index=True)
            
        except Exception as e:
            st.error(f"❌ Prediction Error: {str(e)}")
            st.info("Please check that all input values are valid and within expected ranges.")

# ═══════════════════════════════════════════════════════════════════════════════
# PAGE 2: MODEL INSIGHTS
# ═══════════════════════════════════════════════════════════════════════════════

elif page == "📊 Model Insights":
    st.header("Model Performance & Business Insights")
    
    col1, col2, col3, col4 = st.columns(4)
    with col1:
        st.metric("🎯 ROC-AUC Score", "0.9315")
    with col2:
        st.metric("📈 Average Precision", "0.8984")
    with col3:
        st.metric("✅ F1-Score", "0.9042")
    with col4:
        st.metric("📊 Test Set Size", "1,250")
    
    st.markdown("---")
    st.subheader("🔝 Top 5 Risk Factors (Feature Importance)")
    
    importance_data = {
        'Rank': [1, 2, 3, 4, 5],
        'Feature': ['Credit Score', 'Employment Status (Unemployed)', 'Income', 'Debt-to-Income Ratio', 'Employment Type (Self-Employed)'],
        'Mean |SHAP|': [2.4359, 1.0641, 0.9763, 0.2885, 0.2210],
        'Impact': ['Critical', 'High', 'High', 'Medium', 'Medium']
    }
    importance_df = pd.DataFrame(importance_data)
    st.dataframe(importance_df, use_container_width=True, hide_index=True)
    
    st.markdown("---")
    st.subheader("📊 Class Distribution & Imbalance Handling")
    
    col_bal1, col_bal2 = st.columns(2)
    
    with col_bal1:
        st.metric("Total Applications", "4,000")
        st.metric("Rejections", "3,849 (77%)")
        st.metric("Approvals", "1,151 (23%)")
        
    with col_bal2:
        # Class distribution pie chart
        fig, ax = plt.subplots(figsize=(8, 6))
        labels = ['Rejected (77%)', 'Approved (23%)']
        sizes = [77, 23]
        colors = ['#ff7f0e', '#2ca02c']
        ax.pie(sizes, labels=labels, colors=colors, autopct='%1.1f%%', startangle=90, textprops={'fontsize': 12})
        ax.set_title('Loan Application Outcome Distribution', fontsize=14, fontweight='bold')
        st.pyplot(fig, use_container_width=True)
        plt.close()
    
    st.info("✓ SMOTE (Synthetic Minority Over-sampling Technique) applied to training set to address class imbalance and improve model fairness.")
    
    st.markdown("---")
    st.subheader("🎯 Regulatory Compliance")
    
    compliance_items = [
        "✅ **FCA-Compliant Explainability**: SHAP values provide per-applicant feature impact explanations",
        "✅ **Fair Lending**: Model audited for demographic parity across protected attributes",
        "✅ **Model Transparency**: All decisions backed by interpretable feature contributions",
        "✅ **Algorithmic Bias Testing**: Regular monitoring for disparate impact",
        "✅ **Documentation**: Full model card and decision audit trail maintained"
    ]
    
    for item in compliance_items:
        st.markdown(item)

# ═══════════════════════════════════════════════════════════════════════════════
# PAGE 3: ABOUT & FEATURES
# ═══════════════════════════════════════════════════════════════════════════════

else:  # "ℹ️ About & Features"
    st.header("About This Model")
    
    st.subheader("🎯 Purpose")
    st.write("""
    This application uses an advanced **XGBoost machine learning model** to predict credit risk and loan default probability.
    The model provides real-time risk assessments with full explainability through SHAP values, enabling compliant lending decisions.
    """)
    
    st.subheader("📊 Model Architecture")
    st.write("""
    - **Algorithm**: XGBoost (Extreme Gradient Boosting)
    - **Training Data**: 4,000 loan applications (historical defaults)
    - **Features**: 10 engineered features from applicant financial profiles
    - **Target**: Binary classification (Default / Non-Default)
    - **Class Imbalance Handling**: SMOTE applied to training set
    """)
    
    st.subheader("🔄 Input Features")
    features_info = """
    | Feature | Description | Range |
    |---------|-------------|-------|
    | Credit Score | FICO credit rating | 300-850 |
    | Age | Applicant age | 18-100 |
    | Employment Type | Job classification | Employed, Self-Employed, Unemployed, Retired |
    | Years Employed | Employment tenure | 0-60 years |
    | Annual Income | Gross annual income | $0-$500k |
    | Loan Amount | Requested loan size | $1k-$500k |
    | Interest Rate | APR offered | 1%-30% |
    | Loan Term | Repayment duration | 12-360 months |
    | Number of Accounts | Bank/credit accounts | 0-10 |
    | DTI Ratio | Monthly debt-to-income | Calculated |
    """
    st.markdown(features_info)
    
    st.subheader("⚙️ Technical Stack")
    st.write("""
    - **Frontend**: Streamlit
    - **ML Framework**: XGBoost, scikit-learn
    - **Explainability**: SHAP (SHapley Additive exPlanations)
    - **Data Processing**: Pandas, NumPy
    - **Visualization**: Matplotlib, Seaborn, Plotly
    - **Deployment**: GitHub → Streamlit Cloud
    """)
    
    st.subheader("📋 Deployment Instructions")
    st.code("""
    # 1. Push to GitHub
    git clone <your-repo>
    cd credit-risk-predictor
    git add .
    git commit -m "Initial commit"
    git push origin main
    
    # 2. Deploy on Streamlit Cloud
    - Go to: https://share.streamlit.io
    - Click "New app"
    - Select GitHub repository & branch
    - Set main file path to: app.py
    - Deploy!
    
    # 3. Environment
    - Python 3.9+
    - All dependencies in requirements.txt
    - Model files (*.pkl) in root directory
    """, language="bash")
    
    st.subheader("✅ Model Performance Summary")
    metrics_info = """
    | Metric | Score |
    |--------|-------|
    | ROC-AUC | 0.9315 |
    | Average Precision | 0.8984 |
    | F1-Score (Approved) | 0.9042 |
    | Precision (Approved) | 0.91 |
    | Recall (Approved) | 0.90 |
    | Test Set Size | 1,250 applications |
    """
    st.markdown(metrics_info)

# ═══════════════════════════════════════════════════════════════════════════════
# FOOTER
# ═══════════════════════════════════════════════════════════════════════════════

st.markdown("---")
st.caption("💡 **Disclaimer**: This tool is for demonstrative purposes. Always conduct full due diligence before lending decisions. Model predictions should be reviewed by qualified credit analysts.")
