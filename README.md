Credit Risk & Loan Default Prediction Model

A production-ready FCA-compliant machine learning application for credit risk assessment and loan default prediction. Built with XGBoost and deployed via Streamlit Cloud.

🎯 Features


Real-time Risk Assessment: Predict loan default probability instantly
SHAP Explainability: Understand which factors drive each decision (FCA-compliant)
Interactive Interface: User-friendly Streamlit dashboard for loan officers
Model Insights: View feature importance and model performance metrics
Regulatory Compliance: Built-in fairness checks and audit trail support


📊 Model Performance

MetricScoreROC-AUC Score0.9315Average Precision0.8984F1-Score0.9042Test Set Size1,250 applications

🔝 Top Risk Factors


Credit Score - Strongest predictor (SHAP = 2.44)
Employment Status - Unemployment significantly increases risk (SHAP = 1.06)
Income - Higher income reduces default risk (SHAP = 0.98)
Debt-to-Income Ratio - Financial burden indicator (SHAP = 0.29)
Self-Employment - Creates additional risk (SHAP = 0.22)
