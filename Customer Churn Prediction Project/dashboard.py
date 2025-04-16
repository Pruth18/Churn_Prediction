import streamlit as st
import pandas as pd
from data_preprocessing import clean_data
from feature_engineering import add_features
from utils import encode_categoricals
from modeling import train_best_model
from explainability import explain_with_shap

st.title('Customer Churn Prediction Dashboard')

uploaded_file = st.file_uploader("Upload customer data CSV", type=["csv"])

if uploaded_file:
    df = pd.read_csv(uploaded_file)
    df = clean_data(df)
    df = add_features(df)
    df = encode_categoricals(df)
    # Ensure Churn column is binary for modeling
    if 'Churn' in df.columns:
        df['Churn'] = df['Churn'].map({'No': 0, 'Yes': 1})
    # Example: Churn Distribution Plot
    import matplotlib.pyplot as plt
    st.markdown('## Churn Distribution')
    fig1, ax1 = plt.subplots()
    df['Churn'].value_counts().plot(kind='bar', ax=ax1)
    fig1.suptitle('Churn Distribution')
    st.pyplot(fig1)

    # Example: Monthly Charges Distribution
    st.markdown('## Monthly Charges Distribution')
    fig2, ax2 = plt.subplots()
    df['MonthlyCharges'].hist(ax=ax2, bins=20)
    fig2.suptitle('Monthly Charges Distribution')
    st.pyplot(fig2)

    # Add more plots as needed, each with a unique fig.suptitle and st.markdown section

    X = df.drop(['customerID', 'Churn'], axis=1)
    y = df['Churn'] if 'Churn' in df else None
    model = train_best_model(X, y) if y is not None else None

    if y is not None:
        st.write("## Data Preview", df.head())
        st.write("## Churn Distribution", df['Churn'].value_counts())
        st.write("## Model Performance")
        # Could show metrics here if test data is available
        st.write("Model trained on uploaded data.")
        
        # Model Explainability Section
        if model is not None:
            st.write("## Model Explainability (SHAP)")
            with st.spinner('Generating SHAP explanations...'):
                explain_with_shap(model, X, use_streamlit=True, st=st)
        else:
            st.info('Train a model to see SHAP explanations.')
    else:
        st.write("Upload data with 'Churn' column for full analysis.")
else:
    st.write("Upload a CSV file to begin analysis.")
