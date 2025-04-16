from fastapi import FastAPI
from pydantic import BaseModel
import pandas as pd
import joblib
from feature_engineering import add_features
from utils import encode_categoricals
from data_preprocessing import clean_data
from modeling import train_best_model

app = FastAPI()

class CustomerData(BaseModel):
    data: dict

# Load model at startup (for demo, retrain on each start; in production, load pre-trained model)
model = None

@app.on_event("startup")
def load_model():
    global model
    # For demo, train on full data (in production, load from file)
    df = pd.read_csv('WA_Fn-UseC_-Telco-Customer-Churn.csv')
    df = clean_data(df)
    df = add_features(df)
    df = encode_categoricals(df)
    X = df.drop(['customerID', 'Churn'], axis=1)
    y = df['Churn'].map({'No': 0, 'Yes': 1})
    model = train_best_model(X, y, model_name='XGBoost')

@app.post("/predict")
def predict_churn(customer: CustomerData):
    global model
    input_df = pd.DataFrame([customer.data])
    input_df = add_features(input_df)
    input_df = encode_categoricals(input_df)
    # Align columns with training data
    X = pd.read_csv('WA_Fn-UseC_-Telco-Customer-Churn.csv')
    X = clean_data(X)
    X = add_features(X)
    X = encode_categoricals(X)
    train_cols = X.drop(['customerID', 'Churn'], axis=1).columns
    input_df = input_df.reindex(columns=train_cols, fill_value=0)
    proba = model.predict_proba(input_df)[0, 1]
    return {"churn_probability": float(proba)}
