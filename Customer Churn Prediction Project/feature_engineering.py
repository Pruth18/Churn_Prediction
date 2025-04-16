import pandas as pd
import numpy as np

def add_features(df: pd.DataFrame) -> pd.DataFrame:
    """Create new features for churn prediction."""
    # Tenure buckets
    df['TenureGroup'] = pd.cut(df['tenure'], bins=[0, 12, 24, 48, 72], labels=['0-12', '12-24', '24-48', '48-72'])
    # Number of services
    services = ['PhoneService', 'InternetService', 'OnlineSecurity', 'OnlineBackup',
                'DeviceProtection', 'TechSupport', 'StreamingTV', 'StreamingMovies']
    df['NumServices'] = df[services].apply(lambda x: np.sum(x == 'Yes'), axis=1)
    # Charges per tenure
    df['AvgChargesPerMonth'] = df['TotalCharges'] / (df['tenure'].replace(0, 1))
    # Binary flags
    df['HasStreaming'] = ((df['StreamingTV'] == 'Yes') | (df['StreamingMovies'] == 'Yes')).astype(int)
    df['HasSecurity'] = ((df['OnlineSecurity'] == 'Yes') | (df['DeviceProtection'] == 'Yes')).astype(int)
    # Interaction term
    df['SeniorContract'] = ((df['SeniorCitizen'] == 1) & (df['Contract'] == 'Month-to-month')).astype(int)
    return df
