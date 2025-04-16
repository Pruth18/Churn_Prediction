import pandas as pd

def encode_categoricals(df: pd.DataFrame, drop_first=True) -> pd.DataFrame:
    """Encode categorical features using one-hot encoding."""
    categorical_cols = df.select_dtypes(include=['object', 'category']).columns.tolist()
    categorical_cols = [col for col in categorical_cols if col not in ['customerID', 'Churn']]
    return pd.get_dummies(df, columns=categorical_cols, drop_first=drop_first)


def encode_target(df: pd.DataFrame) -> pd.DataFrame:
    """Encode the target variable 'Churn' as binary."""
    df['Churn'] = df['Churn'].map({'No': 0, 'Yes': 1})
    return df
