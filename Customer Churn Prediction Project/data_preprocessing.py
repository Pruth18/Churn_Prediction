import pandas as pd

def load_data(path: str) -> pd.DataFrame:
    """Load the Telco Customer Churn dataset."""
    return pd.read_csv(path)


def clean_data(df: pd.DataFrame) -> pd.DataFrame:
    """Basic cleaning: handle missing values, convert columns as needed."""
    # Remove spaces in column names
    df.columns = df.columns.str.strip()
    # Remove customerID with missing TotalCharges
    df = df[df['TotalCharges'] != ' '].copy()
    df['TotalCharges'] = pd.to_numeric(df['TotalCharges'])
    return df.reset_index(drop=True)
