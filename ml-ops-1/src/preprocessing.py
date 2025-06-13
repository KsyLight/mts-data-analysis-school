import pandas as pd
import numpy as np

def preprocess(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()

    df['transaction_time'] = pd.to_datetime(df['transaction_time'], errors='coerce')
    df['hour']       = df['transaction_time'].dt.hour
    df['dayofweek']  = df['transaction_time'].dt.dayofweek
    df['month']      = df['transaction_time'].dt.month
    df['is_weekend'] = (df['dayofweek'] >= 5).astype(int)
    df.drop(columns=['transaction_time'], inplace=True, errors='ignore')

    if 'amount' in df.columns:
        df['amount_log'] = np.log1p(df['amount'])

    return df