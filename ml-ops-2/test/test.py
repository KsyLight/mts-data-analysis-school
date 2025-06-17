import pandas as pd
import numpy as np
import pytest

from fraud_detector.src.preprocessing import preprocess

@pytest.fixture
def sample_record():
    return {
        "transaction_id": "123",
        "amount": 150.5,
        "transaction_time": "2025-06-17T10:30:00"
    }

def test_preprocess_contains_expected_columns(sample_record):
    df = preprocess(sample_record)
    expected = {"transaction_id", "amount", "hour", "dayofweek", "month", "is_weekend", "amount_log"}
    assert expected.issubset(set(df.columns))

def test_preprocess_values(sample_record):
    df = preprocess(sample_record)
    assert df.loc[0, "hour"] == 10
    assert df.loc[0, "dayofweek"] == pd.to_datetime(sample_record["transaction_time"]).dayofweek
    assert np.isclose(df.loc[0, "amount_log"], np.log1p(sample_record["amount"]))

def test_missing_transaction_time():
    rec = {"transaction_id": "X", "amount": 0.0}
    df = preprocess(rec)
    assert pd.isna(df.loc[0, "hour"] )
    # is_weekend может быть 0 или 1, проверяем корректность типа
    assert df.loc[0, "is_weekend"] in (0, 1)