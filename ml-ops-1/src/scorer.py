import pandas as pd
import logging
from catboost import CatBoostClassifier

logger = logging.getLogger(__name__)

model = CatBoostClassifier()
model.load_model("./models/cb_ml1_alg.cbm")
logger.info("CatBoost модель загружена.")

THRESHOLD = 0.68

def make_pred(df_proc: pd.DataFrame, path_to_file: str):
    """
    df_proc – DataFrame после preprocess()
    path_to_file – путь к test.csv для чтения оригинальных индексов
    """
    proba = model.predict_proba(df_proc)[:,1]
    preds = (proba > THRESHOLD).astype(int)

    idx = pd.read_csv(path_to_file).index
    return preds, proba, idx

def get_feature_importance(n: int = 5) -> pd.DataFrame:
    """
    Топ-n признаков по важности.
    """
    fi = model.get_feature_importance(prettified=True).head(n)
    return fi