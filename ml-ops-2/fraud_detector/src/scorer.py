import logging
import pandas as pd
from catboost import CatBoostClassifier

# Настраиваем логгер
logger = logging.getLogger(__name__)

# Загружаем сохранённую модель
model = CatBoostClassifier()
model.load_model("./models/cb_ml1_alg.cbm")
logger.info("CatBoost модель загружена из ./models/my_catboost.cbm")

# Порог для бинаризации
THRESHOLD = 0.5

def score(df: pd.DataFrame) -> tuple[float, bool]:
    """
    Предскаиваем proba и флаг fraud_flag:
    - proba  — вероятность мошенничества
    - fraud  — True, если proba > THRESHOLD
    """
    proba = model.predict_proba(df)[:, 1]
    fraud = bool(proba[0] > THRESHOLD)
    return float(proba[0]), fraud