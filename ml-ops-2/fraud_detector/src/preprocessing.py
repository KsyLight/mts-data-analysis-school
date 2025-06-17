import pandas as pd
import numpy as np

def preprocess(rec: dict) -> pd.DataFrame:
    """
    Преобразуем один JSON-рекорд в DataFrame и добавляем фичи:
    - час, день недели, месяц, признак выходного
    - логарифм суммы транзакции
    """
    # Оборачиваем словарь в DataFrame из одной строки
    df = pd.DataFrame([rec])

    # Обрабатываем поле времени транзакции:
    # если ключ отсутствует или некорректен — используем текущее UTC-время
    if 'transaction_time' not in df.columns:
        df['transaction_time'] = pd.Timestamp.utcnow()
    else:
        df['transaction_time'] = pd.to_datetime(df['transaction_time'], errors='coerce')
        df['transaction_time'].fillna(pd.Timestamp.utcnow(), inplace=True)

    # Извлекаем временные признаки
    df['hour']       = df['transaction_time'].dt.hour
    df['dayofweek']  = df['transaction_time'].dt.dayofweek
    df['month']      = df['transaction_time'].dt.month
    df['is_weekend'] = (df['dayofweek'] >= 5).astype(int)

    # Удаляем исходный timestamp
    df.drop(columns=['transaction_time'], inplace=True)

    # Логарифмическое преобразование суммы транзакции
    if 'amount' in df.columns:
        df['amount_log'] = np.log1p(df['amount'])
    else:
        df['amount_log'] = np.nan

    return df