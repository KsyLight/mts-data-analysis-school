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
    # Конвертируем время
    df['transaction_time'] = pd.to_datetime(df['transaction_time'], errors='coerce')
    # Извлекаем признаки из времени
    df['hour']       = df['transaction_time'].dt.hour
    df['dayofweek']  = df['transaction_time'].dt.dayofweek
    df['month']      = df['transaction_time'].dt.month
    df['is_weekend'] = (df['dayofweek'] >= 5).astype(int)
    # Убираем исходный timestamp
    df.drop(columns=['transaction_time'], inplace=True)
    # Логарифмическое преобразование суммы
    if 'amount' in df.columns:
        df['amount_log'] = np.log1p(df['amount'])
    return df