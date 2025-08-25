# Импорт библиотек
#
import pandas as pd

# Визуализация
import matplotlib.pyplot as plt
import seaborn as sns

# 
import re
from typing import Dict, Optional, Tuple
from IPython.display import display

# ====================================================================================

# Функции
# 

def gen_info(df, n: int = 5):
    """
    Простой вывод общей информации.
    """
    df.info()
    df.sample(n)

# ------------------------------------------------------------------------------------

def plot_class_distribution(df: pd.DataFrame, label_column: str = 'label') -> None:
    """
    Визуализирует распределение классов в датасете.

    Параметры:
    df: DataFrame с данными
    label_column: название колонки с метками классов
    """
    class_counts = df[label_column].value_counts().sort_index()

    plt.figure(figsize=(10, 6))
    sns.barplot(x=class_counts.index, y=class_counts.values, hue=['Легит', 'Фрод'], palette=['#4CAF50', '#F44336'])
    plt.xticks([0, 1], ['Легит', 'Фрод'])
    plt.ylabel('Количество примеров')
    plt.xlabel('Класс')
    plt.title('Распределение классов')
    plt.grid(axis='y', linestyle='--', alpha=0.6)
    
    for i, v in enumerate(class_counts.values):
        plt.text(i, v + 1, str(v), ha='center', fontweight='bold')

    plt.show()

# ------------------------------------------------------------------------------------

def show_missing_values(df: pd.DataFrame, name: str = "Dataset") -> None:
    """
    Отображает количество пропусков в абсолютных и относительных значениях.

    Параметры:
    df: DataFrame с данными
    name: название датасета (по умолчанию "Dataset")
    """
    print(f"Количество пропусков в абсолютных и относительных значениях в {name}:")
    display(
        pd.DataFrame({
            'Total NaN': df.isna().sum(),
            'Percentage NaN': df.isna().mean() * 100
        })
        .style.background_gradient('coolwarm')
        .format({'Percentage NaN': '{:.2f}%'})
    )

# ------------------------------------------------------------------------------------

def add_length_columns(df: pd.DataFrame, text_col: str = "text") -> pd.DataFrame:
    """
    Добавляет колонки длины текста: 'char_len' и 'word_len' (копия df).

    Пераметры:
    df: DataFrame с колонкой текста
    text_col: имя колонки с текстом

    Возвращает:
    Новый DataFrame с добавленными колонками
    """
    d = df.copy()
    d['char_len'] = d[text_col].astype(str).str.len()
    d['word_len'] = d[text_col].astype(str).str.split().str.len()
    return d

def describe_lengths(df_with_len: pd.DataFrame) -> pd.DataFrame:
    """
    Возвращает описательные статистики по колонкам 'char_len' и 'word_len'.

    Параметры:
    df_with_len: DataFrame, где уже есть 'char_len' и 'word_len'
    """
    stats = pd.DataFrame([{
        "char_mean": df_with_len['char_len'].mean(),
        "char_median": df_with_len['char_len'].median(),
        "char_p95": df_with_len['char_len'].quantile(0.95),
        "char_p99": df_with_len['char_len'].quantile(0.99),
        "word_mean": df_with_len['word_len'].mean(),
        "word_median": df_with_len['word_len'].median(),
        "word_p95": df_with_len['word_len'].quantile(0.95),
        "word_p99": df_with_len['word_len'].quantile(0.99),
    }])

    return stats

def plot_length_distributions(df_with_len: pd.DataFrame, bins: int = 60, log_scale: bool = False) -> None:
    """
    Рисует гистограммы и boxplot для "char_len" и "word_len" на всём датасете.

    Параметры:
    df_with_len: DataFrame с "char_len" и "word_len"
    bins: число корзин/бинов для гистограмм
    log_scale: если True, ось Y в лог-шкале
    """
    # Гистограммы
    plt.figure(figsize=(10, 6))
    plt.subplot(1, 2, 1)
    plt.hist(df_with_len["char_len"], bins=bins)
    plt.title("Распределение длины (символы)")
    plt.xlabel("символов")
    plt.ylabel("кол-во")
    if log_scale: plt.yscale("log")

    plt.subplot(1, 2, 2)
    plt.hist(df_with_len["word_len"], bins=bins)
    plt.title("Распределение длины (слова)")
    plt.xlabel("слов")
    plt.ylabel("кол-во")
    if log_scale: plt.yscale("log")

    plt.tight_layout()
    plt.show()

    # Boxplot
    plt.figure(figsize=(10, 6))
    plt.boxplot(
        [df_with_len["char_len"], df_with_len["word_len"]],
        labels=["char_len", "word_len"],
        showfliers=True,
        vert=False
    )
    plt.title("Boxplot длин текстов")
    plt.show()

def short_long_shares(df_with_len: pd.DataFrame, short_thresh: int = 5, long_thresh: int = 500) -> None:
    """
    Считает и выводит долю очень коротких (< short_thresh слов) и очень длинных (> long_thresh слов) текстов.

    Параметры:
    df_with_len: DataFrame с 'word_len'
    short_thresh: порог для коротких
    long_thresh: порог для длинных
    """
    share_short = (df_with_len["word_len"] < short_thresh).mean()
    share_long = (df_with_len["word_len"] > long_thresh).mean()

    print(f"Доля коротких (<{short_thresh} слов): {share_short:.2%}")
    print(f"Доля длинных (>{long_thresh} слов): {share_long:.2%}")