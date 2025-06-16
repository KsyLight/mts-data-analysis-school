import os
import pandas as pd
import psycopg2
import matplotlib.pyplot as plt
import streamlit as st

# Подключаемся к БД по переменной окружения
DSN = os.getenv("POSTGRES_DSN")

@st.cache_data(ttl=60)
def load_data(query: str) -> pd.DataFrame:
    conn = psycopg2.connect(DSN)
    df = pd.read_sql(query, conn)
    conn.close()
    return df

st.title("Fraud Dashboard")

# Кнопка для загрузки результатов
if st.button("Посмотреть мошенничества"):
    # Последние 10 фродовых транзакций
    q1 = """
      SELECT transaction_id, score
      FROM scores
      WHERE fraud_flag = TRUE
      ORDER BY ts DESC
      LIMIT 10
    """
    df_fraud = load_data(q1)
    st.subheader("Последние 10 мошеннических транзакций")
    st.table(df_fraud)

    # Гистограмма распределения скор-рейтов последних 100 записей
    q2 = """
      SELECT score
      FROM scores
      ORDER BY ts DESC
      LIMIT 100
    """
    df_scores = load_data(q2)
    st.subheader("Распределение скор-рейтов (последние 100)")
    fig, ax = plt.subplots(figsize=(6,4))
    ax.hist(df_scores["score"], bins=20, density=True, alpha=0.7)
    ax.set_xlabel("Score")
    ax.set_ylabel("Density")
    st.pyplot(fig)