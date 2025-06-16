# интерфейсная служба на Streamlit
FROM python:3.12-slim
WORKDIR /app

# Устанавливаем зависимости
COPY requirements.txt ./
RUN pip install --no-cache-dir -r requirements.txt

# Копируем остальной код
COPY app.py ./

# Копируем конфиг Streamlit
RUN mkdir -p .streamlit
COPY .streamlit/config.toml .streamlit/config.toml

# Открываем порт 8501
EXPOSE 8501

# Запускаем Streamlit
ENTRYPOINT ["streamlit", "run", "app.py", \
            "--server.address=0.0.0.0", "--server.port=8501"]