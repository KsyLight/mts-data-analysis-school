# Берём «лёгкий» образ с Python 3.12
FROM python:3.12-slim

# Рабочая директория внутри контейнера
WORKDIR /app

# Создаём папку для логов и даём права на запись
RUN mkdir -p /app/logs && chmod 777 /app/logs

# Копируем список зависимостей и устанавливаем их
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Копируем код и модель
COPY src/   ./src/
COPY app/   ./app/
COPY models/my_catboost.cbm ./models/

# Чтобы внутри контейнера корректно работал import src.*
ENV PYTHONPATH=/app

# Точки монтирования (если вдруг вам нужно заливать/выгружать файлы вручную)
VOLUME /app/input
VOLUME /app/output

# Запускаем наш скрипт-обработчик
ENTRYPOINT ["python", "app/app.py"]