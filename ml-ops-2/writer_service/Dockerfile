FROM python:3.12-slim
WORKDIR /app

# Устанавливаем зависимости
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Копируем код писателя
COPY src/ ./src/

# Запускаем писатель
ENTRYPOINT ["python", "src/writer.py"]