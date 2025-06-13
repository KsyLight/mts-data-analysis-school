FROM python:3.12-slim

WORKDIR /app

# Чтобы логи валились без проблем
RUN mkdir -p /app/logs && chmod 777 /app/logs

# Устанавливаем зависимости
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Копируем весь проект
COPY . .

# Говорим Python: ищи src/ в /app
ENV PYTHONPATH=/app

# Монтируем папки для входа и выхода
VOLUME /app/input
VOLUME /app/output

# Запускаем сервис
ENTRYPOINT ["python", "app/app.py"]