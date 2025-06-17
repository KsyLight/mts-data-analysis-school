import os
import time
import json
import logging
from kafka import KafkaConsumer
from kafka.errors import NoBrokersAvailable
import psycopg2

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("writer")

# Ждём, пока Kafka не станет доступна
while True:
    try:
        consumer = KafkaConsumer(
            os.getenv("INPUT_TOPIC"),
            bootstrap_servers=os.getenv("KAFKA_BOOTSTRAP_SERVERS"),
            group_id="writer",
            auto_offset_reset="earliest"
        )
        logger.info("Connected to Kafka")
        break
    except NoBrokersAvailable:
        logger.info("Kafka unavailable, retrying in 1s...")
        time.sleep(1)

# Подключаемся к PostgreSQL
dsn = os.getenv("POSTGRES_DSN")
conn = psycopg2.connect(dsn)
cursor = conn.cursor()
logger.info("Connected to Postgres")

# Основной цикл: читаем сообщения и пишем в таблицу scores
for msg in consumer:
    try:
        record = json.loads(msg.value.decode("utf-8"))
        transaction_id = record["transaction_id"]
        score = record["score"]
        fraud_flag = record["fraud_flag"]

        cursor.execute(
            "INSERT INTO scores (transaction_id, score, fraud_flag) VALUES (%s, %s, %s)",
            (transaction_id, score, fraud_flag)
        )
        conn.commit()
        logger.info(f"Written to DB: {transaction_id}, score={score}, fraud={fraud_flag}")
    except Exception as e:
        logger.error(f"Error writing record {msg.value}: {e}")
        conn.rollback()