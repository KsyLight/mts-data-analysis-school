import os
import json
import logging
import psycopg2
from kafka import KafkaConsumer

# Логирование
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(message)s"
)
logger = logging.getLogger("writer")

# Конфиг через окружение
KAFKA_SERVERS = os.getenv("KAFKA_BOOTSTRAP_SERVERS", "kafka:9092")
IN_TOPIC      = os.getenv("INPUT_TOPIC",          "scores")
POSTGRES_DSN  = os.getenv("POSTGRES_DSN")

# Консьюмер Kafka
consumer = KafkaConsumer(
    IN_TOPIC,
    bootstrap_servers=KAFKA_SERVERS,
    value_deserializer=lambda m: json.loads(m.decode("utf-8")),
    auto_offset_reset="earliest",
    group_id="db-writer"
)

# Подключаемся к Postgres
conn = psycopg2.connect(POSTGRES_DSN)
cur  = conn.cursor()
logger.info("Сервис-писатель запущен, слушаем топик `scores`")

for rec in consumer:
    try:
        data = rec.value
        cur.execute(
            """
            INSERT INTO scores(transaction_id, score, fraud_flag, ts)
            VALUES (%s, %s, %s, now())
            ON CONFLICT (transaction_id) DO NOTHING
            """,
            (data["transaction_id"], data["score"], data["fraud_flag"])
        )
        conn.commit()
        logger.info(f"Записано в БД: {data}")
    except Exception as e:
        logger.exception(f"Ошибка записи в БД: {e}")