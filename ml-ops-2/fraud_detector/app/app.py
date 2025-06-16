import os
import time
import json
import logging

from kafka import KafkaConsumer, KafkaProducer
from src.preprocessing import preprocess
from src.scorer        import score

# Конфигурация через переменные окружения
KAFKA_SERVERS = os.getenv("KAFKA_BOOTSTRAP_SERVERS", "kafka:9092")
IN_TOPIC      = os.getenv("KAFKA_TRANSACTIONS_TOPIC", "transactions")
OUT_TOPIC     = os.getenv("KAFKA_SCORING_TOPIC",      "scores")

# Логирование в файл и в консоль
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(message)s",
    handlers=[
        logging.FileHandler("/app/logs/service.log", encoding="utf-8"),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger("fraud-scorer")

# Инициализируем Kafka Consumer и Producer
consumer = KafkaConsumer(
    IN_TOPIC,
    bootstrap_servers=KAFKA_SERVERS,
    value_deserializer=lambda m: json.loads(m.decode("utf-8")),
    auto_offset_reset="earliest",
    group_id="ml-scorer"
)
producer = KafkaProducer(
    bootstrap_servers=KAFKA_SERVERS,
    value_serializer=lambda m: json.dumps(m).encode("utf-8")
)

# --- Основной цикл обработки сообщений ---
logger.info(f"Сервис скоринга запущен, слушаем топик `{IN_TOPIC}`")
while True:
    # Читаем пачку сообщений (timeout_ms=500)
    raw_msgs = consumer.poll(timeout_ms=500)
    if not raw_msgs:
        continue

    for tp, records in raw_msgs.items():
        for rec in records:
            try:
                data = rec.value  # Это dict из JSON
                tx_id = data["transaction_id"]

                # 1. Препроцессинг
                df_proc = preprocess(data)

                # 2. Скоринг
                proba, fraud_flag = score(df_proc)

                # 3. Формируем и отправляем результат
                out = {
                    "transaction_id": tx_id,
                    "score": proba,
                    "fraud_flag": fraud_flag
                }
                producer.send(OUT_TOPIC, out)
                producer.flush()
                logger.info(f"Обработана транзакция {tx_id}: {out}")

            except Exception as e:
                logger.exception(f"Ошибка при обработке {data.get('transaction_id')}: {e}")

    # чуть ждём, чтобы не сломать пропускную способность
    time.sleep(0.01)