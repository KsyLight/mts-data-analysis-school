# Real-Time Fraud Detection System

> Используемые датасеты — из соревнования <https://www.kaggle.com/competitions/teta-ml-1-2025>.

Система выявляет мошеннические транзакции в режиме реального времени. Данные поступают в Kafka-топик **`transactions`**, обрабатываются ML-сервисом и результаты публикуются в топик **`scoring`**.

---

## Архитектура

| Компонент       | Роль                               | Порт (host) |
|-----------------|------------------------------------|-------------|
| **interface**   | Streamlit UI — имитация потока CSV | 8501 |
| **fraud_detector** | ML-сервис (CatBoost + preprocess) | — |
| **writer**      | Запись скорингов в Postgres        | — |
| **postgres**    | Хранилище результатов              | 5432 |
| **kafka / zookeeper** | Брокер сообщений              | 9092 / 2181 |
| **kafka-setup** | Cоздание топиков (*1 раз*)         | — |
| **kafka-ui**    | Веб-обзор Kafka                    | 8080 |

---

## Быстрый старт

### Предварительно

* **Docker 20.10+** и **docker-compose v2**  
* (Опц.) **Git LFS** — если вес модели > 100 MB  
  ```bash
  git lfs install

git clone https://github.com/<YOUR-NAME>/fraud-detection-system.git
cd fraud-detection-system

# при необходимости подтянуть вес модели
git lfs pull

# собрать и запустить всё
docker compose up --build -d
