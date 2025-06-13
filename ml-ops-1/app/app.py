#!/usr/bin/env python
import os
import time
import logging
import sys
import pandas as pd
import matplotlib.pyplot as plt
import mplcyberpunk
from watchdog.observers.polling import PollingObserver as Observer
from watchdog.events import FileSystemEventHandler

from src.preprocessing import preprocess
from src.scorer        import make_pred, get_feature_importance

plt.style.use('cyberpunk')

INPUT_DIR  = "/app/input"
OUTPUT_DIR = "/app/output"
LOG_DIR    = "/app/logs"

logging.basicConfig(
    level=logging.DEBUG,
    format="%(asctime)s %(levelname)s %(message)s",
    handlers=[
        logging.FileHandler(os.path.join(LOG_DIR, "service.log"), encoding="utf-8"),
        logging.StreamHandler(sys.stdout),
    ]
)
logger = logging.getLogger(__name__)

class Handler(FileSystemEventHandler):
    def on_created(self, event):
        logger.debug(f"on_created: {event.src_path}")
        if not event.is_directory and event.src_path.lower().endswith(".csv"):
            self.process(event.src_path)

    def on_modified(self, event):
        logger.debug(f"on_modified: {event.src_path}")
        if not event.is_directory and event.src_path.lower().endswith(".csv"):
            self.process(event.src_path)

    def process(self, path: str):
        try:
            logger.info(f"=== Обработка {path} ===")
            df = pd.read_csv(path)
            if "id" in df.columns:
                df.set_index("id", inplace=True)
                logger.info("Индекс по 'id' установлен")

            X = preprocess(df)
            logger.info(f"После preprocess: {X.shape}")

            preds, proba, idx = make_pred(X, path)
            logger.info("Сделан скоринг")

            os.makedirs(OUTPUT_DIR, exist_ok=True)

            # sample_submission.csv
            sub = pd.DataFrame({"id": idx, "target": preds})
            sub.to_csv(os.path.join(OUTPUT_DIR, "sample_submission.csv"), index=False)
            logger.info("sample_submission.csv сохранён")

            # feature_importance.json
            fi_df = get_feature_importance(5)
            fi = {r["Feature Id"]: float(r["Importances"]) for r in fi_df.to_dict("records")}
            import json
            with open(os.path.join(OUTPUT_DIR, "feature_importance.json"), "w", encoding="utf-8") as f:
                json.dump(fi, f, indent=2, ensure_ascii=False)
            logger.info("feature_importance.json сохранён")

            # pred_density.png
            plt.switch_backend("Agg")
            plt.figure(figsize=(8, 6))
            
            # Гистограмма плотности
            pd.Series(proba, name="proba")\
              .plot(kind="hist", bins=40, density=True, alpha=0.6, label="Гистограмма")
            
            # KDE-кривая
            pd.Series(proba).plot(kind="kde", label="KDE")
            plt.title("Плотность предсказанных вероятностей", fontsize=14)
            plt.xlabel("Вероятность", fontsize=12)
            plt.ylabel("Плотность", fontsize=12)
            plt.grid(True, linestyle="--", linewidth=0.5)
            plt.legend()
            plt.tight_layout()
            beaut_path = os.path.join(OUTPUT_DIR, "pred_density.png")
            plt.savefig(beaut_path, dpi=300)
            plt.close()
            logger.info(f"pred_density.png сохранён ➜ {beaut_path}")


        except Exception:
            logger.exception(f"Ошибка при обработке {path}")

if __name__ == "__main__":
    os.makedirs(INPUT_DIR,  exist_ok=True)
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    os.makedirs(LOG_DIR,    exist_ok=True)

    handler = Handler()

    # Startup scan
    for fn in os.listdir(INPUT_DIR):
        if fn.lower().endswith(".csv"):
            p = os.path.join(INPUT_DIR, fn)
            logger.info(f"[startup scan] обрабатываю {p}")
            handler.process(p)

    observer = Observer()
    observer.schedule(handler, INPUT_DIR, recursive=False)
    observer.start()
    logger.info("Сервис запущен, ждём CSV…")
    try:
        while True:
            time.sleep(1)
    except KeyboardInterrupt:
        observer.stop()
    observer.join()
    logger.info("Сервис остановлен.")