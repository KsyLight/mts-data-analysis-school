CREATE TABLE IF NOT EXISTS scores (
  transaction_id TEXT PRIMARY KEY,
  score          REAL    NOT NULL,
  fraud_flag     BOOLEAN NOT NULL,
  ts             TIMESTAMP NOT NULL DEFAULT now()
);