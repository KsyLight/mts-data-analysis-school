import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedKFold

def make_folds_and_report(y_pool: np.ndarray, n_splits: int, seed: int = 13):
    """
    Возвращает список фолдов [(tr_idx, va_idx), ...] и отчёт DataFrame
    с размерами/балансом классов валидации по каждому фолду.
    """
    skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=seed)
    folds = [(tr, va) for tr, va in skf.split(np.zeros(len(y_pool)), y_pool)]

    # Формируем компактный отчёт по валидациям
    p_full = y_pool.mean()
    rows = []
    for i, (_, va) in enumerate(folds, 1):
        yv = y_pool[va]
        rows.append({
            "fold": i,
            "valid_size": len(va),
            "pos_valid": int(yv.sum()),
            "neg_valid": int(len(yv) - yv.sum()),
            "pos_rate_valid": float(yv.mean()),
            "pos_rate_delta_vs_pool": float(yv.mean() - p_full),
        })
    rep = pd.DataFrame(rows).sort_values("fold").reset_index(drop=True)
    rep.loc["summary", "valid_size"] = rep["valid_size"].sum()
    rep.loc["summary", "pos_valid"] = rep["pos_valid"].sum()
    rep.loc["summary", "neg_valid"] = rep["neg_valid"].sum()
    rep.loc["summary", "pos_rate_valid"] = rep["pos_valid"].sum() / rep["valid_size"].sum()

    # Базовые инварианты фолдов
    all_valid = np.concatenate([va for _, va in folds])
    assert len(np.unique(all_valid)) == len(all_valid), "Пересечение между validation-частями!"
    assert set(all_valid) == set(range(len(y_pool))), "Фолды не покрывают весь train_pool!"
    for i, (_, va) in enumerate(folds, 1):
        yv = y_pool[va]
        assert yv.min() != yv.max(), f"Фолд {i} содержит один класс — уменьшите n_splits или смените seed."

    return folds, rep