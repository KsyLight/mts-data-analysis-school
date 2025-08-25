# Импорт библиотек
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F

from typing import List, Dict, Any, Tuple
from dataclasses import asdict
from tqdm import tqdm

import seaborn as sns
import matplotlib.pyplot as plt

from sklearn.model_selection import StratifiedKFold, StratifiedShuffleSplit
from sklearn.metrics import (
    precision_score, recall_score, f1_score,
    average_precision_score, brier_score_loss, confusion_matrix,
    precision_recall_curve
)

from torch.utils.data import Dataset, DataLoader
from transformers import (
    AutoTokenizer, AutoModelForSequenceClassification,
    get_linear_schedule_with_warmup
)
from peft import LoraConfig, TaskType, get_peft_model

# ------------------------------------------------------------------------------------

def make_folds_and_report(y_pool: np.ndarray, n_splits: int, seed: int = 13):
    skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=seed)
    folds = [(tr, va) for tr, va in skf.split(np.zeros(len(y_pool)), y_pool)]

    p_full = y_pool.mean()
    rows = []
    for i, (_, va) in enumerate(folds, 1):
        yv = y_pool[va]
        rows.append({
            'fold': i,
            'valid_size': len(va),
            'pos_valid': int(yv.sum()),
            'neg_valid': int(len(yv) - yv.sum()),
            'pos_rate_valid': float(yv.mean()),
            'pos_rate_delta_vs_pool': float(yv.mean() - p_full),
        })
    rep = pd.DataFrame(rows).sort_values('fold').reset_index(drop=True)
    rep.loc['summary', 'valid_size'] = rep['valid_size'].sum()
    rep.loc['summary', 'pos_valid'] = rep['pos_valid'].sum()
    rep.loc['summary', 'neg_valid'] = rep['neg_valid'].sum()
    rep.loc['summary', 'pos_rate_valid'] = rep['pos_valid'].sum() / rep['valid_size'].sum()

    all_valid = np.concatenate([va for _, va in folds])
    assert len(np.unique(all_valid)) == len(all_valid), "Пересечения между валидациями"
    assert set(all_valid) == set(range(len(y_pool))), "Фолды не покрывают весь train_pool"
    for i, (_, va) in enumerate(folds, 1):
        yv = y_pool[va]
        assert yv.min() != yv.max(), f"Фолд {i} содержит один класс. Уменьшите n_splits или смените seed"

    return folds, rep

def stratified_lockbox_split(y: np.ndarray, test_size: float, seed: int) -> Tuple[np.ndarray, np.ndarray]:
    sss = StratifiedShuffleSplit(n_splits=1, test_size=test_size, random_state=seed)
    train_pool_idx, test_idx = next(sss.split(np.zeros(len(y)), y))

    assert len(set(train_pool_idx) & set(test_idx)) == 0, "Пересечение train_pool и test"
    assert len(train_pool_idx) + len(test_idx) == len(y), "Потеря или дублирование индексов"
    assert test_idx.size >= 2, "Тест слишком мал для стратификации"
    return train_pool_idx, test_idx

def map_folds_to_global(train_pool_idx: np.ndarray, folds: List[Tuple[np.ndarray, np.ndarray]]) -> List[Dict[str, np.ndarray]]:
    mapped = [
        {
            'train_idx': train_pool_idx[tr].astype(int),
            'valid_idx': train_pool_idx[va].astype(int),
        }
        for tr, va in folds
    ]

    all_valid_mapped = np.concatenate([d['valid_idx'] for d in mapped])
    assert len(np.unique(all_valid_mapped)) == len(all_valid_mapped), "Пересечения между валидациями после проекции"
    assert set(all_valid_mapped) == set(train_pool_idx.tolist()), "Валидации не покрывают train_pool полностью"
    return mapped

def build_validation_plan(df_clean: pd.DataFrame,
                          label_col: str,
                          test_size: float,
                          n_splits_desired: int,
                          seed_lockbox: int,
                          seed_folds: int) -> Dict[str, object]:
    y = df_clean[label_col].to_numpy()
    n_pos = int((y == 1).sum())
    n_neg = int((y == 0).sum())

    n_splits = min(n_splits_desired, n_pos, n_neg)
    assert n_pos > 0 and n_neg > 0, "Нужен минимум по одному объекту каждого класса"
    assert n_splits >= 2, "Слишком мало объектов для k-fold"

    train_pool_idx, test_idx = stratified_lockbox_split(y, test_size, seed_lockbox)

    p_full = y.mean()
    p_test = y[test_idx].mean()
    p_pool = y[train_pool_idx].mean()
    print(f"Размеры: train_pool={len(train_pool_idx)} | test={len(test_idx)}")
    print(f"Доля позитива: full={p_full:.3f} | train_pool={p_pool:.3f} | test={p_test:.3f}")

    df_pool = df_clean.iloc[train_pool_idx].reset_index(drop=True)
    y_pool = df_pool[label_col].to_numpy()

    folds, folds_report = make_folds_and_report(y_pool, n_splits=n_splits, seed=seed_folds)
    display(folds_report)

    folds_idx = map_folds_to_global(train_pool_idx, folds)

    all_train_mapped = np.concatenate([d['train_idx'] for d in folds_idx])
    all_valid_mapped = np.concatenate([d['valid_idx'] for d in folds_idx])
    assert set(test_idx).isdisjoint(all_valid_mapped), "Тест пересекается с валидациями"
    assert set(test_idx).isdisjoint(all_train_mapped), "Тест пересекается с тренировочными частями"

    print(f"Тестовая выборка (lockbox): {len(test_idx)} объектов")
    print(f"Фолдов: {len(folds_idx)}; пример размеров первого фолда — обучение {len(folds_idx[0]['train_idx'])}, валидация {len(folds_idx[0]['valid_idx'])}")
    print("Проверки корректности разбиений пройдены")

    return {
        'TEST_IDX': np.array(test_idx, dtype=int),
        'FOLDS_IDX': folds_idx,
        'folds_report': folds_report,
        'train_pool_idx': train_pool_idx,
        'y_pool': y_pool,
    }

def build_tokenizer(cfg: TrainConfig):
    return AutoTokenizer.from_pretrained(cfg.model_name)

class TextDataset(Dataset):
    def __init__(self, df: pd.DataFrame, text_col: str, label_col: str | None = None):
        self.texts = df[text_col].astype(str).tolist()
        self.labels = None if label_col is None else df[label_col].to_numpy().astype(float)
    def __len__(self): 
        return len(self.texts)
    def __getitem__(self, idx: int):
        item = {"text": self.texts[idx]}
        if self.labels is not None:
            item['label'] = self.labels[idx]
        return item

def build_collate_fn(tokenizer, max_length: int):
    def collate_fn(samples: List[Dict[str, Any]]):
        texts = [s['text'] for s in samples]
        enc = tokenizer(
            texts, padding=True, truncation=True, max_length=max_length, return_tensors="pt"
        )
        labels = None
        if "label" in samples[0]:
            labels = torch.tensor([s['label'] for s in samples], dtype=torch.float)
        return enc, labels
    return collate_fn

def compute_token_lengths(df: pd.DataFrame, tokenizer, text_col: str, batch_size: int = 512) -> pd.Series:
    texts = df[text_col].astype(str).tolist()
    lengths = []
    for i in range(0, len(texts), batch_size):
        batch = texts[i:i+batch_size]
        enc = tokenizer(
            batch,
            add_special_tokens=True,
            padding=False,
            truncation=False,
            return_length=True
        )
        if "length" in enc:
            lens = enc['length']
        else:
            lens = [len(tokenizer.tokenize(t)) + tokenizer.num_special_tokens_to_add(pair=False) for t in batch]
        lengths.extend(lens)
    return pd.Series(lengths, index=df.index, name="tok_len")


def plot_token_length_hist(lengths: pd.Series, max_len: int):
    plt.figure(figsize=(7,4))
    sns.histplot(lengths, bins=60, kde=False)
    for m in [128, 256, 384, 512]:
        if m <= max(int(lengths.max()), 128):
            plt.axvline(m, linestyle='--', color='red', alpha=0.8)
    plt.axvline(max_len, linestyle='--', color='red', linewidth=2, label=f"max_len={max_len}")
    plt.title("Распределение длин текстов (в токенах)")
    plt.xlabel("Длина (токены)")
    plt.ylabel("Количество")
    plt.legend()
    plt.tight_layout()
    plt.show()

def build_model(cfg: TrainConfig) -> nn.Module:
    base = AutoModelForSequenceClassification.from_pretrained(
        cfg.model_name,
        num_labels=1
    )
    try:
        base.gradient_checkpointing_enable()
    except Exception:
        pass

    if cfg.use_lora:
        peft_cfg = LoraConfig(
            task_type=TaskType.SEQ_CLS,
            r=cfg.lora_r,
            lora_alpha=cfg.lora_alpha,
            lora_dropout=cfg.lora_dropout,
            # target_modules=["query","value","key","dense"],
            target_modules=['query', 'value'],
            bias='none',
        )
        base = get_peft_model(base, peft_cfg)
        print("Используется LoRA (SEQ_CLS)")

    class Wrapper(nn.Module):
        def __init__(self, m): 
            super().__init__()
            self.m = m
        def forward(self, input_ids, attention_mask, token_type_ids=None):
            out = self.m(
                input_ids=input_ids,
                attention_mask=attention_mask,
                token_type_ids=token_type_ids
            )
            return out.logits.squeeze(-1)

    return Wrapper(base).to(cfg.device)

class FocalLoss(nn.Module):
    def __init__(self, gamma: float = 2.0, alpha: float = 0.25, reduction: str = "mean"):
        super().__init__()
        self.gamma = gamma
        self.alpha = alpha
        self.reduction = reduction
    def forward(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        bce = F.binary_cross_entropy_with_logits(logits, targets, reduction='none')
        probs = torch.sigmoid(logits)
        pt = torch.where(targets == 1, probs, 1 - probs)
        loss = (self.alpha * (1 - pt) ** self.gamma) * bce
        if self.reduction == "mean":
            return loss.mean()
        if self.reduction == "sum":
            return loss.sum()
        return loss

def build_loss(cfg: TrainConfig, y_train: np.ndarray) -> nn.Module:
    if cfg.loss_type == "focal":
        print("Используется Focal Loss")
        return FocalLoss(gamma=cfg.focal_gamma, alpha=cfg.focal_alpha)
    n_pos = int((y_train == 1).sum())
    n_neg = int((y_train == 0).sum())
    pos_weight = torch.tensor([n_neg / max(1, n_pos)], dtype=torch.float, device=cfg.device)
    print(f"Используется BCEWithLogitsLoss с pos_weight={pos_weight.item():.3f}")
    return nn.BCEWithLogitsLoss(pos_weight=pos_weight)

def logits_to_probs(logits: np.ndarray) -> np.ndarray:
    return 1.0 / (1.0 + np.exp(-logits))

def pr_auc(y_true: np.ndarray, y_scores: np.ndarray) -> float:
    return float(average_precision_score(y_true, y_scores))

def pick_threshold_by_f1(y_true: np.ndarray, y_scores: np.ndarray) -> float:
    prec, rec, thr = precision_recall_curve(y_true, y_scores)
    thr = np.append(thr, 1.0)
    f1 = 2 * prec * rec / np.clip(prec + rec, 1e-12, None)
    best_idx = int(np.nanargmax(f1))
    return float(thr[best_idx])

def pick_threshold_by_precision_floor(y_true: np.ndarray, y_scores: np.ndarray, min_precision: float) -> float:
    prec, rec, thr = precision_recall_curve(y_true, y_scores)
    thr = np.append(thr, 1.0)
    mask = prec >= min_precision
    if not mask.any():
        return pick_threshold_by_f1(y_true, y_scores)
    rec_masked = rec[mask]
    thr_masked = thr[mask]
    best_idx = int(np.argmax(rec_masked))
    return float(thr_masked[best_idx])

def report_at_threshold(y_true: np.ndarray, y_scores: np.ndarray, thr: float) -> Dict[str, Any]:
    y_pred = (y_scores >= thr).astype(int)
    return {
        'threshold': float(thr),
        'precision': float(precision_score(y_true, y_pred, zero_division=0)),
        'recall': float(recall_score(y_true, y_pred, zero_division=0)),
        'f1': float(f1_score(y_true, y_pred, zero_division=0)),
        'pr_auc': float(average_precision_score(y_true, y_scores)),
        'brier': float(brier_score_loss(y_true, y_scores)),
        'confusion_matrix': confusion_matrix(y_true, y_pred).tolist(),
    }

def build_loaders_for_fold(df: pd.DataFrame, train_idx: np.ndarray, valid_idx: np.ndarray,
                           tokenizer, cfg: TrainConfig) -> Tuple[DataLoader, DataLoader, np.ndarray]:
    df_tr = df.iloc[train_idx].reset_index(drop=True)
    df_va = df.iloc[valid_idx].reset_index(drop=True)

    ds_tr = TextDataset(df_tr, cfg.text_col, cfg.label_col)
    ds_va = TextDataset(df_va, cfg.text_col, cfg.label_col)

    collate_fn = build_collate_fn(tokenizer, cfg.max_length)

    loader_tr = DataLoader(
        ds_tr, batch_size=cfg.batch_size, shuffle=True,
        collate_fn=collate_fn,
        num_workers=cfg.num_workers, pin_memory=cfg.pin_memory,
        persistent_workers=bool(cfg.num_workers > 0 and cfg.persistent_workers),
        prefetch_factor=(cfg.prefetch_factor if cfg.num_workers > 0 else None),
    )
    loader_va = DataLoader(
        ds_va, batch_size=cfg.batch_size, shuffle=False,
        collate_fn=collate_fn,
        num_workers=cfg.num_workers, pin_memory=cfg.pin_memory,
        persistent_workers=bool(cfg.num_workers > 0 and cfg.persistent_workers),
        prefetch_factor=(cfg.prefetch_factor if cfg.num_workers > 0 else None),
    )

    y_tr = df_tr[cfg.label_col].to_numpy()
    return loader_tr, loader_va, y_tr

def train_one_fold(df: pd.DataFrame, train_idx: np.ndarray, valid_idx: np.ndarray,
                   tokenizer, cfg: TrainConfig) -> Dict[str, Any]:
    seed_everything(cfg.seed)
    model = build_model(cfg)
    loader_tr, loader_va, y_tr = build_loaders_for_fold(df, train_idx, valid_idx, tokenizer, cfg)
    criterion = build_loss(cfg, y_tr)

    optimizer = torch.optim.AdamW(model.parameters(), lr=cfg.lr, weight_decay=cfg.weight_decay)
    steps_total = cfg.epochs * len(loader_tr)
    warmup = int(cfg.warmup_prop * steps_total)
    scheduler = get_linear_schedule_with_warmup(optimizer, warmup, steps_total)
    scaler = torch.amp.GradScaler('cuda', enabled=cfg.use_amp)
    allowed_keys = {'input_ids', 'attention_mask', 'token_type_ids'}

    best_pr_auc, best_state = -1.0, None
    patience, no_improve = 2, 0

    for epoch in range(1, cfg.epochs + 1):
        model.train()
        running_loss = 0.0
        pbar = tqdm(loader_tr, desc=f"Эпоха {epoch}/{cfg.epochs} — обучение", leave=False)
        for enc, labels in pbar:
            enc = {k: v.to(cfg.device) for k, v in enc.items() if k in allowed_keys}
            labels = labels.to(cfg.device)

            optimizer.zero_grad(set_to_none=True)
            with torch.amp.autocast('cuda', enabled=cfg.use_amp):
                logits = model(**enc)
                loss = criterion(logits, labels)
            scaler.scale(loss).backward()
            nn.utils.clip_grad_norm_(model.parameters(), cfg.grad_clip_norm)
            scaler.step(optimizer)
            scaler.update()
            scheduler.step()

            running_loss += float(loss.item())
            pbar.set_postfix({"loss": f"{running_loss / (pbar.n or 1):.4f}"})

        model.eval()
        val_logits, val_labels = [], []
        with torch.no_grad():
            pbar_val = tqdm(loader_va, desc="Оценка на валидации", leave=False)
            for enc, labels in pbar_val:
                enc = {k: v.to(cfg.device) for k, v in enc.items() if k in allowed_keys}
                labels = labels.to(cfg.device)
                logits = model(**enc)
                val_logits.append(logits.detach().cpu().numpy())
                val_labels.append(labels.detach().cpu().numpy())

        val_logits = np.concatenate(val_logits)
        val_labels = np.concatenate(val_labels)
        val_probs = logits_to_probs(val_logits)
        fold_pr = pr_auc(val_labels, val_probs)
        print(f"Эпоха {epoch}: средний loss={running_loss/len(loader_tr):.4f}, PR-AUC валид={fold_pr:.4f}")

        if fold_pr > best_pr_auc:
            best_pr_auc = fold_pr
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
            no_improve = 0
        else:
            no_improve += 1
            if no_improve > patience:
                print("Ранняя остановка по стагнации PR-AUC")
                break

    return {
        'best_state_dict': best_state,
        'val_logits': val_logits,
        'val_labels': val_labels,
        'best_pr_auc': best_pr_auc,
    }

def train_cv(df: pd.DataFrame, folds_idx: List[Dict[str, np.ndarray]], cfg: TrainConfig) -> Dict[str, Any]:
    tokenizer = build_tokenizer(cfg)

    oof_logits = np.zeros(len(df), dtype=float)
    oof_mask = np.zeros(len(df), dtype=bool)
    fold_states = []
    fold_rows = []

    print("Старт кросс-валидации базовой конфигурации")
    for i, fold in enumerate(folds_idx, 1):
        tr, va = fold['train_idx'], fold['valid_idx']
        print(f"Фолд {i}: обучение на {len(tr)}, валидация на {len(va)}")
        res = train_one_fold(df, tr, va, tokenizer, cfg)
        oof_logits[va] = res['val_logits']
        oof_mask[va] = True
        fold_states.append(res['best_state_dict'])

        yv = df.iloc[va][cfg.label_col].to_numpy()
        pv = logits_to_probs(res['val_logits'])
        fold_rows.append({
            'fold': i,
            'valid_size': len(va),
            'pr_auc': pr_auc(yv, pv),
            'precision@0.5': precision_score(yv, (pv>=0.5).astype(int), zero_division=0),
            'recall@0.5': recall_score(yv, (pv>=0.5).astype(int), zero_division=0),
            'f1@0.5': f1_score(yv, (pv>=0.5).astype(int), zero_division=0),
        })
        print(f"Фолд {i}: лучший PR-AUC валид = {res['best_pr_auc']:.4f}")

    assert oof_mask.sum() == sum(len(f['valid_idx']) for f in folds_idx), "OOF покрывает не все валидации"
    oof_labels = df[cfg.label_col].to_numpy()[oof_mask]
    oof_probs = logits_to_probs(oof_logits[oof_mask])
    oof_pr = pr_auc(oof_labels, oof_probs)

    fold_report = pd.DataFrame(fold_rows)
    print("Сводка по фолдам:")
    display(fold_report)
    print(f"OOF PR-AUC по всем фолдам: {oof_pr:.4f}")

    return {
        'tokenizer': tokenizer,
        'fold_states': fold_states,
        'oof_logits': oof_logits,
        'oof_mask': oof_mask,
        'oof_pr_auc': oof_pr,
        'fold_report': fold_report,
    }

def choose_thresholds_and_report(oof_logits: np.ndarray, oof_mask: np.ndarray,
                                 y: np.ndarray, cfg: TrainConfig) -> Dict[str, Any]:
    y_masked = y[oof_mask]
    scores = logits_to_probs(oof_logits[oof_mask])

    thr_f1 = pick_threshold_by_f1(y_masked, scores)
    thr_prec = pick_threshold_by_precision_floor(y_masked, scores, cfg.target_precision)

    rep_f1 = report_at_threshold(y_masked, scores, thr_f1)
    rep_prec = report_at_threshold(y_masked, scores, thr_prec)

    print("Порог по максимуму F1:")
    print(rep_f1)
    print("Порог по требуемой точности:")
    print(rep_prec)

    return {'thr_f1': thr_f1, 'thr_precision': thr_prec,
            'rep_f1': rep_f1, 'rep_precision': rep_prec}

def build_infer_loader(df: pd.DataFrame, idx: np.ndarray, tokenizer, cfg: TrainConfig) -> DataLoader:
    dfx = df.iloc[idx].reset_index(drop=True)

    class InferDS(Dataset):
        def __init__(self, texts): self.texts = texts
        def __len__(self): return len(self.texts)
        def __getitem__(self, i): return {'text': self.texts[i]}

    ds = InferDS(dfx[cfg.text_col].astype(str).tolist())
    def collate_fn(samples):
        texts = [s['text'] for s in samples]
        enc = tokenizer(
            texts, padding=True, truncation=True, max_length=cfg.max_length, return_tensors="pt"
        )
        return enc

    return DataLoader(
        ds, batch_size=cfg.batch_size, shuffle=False,
        collate_fn=collate_fn,
        num_workers=cfg.num_workers, pin_memory=cfg.pin_memory,
        persistent_workers=bool(cfg.num_workers > 0 and cfg.persistent_workers),
        prefetch_factor=(cfg.prefetch_factor if cfg.num_workers > 0 else None),
    )

def train_full_and_eval(df: pd.DataFrame, folds_idx: List[Dict[str, np.ndarray]],
                        test_idx: np.ndarray, cfg: TrainConfig, threshold: float) -> Dict[str, Any]:
    pool_idx = np.unique(np.concatenate([d['train_idx'] for d in folds_idx] +
                                        [d['valid_idx'] for d in folds_idx]))
    tokenizer = build_tokenizer(cfg)
    model = build_model(cfg)

    df_pool = df.iloc[pool_idx].reset_index(drop=True)
    ds_pool = TextDataset(df_pool, cfg.text_col, cfg.label_col)
    collate_fn = build_collate_fn(tokenizer, cfg.max_length)
    loader_tr = DataLoader(
        ds_pool, batch_size=cfg.batch_size, shuffle=True,
        collate_fn=collate_fn,
        num_workers=cfg.num_workers, pin_memory=cfg.pin_memory,
        persistent_workers=bool(cfg.num_workers > 0 and cfg.persistent_workers),
        prefetch_factor=(cfg.prefetch_factor if cfg.num_workers > 0 else None),
    )

    y_pool = df_pool[cfg.label_col].to_numpy()
    criterion = build_loss(cfg, y_pool)
    optimizer = torch.optim.AdamW(model.parameters(), lr=cfg.lr, weight_decay=cfg.weight_decay)
    steps_total = cfg.epochs * len(loader_tr)
    warmup = int(cfg.warmup_prop * steps_total)
    scheduler = get_linear_schedule_with_warmup(optimizer, warmup, steps_total)
    scaler = torch.amp.GradScaler('cuda', enabled=cfg.use_amp)
    allowed_keys = {'input_ids', 'attention_mask', 'token_type_ids'}

    for epoch in range(1, cfg.epochs + 1):
        model.train()
        running = 0.0
        pbar = tqdm(loader_tr, desc=f"Эпоха {epoch}/{cfg.epochs} — обучение на всех 90%", leave=False)
        for enc, labels in pbar:
            enc = {k: v.to(cfg.device) for k, v in enc.items() if k in allowed_keys}
            labels = labels.to(cfg.device)

            optimizer.zero_grad(set_to_none=True)
            with torch.amp.autocast('cuda', enabled=cfg.use_amp):
                logits = model(**enc)
                loss = criterion(logits, labels)
            scaler.scale(loss).backward()
            nn.utils.clip_grad_norm_(model.parameters(), cfg.grad_clip_norm)
            scaler.step(optimizer)
            scaler.update()
            scheduler.step()

            running += float(loss.item())
            pbar.set_postfix({"loss": f"{running / (pbar.n or 1):.4f}"})
        print(f"Эпоха {epoch}: средний loss={running/len(loader_tr):.4f}")

    loader_test = build_infer_loader(df, test_idx, tokenizer, cfg)
    y_test = df.iloc[test_idx][cfg.label_col].to_numpy()

    model.eval()
    test_logits = []
    with torch.no_grad():
        pbar_t = tqdm(loader_test, desc="Инференс на lockbox-тесте", leave=False)
        for enc in pbar_t:
            enc = {k: v.to(cfg.device) for k, v in enc.items() if k in allowed_keys}
            logits = model(**enc)
            test_logits.append(logits.detach().cpu().numpy())
    test_logits = np.concatenate(test_logits)
    test_probs = logits_to_probs(test_logits)

    rep = report_at_threshold(y_test, test_probs, threshold)
    print("Итоговая оценка на lockbox-тесте:")
    print(rep)
    return {'test_logits': test_logits, 'test_probs': test_probs, 'report': rep}

def sample_space(base: TrainConfig, trial: int) -> TrainConfig:
    cfg = TrainConfig(**asdict(base))
    lrs = [2e-5, 3e-5, 5e-5]
    dropouts = [0.1, 0.2, 0.3]
    epochs = [3, 4]
    loss_types = ['bce', 'focal']
    lora_rs = [8, 16]
    lora_alphas = [16, 32]

    cfg.lr = float(np.random.choice(lrs))
    cfg.dropout = float(np.random.choice(dropouts))
    cfg.epochs = int(np.random.choice(epochs))
    cfg.loss_type = str(np.random.choice(loss_types))
    if cfg.loss_type == "focal":
        cfg.focal_gamma = float(np.random.choice([1.5, 2.0, 2.5]))
        cfg.focal_alpha = float(np.random.choice([0.25, 0.5]))
    if cfg.use_lora:
        cfg.lora_r = int(np.random.choice(lora_rs))
        cfg.lora_alpha = int(np.random.choice(lora_alphas))
    return cfg

def hpo_search(df: pd.DataFrame, folds_idx: List[Dict[str, np.ndarray]],
               base_cfg: TrainConfig, n_trials: int = 6) -> Dict[str, Any]:
    leaderboard = []
    best = None
    for t in range(1, n_trials + 1):
        print(f"Запуск подбора №{t}/{n_trials}")
        cfg = sample_space(base_cfg, t)
        res = train_cv(df, folds_idx, cfg)
        score = res['oof_pr_auc']
        row = {'trial': t, 'oof_pr_auc': score, **asdict(cfg)}
        leaderboard.append(row)
        if best is None or score > best['oof_pr_auc']:
            best = {'result': res, 'config': cfg, 'oof_pr_auc': score}
        print(f"Оценка OOF PR-AUC: {score:.4f}")

    lb = pd.DataFrame(leaderboard).sort_values('oof_pr_auc', ascending=False).reset_index(drop=True)
    print("Лидборд гиперпараметров:")
    display(lb.head(10))
    print(f"Лучшая конфигурация: OOF PR-AUC={best['oof_pr_auc']:.4f}")
    return {'best': best, 'leaderboard': lb}

def assemble_full_scores(
    df: pd.DataFrame,
    cv_res: dict,
    final_res: dict,
    test_idx: np.ndarray,
    label_col: str
) -> pd.DataFrame:
    """
    Возвращает DataFrame со столбцами:
      - y_true: истинные метки
      - score: вероятность класса "фрод" (0..1)
      - where: "oof" для train_pool и "test" для lockbox
    """
    n = len(df)
    out = pd.DataFrame(index=np.arange(n))
    out['y_true'] = df[label_col].astype(int).to_numpy()
    out['score'] = np.full(n, np.nan, dtype=float)
    out['where'] = "none"

    # OOF-часть (валидационные куски фолдов)
    if cv_res is not None:
        oof_mask = cv_res['oof_mask'].astype(bool)
        oof_probs = 1.0 / (1.0 + np.exp(-cv_res['oof_logits'][oof_mask]))
        out.loc[oof_mask, 'score'] = oof_probs
        out.loc[oof_mask, 'where'] = "oof"

    # Тест lockbox
    if final_res is not None and test_idx is not None:
        out.loc[test_idx, 'score'] = final_res['test_probs']
        out.loc[test_idx, 'where'] = "test"

    return out