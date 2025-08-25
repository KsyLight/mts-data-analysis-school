# Импорт библиотек
import re
import gc
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.metrics import (
    silhouette_score,
    calinski_harabasz_score,
    davies_bouldin_score,
    precision_score,
    recall_score,
    f1_score,
    average_precision_score,
    brier_score_loss,
    confusion_matrix,
    precision_recall_curve,
    adjusted_rand_score
)
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.calibration import calibration_curve
from sklearn.manifold import trustworthiness

import torch
from torch import nn
from transformers import AutoModel, AutoTokenizer

try:
    import umap
    use_umap = True
except Exception:
    use_umap = False

from tqdm import tqdm
from typing import List

# ------------------------------------------------------------------------------------

def select_candidates(scores: np.ndarray,
                      min_count: int = 120,
                      fixed_thr: float = 0.80,
                      top_k: int = 200) -> np.ndarray:
    idx = np.where(scores >= fixed_thr)[0]
    if len(idx) < min_count:
        top_idx = np.argsort(scores)[::-1][:top_k]
        idx = np.unique(np.concatenate([idx, top_idx]))
    return idx

def build_embedder(model_name: str, device: str = "cuda"):
    tok = AutoTokenizer.from_pretrained(model_name)
    mdl = AutoModel.from_pretrained(model_name).to(device)
    mdl.eval()
    return tok, mdl

def _mean_pool(last_hidden_state: torch.Tensor, attention_mask: torch.Tensor) -> torch.Tensor:
    mask = attention_mask.unsqueeze(-1).expand(last_hidden_state.size()).float()
    summed = (last_hidden_state * mask).sum(dim=1)
    denom = mask.sum(dim=1).clamp(min=1e-6)
    return summed / denom

@torch.no_grad()
def encode_texts_to_embeddings(texts: List[str],
                               tokenizer,
                               model,
                               max_length: int = 320,
                               batch_size: int = 16,
                               device: str = "cuda") -> np.ndarray:
    out = []
    for start in tqdm(range(0, len(texts), batch_size), desc="Вычисление эмбеддингов", leave=False):
        chunk = texts[start:start+batch_size]
        enc = tokenizer(chunk, padding=True, truncation=True, max_length=max_length, return_tensors="pt")
        enc = {k: v.to(device) for k, v in enc.items() if k in {'input_ids','attention_mask','token_type_ids'}}
        with torch.amp.autocast('cuda', enabled=torch.cuda.is_available()):
            out_enc = model(**enc)
            last = out_enc.last_hidden_state if hasattr(out_enc,'last_hidden_state') else out_enc[0]
            pooled = _mean_pool(last, enc['attention_mask']).detach().cpu().numpy()
            out.append(pooled)
    return np.vstack(out) if out else np.empty((0, model.config.hidden_size), dtype=np.float32)

def choose_k_by_silhouette(X: np.ndarray, k_range=range(3, 9), random_state: int = 42) -> int:
    best_k, best_s = None, -1
    for k in k_range:
        km = KMeans(n_clusters=k, n_init='auto', random_state=random_state)
        lbl = km.fit_predict(X)
        s = silhouette_score(X, lbl, metric='euclidean')
        if s > best_s:
            best_s, best_k = s, k
    return best_k or 3

def plot_clusters(embeddings, X2, labels):
    if len(embeddings) > 0:
        # DataFrame для точек
        df_plot = pd.DataFrame({
            'x': X2[:, 0],
            'y': X2[:, 1],
            'cluster': labels
        })

        # Scatterplot
        plt.figure(figsize=(6, 5))
        sns.scatterplot(
            data=df_plot, x='x', y='y',
            hue='cluster', palette='tab10', s=32, alpha=0.85
        )
        plt.title("Кластеры фрод-паттернов (2D)")
        plt.legend(title='cluster', bbox_to_anchor=(1.02, 1), loc='upper left')
        plt.tight_layout()
        plt.show()

        # Countplot
        plt.figure(figsize=(6, 3))
        sns.countplot(x=pd.Series(labels, name="cluster"))
        plt.title("Число объектов в кластерах")
        plt.xlabel("cluster")
        plt.ylabel("count")
        plt.show()

RUS_STOP = set("""
и в во не на с со что как но а или же бы ли для до по от из за у о об над под при между
я ты вы мы он она они оно этот тот там тут здесь тогда сейчас вчера сегодня завтра
есть был была были будет будут быть это тот эта это эти также ещё уже очень
к вам вам вас ваш ваша ваше ваши мой моя моё мои наш наша наше наши
когда где куда откуда почему потому также ну да нет
""".split())

def normalize_for_kw(s: str) -> str:
    s = s.lower()
    s = re.sub(r"[^a-zа-я0-9\[\]_ ]+", " ", s)
    s = re.sub(r"\s+", " ", s).strip()
    return s

def top_tfidf_terms_per_cluster(texts: List[str], labels: np.ndarray, top_k: int = 12) -> pd.DataFrame:
    df_tmp = pd.DataFrame({'text': texts, 'cluster': labels})
    rows = []
    for c in sorted(np.unique(labels)):
        subset = df_tmp.loc[df_tmp['cluster'] == c, 'text'].tolist()
        name = f"cluster_{c}"
        if len(subset) < 3:
            rows.append({'cluster': name, 'terms': ["мало текстов"]})
            continue
        corpus = [normalize_for_kw(t) for t in subset]
        vec = TfidfVectorizer(analyzer='word', ngram_range=(1,2), max_features=5000, stop_words=list(RUS_STOP))
        X = vec.fit_transform(corpus)
        vocab = np.array(vec.get_feature_names_out())
        scores = np.asarray(X.mean(axis=0)).ravel()
        top_idx = np.argsort(scores)[::-1][:top_k]
        rows.append({'cluster': name, 'terms': vocab[top_idx].tolist()})
    return pd.DataFrame(rows)

def show_cluster_examples(embeddings, df_cand, labels, best_cfg, top_k=12):
    if len(embeddings) > 0:
        tfidf_df = top_tfidf_terms_per_cluster(
            df_cand[best_cfg.text_col].astype(str).tolist(),
            labels,
            top_k=top_k
        )
        display(tfidf_df)

        df_cand_view = df_cand.copy()
        df_cand_view['cluster'] = labels

        for c in sorted(np.unique(labels)):
            print(f"\nКластер {c} — примеры:")
            examples = df_cand_view[df_cand_view['cluster'] == c][best_cfg.text_col].head(5).tolist()
            for t in examples:
                print("—", t[:220].replace("\n", " "))

def clustering_metrics_summary(embeddings: np.ndarray, labels: np.ndarray) -> pd.DataFrame:
    lab = np.asarray(labels)
    ok_mask = lab != -1 if (-1 in lab) else np.ones_like(lab, dtype=bool)
    lab_ok = lab[ok_mask]
    X_ok = embeddings[ok_mask]

    rows = []
    if len(np.unique(lab_ok)) >= 2 and len(lab_ok) >= 10:
        try:
            sil = silhouette_score(X_ok, lab_ok)
        except Exception:
            sil = np.nan
        try:
            ch = calinski_harabasz_score(X_ok, lab_ok)
        except Exception:
            ch = np.nan
        try:
            db = davies_bouldin_score(X_ok, lab_ok)
        except Exception:
            db = np.nan
    else:
        sil, ch, db = np.nan, np.nan, np.nan

    rows.append({'metric': "silhouette", 'value': sil})
    rows.append({'metric': "calinski_harabasz", 'value': ch})
    rows.append({'metric': "davies_bouldin", 'value': db})

    sizes = pd.Series(lab, name="cluster").value_counts().sort_index()
    print("Размеры кластеров:")
    display(sizes.to_frame("count").T)

    return pd.DataFrame(rows)

def projection_trustworthiness(X: np.ndarray, X2: np.ndarray, n_neighbors: int = 10) -> float:
    try:
        tw = trustworthiness(X, X2, n_neighbors=n_neighbors)
    except Exception:
        tw = np.nan
    print(f"Trustworthiness(2D) при k={n_neighbors}: {tw:.3f}" if not np.isnan(tw) else "Trustworthiness не рассчитан")
    return tw

def clustering_stability_ari(X: np.ndarray, base_labels: np.ndarray, n_clusters: int, n_runs: int = 8, sample_frac: float = 0.9, random_state: int = 42) -> pd.DataFrame:
    rng = np.random.RandomState(random_state)
    rows = []
    for r in range(n_runs):
        idx = rng.choice(len(X), size=max(10, int(len(X)*sample_frac)), replace=False)
        km = KMeans(n_clusters=n_clusters, n_init=10, random_state=rng.randint(0, 10_000))
        pred = km.fit_predict(X[idx])
        ari = adjusted_rand_score(base_labels[idx], pred)
        rows.append({"run": r+1, "ari": ari, "n": len(idx)})
    df = pd.DataFrame(rows)
    print("Устойчивость кластеров (ARI):")
    display(df.describe().T)
    fig, ax = plt.subplots(figsize=(5, 3))
    sns.stripplot(data=df, x="run", y="ari", ax=ax)
    ax.set_title("ARI по бутстрап-рандам")
    plt.show()
    return df

def reduce_2d(X: np.ndarray, n_neighbors: int = 15, min_dist: float = 0.1, random_state: int = 42) -> np.ndarray:
    if use_umap:
        reducer = umap.UMAP(n_neighbors=n_neighbors, min_dist=min_dist, n_components=2, random_state=random_state)
        return reducer.fit_transform(X)
    return PCA(n_components=2, random_state=random_state).fit_transform(X)

def test_report(y_true: np.ndarray, scores: np.ndarray, thr: float) -> dict:
    y_pred = (scores >= thr).astype(int)
    return {
        'threshold': float(thr),
        'precision': float(precision_score(y_true, y_pred, zero_division=0)),
        'recall': float(recall_score(y_true, y_pred, zero_division=0)),
        'f1': float(f1_score(y_true, y_pred, zero_division=0)),
        'pr_auc': float(average_precision_score(y_true, scores)),
        'brier': float(brier_score_loss(y_true, scores)),
        'confusion_matrix': confusion_matrix(y_true, y_pred).tolist(),
    }

def plot_pr_curve(y_true, scores, title):
    prec, rec, _ = precision_recall_curve(y_true, scores)
    ap = average_precision_score(y_true, scores)
    plt.figure(figsize=(6,4))
    sns.lineplot(x=rec, y=prec)
    plt.xlabel("Recall")
    plt.ylabel("Precision")
    plt.title(f"{title}\nAP={ap:.3f}")
    plt.grid(True, alpha=0.2)
    plt.show()

def plot_calibration(y_true, scores, n_bins=10, title="Калибровка"):
    prob_true, prob_pred = calibration_curve(y_true, scores, n_bins=n_bins, strategy='quantile')
    plt.figure(figsize=(6,4))
    sns.lineplot(x=prob_pred, y=prob_true, marker='o')
    plt.plot([0,1],[0,1],'--', alpha=0.5)
    plt.xlabel("Предсказанная вероятность")
    plt.ylabel("Доля позитива в бине")
    plt.title(title)
    plt.grid(True, alpha=0.2)
    plt.show()

def plot_confmat(y_true, scores, thr, title):
    y_pred = (scores >= thr).astype(int)
    cm = confusion_matrix(y_true, y_pred)
    plt.figure(figsize=(4,3))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
    plt.title(title + f"\nthreshold={thr:.3f}")
    plt.xlabel("Predicted")
    plt.ylabel("True")
    plt.show()