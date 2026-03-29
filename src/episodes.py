"""
Функции для формирования эпизодов и разметки целевой переменной
"""
import pandas as pd
import numpy as np
from tqdm import tqdm
from src.augmentation import jitter, scaling, time_warp

def make_episodes(
    df: pd.DataFrame,
    feature_cols: list,
    mean: np.ndarray,
    std: np.ndarray,
    L: int = 60,
    H: int = 30,
    n: float = 0.05,
    aug_k: int = 1,
    return_indices: bool = False
) -> tuple:
    """
    Формирует эпизоды (окна) для обучения и размечает их.
    
    Parameters
    ----------
    df : pd.DataFrame
        DataFrame с данными
    feature_cols : list
        Список колонок-признаков
    mean, std : np.ndarray
        Параметры нормализации
    L : int
        Длина исторического окна (минут)
    H : int
        Горизонт прогноза (минут)
    n : float
        Порог скачка (например, 0.1 = 10%)
    aug_k : int
        Коэффициент аугментации для положительного класса (1 = без аугментации)
    return_indices : bool
        Возвращать ли индексы окон
    
    Returns
    -------
    tuple
        (X, y) или (X, y, idx) - эпизоды и метки
    """
    V = df[feature_cols].to_numpy(dtype=float)
    main = df[feature_cols[0]].to_numpy(dtype=float)
    N, F = V.shape
    
    # индексы окон
    idx = np.arange(L, N - H)
    M = len(idx)
    
    # формируем все окна
    X_all = np.lib.stride_tricks.sliding_window_view(V, (L, F))[:, 0, :, :]
    X_all = X_all[:M]
    X_all = (X_all - mean) / std
    
    # формируем метки
    future_blocks = np.lib.stride_tricks.sliding_window_view(main, H)[:M]
    start_vals = future_blocks[:, 0]
    labels = (future_blocks.max(axis=1) >= start_vals * (1 + n)).astype(int)
    
    # аугментации для положительного класса
    if aug_k > 1:
        aug_list = []
        for i in tqdm(range(M), desc="Augmenting", ncols=80):
            if labels[i] == 1:
                for _ in range(aug_k - 1):
                    aug_type = np.random.choice(["jitter", "scaling", "time_warp"])
                    if aug_type == "jitter":
                        x_aug = jitter(X_all[i])
                    elif aug_type == "scaling":
                        x_aug = scaling(X_all[i])
                    elif aug_type == "time_warp":
                        x_aug = time_warp(X_all[i], L=L)
                    aug_list.append((x_aug, 1))
        if aug_list:
            X_aug, y_aug = zip(*aug_list)
            X_all = np.concatenate([X_all, np.array(X_aug)], axis=0)
            labels = np.concatenate([labels, np.array(y_aug)])
    
    if return_indices:
        return X_all, labels, idx
    return X_all, labels


def print_stats(name: str, y: np.ndarray) -> None:
    """
    Выводит статистику по классам.
    
    Parameters
    ----------
    name : str
        Название выборки
    y : np.ndarray
        Метки классов
    """
    total = len(y)
    c = np.bincount(y, minlength=2)
    p0 = (c[0] / total * 100) if total else 0
    p1 = (c[1] / total * 100) if total else 0
    print(f"{name}: {total:,} эпизодов | "
          f"0={c[0]:,} ({p0:.1f}%), 1={c[1]:,} ({p1:.1f}%)")