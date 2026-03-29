"""
Функции для работы с признаками: временные признаки, удаление коррелирующих, нормализация
"""

import numpy as np
import pandas as pd


def add_time_features(df: pd.DataFrame, time_col: str = "_time") -> pd.DataFrame:
    """
    Добавляет временные признаки из колонки времени.
    
    Parameters
    ----------
    df : pd.DataFrame
        Исходный DataFrame с колонкой времени
    time_col : str
        Название колонки с временем
    
    Returns
    -------
    pd.DataFrame
        DataFrame с добавленными временными признаками
    """
    df = df.copy()
    df[time_col] = pd.to_datetime(df[time_col], utc=True, errors="coerce")
    
    df["hour"] = df[time_col].dt.hour
    df["dayofweek"] = df[time_col].dt.dayofweek
    df["month"] = df[time_col].dt.month
    df["is_weekend"] = (df["dayofweek"] >= 5).astype(int)
    
    df["hour_sin"] = np.sin(2 * np.pi * df["hour"] / 24)
    df["hour_cos"] = np.cos(2 * np.pi * df["hour"] / 24)
    
    df["dow_sin"] = np.sin(2 * np.pi * df["dayofweek"] / 7)
    df["dow_cos"] = np.cos(2 * np.pi * df["dayofweek"] / 7)
    
    return df


def remove_highly_correlated(df: pd.DataFrame, threshold: float = 0.8, exclude_pattern: str = "temp") -> pd.DataFrame:
    """
    Удаляет признаки с высокой корреляцией (исключая указанный паттерн).
    
    Parameters
    ----------
    df : pd.DataFrame
        DataFrame с числовыми признаками
    threshold : float
        Порог корреляции для удаления
    exclude_pattern : str
        Паттерн для исключения из удаления (например, "temp")
    
    Returns
    -------
    pd.DataFrame
        DataFrame с удалёнными коррелирующими признаками
    """
    num_df = df.select_dtypes(include=["float64", "int64"])
    corr_matrix = num_df.corr().abs()
    
    # маска для верхнего треугольника
    upper = corr_matrix.where(np.triu(np.ones(corr_matrix.shape), k=1).astype(bool))
    
    # список на удаление
    to_drop = [
        column for column in upper.columns
        if any(upper[column] > threshold) and exclude_pattern not in column.lower()
    ]
    
    print(f"Удаляем {len(to_drop)} признаков с высокой корреляцией (без {exclude_pattern}): {to_drop}")
    
    return df.drop(columns=to_drop)


def compute_normalization(df: pd.DataFrame, feature_cols: list) -> tuple:
    """
    Вычисляет среднее и стандартное отклонение для нормализации.
    
    Parameters
    ----------
    df : pd.DataFrame
        DataFrame с признаками
    feature_cols : list
        Список колонок для нормализации
    
    Returns
    -------
    tuple
        (mean, std) - массивы средних и стандартных отклонений
    """
    V = df[feature_cols].to_numpy(dtype=float)
    mean = V.mean(axis=0, keepdims=True)
    std = V.std(axis=0, keepdims=True)
    std_safe = np.where(std == 0, 1, std)
    return mean, std_safe


def split_into_folds(df: pd.DataFrame, n_folds: int = 10) -> list:
    """
    Разбивает данные на хронологические фолды.
    
    Parameters
    ----------
    df : pd.DataFrame
        DataFrame с колонкой "_time"
    n_folds : int
        Количество фолдов
    
    Returns
    -------
    list
        Список DataFrame'ов - фолдов
    """
    df = df.sort_values("_time").reset_index(drop=True)
    n = len(df)
    fold_size = n // n_folds
    
    folds = []
    for i in range(n_folds):
        start = i * fold_size
        end = (i + 1) * fold_size if i < n_folds - 1 else n
        folds.append(df.iloc[start:end].reset_index(drop=True))
    
    return folds