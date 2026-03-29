"""
Функции аугментации временных рядов
"""

import numpy as np


def jitter(X: np.ndarray, sigma: float = 0.01) -> np.ndarray:
    """
    Добавляет гауссовский шум к данным.
    
    Parameters
    ----------
    X : np.ndarray
        Входные данные
    sigma : float
        Коэффициент шума (относительно std данных)
    
    Returns
    -------
    np.ndarray
        Данные с добавленным шумом
    """
    noise = np.random.normal(0, sigma * np.std(X, axis=0, keepdims=True), size=X.shape)
    return X + noise


def scaling(X: np.ndarray, low: float = 0.95, high: float = 1.05) -> np.ndarray:
    """
    Масштабирует данные на случайный коэффициент.
    
    Parameters
    ----------
    X : np.ndarray
        Входные данные
    low : float
        Нижняя граница коэффициента
    high : float
        Верхняя граница коэффициента
    
    Returns
    -------
    np.ndarray
        Масштабированные данные
    """
    factor = np.random.uniform(low, high)
    return X * factor


def time_warp(X: np.ndarray, max_warp: float = 0.05, L: int = 60) -> np.ndarray:
    """
    Деформирует временную шкалу (time warping).
    
    Parameters
    ----------
    X : np.ndarray
        Входные данные (N, F)
    max_warp : float
        Максимальное искажение временной шкалы
    L : int
        Целевая длина последовательности
    
    Returns
    -------
    np.ndarray
        Деформированные данные (L, F)
    """
    N = X.shape[0]
    orig_idx = np.linspace(0, 1, N)
    warp_factor = np.random.uniform(1 - max_warp, 1 + max_warp)
    warped_idx = np.linspace(0, 1, int(N * warp_factor))
    
    warped = np.empty((len(warped_idx), X.shape[1]))
    for f in range(X.shape[1]):
        warped[:, f] = np.interp(warped_idx, orig_idx, X[:, f])
    
    new_idx = np.linspace(0, 1, L)
    X_new = np.empty((L, X.shape[1]))
    for f in range(X.shape[1]):
        X_new[:, f] = np.interp(new_idx, np.linspace(0, 1, len(warped)), warped[:, f])
    
    return X_new