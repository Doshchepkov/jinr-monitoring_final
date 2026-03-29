"""
Визуализация данных и сохранение графиков
Запуск: python visualize_data.py
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import os

# ==================== ФУНКЦИИ ====================

def add_time_features(df, time_col="_time"):
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

def split_into_folds(df, n_folds=10):
    df = df.sort_values("_time").reset_index(drop=True)
    n = len(df)
    fold_size = n // n_folds
    folds = []
    for i in range(n_folds):
        start = i * fold_size
        end = (i + 1) * fold_size if i < n_folds - 1 else n
        folds.append(df.iloc[start:end].reset_index(drop=True))
    return folds

def plot_correlation_matrix(df, title="Матрица корреляций"):
    num_df = df.select_dtypes(include=["float64", "int64"])
    corr = num_df.corr()
    plt.figure(figsize=(12, 10))
    sns.heatmap(corr, cmap="coolwarm", center=0, annot=False)
    plt.title(title, fontsize=14)
    plt.tight_layout()

def plot_walk_validation(df, value_col, n_folds=10, title="Walk-forward validation"):
    n = len(df)
    fold_size = n // n_folds
    
    plt.figure(figsize=(12, 5))
    plt.plot(range(n), df[value_col].values, color='tab:blue', label=value_col)
    
    train_end = 4 * fold_size
    val_end = 5 * fold_size
    test_end = 6 * fold_size
    
    plt.axvspan(0, train_end, color='green', alpha=0.2, label="Train")
    plt.axvspan(train_end, val_end, color='orange', alpha=0.2, label="Validation")
    plt.axvspan(val_end, test_end, color='red', alpha=0.2, label="Test")
    
    for edge in [0, train_end, val_end, test_end, n]:
        if edge < n:
            plt.axvline(edge, color='black', linestyle='--', linewidth=0.8, alpha=0.5)
    
    plt.xlabel("Время (индекс)", fontsize=12)
    plt.ylabel("Температура, °C", fontsize=12)
    plt.title(title, fontsize=14)
    plt.legend(loc="upper right", fontsize=10)
    plt.grid(True, linestyle="--", alpha=0.5)
    plt.tight_layout()

def plot_positive_episodes(df, value_col="_value", L=60, H=30, n=0.05, title="", max_plots=5, save_dir=None):
    v = df[value_col].to_numpy(dtype=float)
    N = len(v)
    plotted = 0
    
    if save_dir:
        os.makedirs(save_dir, exist_ok=True)
    
    for t in range(L, N - H):
        x_window = v[t-L:t]
        future = v[t:t+H]
        start = future[0]
        threshold = start * (1 + n)
        label = int(np.any(future >= threshold))
        
        if label == 1:
            plt.figure(figsize=(10, 4))
            plt.plot(range(-L, 0), x_window, label="прошлое (60)", color="blue")
            plt.plot(range(0, H), future, label="будущее (30)", color="orange")
            plt.axhline(threshold, color="red", linestyle="--", label=f"порог {threshold:.2f}")
            plt.title(f"{title} | Метка={label} | start={start:.2f}, threshold={threshold:.2f}")
            plt.legend()
            plt.grid(True)
            
            if save_dir:
                filename = f"{title}_positive_episode_{plotted+1}.png"
                filepath = os.path.join(save_dir, filename)
                plt.savefig(filepath, dpi=150, bbox_inches="tight")
                print(f"  Сохранено: {filepath}")
                plt.close()
            else:
                plt.show()
            
            plotted += 1
            if plotted >= max_plots:
                break

# ==================== ОСНОВНОЙ КОД ====================

DATA_PATH = "datasets/merged_dataset2.csv"
SAVE_DIR = "screenshots"

print("Загрузка данных...")
df = pd.read_csv(DATA_PATH)
print(f"Загружено {len(df)} строк, {len(df.columns)} колонок")

df = add_time_features(df, "_time")
df = df.dropna()
print(f"После обработки: {len(df)} строк")

# Разбиваем на фолды для train/val/test
folds = split_into_folds(df, n_folds=10)
train_df = pd.concat(folds[0:4]).reset_index(drop=True)
val_df = folds[4].reset_index(drop=True)
test_df = folds[5].reset_index(drop=True)

print(f"Train: {len(train_df)}, Val: {len(val_df)}, Test: {len(test_df)}")

print("\nСохранение графиков...")
os.makedirs(SAVE_DIR, exist_ok=True)

# 1. Матрица корреляций
print("  - Матрица корреляций...")
plot_correlation_matrix(df)
plt.savefig(f"{SAVE_DIR}/correlation_matrix.png", dpi=150, bbox_inches="tight")
plt.close()

# 2. Walk-forward валидация
print("  - Walk-forward валидация...")
plot_walk_validation(df, "enter_fluid_temp")
plt.savefig(f"{SAVE_DIR}/walk_forward_validation.png", dpi=150, bbox_inches="tight")
plt.close()

# 3. Положительные эпизоды
print("  - Положительные эпизоды (скачки)...")
print("    Train:")
plot_positive_episodes(train_df, "enter_fluid_temp", L=60, H=30, n=0.1, title="Train", max_plots=3, save_dir=SAVE_DIR)

print("    Validation:")
plot_positive_episodes(val_df, "enter_fluid_temp", L=60, H=30, n=0.1, title="Validation", max_plots=3, save_dir=SAVE_DIR)

print("    Test:")
plot_positive_episodes(test_df, "enter_fluid_temp", L=60, H=30, n=0.1, title="Test", max_plots=3, save_dir=SAVE_DIR)

print(f"\n✅ Все графики сохранены в {SAVE_DIR}/")