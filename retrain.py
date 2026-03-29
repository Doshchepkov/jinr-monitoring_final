#!/usr/bin/env python
"""
Скрипт для дообучения модели на положительных примерах из буфера

Запуск:
    python retrain.py --base-data datasets/merged_dataset2.csv
"""

import argparse
import pandas as pd
import numpy as np
import joblib
import os
import sys
from datetime import datetime
from xgboost import XGBClassifier

sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'src'))
from src.features_main import add_time_features, compute_normalization
from src.episodes import make_episodes, print_stats
from src.buffer import PositiveBuffer


def retrain_model(
    base_data_path: str,
    original_model_path: str = "models/final_xgb.pkl",
    new_model_path: str = "models/retrained_xgb.pkl",
    buffer_path: str = "buffer/positive_events.csv",
    L: int = 60,
    H: int = 30,
    n: float = 0.1
):
    """
    Дообучение модели с добавлением положительных примеров из буфера
    """
    
    print("=" * 50)
    print("Дообучение модели XGBoost")
    print("=" * 50)
    
    # 1. Загружаем буфер
    print("\n1. Загрузка буфера...")
    buffer = PositiveBuffer(buffer_path)
    print(f"   Всего эпизодов в буфере: {buffer.size()}")
    print(f"   Неиспользованных: {buffer.unused_size()}")
    
    # 2. Загружаем положительные эпизоды
    print("\n2. Загрузка положительных эпизодов...")
    positive_episodes = buffer.get_for_retraining(max_samples=500)
    print(f"   Загружено {len(positive_episodes)} новых положительных эпизодов")
    
    if len(positive_episodes) == 0:
        print("   Нет новых эпизодов для дообучения")
        return
    
    # 3. Загружаем базовую модель
    print("\n3. Загрузка базовой модели...")
    model_data = joblib.load(original_model_path)
    original_model = model_data["model"]
    mean = model_data["mean"]
    std = model_data["std"]
    feature_cols = model_data["feature_cols"]
    print(f"   Модель загружена. Признаков: {len(feature_cols)}")
    
    # 4. Загружаем базовые данные
    print("\n4. Загрузка базовых данных...")
    df = pd.read_csv(base_data_path)
    df = add_time_features(df, "_time")
    print(f"   Загружено {len(df):,} строк")
    
    # 5. Формируем эпизоды из базовых данных (выборка негативных)
    print("\n5. Формирование эпизодов из базовых данных...")
    # Берём часть данных, чтобы не перегружать память
    sample_df = df.sample(min(50000, len(df)), random_state=42)
    X_base, y_base = make_episodes(sample_df, feature_cols, mean, std, L, H, n, aug_k=1)
    X_base = X_base.reshape(X_base.shape[0], -1)
    
    # Оставляем только негативные классы (чтобы не разбавлять позитивные)
    neg_mask = y_base == 0
    X_neg = X_base[neg_mask]
    y_neg = y_base[neg_mask]
    
    # Берём только часть негативных (чтобы не перекос)
    n_neg = min(len(X_neg), len(positive_episodes) * 5)
    X_neg = X_neg[:n_neg]
    y_neg = y_neg[:n_neg]
    
    print(f"   Негативных примеров: {len(X_neg)}")
    
    # 6. Формируем признаки для положительных эпизодов
    print("\n6. Формирование признаков для положительных эпизодов...")
    X_pos_list = []
    
    for ep_df in positive_episodes:
        ep_df["_time"] = pd.to_datetime(ep_df["timestamp"])
        ep_df = add_time_features(ep_df, "_time")
        
        # Нормализация
        X = ep_df[feature_cols].to_numpy(dtype=float)
        X_norm = (X - mean) / std
        X_pos_list.append(X_norm.reshape(1, -1))
    
    X_pos = np.concatenate(X_pos_list, axis=0)
    y_pos = np.ones(len(X_pos))
    
    print(f"   Положительных примеров: {len(X_pos)}")
    
    # 7. Объединяем данные
    print("\n7. Объединение данных для дообучения...")
    X_combined = np.concatenate([X_neg, X_pos], axis=0)
    y_combined = np.concatenate([y_neg, y_pos], axis=0)
    
    print(f"   Всего примеров: {len(X_combined)}")
    print_stats("   Распределение", y_combined)
    
    # 8. Дообучение модели
    print("\n8. Дообучение модели...")
    model = XGBClassifier(
        n_estimators=50,  # меньше деревьев для дообучения
        learning_rate=0.05,
        max_depth=6,
        subsample=0.8,
        colsample_bytree=0.8,
        random_state=42,
        n_jobs=-1,
        use_label_encoder=False,
        eval_metric="logloss"
    )
    
    model.fit(X_combined, y_combined, xgb_model=original_model)
    
    # 9. Сохраняем обновлённую модель
    print("\n9. Сохранение обновлённой модели...")
    os.makedirs(os.path.dirname(new_model_path), exist_ok=True)
    
    updated_model_data = {
        "model": model,
        "mean": mean,
        "std": std,
        "feature_cols": feature_cols,
        "L": L,
        "H": H,
        "n": n,
        "threshold": model_data.get("threshold", 0.5),
        "retrained_at": datetime.now().isoformat(),
        "new_positive_samples": len(X_pos)
    }
    
    joblib.dump(updated_model_data, new_model_path)
    
    print(f"✅ Модель сохранена в {new_model_path}")
    print(f"   Добавлено {len(X_pos)} новых положительных примеров")
    
    # 10. Опционально: заменить старую модель
    answer = input("\nЗаменить старую модель на новую? (y/n): ")
    if answer.lower() == 'y':
        import shutil
        shutil.copy(new_model_path, original_model_path)
        print(f"✅ Старая модель заменена")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Дообучение модели")
    parser.add_argument("--base-data", required=True, help="Путь к базовому датасету")
    parser.add_argument("--original-model", default="models/final_xgb.pkl")
    parser.add_argument("--new-model", default="models/retrained_xgb.pkl")
    parser.add_argument("--L", type=int, default=60)
    parser.add_argument("--H", type=int, default=30)
    parser.add_argument("--n", type=float, default=0.1)
    
    args = parser.parse_args()
    
    retrain_model(
        base_data_path=args.base_data,
        original_model_path=args.original_model,
        new_model_path=args.new_model,
        L=args.L,
        H=args.H,
        n=args.n
    )