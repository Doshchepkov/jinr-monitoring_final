#!/usr/bin/env python
"""
Обучение модели XGBoost для прогнозирования скачков температуры.

Запуск:
    python train_model.py --data-path datasets/merged_dataset2.csv
    python train_model.py --data-path datasets/merged_dataset2.csv --buffer-path buffer/positive_episodes.csv --buffer-ratio 0.5
"""

import argparse
import pandas as pd
import numpy as np
import joblib
import gc
import time
import os
import json
import sys
from datetime import datetime
from xgboost import XGBClassifier
from sklearn.metrics import fbeta_score
import warnings
warnings.filterwarnings('ignore')

from src.features_main import add_time_features, compute_normalization, split_into_folds
from src.episodes import make_episodes, print_stats


def setup_logging(save_dir: str = "logs") -> tuple:
    """Настраивает логирование"""
    os.makedirs(save_dir, exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_file = os.path.join(save_dir, f"training_{timestamp}.txt")
    
    class Tee:
        def __init__(self, *files):
            self.files = files
        def write(self, obj):
            for f in self.files:
                f.write(obj)
                f.flush()
        def flush(self):
            for f in self.files:
                f.flush()
    
    original_stdout = sys.stdout
    log_f = open(log_file, 'w', encoding='utf-8')
    sys.stdout = Tee(original_stdout, log_f)
    
    print(f"Логи сохраняются в: {log_file}")
    print(f"Дата запуска: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    
    return log_f, log_file


def load_buffer_episodes(buffer_path: str, feature_cols: list, mean: np.ndarray, std: np.ndarray, 
                         L: int = 60, max_samples: int = 500) -> tuple:
    """
    Загружает положительные эпизоды из буфера и преобразует их в формат для обучения.
    
    Parameters
    ----------
    buffer_path : str
        Путь к CSV файлу с буфером
    feature_cols : list
        Список признаков
    mean, std : np.ndarray
        Параметры нормализации
    L : int
        Длина окна
    max_samples : int
        Максимальное количество эпизодов для загрузки
    
    Returns
    -------
    tuple
        (X_buffer, y_buffer) - эпизоды и метки (все метки = 1)
    """
    if not os.path.exists(buffer_path):
        print(f"   ⚠️ Буфер не найден: {buffer_path}")
        return None, None
    
    print(f"   Загрузка буфера: {buffer_path}")
    buffer_df = pd.read_csv(buffer_path)
    
    # Группируем по episode_id
    episode_ids = buffer_df["episode_id"].unique()
    if len(episode_ids) > max_samples:
        episode_ids = episode_ids[:max_samples]
        print(f"   Ограничено до {max_samples} эпизодов из {len(buffer_df['episode_id'].unique())}")
    
    X_list = []
    
    for ep_id in episode_ids:
        # Берём один эпизод (L строк)
        ep_df = buffer_df[buffer_df["episode_id"] == ep_id].sort_values("_time")
        
        if len(ep_df) < L:
            continue
        
        # Берём последние L строк
        ep_df = ep_df.tail(L)
        
        # Проверяем, что все нужные признаки есть
        available_cols = [c for c in feature_cols if c in ep_df.columns]
        if len(available_cols) != len(feature_cols):
            continue
        
        # Формируем X
        X = ep_df[feature_cols].to_numpy(dtype=float)
        X_norm = (X - mean) / std
        X_list.append(X_norm)
    
    if len(X_list) == 0:
        print("   ⚠️ Не удалось загрузить эпизоды из буфера")
        return None, None
    
    X_buffer = np.array(X_list)
    y_buffer = np.ones(len(X_buffer))
    
    print(f"   Загружено {len(X_buffer)} положительных эпизодов из буфера")
    
    return X_buffer, y_buffer


def train_model(
    data_path: str,
    L: int = 60,
    H: int = 30,
    n: float = 0.1,
    save_path: str = "models/final_xgb.pkl",
    logs_dir: str = "logs",
    buffer_path: str = None,
    buffer_ratio: float = 0.0
) -> tuple:
    """
    Основная функция обучения модели XGBoost с сохранением логов.
    
    Parameters
    ----------
    buffer_path : str, optional
        Путь к буферу с положительными эпизодами
    buffer_ratio : float, default=0.0
        Пропорция положительных эпизодов из буфера (0.0 = выключено, 0.5 = 50% позитивных из буфера)
    """
    
    log_f, log_file = setup_logging(logs_dir)
    
    print("=" * 50)
    print("Обучение модели XGBoost")
    print("=" * 50)
    
    params = {
        "data_path": data_path,
        "L": L,
        "H": H,
        "n": n,
        "save_path": save_path,
        "buffer_path": buffer_path if buffer_path else "не используется",
        "buffer_ratio": buffer_ratio if buffer_path else 0.0,
        "timestamp": datetime.now().isoformat()
    }
    print("\nПараметры обучения:")
    for k, v in params.items():
        print(f"  {k}: {v}")
    
    # 1. Загрузка данных
    print("\n1. Загрузка данных...")
    df = pd.read_csv(data_path)
    print(f"   Загружено {len(df):,} строк")
    
    # 2. Добавление временных признаков
    print("\n2. Добавление временных признаков...")
    df = add_time_features(df, "_time")
    
    # 3. Список признаков
    feature_cols = [c for c in df.columns if c not in ["_time"]]
    print(f"   Признаков: {len(feature_cols)}")
    
    # 4. Разбиение на фолды
    print("\n3. Разбиение на фолды...")
    folds = split_into_folds(df, n_folds=10)
    print(f"   Создано {len(folds)} фолдов")
    
    # 5. Walk-forward валидация
    print("\n4. Walk-forward валидация...")
    
    results = []
    best_model = None
    best_mean = None
    best_std = None
    best_thr_final = None
    best_f_macro = -1
    
    # Загружаем буфер (один раз для всех фолдов)
    buffer_X = None
    buffer_y = None
    use_buffer = buffer_path is not None and buffer_ratio > 0
    
    if use_buffer:
        print("\n   Загрузка буфера положительных эпизодов...")
        # Для нормализации нужны mean/std из тренировочных данных,
        # но пока используем временные (позже пересчитаем для каждого фолда)
        temp_mean, temp_std = compute_normalization(df, feature_cols)
        buffer_X, buffer_y = load_buffer_episodes(
            buffer_path, feature_cols, temp_mean, temp_std, L, max_samples=500
        )
    
    for k in range(5):
        print(f"\n   === Fold {k+1}/5 ===")
        
        # Разбиение
        train_df = pd.concat(folds[k:k+4]).reset_index(drop=True)
        val_df = folds[k+4].reset_index(drop=True)
        test_df = folds[k+5].reset_index(drop=True)
        
        # Нормализация
        mean, std = compute_normalization(train_df, feature_cols)
        
        # Формирование эпизодов из основных данных
        X_train, y_train = make_episodes(train_df, feature_cols, mean, std, L, H, n, aug_k=1)
        X_val, y_val = make_episodes(val_df, feature_cols, mean, std, L, H, n, aug_k=1)
        X_test, y_test = make_episodes(test_df, feature_cols, mean, std, L, H, n, aug_k=1)
        
        # Подмешиваем эпизоды из буфера (если включено)
        if use_buffer and buffer_X is not None and len(buffer_X) > 0:
            # Пересчитываем нормализацию для буферных эпизодов с текущими mean/std
            X_buffer_norm = []
            for x_ep in buffer_X:
                # x_ep уже нормализован старыми параметрами, нужно пересчитать
                # но для простоты используем текущие mean/std
                x_reshaped = x_ep.reshape(x_ep.shape[0], -1)
                x_new = (x_ep * buffer_std + buffer_mean - mean) / std  # приблизительно
                X_buffer_norm.append(x_new)
            
            X_buffer_reshaped = np.array(X_buffer_norm).reshape(len(X_buffer_norm), -1)
            y_buffer = np.ones(len(X_buffer_reshaped))
            
            # Ограничиваем количество буферных эпизодов по пропорции
            max_buffer_samples = int(len(y_train) * buffer_ratio / (1 - buffer_ratio)) if buffer_ratio < 1 else len(y_train)
            max_buffer_samples = min(max_buffer_samples, len(y_buffer))
            
            if max_buffer_samples > 0:
                X_buffer_subset = X_buffer_reshaped[:max_buffer_samples]
                y_buffer_subset = y_buffer[:max_buffer_samples]
                
                # Объединяем
                X_train = np.concatenate([X_train, X_buffer_subset], axis=0)
                y_train = np.concatenate([y_train, y_buffer_subset], axis=0)
                
                print(f"   Добавлено {len(X_buffer_subset)} эпизодов из буфера")
        
        # Преобразование в 2D
        X_train = X_train.reshape(X_train.shape[0], -1)
        X_val = X_val.reshape(X_val.shape[0], -1)
        X_test = X_test.reshape(X_test.shape[0], -1)
        
        print(f"   Train: {X_train.shape}, Val: {X_val.shape}, Test: {X_test.shape}")
        print_stats("   Train", y_train)
        print_stats("   Val", y_val)
        print_stats("   Test", y_test)
        
        # Обучение
        print("   Обучение XGBoost...")
        start_time = time.time()
        
        model = XGBClassifier(
            n_estimators=200,
            learning_rate=0.1,
            max_depth=6,
            subsample=0.8,
            colsample_bytree=0.8,
            random_state=42,
            n_jobs=-1,
            use_label_encoder=False,
            eval_metric="logloss"
        )
        
        model.fit(X_train, y_train)
        elapsed = time.time() - start_time
        
        # Предсказания и поиск порога (как в оригинале)
        prob_val = model.predict_proba(X_val)[:, 1]
        
        best_thr = 0.5
        best_f = 0
        
        for thr in np.linspace(0.1, 0.9, 9):
            pred_val = (prob_val >= thr).astype(int)
            f2 = fbeta_score(y_val, pred_val, beta=2, pos_label=1, zero_division=0)
            f1 = fbeta_score(y_val, pred_val, beta=1, pos_label=0, zero_division=0)
            f_macro = (f2 + f1) / 2
            
            if f_macro > best_f:
                best_f = f_macro
                best_thr = thr
        
        # Оценка на тесте
        prob_test = model.predict_proba(X_test)[:, 1]
        pred_test = (prob_test >= best_thr).astype(int)
        
        f2_test = fbeta_score(y_test, pred_test, beta=2, pos_label=1, zero_division=0)
        f1_test = fbeta_score(y_test, pred_test, beta=1, pos_label=0, zero_division=0)
        f_macro_test = (f2_test + f1_test) / 2
        
        print(f"   Лучший порог: {best_thr:.2f}")
        print(f"   Val F-macro: {best_f:.3f}")
        print(f"   Test F-macro: {f_macro_test:.3f}")
        print(f"   Время: {elapsed:.1f} сек")
        
        if f_macro_test > best_f_macro:
            best_f_macro = f_macro_test
            best_model = model
            best_mean = mean
            best_std = std
            best_thr_final = best_thr
        
        results.append({
            "fold": k+1,
            "thr": best_thr,
            "val_f_macro": best_f,
            "test_f_macro": f_macro_test,
            "train_size": len(y_train),
            "val_size": len(y_val),
            "test_size": len(y_test),
            "train_pos_pct": (y_train.sum() / len(y_train) * 100),
            "time_sec": elapsed
        })
        
        del X_train, y_train, X_val, y_val, X_test, y_test
        gc.collect()
    
    # Вывод результатов
    print("\n" + "=" * 50)
    print("ИТОГОВЫЕ РЕЗУЛЬТАТЫ ВАЛИДАЦИИ")
    print("=" * 50)
    
    results_df = pd.DataFrame(results)
    print(results_df.to_string(index=False))
    print(f"\nСредний Test F-macro: {results_df['test_f_macro'].mean():.3f}")
    print(f"Лучший Test F-macro: {best_f_macro:.3f}")
    
    # Сохранение результатов
    results_csv = os.path.join(logs_dir, f"training_results_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv")
    results_df.to_csv(results_csv, index=False)
    print(f"\n✅ Результаты сохранены в: {results_csv}")
    
    # Сохранение модели
    print("\n" + "=" * 50)
    print("Сохранение лучшей модели...")
    print("=" * 50)
    
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    
    model_data = {
        "model": best_model,
        "mean": best_mean,
        "std": best_std,
        "feature_cols": feature_cols,
        "L": L,
        "H": H,
        "n": n,
        "threshold": best_thr_final,
        "best_test_f_macro": best_f_macro,
        "mean_test_f_macro": results_df['test_f_macro'].mean(),
        "timestamp": datetime.now().isoformat()
    }
    
    joblib.dump(model_data, save_path)
    
    model_info_path = os.path.join(logs_dir, f"model_info_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json")
    with open(model_info_path, 'w', encoding='utf-8') as f:
        json.dump({k: str(v) if isinstance(v, np.ndarray) else v for k, v in model_data.items()}, f, indent=2)
    
    print(f"✅ Модель сохранена в {save_path}")
    
    sys.stdout = sys.__stdout__
    log_f.close()
    print(f"\n✅ Полный лог сохранён в: {log_file}")
    
    return best_model, results_df


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Обучение модели XGBoost")
    parser.add_argument("--data-path", required=True, help="Путь к CSV файлу с данными")
    parser.add_argument("--L", type=int, default=60, help="Длина исторического окна (минут)")
    parser.add_argument("--H", type=int, default=30, help="Горизонт прогноза (минут)")
    parser.add_argument("--n", type=float, default=0.1, help="Порог скачка (10% = 0.1)")
    parser.add_argument("--save-path", default="models/final_xgb.pkl", help="Путь для сохранения модели")
    parser.add_argument("--logs-dir", default="logs", help="Папка для сохранения логов")
    parser.add_argument("--buffer-path", default=None, help="Путь к буферу positive_episodes.csv")
    parser.add_argument("--buffer-ratio", type=float, default=0.0, 
                        help="Пропорция положительных эпизодов из буфера (0.0 = выкл, 0.5 = 50% позитивных из буфера)")
    
    args = parser.parse_args()
    
    train_model(
        data_path=args.data_path,
        L=args.L,
        H=args.H,
        n=args.n,
        save_path=args.save_path,
        logs_dir=args.logs_dir,
        buffer_path=args.buffer_path,
        buffer_ratio=args.buffer_ratio
    )