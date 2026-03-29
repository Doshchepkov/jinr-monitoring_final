"""
Создание буфера положительных примеров (скачков) для дообучения
"""

import pandas as pd
import numpy as np
import os
from tqdm import tqdm
from episodes import make_episodes
from features_main import add_time_features, compute_normalization, split_into_folds


def create_positive_buffer(
    data_path: str,
    output_dir: str = "buffer",
    L: int = 60,
    H: int = 30,
    n: float = 0.1,
    max_samples: int = 1000
):
    """
    Создаёт буфер положительных эпизодов из датасета.
    
    Запуск:
        python src/create_buffer.py --data-path datasets/merged_dataset2.csv
    """
    
    print("=" * 50)
    print("Создание буфера положительных эпизодов")
    print("=" * 50)
    
    # 1. Загрузка данных
    print("\n1. Загрузка данных...")
    df = pd.read_csv(data_path)
    print(f"   Загружено {len(df):,} строк")
    df = df.tail(100000) 
    # 2. Добавление временных признаков
    print("\n2. Добавление временных признаков...")
    df = add_time_features(df, "_time")
    
    # 3. Признаки
    feature_cols = [c for c in df.columns if c not in ["_time"]]
    print(f"   Признаков: {len(feature_cols)}")
    
    # 4. Нормализация (по всем данным)
    print("\n3. Нормализация данных...")
    mean, std = compute_normalization(df, feature_cols)
    
    # 5. Формирование эпизодов
    print("\n4. Формирование эпизодов...")
    X, y, idx = make_episodes(
        df, feature_cols, mean, std,
        L=L, H=H, n=n, aug_k=1, return_indices=True
    )
    
    # 6. Отбор положительных
    pos_mask = y == 1
    pos_indices = np.where(pos_mask)[0]
    print(f"   Найдено положительных эпизодов: {len(pos_indices)}")
    
    if len(pos_indices) == 0:
        print("   Нет положительных эпизодов для сохранения")
        return
    
    # 7. Ограничиваем количество
    if len(pos_indices) > max_samples:
        pos_indices = pos_indices[:max_samples]
        print(f"   Ограничено до {max_samples} эпизодов")
    
    # 8. Создаём папку
    os.makedirs(output_dir, exist_ok=True)
    
    # 9. Сохраняем в CSV
    print("\n5. Сохранение эпизодов...")
    
    all_episodes = []
    
    for i, pos_idx in enumerate(tqdm(pos_indices, desc="   Сохранение")):
        t = idx[pos_idx]
        timestamp = df.iloc[t]["_time"]
        
        # Берём окно данных (L строк до t)
        window = df.iloc[t-L:t]
        
        # Добавляем метку времени эпизода
        window = window.copy()
        window["episode_timestamp"] = timestamp
        window["episode_id"] = i
        
        all_episodes.append(window)
    
    # Объединяем все эпизоды
    buffer_df = pd.concat(all_episodes, ignore_index=True)
    
    # Сохраняем
    output_path = os.path.join(output_dir, "positive_episodes.csv")
    buffer_df.to_csv(output_path, index=False)
    
    print(f"\n✅ Буфер сохранён: {output_path}")
    print(f"   Эпизодов: {len(pos_indices)}")
    print(f"   Всего строк: {len(buffer_df)}")
    print(f"   Размер: {len(buffer_df)} строк x {len(buffer_df.columns)} колонок")


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Создание буфера положительных эпизодов")
    parser.add_argument("--data-path", required=True, help="Путь к merged_dataset2.csv")
    parser.add_argument("--output-dir", default="buffer", help="Папка для сохранения")
    parser.add_argument("--L", type=int, default=60, help="Длина окна")
    parser.add_argument("--H", type=int, default=30, help="Горизонт прогноза")
    parser.add_argument("--n", type=float, default=0.1, help="Порог скачка")
    parser.add_argument("--max-samples", type=int, default=1000, help="Максимум эпизодов")
    
    args = parser.parse_args()
    
    create_positive_buffer(
        data_path=args.data_path,
        output_dir=args.output_dir,
        L=args.L,
        H=args.H,
        n=args.n,
        max_samples=args.max_samples
    )