"""
Сборка датасета из исходных CSV файлов
Запуск: python build_dataset.py
"""

import pandas as pd
import os
import re
from glob import glob
from tqdm import tqdm

# ==================== НАСТРОЙКИ ====================
SOURCE_DIR = r"source_datasets"  # папка с исходными файлами
OUTPUT_PATH = r"source_datasets\merged_dataset2.csv"  # куда сохранить

# ==================== ФУНКЦИИ ====================

def normalize_text(s: pd.Series) -> pd.Series:
    """Нормализация текста"""
    s = s.astype(str)
    s = s.str.normalize("NFKC")
    s = s.str.replace(r"\s+", " ", regex=True).str.strip().str.lower()
    return s


def robust_parse_time(series: pd.Series) -> pd.Series:
    """Парсинг времени"""
    series = series.astype(str).str.strip()
    parsed = pd.to_datetime(series, utc=True, errors="coerce")
    bad_mask = parsed.isna() & series.notna()
    if bad_mask.any():
        parsed2 = pd.to_datetime(series[bad_mask].str.slice(0, 19), utc=True, errors="coerce")
        parsed.loc[bad_mask] = parsed2
    return parsed


def load_and_split_source_file(path: str) -> list:
    """Загружает исходный CSV и разбивает по группам (hostname, metric)"""
    df = pd.read_csv(path, usecols=["_time", "_value", "hostname", "metric"])
    print(f"  Строк: {len(df):,}")
    
    # парсим время
    df["_time"] = robust_parse_time(df["_time"])
    
    # нормализация
    df["hostname_norm"] = normalize_text(df["hostname"])
    df["metric_norm"] = normalize_text(df["metric"])
    
    # чистим NaN
    df = df.dropna(subset=["_time", "_value", "hostname_norm", "metric_norm"])
    
    # возвращаем список датафреймов по группам
    result = []
    for (host, metric), g in df.groupby(["hostname_norm", "metric_norm"]):
        g = g[["_time", "_value"]].sort_values("_time").reset_index(drop=True)
        safe_metric = re.sub(r"[^\w\-.]+", "_", metric)
        result.append({
            "name": f"{host}_{safe_metric}",
            "df": g
        })
    return result


# ==================== ОСНОВНОЙ КОД ====================

print("=" * 50)
print("Сборка датасета")
print("=" * 50)

# 1. Находим все исходные файлы
source_files = glob(os.path.join(SOURCE_DIR, "*.csv"))
print(f"Найдено исходных файлов: {len(source_files)}")

# 2. Загружаем все файлы и собираем данные
all_data = {}  # словарь {название_метрики: датафрейм}

for file_path in source_files:
    file_name = os.path.basename(file_path)
    print(f"\nОбработка: {file_name}")
    
    try:
        groups = load_and_split_source_file(file_path)
        for g in groups:
            name = g["name"]
            df = g["df"]
            
            # агрегация до минут
            df["_time"] = df["_time"].dt.floor("min")
            df = df.groupby("_time", as_index=False)["_value"].mean()
            df = df.rename(columns={"_value": name})
            
            if name not in all_data:
                all_data[name] = df
            else:
                # объединяем если уже есть
                all_data[name] = pd.concat([all_data[name], df]).drop_duplicates("_time")
        
        print(f"  ✅ Обработан")
    except Exception as e:
        print(f"  ❌ Ошибка: {e}")

# 3. Объединяем все в один датафрейм
print("\nОбъединение всех признаков...")
result = None

for name, df in tqdm(all_data.items(), desc="Объединение"):
    if result is None:
        result = df
    else:
        result = result.merge(df, on="_time", how="outer")

# 4. Заполняем пропуски и сохраняем
if result is not None:
    result = result.sort_values("_time").ffill()
    result.to_csv(OUTPUT_PATH, index=False)
    print(f"\n✅ Итоговая форма: {result.shape}")
    print(f"✅ Сохранено в: {OUTPUT_PATH}")
    print(f"\nКолонки: {result.columns.tolist()}")
else:
    print("❌ Нет данных для сохранения")