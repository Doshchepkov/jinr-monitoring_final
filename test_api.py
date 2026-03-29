import requests
import pandas as pd

# Загружаем 60 строк из датасета
df = pd.read_csv("datasets/merged_dataset2.csv", nrows=60)

# Преобразуем в формат для API
sensor_data = []
for _, row in df.iterrows():
    record = {"timestamp": row["_time"]}
    # Добавляем все колонки
    for col in df.columns:
        if col != "_time" and pd.notna(row[col]):
            record[col] = float(row[col])
    sensor_data.append(record)

print(f"Отправляем {len(sensor_data)} точек данных...")

# Отправляем запрос
try:
    response = requests.post(
        "http://localhost:8000/predict",
        json={"sensor_data": sensor_data, "lookback_minutes": 60},
        timeout=10
    )
    
    print(f"Статус: {response.status_code}")
    print("Ответ от API:")
    print(response.json())
    
except Exception as e:
    print(f"Ошибка: {e}")