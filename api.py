#!/usr/bin/env python
"""
Максимально простой API для предсказания скачков температуры
"""

import pandas as pd
import numpy as np
import joblib
from fastapi import FastAPI
from pydantic import BaseModel
from typing import List, Optional
import warnings
warnings.filterwarnings('ignore')

# Загружаем модель при старте
print("Загрузка модели...")
model_data = joblib.load("models/final_xgb.pkl")
model = model_data["model"]
FEATURE_COLS = model_data["feature_cols"]
MEAN = model_data["mean"][0]
STD = model_data["std"][0]
THRESHOLD = model_data.get("threshold", 0.5)

print(f"Модель загружена. Признаков: {len(FEATURE_COLS)}")
print(f"Порог: {THRESHOLD}")

# ==================== ФУНКЦИЯ ПРЕДОБРАБОТКИ ====================

def add_time_features(df):
    """Добавляет временные признаки"""
    df = df.copy()
    df["_time"] = pd.to_datetime(df["timestamp"])
    df["hour"] = df["_time"].dt.hour
    df["dayofweek"] = df["_time"].dt.dayofweek
    df["month"] = df["_time"].dt.month
    df["is_weekend"] = (df["dayofweek"] >= 5).astype(int)
    df["hour_sin"] = np.sin(2 * np.pi * df["hour"] / 24)
    df["hour_cos"] = np.cos(2 * np.pi * df["hour"] / 24)
    df["dow_sin"] = np.sin(2 * np.pi * df["dayofweek"] / 7)
    df["dow_cos"] = np.cos(2 * np.pi * df["dayofweek"] / 7)
    return df

def prepare_data(json_data, lookback=60):
    """Превращает JSON в нормализованный вектор для модели"""
    # Создаём DataFrame
    df = pd.DataFrame(json_data)
    
    # Добавляем временные признаки
    df = add_time_features(df)
    
    # Берём последние lookback минут
    if len(df) < lookback:
        raise ValueError(f"Нужно {lookback} точек, получено {len(df)}")
    df = df.tail(lookback)
    
    # Проверяем наличие всех признаков
    missing = [c for c in FEATURE_COLS if c not in df.columns]
    if missing:
        print(f"Предупреждение: отсутствуют признаки: {missing[:5]}")
    
    # Берём только те признаки, которые есть
    available = [c for c in FEATURE_COLS if c in df.columns]
    X = df[available].to_numpy(dtype=float)
    
    # Нормализация (только для доступных признаков)
    mean_subset = MEAN[:len(available)]
    std_subset = STD[:len(available)]
    X_norm = (X - mean_subset) / std_subset
    
    # Превращаем окно в плоский вектор
    X_flat = X_norm.reshape(1, -1)
    
    return X_flat

# ==================== PYDANTIC МОДЕЛИ ====================

class SensorDataPoint(BaseModel):
    timestamp: str
    class Config:
        extra = "allow"  # разрешаем любые поля

class PredictRequest(BaseModel):
    sensor_data: List[SensorDataPoint]
    lookback_minutes: Optional[int] = 60

class PredictResponse(BaseModel):
    probability: float
    is_anomaly: bool
    threshold: float

# ==================== API ====================

app = FastAPI()

@app.get("/")
def root():
    return {"status": "ok", "n_features": len(FEATURE_COLS), "threshold": THRESHOLD}

@app.post("/predict", response_model=PredictResponse)
def predict(request: PredictRequest):
    try:
        # Преобразуем в список словарей
        data = [p.dict() for p in request.sensor_data]
        
        # Подготавливаем данные
        X = prepare_data(data, request.lookback_minutes)
        
        # Предсказываем
        prob = model.predict_proba(X)[0, 1]
        is_anomaly = prob >= THRESHOLD
        
        return {
            "probability": float(prob),
            "is_anomaly": is_anomaly,
            "threshold": THRESHOLD
        }
    except Exception as e:
        return {"error": str(e)}

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)