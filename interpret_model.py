#!/usr/bin/env python
"""
Интерпретация модели XGBoost с помощью SHAP
"""

import argparse
import os
import pandas as pd
import numpy as np
import joblib
import matplotlib.pyplot as plt
import shap
import warnings

warnings.filterwarnings("ignore")

from src.features_main import add_time_features
from src.episodes import make_episodes


def build_feature_names(feature_cols, L):
    """
    Создаёт имена признаков для лагов времени
    """
    names = []

    for lag in range(L):
        for f in feature_cols:
            names.append(f"{f}_t-{L-lag}")

    return names


def interpret_model(
    model_path,
    data_path,
    output_dir="screenshots",
    n_samples=1000,
    L=60,
    H=30,
    n=0.1,
):

    print("=" * 50)
    print("SHAP интерпретация модели")
    print("=" * 50)

    # ------------------------------------------------
    # 1 Загрузка модели
    # ------------------------------------------------

    print("\n1. Загрузка модели...")

    model_data = joblib.load(model_path)

    model = model_data["model"]
    feature_cols = model_data["feature_cols"]
    mean = model_data["mean"]
    std = model_data["std"]

    print(f"   Модель загружена. Признаков: {len(feature_cols)}")

    # ------------------------------------------------
    # 2 Загрузка данных
    # ------------------------------------------------

    print("\n2. Загрузка данных...")

    df = pd.read_csv(data_path)
    df = add_time_features(df, "_time")

    print(f"   Загружено {len(df):,} строк")

    # ------------------------------------------------
    # 3 Формирование эпизодов
    # ------------------------------------------------

    print("\n3. Формирование эпизодов...")

    df_sample = df.tail(min(n_samples + L + H, len(df)))

    X, y = make_episodes(df_sample, feature_cols, mean, std, L, H, n, aug_k=1)

    X_flat = X.reshape(X.shape[0], -1)

    print(f"   Сформировано {X_flat.shape[0]} эпизодов")

    # ------------------------------------------------
    # 4 SHAP
    # ------------------------------------------------

    print("\n4. Вычисление SHAP значений...")

    explainer = shap.TreeExplainer(model)

    n_shap = min(200, X_flat.shape[0])
    X_shap = X_flat[:n_shap]

    print(f"   Используем {n_shap} эпизодов")

    shap_values = explainer.shap_values(X_shap)

    # ------------------------------------------------
    # имена признаков
    # ------------------------------------------------

    expanded_feature_names = build_feature_names(feature_cols, L)

    print(f"   Реальных признаков: {len(expanded_feature_names)}")

    # ------------------------------------------------
    # папка вывода
    # ------------------------------------------------

    os.makedirs(output_dir, exist_ok=True)

    # ------------------------------------------------
    # summary plot
    # ------------------------------------------------

    print("\n5. Сохранение SHAP графиков...")

    plt.figure(figsize=(12, 8))

    shap.summary_plot(
        shap_values,
        X_shap,
        feature_names=expanded_feature_names,
        show=False
    )

    plt.tight_layout()

    plt.savefig(
        f"{output_dir}/shap_summary_beeswarm.png",
        dpi=150,
        bbox_inches="tight"
    )

    plt.close()

    print("   shap_summary_beeswarm.png")

    # ------------------------------------------------
    # bar importance
    # ------------------------------------------------

    plt.figure(figsize=(12, 8))

    shap.summary_plot(
        shap_values,
        X_shap,
        feature_names=expanded_feature_names,
        plot_type="bar",
        show=False
    )

    plt.tight_layout()

    plt.savefig(
        f"{output_dir}/shap_summary_bar.png",
        dpi=150,
        bbox_inches="tight"
    )

    plt.close()

    print("   shap_summary_bar.png")

    # ------------------------------------------------
    # важность признаков
    # ------------------------------------------------

    print("\n6. Топ 10 признаков")

    feature_importance = np.abs(shap_values).mean(axis=0)

    top_idx = np.argsort(feature_importance)[-10:][::-1]

    for i in top_idx:
        print(
            f"{expanded_feature_names[i]} : {feature_importance[i]:.4f}"
        )

    importance_df = pd.DataFrame(
        {
            "feature": expanded_feature_names,
            "importance": feature_importance,
        }
    ).sort_values("importance", ascending=False)

    importance_df.to_csv(
        f"{output_dir}/shap_importance.csv",
        index=False
    )

    print("   shap_importance.csv")

    # ------------------------------------------------
    # waterfall
    # ------------------------------------------------

    print("\n7. Waterfall пример")

    probs = model.predict_proba(X_shap)[:, 1]

    idx = np.argmax(probs)

    shap.waterfall_plot(
        shap.Explanation(
            values=shap_values[idx],
            base_values=explainer.expected_value,
            data=X_shap[idx],
            feature_names=expanded_feature_names,
        ),
        max_display=10,
        show=False,
    )

    plt.tight_layout()

    plt.savefig(
        f"{output_dir}/shap_waterfall_example.png",
        dpi=150,
        bbox_inches="tight"
    )

    plt.close()

    print("   shap_waterfall_example.png")

    print("\nSHAP интерпретация завершена")
    print(f"Графики сохранены в {output_dir}")


if __name__ == "__main__":

    parser = argparse.ArgumentParser()

    parser.add_argument("--model", default="models/final_xgb.pkl")

    parser.add_argument("--data", required=True)

    parser.add_argument("--output", default="screenshots")

    parser.add_argument("--samples", type=int, default=1000)

    parser.add_argument("--L", type=int, default=60)

    parser.add_argument("--H", type=int, default=30)

    parser.add_argument("--n", type=float, default=0.1)

    args = parser.parse_args()

    interpret_model(
        args.model,
        args.data,
        args.output,
        args.samples,
        args.L,
        args.H,
        args.n,
    )