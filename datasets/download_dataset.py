#!/usr/bin/env python
import requests
#200 мб примерно
# ====== Прямая ссылка на релиз GitHub ======
url = "https://github.com/Doshchepkov/jinr-monitoring/releases/download/dataset/merged_dataset2.csv"
output_file = "datasets/merged_dataset2.csv"

print(f"Скачиваем с: {url}")
resp = requests.get(url, stream=True)
resp.raise_for_status()

with open(output_file, "wb") as f:
    for chunk in resp.iter_content(chunk_size=8192):
        f.write(chunk)

print(f"Файл сохранён: {output_file}")