# Итоговый проект по курсу «Инженерия Искусственного Интеллекта»

## 1. Паспорт проекта

- **Название проекта:** `Сравнительный анализ архитектур нейронных сетей для сегментации пневмоторакса`
- **Автор:** `Савельев Роман Сергеевич`
- **Группа:** `БФБО-01-23`
- **Контакт:** `frasermuck@gmail.com`
- **Ссылка на репозиторий:** `https://github.com/CrystalRoman/aie-group.git`

**Краткое описание:**  
Проект посвящён задаче автоматической сегментации пневмоторакса на рентгеновских снимках грудной клетки.  
В работе реализованы и сравнены три архитектуры сегментации: `UNet`, `AttentionUNet` и `TransformerUNet`.  
Финальная версия проекта включает:
- обучение моделей;
- сохранение артефактов и метрик;
- EDA;
- сравнение моделей;
- FastAPI-сервис для инференса;
- sanity-check тесты.

---

## 2. Структура проекта

```text
project/
├─ README.md
├─ report.md
├─ self-checklist.md
├─ requirements.txt
├─ .env.example
├─ .gitignore
├─ configs/
├─ data/
├─ notebooks/
├─ src/
├─ tests/
├─ artifacts/
└─ scripts/
```

### Назначение папок

- `configs/` — YAML-конфиги обучения, сервиса и smoke-проверки.
- `data/` — небольшой sample-поднабор данных для демонстрации и локального запуска.
- `notebooks/` — ноутбуки EDA, анализа метрик и проверки API.
- `src/` — основной код проекта.
- `tests/` — sanity-check и unit-тесты.
- `artifacts/` — веса моделей, метрики, графики, prediction examples, логи.
- `scripts/` — вспомогательные скрипты для подготовки окружения, запуска тестов, API и быстрой проверки.

---

## 3. Требования и окружение

### 3.1. Установка зависимостей локально

#### Windows CMD / PowerShell

```bat
python -m venv .venv
.venv\Scripts\activate
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
```

Или подготовленный bat-скрипт:

```bat
scripts\setup_env.bat
```

#### Linux / macOS / Git Bash

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
```

или:

```bash
bash scripts/setup_env.sh
```

### 3.2. Работа в Datasphere

В Datasphere отдельное `venv` обычно не требуется.  
Правильный способ - установить зависимости в текущее окружение ядра:

```python
import sys, subprocess
subprocess.check_call([sys.executable, "-m", "pip", "install", "-r", "requirements.txt"])
```

После этого:
- EDA и сравнение моделей запускаются из ноутбуков;
- сервис проверяется через `notebooks/service_api_check.ipynb`.

---

## 4. Как запустить проект

### 4.1. Запуск тестов

#### Windows

```bat
.venv\Scripts\activate
python -m pytest tests -v
```

или:

```bat
scripts\run_tests.bat
```

#### Linux / macOS

```bash
source .venv/bin/activate
pytest tests -v
```

или:

```bash
bash scripts/run_tests.sh
```

### 4.2. Быстрая проверка проекта

Smoke-проверка:
- тесты;
- короткий запуск обучения через `_tmp_smoke.yaml`.

#### Windows

```bat
.venv\Scripts\activate
python -m pytest tests -v
python -m src.train --config configs\_tmp_smoke.yaml
```

или:

```bat
scripts\check_project.bat
```

#### Linux / macOS

```bash
source .venv/bin/activate
pytest tests -v
python -m src.train --config configs/_tmp_smoke.yaml
```

или:

```bash
bash scripts/check_project.sh
```

### 4.3. Запуск обучения модели

#### Windows

UNet:
```bat
.venv\Scripts\activate
python -m src.train --config configs\training_unet.yaml
```

AttentionUNet:
```bat
.venv\Scripts\activate
python -m src.train --config configs\training_attention_unet.yaml
```

TransformerUNet:
```bat
.venv\Scripts\activate
python -m src.train --config configs\training_transformer_unet.yaml
```

#### Linux / macOS

```bash
source .venv/bin/activate
python -m src.train --config configs/training_unet.yaml
python -m src.train --config configs/training_attention_unet.yaml
python -m src.train --config configs/training_transformer_unet.yaml
```

#### Datasphere / Jupyter

```python
import sys
import subprocess

cmd = [sys.executable, "-u", "-m", "src.train", "--config", "configs/training_transformer_unet.yaml"]

process = subprocess.Popen(
    cmd,
    stdout=subprocess.PIPE,
    stderr=subprocess.STDOUT,
    text=True,
    bufsize=1
)

for line in process.stdout:
    print(line, end="")

return_code = process.wait()
print(f"\nProcess finished with code: {return_code}")
```

### 4.4. Запуск API

#### Windows

```bat
.venv\Scripts\activate
python -m uvicorn src.service:app --host 0.0.0.0 --port 8000
```

или:

```bat
scripts\run_api.bat
```

#### Linux / macOS

```bash
source .venv/bin/activate
python -m uvicorn src.service:app --host 0.0.0.0 --port 8000
```

или:

```bash
bash scripts/run_api.sh
```

### 4.5. Как открыть сайт после запуска сервера

Если сервер запущен локально на той же машине, Swagger UI открывается по адресу:

```text
http://127.0.0.1:8000/docs
```

или:

```text
http://localhost:8000/docs
```

Для проверки состояния сервиса:
```text
http://127.0.0.1:8000/health
```

### 4.6. Как проверить API вручную

#### `/health`

Windows / Linux:
```bash
curl http://127.0.0.1:8000/health
```

#### `/predict`

Windows:
```bash
curl -X POST "http://127.0.0.1:8000/predict" -F "file=@data\png_images\0_test_1_.png"
```

Linux / macOS:
```bash
curl -X POST "http://127.0.0.1:8000/predict" -F "file=@data/png_images/0_test_1_.png"
```

### 4.7. Реализованные endpoints

- `GET /health` — проверка, что сервис поднят и какая модель загружена.
- `POST /predict` — загрузка PNG/JPG рентгеновского снимка и получение предсказанной маски.
- `POST /reload` — перезагрузка модели и конфига сервиса без полного рестарта процесса.

### 4.8. Как работает threshold в API

В Swagger UI для `/predict` поле `threshold` уже заполнено **лучшим найденным порогом**, который сервис берёт из артефактов обучения (`train_summary_*.json`) для текущей модели.  
Если пользователь вручную вводит threshold, сервис принимает только допустимые значения из безопасного диапазона.

### 4.9. Какие веса моделей входят в репозиторий

Для воспроизводимой демонстрации API в репозиторий включён вес `UNet`, и именно эта модель рекомендуется как демонстрационная конфигурация сервиса.

Веса `AttentionUNet` и `TransformerUNet` в git-репозиторий не включены из-за большого размера файлов.  
Поэтому для публичного репозитория и локального запуска API по умолчанию используется `UNet`, а более тяжёлые веса:
- хранятся отдельно;
- могут быть показаны отдельно на защите;
- при необходимости могут быть вручную положены в папку `artifacts/.../models/`.

Если требуется запуск сервиса с другой моделью, достаточно:
1. положить соответствующий файл `best_*.pt` в нужную папку `artifacts/<имя_эксперимента>/models/`;
2. поменять `training_config_path` в `configs/service.yaml`;
3. перезапустить сервис или вызвать `/reload`.

---

## 5. Данные

В репозиторий рекомендуется класть **не полный датасет**, а только небольшой sample.

### Структура

```text
data/
├─ stage_1_train_images.csv
├─ stage_1_test_images.csv
├─ png_images/
├─ png_masks/
├─ README.md
└─ data_description.md
```

**Примечание:** если в старых артефактах (`train_summary_*.json`, `*_split.csv`) сохранились исторические пути `data_full/...`, это допустимо. Это результаты прошлых запусков, а не активные конфиги проекта.

---

## 6. Ноутбуки проекта

- `notebooks/exp01_eda_pneumothorax_dataset.ipynb` — EDA и проверка данных.
- `notebooks/exp02_metrics_model_comparison.ipynb` — сравнение моделей по метрикам.
- `notebooks/service_api_check.ipynb` — проверка `GET /health` и `POST /predict`.

---

## 7. Тесты

В папке `tests/` лежат:

- `test_metrics.py`
- `test_model_forward.py`
- `test_dataset.py`
- `test_config_loading.py`
- `conftest.py`

Запуск:
```bash
pytest tests -v
```

---

## 8. Демонстрация на защите

На защите покажу:

1. структуру проекта;
2. `notebooks/exp02_metrics_model_comparison.ipynb`;
3. Запущу сервис через `python -m uvicorn src.service:app --host 0.0.0.0 --port 8000` или `notebooks/service_api_check.ipynb`, покажу пару запросов через Swagger UI;
4. `GET /health`;
5. `POST /predict`;
6. сохранённую `overlay`-картинку.

---

## 9. Ограничения и дальнейшая работа

- Финальная схема обучения использовала positive-only режим и фильтрацию маленьких масок для устойчивой сегментации.
- API реализован как минимальный FastAPI-сервис.
- В Datasphere Swagger UI может быть недоступен из-за ограничения порта, поэтому сервис дополнительно проверяется из ноутбука.
- В дальнейшем можно усилить инференс-пайплайн, добавить Docker-развёртывание и расширить тесты.
