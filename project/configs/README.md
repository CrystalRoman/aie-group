# Папка `configs`

В этой папке находятся YAML-конфиги проекта.  
Именно через них задаются основные параметры обучения, инференса и сервиса.

## Основные конфиги

В проекте используются такие конфигурационные файлы:

- `training_unet.yaml`
- `training_attention_unet.yaml`
- `training_transformer_unet.yaml`
- `service.yaml`
- `_tmp_smoke.yaml`

При необходимости могут добавляться дополнительные конфиги для отладки или отдельных прогонов.

## Что хранится в training-конфигах

В training-конфигах обычно задаются:

### 1. Пути к данным
Например:
- `train_csv_path`
- `test_csv_path`
- `images_dir`
- `masks_dir`
- `artifacts_dir`

### 2. Параметры данных
Например:
- `image_size`
- `batch_size`
- `num_workers`
- `val_size`
- `positive_only`
- `min_mask_coverage`

### 3. Параметры модели и обучения
Например:
- `model_name`
- `base_channels`
- `encoder_name`
- `loss_name`
- `learning_rate`
- `num_epochs`
- `threshold`
- `threshold_candidates`
- `scheduler_patience`
- `early_stopping_patience`
- `device`

## Назначение отдельных файлов

### `training_unet.yaml`
Основной конфиг baseline-модели `UNet`.

### `training_attention_unet.yaml`
Конфиг для `AttentionUNet`.

### `training_transformer_unet.yaml`
Конфиг для `TransformerUNet`.  
Если трансформерная реализация использует `timm`, библиотека должна быть установлена в окружении.

### `service.yaml`
Конфиг FastAPI-сервиса.  
Обычно в нём задаются:
- `training_config_path`
- `predictions_dir`
- `device` (если нужно переопределить)

Именно через этот файл сервис понимает, какую модель ему загружать.

### `_tmp_smoke.yaml`
Укороченный конфиг для быстрого smoke-прогона.  
Нужен для короткой проверки того, что:
- данные читаются;
- модель создаётся;
- цикл обучения стартует;
- проект не падает на базовых шагах.
