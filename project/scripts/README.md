# Папка `scripts`

В этой папке лежат вспомогательные скрипты запуска.

## Linux / macOS / Git Bash
- `setup_env.sh` — создать `.venv` и установить зависимости
- `run_tests.sh` — запустить `pytest tests -v`
- `run_api.sh` — поднять FastAPI
- `check_project.sh` — тесты + smoke-проверка

## Windows
- `setup_env.bat`
- `run_tests.bat`
- `run_api.bat`
- `check_project.bat`
- `train_unet.bat`
- `train_attention_unet.bat`
- `train_transformer_unet.bat`

## Важно

`.sh`-скрипты запускаются через `bash`, а не через `python`.  
На Windows удобнее использовать `.bat`-скрипты или обычные команды вручную.
