@echo off
setlocal

call .venv\Scripts\activate
python -m pytest tests -v
python -m src.train --config configs\_tmp_smoke.yaml
