@echo off
setlocal

call .venv\Scripts\activate
python -m uvicorn src.service:app --host 0.0.0.0 --port 8000
