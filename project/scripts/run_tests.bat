@echo off
setlocal

call .venv\Scripts\activate
python -m pytest tests -v
