#!/bin/bash
# Скрипт запуска Streamlit приложения

cd "$(dirname "$0")"
source .venv/bin/activate
streamlit run src/app.py --server.port 8501 --server.address localhost --server.headless true
