@echo off
REM Скрипт запуска Streamlit приложения для Windows
chcp 65001 >nul

echo Запуск RAG приложения...
echo.

REM Активация виртуального окружения
if exist .venv\Scripts\activate.bat (
    call .venv\Scripts\activate.bat
) else (
    echo Ошибка: виртуальное окружение не найдено!
    echo Создайте его командой: python -m venv .venv
    pause
    exit /b 1
)

REM Запуск Streamlit
streamlit run src/app.py --server.port 8501 --server.address localhost

pause
