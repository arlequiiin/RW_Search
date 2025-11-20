@echo off
REM Остановка Streamlit приложения для Windows
chcp 65001 >nul

echo Остановка всех процессов Streamlit...

REM Находим и убиваем все процессы streamlit
tasklist /FI "IMAGENAME eq python.exe" /FO CSV | findstr /I "streamlit" >nul
if %errorlevel%==0 (
    echo Найдены запущенные процессы Streamlit
    for /f "tokens=2 delims=," %%a in ('tasklist /FI "IMAGENAME eq python.exe" /FO CSV ^| findstr /I "streamlit"') do (
        set PID=%%a
        set PID=!PID:"=!
        echo Останавливаю процесс !PID!
        taskkill /PID !PID! /F >nul 2>&1
    )
    echo Готово!
) else (
    echo Streamlit не запущен
)

timeout /t 2 >nul
