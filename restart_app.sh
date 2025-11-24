echo "Остановка Streamlit..."
pkill -f "streamlit run"
sleep 2

echo "Запуск Streamlit с новой конфигурацией..."
cd "$(dirname "$0")"
source .venv/bin/activate
nohup streamlit run src/app.py --server.port 8501 --server.address 0.0.0.0 --server.headless true > /tmp/streamlit.log 2>&1 &

sleep 3

if pgrep -f "streamlit run" > /dev/null; then
    echo "✅ Streamlit успешно запущен!"
    echo "📡 Адрес: http://localhost:8501"
else
    echo "❌ Ошибка запуска Streamlit"
    cat /tmp/streamlit.log
fi
