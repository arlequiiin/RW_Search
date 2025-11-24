# Как использовать проект

## Быстрый старт

### 1. Установка зависимостей (на рабочем ПК в WSL)
```bash
# Создание виртуального окружения
python -m venv .venv

# Активация виртуального окружения
source .venv/bin/activate

# Установка зависимостей
pip install -r requirements.txt
```

### 2. Настройка удаленного доступа через VSCode (5 минут)

**На рабочем ПК (в WSL):**
```bash
# Убедитесь, что SSH сервер установлен
sudo apt update
sudo apt install openssh-server
sudo service ssh start
```

**На ноутбуке:**
1. Установите расширение **Remote - SSH** в VSCode
2. Нажмите `F1` → `Remote-SSH: Connect to Host` → `Add New SSH Host`
3. Введите: `ssh ваш_пользователь@IP_рабочего_ПК`
4. Подключитесь (введите пароль)
5. VSCode автоматически пробросит порты!

**Узнать IP рабочего ПК:**
```bash
# В WSL на рабочем ПК
hostname -I
```

### 3. Запуск приложения

**В WSL на рабочем ПК (или через VSCode Remote):**
```bash
# Простой запуск
./run_app.sh

# Перезапуск (если порт занят)
./restart_app.sh
```

**Ручной запуск:**
```bash
streamlit run src/app.py --server.port 8501
```

### 4. Доступ к приложению

**Если работаете через VSCode Remote SSH:**
```
http://localhost:8501  # VSCode автоматически пробросит порт
```

**Если работаете напрямую:**
```
http://IP_РАБОЧЕГО_ПК:8501
```

## Решение проблем

### Проблема: порт 8501 занят

**Способ 1 (перезапуск):**
```bash
./restart_app.sh
```

**Способ 2 (найти и завершить процесс):**
```bash
# Найти процесс
lsof -ti:8501

# Убить процесс
lsof -ti:8501 | xargs kill -9
```

**Способ 3 (запуск на другом порту):**
```bash
streamlit run src/app.py --server.port 8502
```