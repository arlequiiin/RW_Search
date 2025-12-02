"""
Скрипт для парсинга экспорта Telegram чата в формат инструкций для RAG-системы

Этапы обработки:
1. Фильтрация мусора (стикеры, короткие сообщения, приветствия)
2. Группировка сообщений по временным интервалам (диалоги)
3. Извлечение технических диалогов по ключевым словам
4. Копирование изображений в data/images/
5. Создание .md файла с инструкциями
"""

import json
import os
import re
import shutil
from datetime import datetime
from pathlib import Path
from typing import List, Dict, Optional


# =============== КОНФИГУРАЦИЯ ===============

# Путь к JSON файлу экспорта
JSON_FILE = "result.json"

# Папка с фотографиями из экспорта
PHOTOS_SOURCE_DIR = "photos"

# Выходной файл
OUTPUT_FILE = "data/docs/руководитель_инструкции.md"

# Папка для копирования изображений
IMAGES_DIR = "data/images"

# Временной интервал для группировки диалога (в секундах)
DIALOGUE_TIME_WINDOW = 3 * 60 * 60  # 3 часа

# Минимальная длина сообщения (символы)
MIN_MESSAGE_LENGTH = 10

# Ключевые слова для определения технических сообщений
TECHNICAL_KEYWORDS = [
    # ЕГАИС
    "егаис", "утм", "марк", "остатки", "фтп", "ftp", "справочник",
    "утилита", "fsrar", "алкоголь",

    # 1С
    "1с", "робот", "загрузка", "выгрузка", "обработка", "база", "бд",
    "конфигурация", "обновление", "синхронизация",

    # Ошибки и проблемы
    "ошибка", "error", "не работает", "проблема", "баг", "падает",
    "вылетает", "зависает", "глюк", "сбой", "крэш",

    # Кассы и оборудование
    "ккм", "касса", "чек", "фискал", "терминал", "принтер",
    "сканер", "весы", "оборудование",

    # Системное
    "сервер", "комп", "виндовс", "windows", "драйвер", "служба",
    "порт", "ip", "сеть", "подключение", "настройка",

    # Документы и операции
    "накладная", "приход", "расход", "инвентаризация", "акт",
    "документ", "товар", "номенклатура", "ценник",

    # Действия
    "переустановка", "установка", "удаление", "перезагрузка",
    "запуск", "остановка", "проверка", "настройка"
]

# Слова-исключения (убираем из обработки)
SKIP_PHRASES = [
    "привет", "здравствуй", "спасибо", "пасиб", "ок", "окей",
    "да", "нет", "хорошо", "понял", "ясно", "норм", "отлично",
    "👍", "👌", "🙏", "😊", "😁", "+", "++", "+++",
    "спокойной ночи", "доброе утро", "добрый день", "пока"
]


# =============== УТИЛИТЫ ===============

def clean_text(text: str) -> str:
    """Очистка текста от лишних пробелов и символов"""
    if not text:
        return ""
    # Убираем множественные пробелы
    text = re.sub(r'\s+', ' ', text)
    return text.strip()


def is_skip_message(text: str) -> bool:
    """Проверка, является ли сообщение служебным/неинформативным"""
    if not text:
        return True

    text_lower = text.lower().strip()

    # Слишком короткое
    if len(text) < MIN_MESSAGE_LENGTH:
        return True

    # Точное совпадение с фразами-исключениями
    for phrase in SKIP_PHRASES:
        if text_lower == phrase or text_lower == phrase + ".":
            return True

    return False


def has_technical_content(text: str) -> bool:
    """Проверка наличия технического контента"""
    if not text:
        return False

    text_lower = text.lower()

    # Ищем ключевые слова
    for keyword in TECHNICAL_KEYWORDS:
        if keyword in text_lower:
            return True

    # Ищем паттерны (коды ошибок, IP-адреса, версии)
    patterns = [
        r'\d{1,3}\.\d{1,3}\.\d{1,3}\.\d{1,3}',  # IP адрес
        r'ошибк[аи]?\s*\d+',                     # "ошибка 409"
        r'error\s*\d+',                          # "error 500"
        r'версия\s*\d+',                         # "версия 8"
        r'код\s*\d+',                            # "код 123"
    ]

    for pattern in patterns:
        if re.search(pattern, text_lower):
            return True

    return False


def format_datetime(unix_timestamp: str) -> str:
    """Форматирование даты из unix timestamp"""
    try:
        dt = datetime.fromtimestamp(int(unix_timestamp))
        return dt.strftime("%Y-%m-%d %H:%M")
    except:
        return unix_timestamp


# =============== ОСНОВНАЯ ЛОГИКА ===============

def load_json(filepath: str) -> Dict:
    """Загрузка JSON файла"""
    with open(filepath, 'r', encoding='utf-8') as f:
        return json.load(f)


def filter_messages(messages: List[Dict]) -> List[Dict]:
    """
    Фильтрация сообщений:
    - Удаляем стикеры, служебные сообщения
    - Удаляем короткие и неинформативные
    - Оставляем только технические
    """
    filtered = []

    for msg in messages:
        # Пропускаем стикеры
        if msg.get('media_type') == 'sticker':
            continue

        # Пропускаем голосовые
        if msg.get('media_type') == 'voice_message':
            continue

        # Пропускаем служебные
        if msg.get('type') != 'message':
            continue

        # Извлекаем текст
        text = msg.get('text', '')
        if isinstance(text, list):
            # Иногда text - это массив объектов
            text = ' '.join([t.get('text', '') if isinstance(t, dict) else str(t) for t in text])

        text = clean_text(text)

        # Есть ли фото?
        has_photo = 'photo' in msg

        # Если есть фото - оставляем даже без текста
        if has_photo:
            msg['cleaned_text'] = text
            msg['has_photo'] = True
            filtered.append(msg)
            continue

        # Проверяем текст
        if is_skip_message(text):
            continue

        # Проверяем технический контент
        if not has_technical_content(text):
            continue

        # Добавляем очищенный текст
        msg['cleaned_text'] = text
        msg['has_photo'] = False
        filtered.append(msg)

    return filtered


def group_into_dialogues(messages: List[Dict]) -> List[List[Dict]]:
    """
    Группировка сообщений в диалоги по временным интервалам
    """
    if not messages:
        return []

    dialogues = []
    current_dialogue = [messages[0]]

    for i in range(1, len(messages)):
        prev_msg = messages[i - 1]
        curr_msg = messages[i]

        prev_time = int(prev_msg.get('date_unixtime', 0))
        curr_time = int(curr_msg.get('date_unixtime', 0))

        time_diff = curr_time - prev_time

        # Если разница во времени больше порога - новый диалог
        if time_diff > DIALOGUE_TIME_WINDOW:
            if current_dialogue:
                dialogues.append(current_dialogue)
            current_dialogue = [curr_msg]
        else:
            current_dialogue.append(curr_msg)

    # Добавляем последний диалог
    if current_dialogue:
        dialogues.append(current_dialogue)

    return dialogues


def extract_topic(dialogue: List[Dict]) -> str:
    """
    Извлечение темы диалога (для заголовка инструкции)
    Берём первое сообщение как описание проблемы
    """
    if not dialogue:
        return "Общий вопрос"

    first_msg = dialogue[0]
    text = first_msg.get('cleaned_text', '')

    # Ограничиваем длину заголовка
    if len(text) > 80:
        return text[:77] + "..."

    return text if text else "Техническая консультация"


def copy_images(dialogue: List[Dict], dialogue_idx: int) -> List[str]:
    """
    Копирование изображений из диалога в data/images/
    Возвращает список плейсхолдеров для .md файла
    """
    image_placeholders = []

    for msg_idx, msg in enumerate(dialogue):
        if not msg.get('has_photo'):
            continue

        photo_path = msg.get('photo', '')
        if not photo_path:
            continue

        # Проверяем существование исходного файла
        source_path = Path(PHOTOS_SOURCE_DIR) / Path(photo_path).name

        if not source_path.exists():
            # Пробуем относительный путь
            source_path = Path(photo_path)
            if not source_path.exists():
                print(f"⚠️  Изображение не найдено: {photo_path}")
                continue

        # Создаём уникальное имя для изображения
        ext = source_path.suffix
        new_name = f"telegram_d{dialogue_idx}_m{msg_idx}{ext}"

        # Путь назначения
        dest_path = Path(IMAGES_DIR) / new_name

        # Создаём папку если не существует
        dest_path.parent.mkdir(parents=True, exist_ok=True)

        # Копируем файл
        try:
            shutil.copy2(source_path, dest_path)
            # Относительный путь для плейсхолдера
            rel_path = f"images/{new_name}"
            image_placeholders.append(f"[[image: {rel_path}]]")
        except Exception as e:
            print(f"⚠️  Ошибка копирования {source_path}: {e}")

    return image_placeholders


def format_dialogue_to_markdown(dialogue: List[Dict], dialogue_idx: int) -> str:
    """
    Форматирование диалога в markdown инструкцию
    """
    if not dialogue:
        return ""

    # Заголовок
    topic = extract_topic(dialogue)

    # Дата
    first_date = format_datetime(dialogue[0].get('date_unixtime', '0'))

    # Копируем изображения
    images = copy_images(dialogue, dialogue_idx)

    # Собираем текст диалога
    conversation = []
    for msg in dialogue:
        author = msg.get('from', 'Unknown')
        text = msg.get('cleaned_text', '')

        if text or msg.get('has_photo'):
            conversation.append(f"**{author}:** {text}")

    # Формируем markdown
    md = f"# {topic}\n\n"
    md += f"**Дата:** {first_date}\n\n"

    # Добавляем изображения в начало (если есть)
    if images:
        md += "**Скриншоты:**\n"
        for img in images:
            md += f"{img}\n"
        md += "\n"

    # Добавляем диалог
    md += "**Диалог:**\n\n"
    md += "\n\n".join(conversation)
    md += "\n"

    return md


def process_chat(json_file: str, output_file: str):
    """
    Основная функция обработки чата
    """
    print("🚀 Начинаем обработку Telegram чата...")

    # Загружаем JSON
    print(f"📂 Загружаем {json_file}...")
    data = load_json(json_file)
    messages = data.get('messages', [])
    print(f"   Всего сообщений: {len(messages)}")

    # Фильтруем сообщения
    print("🔍 Фильтруем технические сообщения...")
    filtered = filter_messages(messages)
    print(f"   Осталось после фильтрации: {len(filtered)} ({len(filtered)/len(messages)*100:.1f}%)")

    # Группируем в диалоги
    print("📊 Группируем в диалоги...")
    dialogues = group_into_dialogues(filtered)
    print(f"   Создано диалогов: {len(dialogues)}")

    # Генерируем markdown
    print("📝 Создаём markdown файл...")
    markdown_content = []

    for idx, dialogue in enumerate(dialogues):
        md = format_dialogue_to_markdown(dialogue, idx)
        if md:
            markdown_content.append(md)

    # Объединяем с разделителем
    final_md = "\n---\n\n".join(markdown_content)

    # Сохраняем файл
    output_path = Path(output_file)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    with open(output_path, 'w', encoding='utf-8') as f:
        f.write(final_md)

    print(f"\n✅ Готово!")
    print(f"   Создано инструкций: {len(markdown_content)}")
    print(f"   Выходной файл: {output_file}")
    print(f"   Размер файла: {output_path.stat().st_size / 1024:.1f} КБ")
    print(f"\n💡 Следующий шаг: просмотрите файл и уберите лишнее вручную (~30-40% ручной работы)")


# =============== ЗАПУСК ===============

if __name__ == "__main__":
    # Проверяем наличие JSON файла
    if not os.path.exists(JSON_FILE):
        print(f"❌ Файл {JSON_FILE} не найден!")
        print(f"   Убедитесь, что result.json находится в корне проекта")
        exit(1)

    # Запускаем обработку
    process_chat(JSON_FILE, OUTPUT_FILE)
