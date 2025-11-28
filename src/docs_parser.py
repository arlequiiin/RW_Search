import os
import uuid
from docx import Document
from typing import Tuple, List, Dict

def read_txt_md(file_path: str) -> str:
    with open(file_path, "r", encoding="utf-8") as f:
        return f.read()

def read_docx(file_path: str) -> str:
    doc = Document(file_path)
    text = "\n".join(p.text for p in doc.paragraphs)
    return text

def extract_text(file_path: str) -> str:
    """
    Извлечение текста из файла

    Args:
        file_path: Путь к файлу

    Returns:
        Извлеченный текст
    """
    ext = os.path.splitext(file_path)[1].lower()
    if ext == ".txt" or ext == ".md":
        return read_txt_md(file_path)
    elif ext == ".docx":
        return read_docx(file_path)
    else:
        raise ValueError(f"Неподдерживаемый формат файла: {ext}")

def extract_text_with_filename(file_path: str) -> Tuple[str, str]:
    """
    Извлечение текста из файла с названием файла

    Args:
        file_path: Путь к файлу

    Returns:
        Tuple (текст, название файла без расширения)
    """
    text = extract_text(file_path)
    filename = os.path.basename(file_path)
    # Убираем расширение из названия
    filename_without_ext = os.path.splitext(filename)[0]
    return text, filename_without_ext

def prepare_text_for_chunking(text: str, filename_without_ext: str) -> str:
    """
    Подготовка текста для разбиения на чанки с добавлением названия документа

    Args:
        text: Исходный текст документа
        filename_without_ext: Название файла без расширения

    Returns:
        Текст с добавленным названием документа
    """
    # Добавляем название документа в начало
    header = f"Документ: {filename_without_ext}\n\n"
    return header + text


# === Новые функции для работы с двумя типами документов ===

def parse_single_instruction(
    file_path: str,
    tags: List[str] = None,
    author: str = "Admin"
) -> List[Dict]:
    """
    Обработка файла с одной инструкцией
    Название файла = название инструкции

    Args:
        file_path: Путь к файлу
        tags: Список тегов
        author: Автор документа

    Returns:
        Список из одной инструкции с метаданными
    """
    text, filename_without_ext = extract_text_with_filename(file_path)
    file_format = os.path.splitext(file_path)[1][1:]  # без точки

    instruction_data = {
        'id': str(uuid.uuid4()),
        'doc_id': str(uuid.uuid4()),
        'title': filename_without_ext,
        'file_path': file_path,
        'file_format': file_format,
        'source_type': 'single_file',
        'separator_index': None,
        'text': text,
        'tags': tags or [],
        'author': author,
        'images': []  # TODO: будет заполнено при извлечении изображений
    }

    return [instruction_data]


def parse_multi_instructions(
    file_path: str,
    tags: List[str] = None,
    author: str = "Admin"
) -> List[Dict]:
    """
    Обработка файла с множеством инструкций
    Разделитель: ---
    Первая строка после --- = название инструкции

    Args:
        file_path: Путь к файлу
        tags: Список тегов (применяются ко всем инструкциям)
        author: Автор документа

    Returns:
        Список инструкций с метаданными
    """
    text = extract_text(file_path)
    file_format = os.path.splitext(file_path)[1][1:]
    doc_id = str(uuid.uuid4())  # Общий doc_id для всех инструкций

    # Разделение по ---
    sections = text.split('---')
    instructions = []

    for idx, section in enumerate(sections):
        section = section.strip()
        if not section:
            continue

        # Первая строка = название инструкции
        lines = section.split('\n', 1)
        title = lines[0].strip()
        content = lines[1].strip() if len(lines) > 1 else ""

        if not title:
            # Если нет названия, пропускаем
            continue

        instruction_data = {
            'id': str(uuid.uuid4()),
            'doc_id': doc_id,
            'title': title,
            'file_path': file_path,
            'file_format': file_format,
            'source_type': 'multi_instruction',
            'separator_index': idx,
            'text': content,
            'tags': tags or [],
            'author': author,
            'images': []
        }

        instructions.append(instruction_data)

    return instructions


def parse_document(
    file_path: str,
    source_type: str,
    tags: List[str] = None,
    author: str = "Admin"
) -> List[Dict]:
    """
    Универсальная функция для парсинга документа

    Args:
        file_path: Путь к файлу
        source_type: Тип источника ('single_file' или 'multi_instruction')
        tags: Список тегов
        author: Автор документа

    Returns:
        Список инструкций с метаданными
    """
    if source_type == 'single_file':
        return parse_single_instruction(file_path, tags, author)
    elif source_type == 'multi_instruction':
        return parse_multi_instructions(file_path, tags, author)
    else:
        raise ValueError(f"Неподдерживаемый тип источника: {source_type}")
