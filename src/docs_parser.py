import os
from docx import Document
from typing import Tuple

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
