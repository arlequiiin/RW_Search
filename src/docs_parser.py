import os
from docx import Document

def read_txt_md(file_path: str) -> str:
    with open(file_path, "r", encoding="utf-8") as f:
        return f.read()

def read_docx(file_path: str) -> str:
    doc = Document(file_path)
    text = "\n".join(p.text for p in doc.paragraphs)
    return text

def extract_text(file_path: str) -> str:
    ext = os.path.splitext(file_path)[1].lower()
    if ext == ".txt" or ext == ".md":
        return read_txt_md(file_path)
    elif ext == ".docx":
        return read_docx(file_path)
    else:
        raise ValueError(f"Неподдерживаемый формат файла: {ext}")
