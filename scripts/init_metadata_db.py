"""
Скрипт для инициализации базы данных метаданных
"""
import sqlite3
import os
import sys

# Добавляем корневую директорию проекта в путь
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.config import METADATA_DB, DATA_DIR

def init_metadata_database():
    """Создание структуры базы данных метаданных"""

    # Создаём директорию data, если не существует
    os.makedirs(DATA_DIR, exist_ok=True)

    conn = sqlite3.connect(METADATA_DB)
    cursor = conn.cursor()

    # Таблица инструкций
    cursor.execute('''
    CREATE TABLE IF NOT EXISTS instructions (
        id TEXT PRIMARY KEY,
        doc_id TEXT NOT NULL,
        title TEXT NOT NULL,
        file_path TEXT NOT NULL,
        file_format TEXT,
        source_type TEXT,
        separator_index INTEGER,
        active BOOLEAN DEFAULT 1,
        version INTEGER DEFAULT 1,
        author TEXT,
        created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
        updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
    )
    ''')

    # Таблица тегов
    cursor.execute('''
    CREATE TABLE IF NOT EXISTS tags (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        name TEXT UNIQUE NOT NULL,
        category TEXT
    )
    ''')

    # Связь инструкций и тегов (many-to-many)
    cursor.execute('''
    CREATE TABLE IF NOT EXISTS instruction_tags (
        instruction_id TEXT,
        tag_id INTEGER,
        PRIMARY KEY (instruction_id, tag_id),
        FOREIGN KEY (instruction_id) REFERENCES instructions(id) ON DELETE CASCADE,
        FOREIGN KEY (tag_id) REFERENCES tags(id) ON DELETE CASCADE
    )
    ''')

    # Таблица для хранения связей с изображениями
    cursor.execute('''
    CREATE TABLE IF NOT EXISTS instruction_images (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        instruction_id TEXT,
        image_path TEXT,
        image_index INTEGER,
        placeholder TEXT,
        FOREIGN KEY (instruction_id) REFERENCES instructions(id) ON DELETE CASCADE
    )
    ''')

    # История изменений (для версионирования)
    cursor.execute('''
    CREATE TABLE IF NOT EXISTS instruction_history (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        instruction_id TEXT,
        version INTEGER,
        backup_path TEXT,
        changed_by TEXT,
        change_description TEXT,
        changed_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
        FOREIGN KEY (instruction_id) REFERENCES instructions(id) ON DELETE CASCADE
    )
    ''')

    # Индексы для быстрого поиска
    cursor.execute('CREATE INDEX IF NOT EXISTS idx_instructions_doc_id ON instructions(doc_id)')
    cursor.execute('CREATE INDEX IF NOT EXISTS idx_instructions_active ON instructions(active)')
    cursor.execute('CREATE INDEX IF NOT EXISTS idx_instructions_title ON instructions(title)')
    cursor.execute('CREATE INDEX IF NOT EXISTS idx_tags_name ON tags(name)')

    # Добавляем предустановленные теги
    default_tags = [
        ('ЕГАИС', 'система'),
        ('1С', 'система'),
        ('SQL', 'система'),
    ]

    if default_tags:
        cursor.executemany(
            'INSERT OR IGNORE INTO tags (name, category) VALUES (?, ?)',
            default_tags
        )

    conn.commit()
    conn.close()

    print(f"База данных метаданных создана: {METADATA_DB}")
    if default_tags:
        print(f"Добавлено {len(default_tags)} предустановленных тегов")

if __name__ == "__main__":
    init_metadata_database()
