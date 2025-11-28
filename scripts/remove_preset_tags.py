"""
Скрипт для удаления предустановленных тегов из базы данных
"""
import sqlite3
import os
import sys

# Добавляем корневую директорию проекта в путь
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.config import METADATA_DB

def remove_preset_tags():
    """Удаление предустановленных тегов"""

    if not os.path.exists(METADATA_DB):
        print(f"База данных не найдена: {METADATA_DB}")
        return

    conn = sqlite3.connect(METADATA_DB)
    cursor = conn.cursor()

    # Удаляем предустановленные теги
    preset_tags = ['ЕГАИС', '1С']

    for tag in preset_tags:
        cursor.execute('DELETE FROM tags WHERE name = ?', (tag,))
        print(f"Удалён тег: {tag}")

    conn.commit()
    conn.close()

    print("Предустановленные теги удалены")

if __name__ == "__main__":
    remove_preset_tags()
