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

    # Удаляем ВСЕ теги, которые не связаны с инструкциями
    cursor.execute('''
        DELETE FROM tags
        WHERE id NOT IN (SELECT DISTINCT tag_id FROM instruction_tags)
    ''')
    deleted_count = cursor.rowcount

    conn.commit()
    conn.close()

    print(f"Удалено неиспользуемых тегов: {deleted_count}")

if __name__ == "__main__":
    remove_preset_tags()
