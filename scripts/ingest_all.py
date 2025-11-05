#!/usr/bin/env python3
"""
Скрипт для массовой загрузки всех документов из папки
"""
import sys
import os
from pathlib import Path

# Добавляем корневую директорию в PATH
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from scripts.ingest_file import ingest_document

def ingest_all_documents(docs_dir: str = "data/docs"):
    """
    Загрузка всех документов из папки

    Args:
        docs_dir: Путь к папке с документами
    """
    docs_path = Path(docs_dir)
    
    if not docs_path.exists():
        print(f"❌ Папка не найдена: {docs_dir}")
        return

    # Поддерживаемые форматы
    supported_extensions = ['.docx', '.md', '.txt']
    
    # Получаем список файлов
    files = []
    for ext in supported_extensions:
        files.extend(docs_path.glob(f'*{ext}'))
    
    total = len(files)
    print(f"\n📁 Найдено файлов: {total}")
    print("=" * 60)
    
    success_count = 0
    error_count = 0
    errors = []
    
    for i, file_path in enumerate(files, 1):
        print(f"\n[{i}/{total}] Обработка: {file_path.name}")
        
        try:
            success = ingest_document(
                str(file_path),
                active=True,
                author="Admin",
                tags="автозагрузка"
            )
            
            if success:
                success_count += 1
            else:
                error_count += 1
                errors.append(file_path.name)
                
        except Exception as e:
            print(f"❌ КРИТИЧЕСКАЯ ОШИБКА: {e}")
            error_count += 1
            errors.append(f"{file_path.name} ({str(e)})")
    
    # Итоги
    print("\n" + "=" * 60)
    print("📊 ИТОГИ ЗАГРУЗКИ:")
    print(f"   ✅ Успешно загружено: {success_count}")
    print(f"   ❌ Ошибок: {error_count}")
    
    if errors:
        print("\n⚠️  Файлы с ошибками:")
        for error in errors:
            print(f"   - {error}")
    
    print("=" * 60)


if __name__ == "__main__":
    docs_dir = sys.argv[1] if len(sys.argv) > 1 else "data/docs"
    ingest_all_documents(docs_dir)
