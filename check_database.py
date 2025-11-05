#!/usr/bin/env python3
"""Проверка наличия файла в базе"""

from src.storage import get_chroma

client, collection = get_chroma()

# Получаем все документы
results = collection.get()

# Ищем файл Расхождения.md
found = []
for i, metadata in enumerate(results["metadatas"]):
    filename = metadata.get("filename", "")
    text = results["documents"][i]
    
    if "расхожден" in filename.lower() or "расхожден" in text.lower():
        found.append({
            "id": results["ids"][i],
            "filename": filename,
            "text": text[:200]
        })

print(f"Найдено чанков с 'расхожден': {len(found)}")

if found:
    for item in found[:3]:
        print(f"\nФайл: {item['filename']}")
        print(f"ID: {item['id']}")
        print(f"Text: {item['text']}...")
else:
    print("\n❌ Файл 'Расхождения.md' НЕ НАЙДЕН в базе!")
    print("\nВозможно, файл не был загружен. Проверим:")
    
import os
if os.path.exists("data/docs/Расхождения.md"):
    print("✅ Файл существует в data/docs/")
else:
    print("❌ Файл НЕ существует в data/docs/")
