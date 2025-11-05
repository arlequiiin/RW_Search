#!/usr/bin/env python3
"""Сравнение разных моделей эмбеддингов для русского"""

from sentence_transformers import SentenceTransformer, util

print("=" * 70)
print("СРАВНЕНИЕ МОДЕЛЕЙ ЭМБЕДДИНГОВ ДЛЯ РУССКОГО ЯЗЫКА")
print("=" * 70)

query = "как написать расхождения?"
documents = [
    "Каждый день отдел закрытия пишут расхождения между 1 и 3 регистром",
    "Перед выполнением операции нужно открыть базу",
    "Для создания обращения укажите номер расхождения",
]

models_to_test = [
    ("all-MiniLM-L6-v2", "Текущая модель (EN focus)"),
    ("sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2", "Multilingual MiniLM"),
    ("intfloat/multilingual-e5-small", "Multilingual E5 Small"),
]

for model_name, description in models_to_test:
    print(f"\n{'='*70}")
    print(f"🔬 Модель: {model_name}")
    print(f"   {description}")
    print(f"{'='*70}")
    
    try:
        print(f"   Загрузка модели...")
        model = SentenceTransformer(model_name)
        
        query_emb = model.encode(query, convert_to_tensor=True)
        doc_embs = model.encode(documents, convert_to_tensor=True)
        
        similarities = util.cos_sim(query_emb, doc_embs)[0]
        
        print(f"\n   📊 Результаты:")
        for i, sim in enumerate(similarities, 1):
            marker = "✅" if i == 1 else "  "
            print(f"   {marker} [{i}] {sim.item()*100:.2f}%")
        
        # Проверка: правильный ли документ на первом месте
        best_idx = similarities.argmax().item()
        if best_idx == 0:
            print(f"\n   ✅ ПРАВИЛЬНО: Нашёл документ с 'расхождения'")
        else:
            print(f"\n   ❌ ОШИБКА: Лучший результат - документ [{best_idx+1}]")
            
    except Exception as e:
        print(f"   ❌ Ошибка: {e}")

print(f"\n{'='*70}")
