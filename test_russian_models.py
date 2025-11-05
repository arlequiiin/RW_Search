#!/usr/bin/env python3
"""Тест русскоязычных моделей"""

from sentence_transformers import SentenceTransformer, util

print("=" * 70)
print("ТЕСТ ЛУЧШИХ МОДЕЛЕЙ ДЛЯ РУССКОГО")
print("=" * 70)

query = "как написать расхождения?"
documents = [
    "Каждый день отдел закрытия пишут расхождения между 1 и 3 регистром",
    "Перед выполнением операции нужно открыть базу",
    "Для создания обращения укажите номер расхождения",
]

# Модели оптимизированные для русского
models_to_test = [
    "sentence-transformers/paraphrase-multilingual-mpnet-base-v2",
    "cointegrated/rubert-tiny2",
]

for model_name in models_to_test:
    print(f"\n🔬 Модель: {model_name}")
    
    try:
        model = SentenceTransformer(model_name)
        
        query_emb = model.encode(query, convert_to_tensor=True)
        doc_embs = model.encode(documents, convert_to_tensor=True)
        
        similarities = util.cos_sim(query_emb, doc_embs)[0]
        
        for i, sim in enumerate(similarities, 1):
            marker = "✅" if i == 1 else "  "
            print(f"{marker} [{i}] {sim.item()*100:.2f}% - {documents[i-1][:60]}...")
        
        best_idx = similarities.argmax().item()
        if best_idx == 0:
            print(f"✅ Правильно!")
        else:
            print(f"❌ Ошибка: выбрал [{best_idx+1}]")
            
    except Exception as e:
        print(f"❌ Ошибка загрузки: {e}")

print(f"\n{'='*70}")
