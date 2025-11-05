#!/usr/bin/env python3
"""Тест лучших русскоязычных моделей эмбеддингов"""

from sentence_transformers import SentenceTransformer, util
import time

print("=" * 70)
print("ПОИСК ЛУЧШЕЙ МОДЕЛИ ДЛЯ РУССКОГО ЯЗЫКА")
print("=" * 70)

# Реальные тексты из твоей базы
query = "расхождения"
documents = [
    # Должен быть №1 - файл Расхождения.md
    "Каждый день отдел закрытия пишут расхождения между 1 и 3 регистром",
    # Нерелевантные
    "Перед выполнением операции нужно открыть базу данных",
    "Отключение выгрузки справочника цен в систему",
    "Восстановление документа комплектации товаров"
]

# Модели для тестирования (от легких к тяжелым)
models_to_test = [
    ("all-MiniLM-L6-v2", "Текущая (EN-focused, плохо для RU)"),
    ("sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2", "Multilingual MiniLM (быстрая)"),
    ("sentence-transformers/paraphrase-multilingual-mpnet-base-v2", "Multilingual MPNet (качество)"),
    ("intfloat/multilingual-e5-small", "E5 Small (новая, хорошая)"),
]

results = []

for model_name, description in models_to_test:
    print(f"\n{'='*70}")
    print(f"🔬 {model_name}")
    print(f"   {description}")
    print(f"{'='*70}")
    
    try:
        start = time.time()
        model = SentenceTransformer(model_name)
        load_time = time.time() - start
        
        # Тест на скорость
        start = time.time()
        query_emb = model.encode(query, convert_to_tensor=True)
        doc_embs = model.encode(documents, convert_to_tensor=True)
        encode_time = time.time() - start
        
        similarities = util.cos_sim(query_emb, doc_embs)[0]
        
        # Проверка: правильный ли документ на первом месте?
        best_idx = similarities.argmax().item()
        is_correct = (best_idx == 0)
        
        print(f"\n   📊 Результаты:")
        for i, sim in enumerate(similarities, 1):
            marker = "✅" if i == 1 else ("🟡" if sim > 0.5 else "  ")
            print(f"   {marker} [{i}] {sim.item()*100:.1f}%")
        
        print(f"\n   ⏱️  Время загрузки: {load_time:.2f}с")
        print(f"   ⚡ Время эмбеддингов: {encode_time:.3f}с")
        
        if is_correct:
            print(f"   ✅ ПРАВИЛЬНО: Нашёл 'расхождения' на 1 месте!")
            verdict = "✅ ОТЛИЧНО"
        else:
            print(f"   ❌ ОШИБКА: Лучший = документ [{best_idx+1}]")
            verdict = "❌ ПЛОХО"
        
        results.append({
            'model': model_name,
            'correct': is_correct,
            'top1_score': similarities[0].item(),
            'load_time': load_time,
            'encode_time': encode_time,
            'verdict': verdict
        })
            
    except Exception as e:
        print(f"   ❌ Ошибка: {e}")
        results.append({
            'model': model_name,
            'correct': False,
            'verdict': f"❌ ОШИБКА: {str(e)[:50]}"
        })

# Итоговая таблица
print(f"\n{'='*70}")
print("📊 ИТОГОВАЯ ТАБЛИЦА:")
print(f"{'='*70}")
print(f"{'Модель':<50} {'Результат':<15} {'Точность'}")
print("-" * 70)

for r in results:
    if 'top1_score' in r:
        print(f"{r['model'][:48]:<50} {r['verdict']:<15} {r['top1_score']*100:.1f}%")
    else:
        print(f"{r['model'][:48]:<50} {r['verdict']}")

print(f"{'='*70}")

# Рекомендация
print("\n🎯 РЕКОМЕНДАЦИЯ:")
correct_models = [r for r in results if r.get('correct')]
if correct_models:
    best = max(correct_models, key=lambda x: x.get('top1_score', 0))
    print(f"   Лучшая модель: {best['model']}")
    print(f"   Точность: {best['top1_score']*100:.1f}%")
else:
    print("   ⚠️  Ни одна модель не дала правильный результат!")
    print("   Рекомендую: sentence-transformers/paraphrase-multilingual-mpnet-base-v2")

print(f"{'='*70}")
