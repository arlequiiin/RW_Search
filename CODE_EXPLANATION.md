# Подробное объяснение кода RAG-системы

## 📋 Содержание
1. [Общая архитектура](#общая-архитектура)
2. [Поток данных](#поток-данных)
3. [Детальный разбор модулей](#детальный-разбор-модулей)
4. [Сложные конструкции](#сложные-конструкции)
5. [Примеры работы](#примеры-работы)

---

## 🏗️ Общая архитектура

### Схема вызовов (что кого вызывает)

```
app.py (Streamlit UI)
    │
    ├──> create_rag_pipeline() [из rag_pipeline.py]
    │       │
    │       ├──> EmbeddingModel() [из embeddings.py]
    │       │       └──> SentenceTransformer (библиотека)
    │       │
    │       ├──> get_chroma() [из storage.py]
    │       │       └──> ChromaDB (библиотека)
    │       │
    │       └──> get_llm_client() [из llm_client.py]
    │               └──> Ollama (библиотека)
    │
    ├──> extract_text_with_filename() [из docs_parser.py]
    │       └──> read_docx() или read_txt_md()
    │
    ├──> prepare_text_for_chunking() [из docs_parser.py]
    │
    └──> split_text() [из chunker.py]
```

### Структура проекта (модули)

| Файл | Назначение | Что возвращает |
|------|-----------|----------------|
| `config.py` | Конфигурация (константы) | Константы |
| `embeddings.py` | Работа с эмбеддингами | numpy массивы |
| `storage.py` | ChromaDB клиент | client, collection |
| `llm_client.py` | Генерация ответов | строка (ответ) |
| `docs_parser.py` | Парсинг документов | текст |
| `chunker.py` | Разбиение на чанки | список строк |
| `rag_pipeline.py` | Основная логика RAG | словарь с ответом |
| `hybrid_search.py` | Гибридный поиск | список результатов |
| `app.py` | Веб-интерфейс | None (рендерит UI) |

---

## 🔄 Поток данных

### Сценарий 1: Добавление документа

```
1. Пользователь загружает файл через Streamlit
   ↓
2. app.py: сохраняет файл в data/docs/
   ↓
3. docs_parser.extract_text_with_filename(file_path)
   → возвращает: (текст, "Название_файла")
   ↓
4. docs_parser.prepare_text_for_chunking(text, filename)
   → добавляет заголовок: "Документ: Название_файла\n\nТекст..."
   ↓
5. chunker.split_text(text, max_length=2000, overlap=200)
   → возвращает: ["чанк1", "чанк2", ...]
   ↓
6. EmbeddingModel.encode(chunks)
   → возвращает: numpy array [[0.1, 0.2, ...], [0.3, 0.4, ...]]
   ↓
7. collection.add(documents=chunks, embeddings=embeddings, metadatas=metadata, ids=ids)
   → сохраняет в ChromaDB
```

### Сценарий 2: Поиск (Query)

```
1. Пользователь вводит запрос в Streamlit
   ↓
2. app.py: rag.query(user_query, top_k=3)
   ↓
3. RAGPipeline.search_similar(query)
   ├──> EmbeddingModel.encode([query])
   │    → возвращает: numpy array [0.5, 0.6, ...]
   │
   └──> collection.query(query_embeddings=[...], n_results=3)
        → ChromaDB ищет похожие векторы
        → возвращает: {documents: [...], metadatas: [...], distances: [...]}
   ↓
4. RAGPipeline.format_context(documents)
   → формирует контекст: "[Документ 1: ...]\nТекст\n---\n[Документ 2: ...]"
   ↓
5. LLMClient.generate_rag_answer(query, context)
   ├──> формирует промпт с системной инструкцией
   │
   └──> ollama.chat(model="qwen2.5:14b", messages=[...])
        → возвращает: "Ответ от LLM"
   ↓
6. Возврат в app.py
   → отображение ответа + источников
```

---

## 🔍 Детальный разбор модулей

### 1. config.py - Конфигурация

```python
import os
BASE_DIR = os.path.dirname(os.path.dirname(__file__))
# __file__ = /home/rinat/GitHub/src/config.py
# os.path.dirname(__file__) = /home/rinat/GitHub/src
# os.path.dirname(dirname) = /home/rinat/GitHub
```

**Что происходит:**
- `__file__` - путь к текущему файлу
- `os.path.dirname(__file__)` - получаем директорию (src/)
- Повторный `dirname` - поднимаемся на уровень выше (корень проекта)

**Константы:**
```python
EMBEDDING_MODEL_NAME = "intfloat/multilingual-e5-large"
# Модель для преобразования текста в векторы (384 измерения)

CHUNK_SIZE_TOKENS = 500
# Размер одного чанка (куска текста) в токенах
# 1 токен ≈ 0.75 слова, значит ~375 слов, ~2000 символов

CHUNK_OVERLAP_TOKENS = 50
# Перекрытие между чанками (чтобы не терять контекст на границах)

TOP_K = 5
# Сколько самых похожих документов возвращать при поиске

LLM_MODEL_NAME = "qwen2.5:14b-instruct-q4_K_M"
# Языковая модель для генерации ответов (14 миллиардов параметров, квантизированная)
```

---

### 2. embeddings.py - Работа с векторными представлениями

```python
from sentence_transformers import SentenceTransformer

class EmbeddingModel:
    def __init__(self, model_name="all-MiniLM-L6-v2"):
        # Загружаем предобученную модель из HuggingFace
        self.model = SentenceTransformer(model_name)
        # При первом запуске скачивает модель (~400MB) в ~/.cache/
```

**Метод encode:**
```python
def encode(self, texts):
    return self.model.encode(texts, convert_to_numpy=True)
```

**Что происходит внутри:**
1. `texts` - может быть строка или список строк: `["текст 1", "текст 2"]`
2. Модель прогоняет каждый текст через Transformer:
   - Токенизация (разбиение на слова/подслова)
   - Прогон через BERT-подобную нейронную сеть
   - Pooling (усреднение) выходов последнего слоя
3. **Возвращает:** numpy массив формы `(N, 384)`, где N - количество текстов

**Пример:**
```python
embedder = EmbeddingModel("intfloat/multilingual-e5-large")
vectors = embedder.encode(["Привет мир", "Hello world"])
# vectors.shape = (2, 384)
# vectors[0] = [0.123, -0.456, 0.789, ..., 0.234]  # 384 числа
```

**Зачем это нужно:**
- Тексты с похожим смыслом будут иметь близкие векторы
- Можем искать похожие документы через косинусное расстояние

---

### 3. storage.py - Векторная база данных

```python
import chromadb
from src.config import CHROMA_DIR

def get_chroma():
    # Создаём постоянный клиент (данные сохраняются на диск)
    client = chromadb.PersistentClient(path=CHROMA_DIR)
    # CHROMA_DIR = "/home/rinat/GitHub/chroma_db"

    # Получаем или создаём коллекцию "documents"
    collection = client.get_or_create_collection("documents")

    return client, collection
```

**Что такое ChromaDB:**
- Специализированная база для хранения векторов + метаданных
- Использует индекс HNSW (быстрый поиск ближайших соседей)
- Автоматически вычисляет косинусное расстояние между векторами

**Как хранятся данные:**
```
chroma_db/
  └── chroma.sqlite3           # Метаданные
  └── index/                    # HNSW индекс
      └── id_to_uuid/
      └── uuid_to_id/
```

**Методы коллекции:**

1. **Добавление:**
```python
collection.add(
    documents=["текст чанка 1", "текст чанка 2"],
    embeddings=[[0.1, 0.2, ...], [0.3, 0.4, ...]],  # numpy arrays
    metadatas=[{"doc_id": "123", "chunk_index": 0}, {...}],
    ids=["uuid-1", "uuid-2"]
)
```

2. **Поиск:**
```python
results = collection.query(
    query_embeddings=[[0.5, 0.6, ...]],  # вектор запроса
    n_results=5,                          # TOP-5
    where={"active": True}                # фильтр по метаданным
)
# Возвращает:
# {
#   'documents': [["текст чанка 3", "текст чанка 1", ...]],
#   'metadatas': [[{...}, {...}]],
#   'distances': [[0.12, 0.34, ...]]  # косинусное расстояние (0=идентичны, 2=противоположны)
# }
```

---

### 4. llm_client.py - Генерация ответов

#### Класс LLMClient

**Инициализация:**
```python
def __init__(self, model_name: str = LLM_MODEL_NAME):
    self.model_name = model_name
    self._verify_model()
```

**Проверка модели:**
```python
def _verify_model(self):
    try:
        models = ollama.list()
        # Запрос к Ollama API: GET http://localhost:11434/api/tags
        # Возвращает: {'models': [{'name': 'qwen2.5:14b-instruct-q4_K_M'}, ...]}

        available_models = [m['name'] for m in models.get('models', [])]
        # List comprehension: создаём список названий моделей

        if self.model_name not in available_models:
            raise ValueError(f"Модель {self.model_name} не найдена")
    except Exception as e:
        print(f"⚠️ Предупреждение: {e}")
```

**Метод generate:**
```python
def generate(self, prompt: str, system_prompt: str = None,
             max_tokens: int = 1024, temperature: float = 0.7) -> str:

    messages = []

    # 1. Добавляем системное сообщение (если есть)
    if system_prompt:
        messages.append({'role': 'system', 'content': system_prompt})

    # 2. Добавляем пользовательский запрос
    messages.append({'role': 'user', 'content': prompt})

    # 3. Отправляем в Ollama
    response = ollama.chat(
        model=self.model_name,
        messages=messages,
        options={
            'num_predict': max_tokens,  # макс. длина ответа
            'temperature': temperature   # случайность (0=детерминированный, 1=креативный)
        }
    )

    # 4. Извлекаем текст ответа
    return response['message']['content']
```

**Как работает Ollama:**
1. Запускается локальный сервер на порту 11434
2. Модель загружена в память (занимает ~8GB RAM для qwen2.5:14b-q4)
3. При вызове `ollama.chat()`:
   - Отправляет POST запрос к http://localhost:11434/api/chat
   - Передаёт историю сообщений
   - Модель генерирует ответ автoregressive (токен за токеном)
   - Возвращает полный ответ

**Метод generate_rag_answer:**
```python
def generate_rag_answer(self, query: str, context: str, max_tokens: int = 1024) -> str:

    # 1. Системный промпт (инструкция для модели)
    system_prompt = """Ты — помощник по поиску информации в базе знаний инструкций.

ВАЖНЫЕ ПРАВИЛА:
1. Используй ТОЛЬКО информацию из предоставленного контекста
2. Если в контексте нет ответа — честно скажи "В базе знаний нет информации"
3. Не придумывай информацию
4. Отвечай четко, структурированно
"""

    # 2. Пользовательский промпт (контекст + вопрос)
    prompt = f"""КОНТЕКСТ ИЗ БАЗЫ ЗНАНИЙ:
{context}

---

ВОПРОС ПОЛЬЗОВАТЕЛЯ:
{query}

---

ОТВЕТ (используй только информацию из контекста выше):"""

    # 3. Генерация с низкой температурой (для точности)
    return self.generate(
        prompt=prompt,
        system_prompt=system_prompt,
        max_tokens=max_tokens,
        temperature=0.3  # Низкая = более предсказуемый ответ
    )
```

**Зачем temperature=0.3:**
- При temperature=0: модель всегда выбирает самый вероятный токен (детерминированно)
- При temperature=1: модель более креативна, может галлюцинировать
- 0.3 - баланс: точность + немного вариативности

---

### 5. chunker.py - Разбиение на чанки

```python
def split_text(text: str, max_length=500, overlap=50):
    chunks = []
    start = 0

    while start < len(text):
        end = min(len(text), start + max_length)
        chunk = text[start:end]
        chunks.append(chunk)
        start += max_length - overlap  # Сдвигаем на (max_length - overlap)

    return chunks
```

**Пошаговый пример:**
```python
text = "0123456789ABCDEFGHIJ"  # 20 символов
chunks = split_text(text, max_length=10, overlap=3)

# Итерация 1:
#   start=0, end=10, chunk="0123456789", start_next=0+10-3=7
# Итерация 2:
#   start=7, end=17, chunk="789ABCDEFG", start_next=7+10-3=14
# Итерация 3:
#   start=14, end=20, chunk="EFGHIJ", start_next=14+10-3=21 > len(text) → выход

# Результат:
# ["0123456789", "789ABCDEFG", "EFGHIJ"]
#   ^^^                 ^^^         перекрытие "789"
```

**Зачем overlap (перекрытие):**
- Чтобы не терять контекст на границах чанков
- Если важное предложение оказывается на границе, оно попадёт в оба чанка

**Почему max_length=500 токенов → 2000 символов:**
- 1 токен ≈ 4 символа для русского текста
- 500 токенов × 4 = 2000 символов

---

### 6. docs_parser.py - Парсинг документов

**Чтение .txt и .md:**
```python
def read_txt_md(file_path: str) -> str:
    with open(file_path, "r", encoding="utf-8") as f:
        return f.read()
```
- Простое чтение файла целиком
- `encoding="utf-8"` - для поддержки кириллицы

**Чтение .docx:**
```python
from docx import Document

def read_docx(file_path: str) -> str:
    doc = Document(file_path)
    # Document - объект python-docx, представляет Word документ

    text = "\n".join(p.text for p in doc.paragraphs)
    # List comprehension + generator expression
    # Проходим по всем параграфам и объединяем их через \n

    return text
```

**Разбор конструкции:**
```python
# Эквивалентный код без list comprehension:
paragraphs_texts = []
for p in doc.paragraphs:
    paragraphs_texts.append(p.text)
text = "\n".join(paragraphs_texts)
```

**Что такое doc.paragraphs:**
- Список всех параграфов в Word документе
- Каждый `p` - объект Paragraph с атрибутом `.text`

**Универсальная функция extract_text:**
```python
def extract_text(file_path: str) -> str:
    ext = os.path.splitext(file_path)[1].lower()
    # os.path.splitext("/path/to/file.DOCX") = ("/path/to/file", ".DOCX")
    # [1] берём расширение, .lower() приводим к нижнему регистру

    if ext == ".txt" or ext == ".md":
        return read_txt_md(file_path)
    elif ext == ".docx":
        return read_docx(file_path)
    else:
        raise ValueError(f"Неподдерживаемый формат: {ext}")
```

**С названием файла:**
```python
def extract_text_with_filename(file_path: str) -> Tuple[str, str]:
    text = extract_text(file_path)

    filename = os.path.basename(file_path)
    # os.path.basename("/path/to/Инструкция.docx") = "Инструкция.docx"

    filename_without_ext = os.path.splitext(filename)[0]
    # os.path.splitext("Инструкция.docx") = ("Инструкция", ".docx")
    # [0] берём часть без расширения

    return text, filename_without_ext
```

**Подготовка для чанкинга:**
```python
def prepare_text_for_chunking(text: str, filename_without_ext: str) -> str:
    header = f"Документ: {filename_without_ext}\n\n"
    return header + text
```

**Зачем добавлять заголовок:**
- Чтобы в каждом чанке было название документа
- LLM будет знать, из какого документа информация
- Улучшает контекст при поиске

---

### 7. hybrid_search.py - Гибридный поиск

#### Зачем нужен гибридный поиск:

**Проблема семантического поиска:**
- Если пользователь ищет "ошибка 409", а в документе написано "error 409"
- Векторный поиск может не найти (разные языки)

**Решение - BM25:**
- Статистический алгоритм (TF-IDF подобный)
- Хорош для точного совпадения терминов, чисел, кодов

**Гибридный = Semantic + BM25**

#### Токенизация:
```python
def tokenize_russian(text: str) -> List[str]:
    import re
    # Регулярное выражение \b\w+\b:
    #   \b - граница слова (word boundary)
    #   \w+ - один или более буквенно-цифровых символов
    # Пример: "Ошибка-409!" → ["ошибка", "409"]

    tokens = re.findall(r'\b\w+\b', text.lower())
    return tokens
```

**Пример:**
```python
text = "Загрузка справочников вручную. Версия 1.2"
tokens = tokenize_russian(text)
# ["загрузка", "справочников", "вручную", "версия", "1", "2"]
```

#### Класс HybridSearcher:

**Инициализация:**
```python
def __init__(self, documents: List[str], document_ids: List[str]):
    self.documents = documents
    self.document_ids = document_ids

    # Токенизируем все документы
    tokenized_docs = [tokenize_russian(doc) for doc in documents]
    # [["загрузка", "справочников"], ["ошибка", "409"], ...]

    # Создаём BM25 индекс
    self.bm25 = BM25Okapi(tokenized_docs)
    # BM25Okapi - класс из библиотеки rank-bm25
    # Строит обратный индекс: слово → [док1, док3, ...]
```

**Поиск через BM25:**
```python
def search_bm25(self, query: str, top_k: int = 10) -> List[Dict]:
    # 1. Токенизируем запрос
    tokenized_query = tokenize_russian(query)
    # "ошибка 409" → ["ошибка", "409"]

    # 2. Вычисляем BM25 scores для всех документов
    scores = self.bm25.get_scores(tokenized_query)
    # numpy array: [0.0, 2.3, 0.1, 4.5, ...]

    # 3. Сортируем индексы по убыванию scores
    top_indices = np.argsort(scores)[::-1][:top_k]
    # np.argsort([0.0, 2.3, 0.1, 4.5]) = [0, 2, 1, 3]
    # [::-1] разворачиваем: [3, 1, 2, 0]
    # [:top_k] берём первые top_k

    # 4. Формируем результаты
    results = []
    for idx in top_indices:
        if scores[idx] > 0:  # Только релевантные
            results.append({
                'doc_id': self.document_ids[idx],
                'text': self.documents[idx],
                'bm25_score': float(scores[idx])
            })

    return results
```

**Комбинирование результатов:**
```python
@staticmethod
def combine_scores(semantic_results, bm25_results,
                   semantic_weight=0.5, bm25_weight=0.5):

    # 1. Нормализация scores (приведение к [0, 1])
    def normalize_scores(results, score_key):
        scores = [r[score_key] for r in results]
        min_score, max_score = min(scores), max(scores)

        # Нормализация: (x - min) / (max - min)
        # Пример: [10, 20, 30] → [0.0, 0.5, 1.0]
        for r in results:
            r[f'{score_key}_norm'] = (r[score_key] - min_score) / (max_score - min_score)
        return results

    # 2. Для semantic: преобразуем distance в score
    for r in semantic_results:
        r['semantic_score'] = 1 - r.get('distance', 0)
        # distance=0.2 → score=0.8

    semantic_results = normalize_scores(semantic_results, 'semantic_score')
    bm25_results = normalize_scores(bm25_results, 'bm25_score')

    # 3. Объединяем по doc_id в словарь
    combined = {}

    for r in semantic_results:
        doc_id = r.get('id') or r.get('doc_id')
        combined[doc_id] = {
            'semantic_norm': r.get('semantic_score_norm', 0),
            'bm25_norm': 0,  # пока 0, обновим если есть в bm25_results
            # ... другие поля
        }

    for r in bm25_results:
        doc_id = r['doc_id']
        if doc_id in combined:
            combined[doc_id]['bm25_norm'] = r.get('bm25_score_norm', 0)
        else:
            # Документ только в BM25, не в semantic
            combined[doc_id] = {
                'semantic_norm': 0,
                'bm25_norm': r.get('bm25_score_norm', 0),
                # ...
            }

    # 4. Вычисляем финальный hybrid_score
    for doc_id in combined:
        combined[doc_id]['hybrid_score'] = (
            semantic_weight * combined[doc_id]['semantic_norm'] +
            bm25_weight * combined[doc_id]['bm25_norm']
        )
        # hybrid = 0.5 × 0.8 + 0.5 × 0.6 = 0.7

    # 5. Сортируем по hybrid_score (от большего к меньшему)
    results = sorted(combined.values(), key=lambda x: x['hybrid_score'], reverse=True)

    return results
```

**Зачем нормализация:**
- Semantic distance: от 0 до 2 (косинусное)
- BM25 score: от 0 до ~100 (зависит от документов)
- Без нормализации BM25 "задавит" semantic
- После нормализации: оба от 0 до 1, можно комбинировать

---

### 8. rag_pipeline.py - Основная логика

#### Класс RAGPipeline:

**Инициализация:**
```python
def __init__(self, embedding_model_name: str = EMBEDDING_MODEL_NAME, top_k: int = TOP_K):
    print("Инициализация RAG pipeline...")

    # 1. Загружаем модель эмбеддингов
    self.embedding_model = EmbeddingModel(embedding_model_name)
    # Загружает ~400MB модель из HuggingFace

    # 2. Подключаемся к ChromaDB
    self.client, self.collection = get_chroma()
    # Открывает /chroma_db/chroma.sqlite3

    # 3. Создаём LLM клиент
    self.llm_client = get_llm_client()
    # Проверяет, что Ollama запущен

    self.top_k = top_k
    print("✅ RAG pipeline готов")
```

**Поиск похожих документов:**
```python
def search_similar(self, query: str, top_k: int = None, filter_active: bool = True):
    if top_k is None:
        top_k = self.top_k

    # 1. Создаём эмбеддинг запроса
    query_embedding = self.embedding_model.encode([query])[0].tolist()
    # encode([query]) возвращает numpy array формы (1, 384)
    # [0] берём первый элемент → array формы (384,)
    # .tolist() преобразуем в список Python [0.1, 0.2, ...]

    # 2. Формируем фильтр по метаданным
    where_filter = {"active": True} if filter_active else None
    # Если True, то ищем только среди active=True документов

    # 3. Поиск в ChromaDB
    results = self.collection.query(
        query_embeddings=[query_embedding],  # список векторов (можем искать несколько)
        n_results=top_k,
        where=where_filter if where_filter else {}
    )

    # 4. Форматирование результатов
    documents = []
    if results['documents'] and len(results['documents']) > 0:
        # results['documents'] = [["текст1", "текст2", ...]]  # вложенный список!
        # results['metadatas'] = [[{meta1}, {meta2}, ...]]
        # results['distances'] = [[0.12, 0.34, ...]]

        for i in range(len(results['documents'][0])):
            doc = {
                'text': results['documents'][0][i],
                'metadata': results['metadatas'][0][i] if results['metadatas'] else {},
                'distance': results['distances'][0][i] if results['distances'] else None,
                'id': results['ids'][0][i] if results['ids'] else None
            }
            documents.append(doc)

    return documents
```

**Почему вложенные списки:**
- ChromaDB позволяет искать несколько запросов сразу
- `query_embeddings=[[vec1], [vec2]]` → два запроса
- Ответ: `{'documents': [[результаты_для_vec1], [результаты_для_vec2]]}`
- У нас один запрос → берём `[0]`

**Форматирование контекста:**
```python
def format_context(self, documents: List[Dict]) -> Tuple[str, List[Dict]]:
    if not documents:
        return "Контекст отсутствует.", []

    context_parts = []
    sources = []

    for i, doc in enumerate(documents, 1):
        # enumerate(documents, 1) → (1, doc1), (2, doc2), ...

        text = doc['text']
        metadata = doc.get('metadata', {})

        # Формируем источник
        source_info = {
            'index': i,
            'filename': metadata.get('filename', 'Неизвестный документ'),
            'doc_id': metadata.get('doc_id', ''),
            'distance': doc.get('distance', 0.0)
        }
        sources.append(source_info)

        # Формируем контекст
        context_part = f"""[Документ {i}: {source_info['filename']}]
{text}
"""
        context_parts.append(context_part)

    # Объединяем через разделитель
    context = "\n---\n".join(context_parts)

    return context, sources
```

**Пример контекста:**
```
[Документ 1: Загрузка справочников]
Документ: Загрузка справочников

Для загрузки справочников вручную...

---

[Документ 2: Ошибки ККМ]
Документ: Ошибки ККМ

Ошибка 409: неверный код маркировки...
```

**Основной метод query:**
```python
def query(self, user_query: str, top_k: int = None) -> Dict:
    print(f"\n🔍 Поиск по запросу: {user_query}")

    # 1. Поиск похожих документов
    documents = self.search_similar(user_query, top_k=top_k)

    if not documents:
        return {
            'answer': "В базе знаний не найдено релевантной информации",
            'context': "",
            'sources': [],
            'documents': []
        }

    print(f"✅ Найдено документов: {len(documents)}")

    # 2. Форматирование контекста
    context, sources = self.format_context(documents)

    # 3. Генерация ответа с помощью LLM
    print("🤖 Генерация ответа...")
    answer = self.llm_client.generate_rag_answer(
        query=user_query,
        context=context
    )

    print("✅ Ответ готов")

    return {
        'answer': answer,
        'context': context,
        'sources': sources,
        'documents': documents
    }
```

---

### 9. app.py - Streamlit интерфейс

**Кеширование RAG pipeline:**
```python
@st.cache_resource
def get_rag_pipeline():
    return create_rag_pipeline()
```

**Что делает @st.cache_resource:**
- При первом вызове функции выполняет её и сохраняет результат
- При повторных вызовах возвращает сохранённый результат
- Не пересоздаёт модели при каждом обновлении страницы
- Важно для тяжёлых ресурсов (модели нейронных сетей)

**Основная функция:**
```python
def main():
    st.title("📚 RAG Поиск по базе инструкций")

    # Инициализация RAG pipeline (один раз благодаря кешу)
    rag = get_rag_pipeline()

    # Боковая панель
    with st.sidebar:
        stats = rag.get_stats()
        st.metric("Всего чанков", stats['total_chunks'])
        top_k = st.slider("Количество результатов", 1, 10, 3)

    # Вкладки
    tab1, tab2, tab3 = st.tabs(["🔍 Поиск", "📄 Загрузка", "📊 База знаний"])

    with tab1:
        query = st.text_input("Введите вопрос:")

        if st.button("🔍 Найти"):
            if not query:
                st.warning("⚠️ Введите вопрос")
            else:
                with st.spinner("Поиск..."):
                    result = rag.query(query, top_k=top_k)

                    # Отображение ответа
                    st.success(result['answer'])

                    # Отображение источников
                    for source in result['sources']:
                        with st.expander(f"📄 {source['filename']}"):
                            st.text(result['documents'][source['index'] - 1]['text'])
```

**Загрузка документа:**
```python
with tab2:
    uploaded_file = st.file_uploader("Выберите файл", type=['docx', 'md', 'txt'])

    if st.button("📤 Загрузить"):
        # 1. Сохраняем файл
        file_path = f"data/docs/{uploaded_file.name}"
        with open(file_path, "wb") as f:
            f.write(uploaded_file.getbuffer())

        # 2. Извлекаем текст
        text, filename_without_ext = extract_text_with_filename(file_path)

        # 3. Подготовка с заголовком
        text_with_header = prepare_text_for_chunking(text, filename_without_ext)

        # 4. Разбиение на чанки
        chunks = split_text(text_with_header, max_length=2000, overlap=200)

        # 5. Создание эмбеддингов
        embedding_model = EmbeddingModel(EMBEDDING_MODEL_NAME)
        embeddings = embedding_model.encode(chunks)

        # 6. Подготовка метаданных
        doc_id = str(uuid.uuid4())  # уникальный ID

        for i, chunk in enumerate(chunks):
            chunk_id = f"{doc_id}_chunk_{i}"
            metadata = {
                'doc_id': doc_id,
                'filename': uploaded_file.name,
                'chunk_index': i,
                'total_chunks': len(chunks),
                'active': True,
                # ...
            }

        # 7. Добавление в ChromaDB
        collection.add(
            documents=chunks,
            embeddings=embeddings.tolist(),
            metadatas=metadatas,
            ids=chunk_ids
        )

        st.success("🎉 Документ добавлен!")
        st.cache_resource.clear()  # Очистка кеша
```

---

## 🧩 Сложные конструкции

### 1. List Comprehension

**Базовый синтаксис:**
```python
result = [выражение for элемент in итерируемое if условие]
```

**Примеры из кода:**

```python
# 1. Извлечение названий моделей
available_models = [m['name'] for m in models.get('models', [])]

# Эквивалент:
available_models = []
for m in models.get('models', []):
    available_models.append(m['name'])
```

```python
# 2. Токенизация всех документов
tokenized_docs = [tokenize_russian(doc) for doc in documents]

# Эквивалент:
tokenized_docs = []
for doc in documents:
    tokenized_docs.append(tokenize_russian(doc))
```

```python
# 3. С условием
results = [r for r in all_results if r['score'] > 0.5]

# Эквивалент:
results = []
for r in all_results:
    if r['score'] > 0.5:
        results.append(r)
```

### 2. Generator Expression

```python
text = "\n".join(p.text for p in doc.paragraphs)
```

**Отличие от list comprehension:**
- `[p.text for p in doc.paragraphs]` - создаёт список в памяти
- `(p.text for p in doc.paragraphs)` - генератор (вычисляет по требованию)
- `"\n".join()` принимает итерируемое, поэтому можно использовать генератор

### 3. Conditional Expression (тернарный оператор)

```python
where_filter = {"active": True} if filter_active else None

# Эквивалент:
if filter_active:
    where_filter = {"active": True}
else:
    where_filter = None
```

```python
metadata = doc.get('metadata', {})
# Если ключ 'metadata' есть - вернёт его значение
# Если нет - вернёт {} (пустой словарь)
```

### 4. Slicing и шаг

```python
top_indices = np.argsort(scores)[::-1][:top_k]
```

**Разбор:**
```python
arr = [1, 2, 3, 4, 5]

arr[::-1]     # Разворот: [5, 4, 3, 2, 1]
arr[::2]      # Каждый второй: [1, 3, 5]
arr[1:4]      # С 1 по 3: [2, 3, 4]
arr[:3]       # Первые 3: [1, 2, 3]
```

### 5. Enumerate

```python
for i, doc in enumerate(documents, 1):
    print(f"Документ {i}")
```

**Что происходит:**
```python
documents = ['doc1', 'doc2', 'doc3']

# enumerate(documents) → (0, 'doc1'), (1, 'doc2'), (2, 'doc3')
# enumerate(documents, 1) → (1, 'doc1'), (2, 'doc2'), (3, 'doc3')

for i, doc in enumerate(documents, 1):
    # i=1, doc='doc1'
    # i=2, doc='doc2'
    # i=3, doc='doc3'
```

### 6. Lambda функции

```python
results = sorted(combined.values(), key=lambda x: x['hybrid_score'], reverse=True)
```

**Что такое lambda:**
```python
# Lambda - анонимная функция (функция без имени)
lambda x: x['hybrid_score']

# Эквивалент:
def get_score(x):
    return x['hybrid_score']

# Использование:
sorted(combined.values(), key=get_score, reverse=True)
```

**Зачем lambda:**
- Короткая запись для простых функций
- Не нужно давать имя

### 7. Контекстный менеджер (with)

```python
with open(file_path, "r", encoding="utf-8") as f:
    return f.read()
```

**Что происходит:**
```python
# Эквивалент:
f = open(file_path, "r", encoding="utf-8")
try:
    result = f.read()
    return result
finally:
    f.close()  # Гарантированно закроется, даже если ошибка
```

**Преимущества with:**
- Автоматически закрывает файл
- Даже если произошла ошибка
- Короче и понятнее

### 8. F-strings (форматирование строк)

```python
header = f"Документ: {filename_without_ext}\n\n"
```

**Различные способы:**
```python
name = "Иван"
age = 25

# 1. F-string (современный, Python 3.6+)
text = f"Меня зовут {name}, мне {age} лет"

# 2. format() (старый способ)
text = "Меня зовут {}, мне {} лет".format(name, age)

# 3. % оператор (очень старый)
text = "Меня зовут %s, мне %d лет" % (name, age)
```

**Выражения в f-string:**
```python
f"Результат: {2 + 2}"  # "Результат: 4"
f"Релевантность: {1 - distance:.2%}"  # "Релевантность: 85.30%"
```

### 9. Распаковка аргументов

```python
client, collection = get_chroma()
```

**Что происходит:**
```python
# get_chroma() возвращает tuple: (client_obj, collection_obj)
result = get_chroma()
# result = (client_obj, collection_obj)

# Распаковка:
client, collection = result
# client = result[0]
# collection = result[1]
```

**Другие примеры:**
```python
# Множественная распаковка
a, b, c = [1, 2, 3]

# Игнорирование значений
first, _, third = [1, 2, 3]  # _ используется для неиспользуемых значений

# Распаковка с остатком
first, *rest = [1, 2, 3, 4]
# first = 1, rest = [2, 3, 4]
```

### 10. Dictionary .get() с default

```python
metadata.get('filename', 'Неизвестный документ')
```

**Разница:**
```python
metadata = {'doc_id': '123'}

# 1. Прямое обращение - ошибка если нет ключа
filename = metadata['filename']  # KeyError!

# 2. get() - возвращает None если нет ключа
filename = metadata.get('filename')  # None

# 3. get() с default - возвращает default если нет ключа
filename = metadata.get('filename', 'Неизвестный')  # 'Неизвестный'
```

---

## 📊 Примеры работы

### Пример 1: Добавление документа

**Входные данные:**
```
Файл: "Загрузка справочников.docx"
Содержимое:
"Для загрузки справочников вручную зайдите на ftp://...
Скопируйте файлы на флешку.
На магазине положите в корень диска C:."
```

**Процесс:**

1. **Парсинг** (docs_parser.py):
```python
text, filename = extract_text_with_filename("Загрузка справочников.docx")
# text = "Для загрузки справочников вручную..."
# filename = "Загрузка справочников"
```

2. **Добавление заголовка**:
```python
text_with_header = prepare_text_for_chunking(text, filename)
# "Документ: Загрузка справочников\n\nДля загрузки справочников..."
```

3. **Разбиение на чанки** (chunker.py):
```python
chunks = split_text(text_with_header, max_length=2000, overlap=200)
# [
#   "Документ: Загрузка справочников\n\nДля загрузки...",
#   "...скопируйте файлы на флешку...",
#   "...положите в корень диска C:."
# ]
```

4. **Создание эмбеддингов** (embeddings.py):
```python
embeddings = embedding_model.encode(chunks)
# numpy array формы (3, 384)
# [
#   [0.123, -0.456, 0.789, ...],  # 384 числа для чанка 1
#   [0.234, -0.567, 0.890, ...],  # 384 числа для чанка 2
#   [0.345, -0.678, 0.901, ...]   # 384 числа для чанка 3
# ]
```

5. **Сохранение в ChromaDB** (storage.py):
```python
collection.add(
    documents=chunks,
    embeddings=embeddings.tolist(),
    metadatas=[
        {'doc_id': 'uuid-123', 'chunk_index': 0, 'filename': 'Загрузка справочников.docx'},
        {'doc_id': 'uuid-123', 'chunk_index': 1, 'filename': 'Загрузка справочников.docx'},
        {'doc_id': 'uuid-123', 'chunk_index': 2, 'filename': 'Загрузка справочников.docx'}
    ],
    ids=['uuid-123_chunk_0', 'uuid-123_chunk_1', 'uuid-123_chunk_2']
)
```

### Пример 2: Поиск

**Запрос пользователя:**
```
"Как загрузить справочники вручную?"
```

**Процесс:**

1. **Создание эмбеддинга запроса**:
```python
query_embedding = embedding_model.encode(["Как загрузить справочники вручную?"])
# array формы (1, 384) → берём [0] → (384,)
# [0.789, -0.123, 0.456, ...]
```

2. **Поиск в ChromaDB**:
```python
results = collection.query(
    query_embeddings=[[0.789, -0.123, ...]],
    n_results=3
)

# ChromaDB вычисляет косинусное расстояние между query_embedding и всеми векторами в БД
# Формула: distance = 1 - (A · B) / (||A|| × ||B||)
# Где A · B - скалярное произведение векторов

# Результат:
# {
#   'documents': [["Документ: Загрузка справочников\n\nДля загрузки...", ...]],
#   'metadatas': [[{'filename': 'Загрузка справочников.docx', ...}]],
#   'distances': [[0.12, 0.34, 0.56]]  # чем меньше, тем похожее
# }
```

3. **Форматирование контекста**:
```python
context = """[Документ 1: Загрузка справочников.docx]
Документ: Загрузка справочников

Для загрузки справочников вручную зайдите на ftp://...

---

[Документ 2: ...]
..."""
```

4. **Генерация ответа через LLM**:
```python
prompt = f"""КОНТЕКСТ:
{context}

ВОПРОС: Как загрузить справочники вручную?

ОТВЕТ:"""

answer = llm_client.generate(prompt, system_prompt="Используй только контекст...")
# "Для загрузки справочников вручную необходимо:
#  1. Зайти на ftp://sps-holding.ru/kb_y/SPRmanually/
#  2. Скопировать файлы на флешку
#  3. На магазине положить в корень диска C:"
```

5. **Возврат результата**:
```python
return {
    'answer': "Для загрузки справочников вручную необходимо...",
    'sources': [
        {'filename': 'Загрузка справочников.docx', 'distance': 0.12}
    ],
    'documents': [...]
}
```

---

## 🎯 Ключевые моменты для понимания

### 1. Почему векторный поиск работает:
- Тексты с похожим смыслом имеют близкие векторы
- Модель обучена на миллионах примеров
- Понимает синонимы, парафразы, контекст

### 2. Почему нужен чанкинг:
- Модель эмбеддингов имеет ограничение на длину текста (512 токенов)
- Маленькие чанки = точнее поиск (меньше шума)
- Большие чанки = больше контекста

### 3. Почему overlap важен:
- Предложение на границе чанка может быть разрезано
- С overlap оно попадёт в оба соседних чанка

### 4. Почему системный промпт критичен:
- Без него LLM может "галлюцинировать" (придумывать факты)
- Промпт инструктирует использовать только контекст
- Temperature=0.3 снижает креативность, повышает точность

### 5. Почему ChromaDB, а не обычная БД:
- Обычная БД: поиск по точному совпадению (SQL LIKE)
- ChromaDB: поиск по семантическому сходству
- Использует специализированные индексы (HNSW) для быстрого ANN

---

## 📝 Итоговая схема

```
ПОЛЬЗОВАТЕЛЬ
    ↓ (вводит запрос)
STREAMLIT UI (app.py)
    ↓ (вызывает)
RAG PIPELINE (rag_pipeline.py)
    ├─→ EMBEDDINGS (embeddings.py)
    │   └─→ query → vector [0.1, 0.2, ...]
    │
    ├─→ CHROMA DB (storage.py)
    │   └─→ vector → TOP-K документов
    │
    └─→ LLM CLIENT (llm_client.py)
        └─→ контекст + запрос → ОТВЕТ
            ↓
        OLLAMA (qwen2.5:14b)
            ↓
ПОЛЬЗОВАТЕЛЬ
    ↓ (видит ответ + источники)
```

---

Это полное объяснение архитектуры и кода! Если нужны пояснения по конкретным частям - спрашивай!
