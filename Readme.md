# 1. Краткое описание задачи (цель)

Собрать локальную RAG-систему (Retrieval-Augmented Generation) для базы инструкций (.docx, .md, .txt) с Web-UI на Streamlit, позволяющую:

- искать по смыслу (RAG Basic),
- показывать связанные изображения,
- добавлять/редактировать/помечать инструкции неактуальными,
- работать полностью локально (эмбеддинги + векторная БД + LLM оффлайн).

---

# 2. Общая архитектура (высокоуровнево)

1. **Frontend** — Streamlit (локальный веб-интерфейс).
    
2. **Backend / App logic** — Python-скрипты (RAG pipeline): загрузка документов, парсинг, разделение, эмбеддинги, индексирование, запросы к LLM.
    
3. **Векторная база (локальная)** — Chroma (persisted directory) — хранит векторы + метаданные.
    
4. **Эмбеддинги (локальные)** — SentenceTransformers (`all-MiniLM-L6-v2` рекомендуем; опция: `multi-qa-MiniLM-L6-cos-v1`).
    
5. **LLM (локальная генерация)** — lama3:8b
    
6. **Хранилище исходных файлов и изображений** — локальная папка `data/docs` и `data/images`.
    
7. **Метаданные** — JSON/SQLite файл для списка документов и их статусов/версий (можно использовать SQLite для удобства).
    

Диаграмма (упрощённо):

![[Pasted image 20251103181809.png]]

---

# 3. Выбранный стек (конкретно)

- Язык: Python 3.10+
    
- UI: Streamlit
    
- Векторная БД: Chroma (локальный, persistent)
    
- Эмбеддинги: SentenceTransformers `all-MiniLM-L6-v2`
    
- Оболочка для RAG/взаимодействия: LangChain (необязательно, но сильно упрощает интеграцию)
    
- LLM (генерация): локальная модель через Ollama llama3:8b
    
- Парсинг docx: `python-docx`
    
- Markdown: `markdown` пакет / direct read
    
- Форматирование вывода: Markdown rendering в Streamlit (`st.markdown`)
    
- Виртуальная среда: `venv`
    
- Контроль версий/хранение: Git (VS Code).
    
---

# 4. Структура проекта (файлы и папки)

![[Pasted image 20251103181615.png]]

---

# 5. Метаданные и схема хранения

Для каждого **original document** сохранять:

![[Pasted image 20251103181829.png]]

---

# 6. Детали пайплайна (шаг за шагом)

## 6.1 Настройка окружения (Windows, VS Code)

1. Установить Python 3.14
    
2. Создать venv:
```python
python -m venv .venv 
.venv\Scripts\activate 
pip install --upgrade pip
```
3. Установить зависимости (requirements.txt будет содержать):
    
    - streamlit
    - langchain
    - chromadb
    - sentence-transformers
    - python-docx
    - pdfminer.six (если будут PDF в будущем)
    - ollama-python (если используешь Ollama) _или_ ctransformers bindings
    - transformers (для ряда функций, опционально)
    - tqdm
    - pydantic
    - sqlalchemy (если выберешь SQLite)
    - uvicorn, fastapi (если позже захочешь API)

```python     
pip install -r requirements.txt
```

## 6.2 Парсинг документов

- `docs_parser.py`:
    
    - Для `.docx`: использовать `python-docx` чтобы извлечь текст и встроенные изображения (save to `data/images/{doc_id}_{img_idx}.png`).
        
    - Для `.md`: прочитать как текст, искать синтаксис `![](...)` либо относительные пути для картинок, копировать их в `data/images`.
        
    - Для `.txt`: просто читать.
        
    - Делить файл по `---` (если есть) в первую очередь: каждая часть считается отдельной “инструкцией” и получает собственный `doc_id` / version.
        
    - Нормализация: trim whitespace, убрать лишние спец-символы.
        

## 6.3 Разбиение на чанки (chunking)

- Подход: разделять по семантическим границам (параграф/письменные разделы), затем обрезать до размера:
    
    - `chunk_size_tokens = 500` (пример)
        
    - `chunk_overlap_tokens = 50`
        
- Если используешь LangChain — `RecursiveCharacterTextSplitter` с max_chars соответствующим примерно 500 токенам (1 токен ≈ 4 chars для EN; для русск. — ориентируйся вручную) — можно ставить `max_chars=2000` и overlap `200`.
    
- Для каждой части/чанка формировать:
    
    - `chunk_text`
        
    - `chunk_id`
        
    - `metadata` (doc_id, chunk_index, images)
        

## 6.4 Создание эмбеддингов и индекс

- Инициализация эмбеддингов:

```python
from sentence_transformers import SentenceTransformer 
model = SentenceTransformer("all-MiniLM-L6-v2") 
embeddings = [model.encode(text) for text in chunks]
```

- Добавление в Chroma:
    
    - Использовать `chromadb.Client()` или LangChain wrapper:
```python 
from chromadb import Client
client = Client() collection = client.create_collection("instructions", persist_directory="chroma_db")
collection.add(documents=chunks_text, metadatas=chunks_meta, ids=chunks_ids, embeddings=embeddings)
client.persist()
```
- В metadata указывать `doc_id`, `filename`, `active`.


## 6.5 Поиск и ранжирование при запросе

- При user query:
    
    1. Преобразовать query к эмбеддингу.
        
    2. `collection.query(query_embeddings=..., n_results=top_k, where={"active": True})` (фильтр на active).
        
    3. Получить top_k чанк(ов) (рекомендуется 3–5).
        
    4. Собрать контекст: склеить retrieved chunks в аккуратном порядке (по релевантности, возможно по doc/version).
        
    5. Подготовить промпт для LLM, включив:
        
        - системную инструкцию: «Используй только информацию из контекста; если нет ответа — честно скажи “не знаю”»
            
        - контекст (подписи к картинкам: `[[image: path/to.png]]`)
            
        - сам вопрос пользователя
            
    6. Отправить prompt в LLM client (`llm_client.py`), получить ответ.
        
    7. Вернуть ответ + ссылки на источники + опционально кнопки “Показать картинку”.
        

## 6.6 Добавление новой инструкции (UI flow)

- Streamlit: file_uploader -> сохранение -> docs_parser -> chunking -> embeddings -> collection.add -> metadata update
    
- Ответы/статусы показывать в UI.
    

## 6.7 Пометка как неактуальная (UI flow)

- Streamlit: selectbox с active/inactive фильтром -> при пометке `active=false` обновляем метаданные в Chroma (или в metadata DB) — при поиске фильтруем.
    

## 6.8 Редактирование инструкции (UI flow)

- Streamlit: select document -> show text area (pre-filled) -> edit -> save -> backend:
    
    - при сохранении:
        
        - пометить старую версию `active=false` и сохранить копию в `backups/`
            
        - сохранить новый файл в `data/docs/` (новый `doc_id` или increment version)
            
        - прогнать через ingest pipeline (парсинг -> chunk -> embed -> add)
            
    - Это даёт версионирование и безопасное восстановление.
        

---

# 7. UI детали (Streamlit) — компоненты и страницы

1. **Главная страница — Поиск**
    
    - `st.text_input` (вопрос)
        
    - `st.button("Найти")`
        
    - Результат: `st.markdown` с отформатированным ответом, ниже — блок «Источники» с кнопками `st.button` для каждого источника (показать/скрыть) и `st.image()` для картинок.
        
2. **Управление базой (вкладка)**
    
    - **Добавить инструкцию**: `st.file_uploader` + метаданные (author, tags) + загрузка.
        
    - **Редактировать инструкцию**: `st.selectbox` → `st.text_area` → `st.button("Сохранить")
        
    - **Пометить неактуальной**: `st.selectbox` → `st.button("Пометить неактуальной")`.
        
    - **Журнал изменений (опционально)**: отображение `metadata.db` записей.
        
3. **Инструменты администратора**
    
    - Кнопка `Rebuild index` (перестроить весь индекс) — использовать осторожно, может быть долго.
        
    - Очистить Chroma (danger!).
        

---

# 8. Конфигурации и значения по умолчанию (важно)

- `CHUNK_SIZE_TOKENS = 500`
    
- `CHUNK_OVERLAP_TOKENS = 50`
    
- `EMBEDDING_MODEL = "all-MiniLM-L6-v2"`
    
- `TOP_K = 3` (количество возвращаемых чанков)
    
- `CHROMA_PERSIST_DIR = "./chroma_db"`
    
- `DATA_DIR = "./data"`
    
- `IMAGES_DIR = "./data/images"`
    
- `METADATA_DB = "./data/metadata.db"` (SQLite)
    

---

# 9. Логирование и мониторинг

- Логи для действий: добавление файла, удаление, пометка, перестроение индекса, ошибки LLM.
- Хранить простые логи в `logs/app.log`.

---

# 10. Тесты и валидация

- Unit tests:
    - `docs_parser` для разных форматов.
    - `chunking` на edge-cases (очень короткие/очень длинные тексты).
    - `storage` операций: add/remove/update.
- Интеграционные тесты:
    - ingest → query → проверить, что релевантный chunk возвращается.

---

# 11. Производительность и оптимизации (практические подсказки)

- Для 20–1000 документов Chroma + MiniLM работают очень быстро на CPU.
- Если заметишь деградацию поиска —:
    - увеличить `TOP_K`, или
    - сменить на `all-mpnet-base-v2` (лучше качество, выше время/память).
- Для LLM inference на GTX1660 (6GB): использовать quantized или Q4 модели (формат GGUF с 4-bit) — это снизит потребление VRAM.
- После апгрейда до RTX5060Ti (16GB) — можно запускать более тяжёлые модели/лучшее качество.