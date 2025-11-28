# Быстрая шпаргалка

## Команды

### Инициализация (первый раз)
```bash
python scripts/init_metadata_db.py
```

### Запуск приложения
```bash
streamlit run src/app.py --server.port 8502
```

### Проверка БД
```bash
sqlite3 data/metadata.db
```

## Формат файлов

### Один файл = одна инструкция
```markdown
Название файла: "Настройка_ЕГАИС.md"
→ Название инструкции: "Настройка_ЕГАИС"

Содержимое — это весь текст инструкции.
```

### Множество инструкций (разделитель `---`)
```markdown
Настройка фильтра ЕГАИС

Текст инструкции 1...

---

Решение ошибки подключения

Текст инструкции 2...

---

Создание отчета

Текст инструкции 3...
```

## SQL запросы

### Список всех инструкций
```sql
SELECT title, source_type, active FROM instructions;
```

### Инструкции по тегу
```sql
SELECT i.title, t.name
FROM instructions i
JOIN instruction_tags it ON i.id = it.instruction_id
JOIN tags t ON it.tag_id = t.id
WHERE t.name = 'ЕГАИС';
```

### Все теги
```sql
SELECT * FROM tags ORDER BY name;
```

### Статистика
```sql
SELECT
  COUNT(CASE WHEN active = 1 THEN 1 END) as active,
  COUNT(CASE WHEN active = 0 THEN 1 END) as inactive
FROM instructions;
```

### Удалить все инструкции
```sql
DELETE FROM instructions;
```

## Python API

### Добавить инструкцию
```python
from src.docs_parser import parse_document
from src.metadata_manager import MetadataManager

instructions = parse_document(
    'data/docs/file.md',
    source_type='multi_instruction',
    tags=['ЕГАИС', 'фильтр'],
    author='Admin'
)

mm = MetadataManager()
for inst in instructions:
    mm.add_instruction(**inst)
```

### Получить по тегу
```python
from src.metadata_manager import MetadataManager

mm = MetadataManager()
egais_inst = mm.get_instructions_by_tag('ЕГАИС')

for inst in egais_inst:
    print(inst['title'])
```

### Статистика
```python
from src.metadata_manager import MetadataManager

mm = MetadataManager()
print(mm.get_stats())
```

## Структура метаданных

### В ChromaDB (каждый чанк)
```python
{
    'instruction_id': 'uuid',
    'doc_id': 'uuid',
    'title': 'Название',
    'tags': 'ЕГАИС,фильтр',
    'active': True,
    'author': 'Admin',
    ...
}
```

### В SQLite (инструкция)
```sql
instructions:
  id, doc_id, title, file_path, file_format,
  source_type, separator_index, active, version,
  author, created_at, updated_at

tags:
  id, name, category

instruction_tags:
  instruction_id, tag_id
```

## Типичные задачи

### Добавить новый тег
```python
from src.metadata_manager import MetadataManager
mm = MetadataManager()
mm.add_tag('новый_тег', category='функция')
```

### Пометить неактуальной
```python
mm.mark_instruction_inactive('instruction_id')
```

### Удалить инструкцию
```python
mm.delete_instruction('instruction_id')
```

### Получить все инструкции
```python
all_inst = mm.get_all_instructions(active_only=True)
```

## Troubleshooting

| Проблема | Решение |
|----------|---------|
| `no such table: instructions` | `python scripts/init_metadata_db.py` |
| Теги не видны | Проверить: `sqlite3 data/metadata.db "SELECT * FROM tags;"` |
| ModuleNotFoundError | `export PYTHONPATH=$(pwd):$PYTHONPATH` |
| Инструкции не разделились | Проверить разделитель `---` и тип документа |

## Полезные ссылки

- [USAGE_GUIDE.md](USAGE_GUIDE.md) — подробное руководство
- [TESTING_GUIDE.md](TESTING_GUIDE.md) — инструкция по тестированию
- [CHANGELOG.md](CHANGELOG.md) — история изменений
