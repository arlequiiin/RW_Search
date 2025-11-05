#!/usr/bin/env python3
"""
Streamlit Web UI для RAG-системы поиска по инструкциям
"""
import streamlit as st
import os
import sys
from datetime import datetime

# Добавляем корневую директорию в PATH
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.rag_pipeline import create_rag_pipeline
from src.docs_parser import extract_text_with_filename, prepare_text_for_chunking
from src.chunker import split_text
from src.embeddings import EmbeddingModel
from src.storage import get_chroma
from src.config import EMBEDDING_MODEL_NAME, CHUNK_SIZE_TOKENS, CHUNK_OVERLAP_TOKENS
import uuid


# Конфигурация страницы
st.set_page_config(
    page_title="RAG Поиск по инструкциям",
    page_icon="📚",
    layout="wide"
)


@st.cache_resource
def get_rag_pipeline():
    """Кешированная инициализация RAG pipeline"""
    return create_rag_pipeline()


def main():
    """Главная функция приложения"""
    
    st.title("📚 RAG Поиск по базе инструкций")
    st.markdown("---")

    # Инициализация RAG pipeline
    try:
        rag = get_rag_pipeline()
    except Exception as e:
        st.error(f"❌ Ошибка инициализации системы: {e}")
        return

    # Боковая панель со статистикой
    with st.sidebar:
        st.header("ℹ️ Информация")
        
        stats = rag.get_stats()
        st.metric("Всего чанков в базе", stats['total_chunks'])
        st.metric("Коллекция", stats['collection_name'])
        
        st.markdown("---")
        st.markdown("### ⚙️ Настройки поиска")
        top_k = st.slider("Количество результатов", 1, 10, 3)

    # Основные вкладки
    tab1, tab2, tab3 = st.tabs(["🔍 Поиск", "📄 Загрузка документов", "📊 База знаний"])

    # === ВКЛАДКА 1: ПОИСК ===
    with tab1:
        st.header("Поиск по базе знаний")
        
        query = st.text_input(
            "Введите ваш вопрос:",
            placeholder="Например: Как решить ошибку фильтра?"
        )

        if st.button("🔍 Найти", type="primary"):
            if not query:
                st.warning("⚠️ Введите вопрос")
            else:
                with st.spinner("Поиск и генерация ответа..."):
                    try:
                        result = rag.query(query, top_k=top_k)
                        
                        # Отображение ответа
                        st.markdown("### 💬 Ответ:")
                        st.success(result['answer'])
                        
                        # Отображение источников
                        if result['sources']:
                            st.markdown("### 📚 Источники:")
                            for source in result['sources']:
                                with st.expander(
                                    f"📄 {source['filename']} (релевантность: {1 - source['distance']:.2%})"
                                ):
                                    doc = result['documents'][source['index'] - 1]
                                    st.text(doc['text'])
                                    
                                    # Метаданные
                                    metadata = doc.get('metadata', {})
                                    st.caption(f"Doc ID: {metadata.get('doc_id', 'N/A')}")
                                    st.caption(f"Чанк: {metadata.get('chunk_index', 0) + 1}/{metadata.get('total_chunks', 1)}")
                        else:
                            st.info("Источники не найдены")
                    
                    except Exception as e:
                        st.error(f"❌ Ошибка при поиске: {e}")

    # === ВКЛАДКА 2: ЗАГРУЗКА ДОКУМЕНТОВ ===
    with tab2:
        st.header("Загрузка новых документов")
        
        uploaded_file = st.file_uploader(
            "Выберите файл (.docx, .md, .txt)",
            type=['docx', 'md', 'txt']
        )

        col1, col2 = st.columns(2)
        with col1:
            author = st.text_input("Автор документа", value="Admin")
        with col2:
            tags = st.text_input("Теги (через запятую)", value="")

        if st.button("📤 Загрузить документ", type="primary"):
            if uploaded_file is None:
                st.warning("⚠️ Выберите файл для загрузки")
            else:
                try:
                    # Сохранение файла
                    file_path = f"data/docs/{uploaded_file.name}"
                    with open(file_path, "wb") as f:
                        f.write(uploaded_file.getbuffer())

                    with st.spinner("Обработка документа..."):
                        # Извлечение текста с названием файла
                        text, filename_without_ext = extract_text_with_filename(file_path)
                        st.info(f"✅ Извлечено {len(text)} символов из документа: {filename_without_ext}")

                        # Подготовка текста с названием документа
                        text_with_header = prepare_text_for_chunking(text, filename_without_ext)

                        # Разбиение на чанки
                        chunks = split_text(
                            text_with_header,
                            max_length=CHUNK_SIZE_TOKENS * 4,
                            overlap=CHUNK_OVERLAP_TOKENS * 4
                        )
                        st.info(f"✅ Создано {len(chunks)} чанков")
                        
                        # Создание эмбеддингов
                        embedding_model = EmbeddingModel(EMBEDDING_MODEL_NAME)
                        embeddings = embedding_model.encode(chunks)
                        st.info(f"✅ Создано {len(embeddings)} эмбеддингов")
                        
                        # Подготовка метаданных
                        doc_id = str(uuid.uuid4())
                        created_at = datetime.now().isoformat()
                        
                        # Добавление в ChromaDB
                        client, collection = get_chroma()
                        
                        chunk_ids = []
                        metadatas = []
                        
                        for i, chunk in enumerate(chunks):
                            chunk_id = f"{doc_id}_chunk_{i}"
                            chunk_ids.append(chunk_id)
                            
                            metadata = {
                                'doc_id': doc_id,
                                'filename': uploaded_file.name,
                                'file_path': file_path,
                                'chunk_index': i,
                                'total_chunks': len(chunks),
                                'active': True,
                                'author': author,
                                'tags': tags,
                                'created_at': created_at
                            }
                            metadatas.append(metadata)
                        
                        collection.add(
                            documents=chunks,
                            embeddings=embeddings.tolist(),
                            metadatas=metadatas,
                            ids=chunk_ids
                        )
                        
                        st.success(f"🎉 Документ успешно добавлен! Doc ID: {doc_id}")
                        st.balloons()
                        
                        # Очистка кеша
                        st.cache_resource.clear()
                        
                except Exception as e:
                    st.error(f"❌ Ошибка при загрузке: {e}")

    # === ВКЛАДКА 3: БАЗА ЗНАНИЙ ===
    with tab3:
        st.header("Управление базой знаний")
        
        st.info("🚧 Функционал в разработке")
        st.markdown("""
        Планируемые функции:
        - Просмотр всех документов
        - Удаление документов
        - Пометка документов как неактуальных
        - Редактирование метаданных
        - Экспорт базы знаний
        """)
        
        if st.button("🗑️ Очистить всю базу (ОПАСНО)", type="secondary"):
            st.warning("⚠️ Эта функция пока не реализована")


if __name__ == "__main__":
    main()
