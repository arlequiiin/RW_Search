import streamlit as st
import os
import sys
import re
from datetime import datetime

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.rag_pipeline import create_rag_pipeline
from src.docs_parser import parse_document, prepare_text_for_chunking
from src.chunker import split_text
from src.embeddings import EmbeddingModel
from src.storage import get_chroma
from src.metadata_manager import MetadataManager
from src.config import EMBEDDING_MODEL_NAME, CHUNK_SIZE_TOKENS, CHUNK_OVERLAP_TOKENS


def render_answer_with_images(answer_text: str, available_images: list):
    """
    Парсинг ответа LLM и отображение изображений в нужных местах

    Args:
        answer_text: Текст ответа от LLM
        available_images: Список доступных путей к изображениям
    """
    # Паттерн для поиска плейсхолдеров [[image: путь]]
    image_pattern = r'\[\[image:\s*([^\]]+)\]\]'

    # Разбиваем текст на части по плейсхолдерам
    parts = re.split(image_pattern, answer_text)

    for i, part in enumerate(parts):
        if i % 2 == 0:
            # Это текстовая часть
            if part.strip():
                st.markdown(part.strip())
        else:
            # Это путь к изображению
            image_path = part.strip()
            # Проверяем, есть ли это изображение в доступных
            if image_path in available_images or any(img.endswith(image_path) for img in available_images):
                # Находим полный путь
                full_path = None
                for img in available_images:
                    if img == image_path or img.endswith(image_path):
                        full_path = img
                        break

                if full_path:
                    try:
                        st.image(full_path, use_column_width=True)
                    except Exception as e:
                        st.caption(f"⚠️ Не удалось загрузить изображение: {image_path}")
            else:
                # Изображение не найдено в доступных - показываем плейсхолдер
                st.caption(f"🖼️ Изображение: {image_path}")


def display_answer_with_inline_images(answer: str, images: list, instruction_title: str = None):
    """
    Отображение ответа с встроенными изображениями

    Args:
        answer: Текст ответа от LLM
        images: Список путей к изображениям
        instruction_title: Название топ-1 инструкции
    """
    # Проверяем наличие плейсхолдеров изображений в ответе
    has_image_placeholders = bool(re.search(r'\[\[image:', answer))

    if has_image_placeholders:
        # Есть плейсхолдеры - рендерим с изображениями в тексте
        if instruction_title:
            st.markdown(f"**Источник изображений:** {instruction_title}")
        render_answer_with_images(answer, images)
    else:
        # Нет плейсхолдеров - показываем текст и изображения отдельно
        st.markdown(answer)

        if images:
            st.markdown("---")
            if instruction_title:
                st.markdown(f"### 🖼️ Изображения из инструкции: **{instruction_title}**")
            else:
                st.markdown("### 🖼️ Связанные изображения:")

            # Отображаем изображения в колонках (по 2 в ряд)
            cols_per_row = 2
            for idx in range(0, len(images), cols_per_row):
                cols = st.columns(cols_per_row)
                for col_idx, col in enumerate(cols):
                    img_idx = idx + col_idx
                    if img_idx < len(images):
                        image_path = images[img_idx]
                        with col:
                            try:
                                st.image(image_path, caption=f"Изображение {img_idx + 1}", width='stretch')
                            except Exception as e:
                                st.warning(f"⚠️ Не удалось загрузить изображение: {image_path}")


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

                        # Отображение ответа с изображениями
                        st.markdown("### 💬 Ответ:")

                        # Используем новую функцию для отображения ответа с встроенными изображениями
                        display_answer_with_inline_images(
                            answer=result['answer'],
                            images=result.get('images', []),
                            instruction_title=result.get('best_instruction_title')
                        )

                        # Отображение источников
                        if result['sources']:
                            st.markdown("### 📚 Источники:")
                            for source in result['sources']:
                                # Показываем наличие изображений в источнике
                                has_images = len(source.get('images', [])) > 0
                                images_indicator = " 🖼️" if has_images else ""

                                # Отмечаем топ-1 инструкцию
                                is_best = source.get('is_best', False)
                                best_indicator = " ⭐ (основной источник)" if is_best else ""

                                with st.expander(
                                    f"📄 {source['filename']}{images_indicator}{best_indicator} (релевантность: {1 - source['distance']:.2%})"
                                ):
                                    doc = result['documents'][source['index'] - 1]
                                    st.text(doc['text'])

                                    # Метаданные
                                    metadata = doc.get('metadata', {})
                                    st.caption(f"Название: {source.get('title', 'N/A')}")
                                    st.caption(f"Doc ID: {metadata.get('doc_id', 'N/A')}")
                                    st.caption(f"Чанк: {metadata.get('chunk_index', 0) + 1}/{metadata.get('total_chunks', 1)}")

                                    # Показываем изображения конкретного источника
                                    if source.get('images'):
                                        st.markdown("**Изображения в этом источнике:**")
                                        for img_path in source['images']:
                                            try:
                                                st.image(img_path, width='stretch')
                                            except Exception as e:
                                                st.caption(f"⚠️ Изображение: {img_path} (не удалось загрузить)")
                        else:
                            st.info("Источники не найдены")
                    
                    except Exception as e:
                        st.error(f"❌ Ошибка при поиске: {e}")

    with tab2:
        st.header("Загрузка новых документов")

        uploaded_file = st.file_uploader(
            "Выберите файл (.docx, .md, .txt)",
            type=['docx', 'md', 'txt']
        )

        # Тип документа
        source_type = st.radio(
            "Тип документа:",
            options=['single_file', 'multi_instruction'],
            format_func=lambda x: "Один файл = одна инструкция" if x == 'single_file' else "Множество инструкций (разделитель ---)",
            help="Выберите тип документа для правильной обработки"
        )

        # Метаданные
        col1, col2 = st.columns(2)
        with col1:
            author = st.text_input("Автор документа", value="Admin")
        with col2:
            # Получаем все доступные теги
            metadata_manager = MetadataManager()
            available_tags = metadata_manager.get_all_tags()

            selected_tags = st.multiselect(
                "Выберите теги",
                options=available_tags,
                default=[]
            )

        # Дополнительные теги
        custom_tags_input = st.text_input(
            "Дополнительные теги (через запятую)",
            placeholder="новый_тег1, новый_тег2"
        )

        # Объединяем выбранные и пользовательские теги
        custom_tags = [t.strip() for t in custom_tags_input.split(',') if t.strip()]
        all_tags = selected_tags + custom_tags

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
                        # Парсинг документа (одна или несколько инструкций)
                        instructions = parse_document(
                            file_path=file_path,
                            source_type=source_type,
                            tags=all_tags,
                            author=author
                        )

                        st.info(f"Найдено инструкций: {len(instructions)}")

                        # Инициализация
                        metadata_manager = MetadataManager()
                        embedding_model = EmbeddingModel(EMBEDDING_MODEL_NAME)
                        client, collection = get_chroma()

                        # Обработка каждой инструкции
                        for instruction in instructions:
                            st.write(f"Обработка: **{instruction['title']}**")

                            # Подготовка текста с заголовком
                            text_with_header = prepare_text_for_chunking(
                                instruction['text'],
                                instruction['title']
                            )

                            # Разбиение на чанки
                            chunks = split_text(
                                text_with_header,
                                max_length=CHUNK_SIZE_TOKENS * 4,
                                overlap=CHUNK_OVERLAP_TOKENS * 4
                            )

                            # Создание эмбеддингов
                            embeddings = embedding_model.encode(chunks)

                            # Добавление в ChromaDB
                            chunk_ids = []
                            metadatas = []
                            created_at = datetime.now().isoformat()

                            for i, chunk in enumerate(chunks):
                                chunk_id = f"{instruction['id']}_chunk_{i}"
                                chunk_ids.append(chunk_id)

                                metadata = {
                                    'instruction_id': instruction['id'],
                                    'doc_id': instruction['doc_id'],
                                    'title': instruction['title'],
                                    'filename': uploaded_file.name,
                                    'file_path': file_path,
                                    'chunk_index': i,
                                    'total_chunks': len(chunks),
                                    'active': True,
                                    'author': author,
                                    'tags': ','.join(all_tags),
                                    'created_at': created_at,
                                    'images': ','.join(instruction.get('images', []))
                                }
                                metadatas.append(metadata)

                            collection.add(
                                documents=chunks,
                                embeddings=embeddings.tolist(),
                                metadatas=metadatas,
                                ids=chunk_ids
                            )

                            # Сохранение метаданных в БД
                            metadata_manager.add_instruction(
                                instruction_id=instruction['id'],
                                doc_id=instruction['doc_id'],
                                title=instruction['title'],
                                file_path=file_path,
                                file_format=instruction['file_format'],
                                source_type=instruction['source_type'],
                                separator_index=instruction['separator_index'],
                                author=author,
                                tags=all_tags,
                                images=instruction['images']
                            )

                            st.success(f"✓ {instruction['title']} ({len(chunks)} чанков)")

                        st.success(f"🎉 Загрузка завершена! Добавлено инструкций: {len(instructions)}")
                        st.balloons()

                        # Очистка кеша
                        st.cache_resource.clear()

                except Exception as e:
                    st.error(f"❌ Ошибка при загрузке: {e}")
                    import traceback
                    st.code(traceback.format_exc())

    with tab3:
        st.header("Управление базой знаний")

        metadata_manager = MetadataManager()
        stats = metadata_manager.get_stats()

        # Статистика
        col1, col2, col3 = st.columns(3)
        with col1:
            st.metric("Активных инструкций", stats['active_instructions'])
        with col2:
            st.metric("Неактуальных", stats['inactive_instructions'])
        with col3:
            st.metric("Всего тегов", stats['total_tags'])

        st.markdown("---")

        # Фильтры
        col1, col2 = st.columns(2)
        with col1:
            show_inactive = st.checkbox("Показать неактуальные", value=False)
        with col2:
            filter_tag = st.selectbox(
                "Фильтр по тегу",
                options=["Все"] + metadata_manager.get_all_tags()
            )

        # Получение инструкций
        if filter_tag == "Все":
            instructions = metadata_manager.get_all_instructions(active_only=not show_inactive)
        else:
            instructions = metadata_manager.get_instructions_by_tag(filter_tag)
            if not show_inactive:
                instructions = [i for i in instructions if i['active']]

        st.markdown(f"### Найдено инструкций: {len(instructions)}")

        # Отображение инструкций
        for inst in instructions:
            with st.expander(
                f"{'❌' if not inst['active'] else '✅'} {inst['title']} ({inst['file_format']})",
                expanded=False
            ):
                col1, col2 = st.columns([3, 1])

                with col1:
                    st.write(f"**ID:** `{inst['id']}`")
                    st.write(f"**Файл:** {inst['file_path']}")
                    st.write(f"**Автор:** {inst['author']}")
                    st.write(f"**Создано:** {inst['created_at']}")
                    st.write(f"**Тип:** {inst['source_type']}")

                    if inst['tags']:
                        tags_str = ", ".join([f"`{tag}`" for tag in inst['tags']])
                        st.write(f"**Теги:** {tags_str}")

                with col2:
                    if inst['active']:
                        if st.button("Пометить неактуальной", key=f"deactivate_{inst['id']}"):
                            if metadata_manager.mark_instruction_inactive(inst['id']):
                                st.success("Помечена как неактуальная")
                                st.rerun()
                            else:
                                st.error("Ошибка")

                    if st.button("🗑️ Удалить", key=f"delete_{inst['id']}", type="secondary"):
                        try:
                            # Удаляем из метаданных SQLite
                            if metadata_manager.delete_instruction(inst['id']):
                                # Удаляем чанки из ChromaDB
                                client, collection = get_chroma()
                                # Получаем все чанки этой инструкции
                                results = collection.get(
                                    where={"instruction_id": inst['id']}
                                )
                                if results and results['ids']:
                                    collection.delete(ids=results['ids'])
                                    st.success(f"✅ Удалена инструкция и {len(results['ids'])} чанков")
                                else:
                                    st.success("✅ Удалена инструкция (чанки не найдены)")
                                st.rerun()
                            else:
                                st.error("❌ Ошибка при удалении метаданных")
                        except Exception as e:
                            st.error(f"❌ Ошибка при удалении: {e}")

        st.markdown("---")

        # Опасная зона
        with st.expander("⚠️ Опасная зона", expanded=False):
            st.warning("Эти действия необратимы!")

            if st.button("🗑️ Очистить всю базу (ChromaDB + метаданные)", type="secondary"):
                try:
                    # Очищаем ChromaDB
                    client, collection = get_chroma()
                    client.delete_collection("documents")

                    # Очищаем метаданные SQLite
                    metadata_manager = MetadataManager()
                    metadata_manager.clear_all_data()

                    st.success("✅ База данных полностью очищена (ChromaDB + метаданные)")
                    st.cache_resource.clear()
                    st.rerun()
                except Exception as e:
                    st.error(f"❌ Ошибка при очистке: {e}")


if __name__ == "__main__":
    main()
