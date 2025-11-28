
from typing import List, Dict, Tuple
from src.embeddings import EmbeddingModel
from src.storage import get_chroma
from src.llm_client import get_llm_client
from src.config import TOP_K, EMBEDDING_MODEL_NAME
from src.hybrid_search import HybridSearcher


class RAGPipeline:
    def __init__(
        self,
        embedding_model_name: str = EMBEDDING_MODEL_NAME,
        top_k: int = TOP_K
    ):
        print("Инициализация RAG pipeline...")
        self.embedding_model = EmbeddingModel(embedding_model_name)
        self.client, self.collection = get_chroma()
        self.llm_client = get_llm_client()
        self.top_k = top_k
        print("✅ RAG pipeline готов")

    def search_similar(
        self,
        query: str,
        top_k: int = None,
        filter_active: bool = True
    ) -> List[Dict]:
        """
        Поиск похожих документов в векторной базе

        Args:
            query: Поисковый запрос
            top_k: Количество результатов (если None, используется self.top_k)
            filter_active: Фильтровать только активные документы

        Returns:
            Список найденных документов с метаданными и скорами
        """
        if top_k is None:
            top_k = self.top_k

        # эмбеддинг запроса
        query_embedding = self.embedding_model.encode([query])[0].tolist()

        # подготовка фильтра
        where_filter = {"active": True} if filter_active else None

        # поиск в ChromaDB
        results = self.collection.query(
            query_embeddings=[query_embedding],
            n_results=top_k,
            where=where_filter if where_filter else {}
        )

        # форматирование результатов
        documents = []
        if results['documents'] and len(results['documents']) > 0:
            for i in range(len(results['documents'][0])):
                doc = {
                    'text': results['documents'][0][i],
                    'metadata': results['metadatas'][0][i] if results['metadatas'] else {},
                    'distance': results['distances'][0][i] if results['distances'] else None,
                    'id': results['ids'][0][i] if results['ids'] else None
                }
                documents.append(doc)

        return documents

    def format_context(self, documents: List[Dict]) -> Tuple[str, List[Dict], List[str], str]:
        """
        Форматирование найденных документов в контекст для LLM

        Args:
            documents: Список документов из поиска

        Returns:
            Tuple (отформатированный контекст, источники, список путей к изображениям, instruction_id топ-1 инструкции)
        """
        if not documents:
            return "Контекст отсутствует.", [], [], None

        context_parts = []
        sources = []
        all_images = []

        # Группируем документы по instruction_id и считаем средний score
        from collections import defaultdict
        instruction_scores = defaultdict(list)
        instruction_docs = defaultdict(list)

        for doc in documents:
            metadata = doc.get('metadata', {})
            instruction_id = metadata.get('instruction_id', metadata.get('doc_id', ''))
            distance = doc.get('distance', 1.0)

            instruction_scores[instruction_id].append(distance)
            instruction_docs[instruction_id].append(doc)

        # Находим instruction_id с наилучшим средним score (наименьший distance)
        best_instruction_id = None
        best_avg_distance = float('inf')

        for instruction_id, distances in instruction_scores.items():
            avg_distance = sum(distances) / len(distances)
            if avg_distance < best_avg_distance:
                best_avg_distance = avg_distance
                best_instruction_id = instruction_id

        for i, doc in enumerate(documents, 1):
            text = doc['text']
            metadata = doc.get('metadata', {})
            instruction_id = metadata.get('instruction_id', metadata.get('doc_id', ''))

            # Извлекаем изображения из метаданных
            images_str = metadata.get('images', '')
            # Преобразуем строку обратно в список
            images = [img.strip() for img in images_str.split(',') if img.strip()] if images_str else []

            # Добавляем изображения ТОЛЬКО из топ-1 инструкции
            if images and instruction_id == best_instruction_id:
                all_images.extend(images)

            # Формируем источник
            source_info = {
                'index': i,
                'filename': metadata.get('filename', 'Неизвестный документ'),
                'title': metadata.get('title', metadata.get('filename', 'Неизвестный документ')),
                'doc_id': metadata.get('doc_id', ''),
                'instruction_id': instruction_id,
                'distance': doc.get('distance', 0.0),
                'images': images,
                'is_best': instruction_id == best_instruction_id
            }
            sources.append(source_info)

            # Формируем контекст
            context_part = f"""[Документ {i}: {source_info['filename']}]
{text}
"""
            context_parts.append(context_part)

        context = "\n---\n".join(context_parts)
        return context, sources, all_images, best_instruction_id

    def query(self, user_query: str, top_k: int = None) -> Dict:
        """
        Основной метод для выполнения RAG запроса

        Args:
            user_query: Вопрос пользователя
            top_k: Количество документов для поиска

        Returns:
            Словарь с ответом, контекстом и источниками
        """
        print(f"\n🔍 Поиск по запросу: {user_query}")

        # 1. Поиск похожих документов
        documents = self.search_similar(user_query, top_k=top_k)

        if not documents:
            return {
                'answer': "К сожалению, в базе знаний не найдено релевантной информации по вашему запросу.",
                'context': "",
                'sources': [],
                'documents': []
            }

        print(f"✅ Найдено документов: {len(documents)}")

        # 2. Форматирование контекста
        context, sources, images, best_instruction_id = self.format_context(documents)

        # 3. Генерация ответа с помощью LLM
        print("🤖 Генерация ответа...")
        answer = self.llm_client.generate_rag_answer(
            query=user_query,
            context=context
        )

        print("✅ Ответ готов")

        # Получаем название топ-1 инструкции для отображения
        best_instruction_title = None
        for source in sources:
            if source.get('is_best'):
                best_instruction_title = source.get('title')
                break

        return {
            'answer': answer,
            'context': context,
            'sources': sources,
            'documents': documents,
            'images': images,
            'best_instruction_id': best_instruction_id,
            'best_instruction_title': best_instruction_title
        }

    def get_stats(self) -> Dict:
        """
        Получение статистики базы знаний

        Returns:
            Словарь со статистикой
        """
        count = self.collection.count()
        return {
            'total_chunks': count,
            'collection_name': self.collection.name
        }


# создание пайплайна
def create_rag_pipeline() -> RAGPipeline:
    """Создание и возврат RAG пайплайна"""
    return RAGPipeline()
