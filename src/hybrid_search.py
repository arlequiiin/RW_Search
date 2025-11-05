"""
Гибридный поиск: Embeddings (семантический) + BM25 (ключевые слова)
"""
from typing import List, Dict
from rank_bm25 import BM25Okapi
import numpy as np


def tokenize_russian(text: str) -> List[str]:
    """Простая токенизация для русского текста"""
    import re
    # Приводим к нижнему регистру и разбиваем по пробелам/знакам
    tokens = re.findall(r'\b\w+\b', text.lower())
    return tokens


class HybridSearcher:
    """
    Гибридный поиск, комбинирующий:
    - Semantic search (embeddings) - понимание смысла
    - Keyword search (BM25) - точное совпадение слов
    """
    
    def __init__(self, documents: List[str], document_ids: List[str]):
        """
        Args:
            documents: Список текстов документов
            document_ids: ID документов (для маппинга)
        """
        self.documents = documents
        self.document_ids = document_ids
        
        # Токенизация для BM25
        tokenized_docs = [tokenize_russian(doc) for doc in documents]
        self.bm25 = BM25Okapi(tokenized_docs)
    
    def search_bm25(self, query: str, top_k: int = 10) -> List[Dict]:
        """
        Keyword-based поиск через BM25
        
        Returns:
            List of {doc_id, text, score}
        """
        tokenized_query = tokenize_russian(query)
        scores = self.bm25.get_scores(tokenized_query)
        
        # Сортируем по score
        top_indices = np.argsort(scores)[::-1][:top_k]
        
        results = []
        for idx in top_indices:
            if scores[idx] > 0:  # Только релевантные
                results.append({
                    'doc_id': self.document_ids[idx],
                    'text': self.documents[idx],
                    'bm25_score': float(scores[idx])
                })
        
        return results
    
    @staticmethod
    def combine_scores(
        semantic_results: List[Dict],
        bm25_results: List[Dict],
        semantic_weight: float = 0.5,
        bm25_weight: float = 0.5
    ) -> List[Dict]:
        """
        Комбинирование результатов из двух методов
        
        Args:
            semantic_results: Результаты от embeddings (с distance)
            bm25_results: Результаты от BM25 (с bm25_score)
            semantic_weight: Вес семантического поиска (0-1)
            bm25_weight: Вес BM25 поиска (0-1)
        
        Returns:
            Объединённые и отранжированные результаты
        """
        # Нормализация scores
        def normalize_scores(results, score_key):
            if not results:
                return results
            scores = [r[score_key] for r in results]
            min_score, max_score = min(scores), max(scores)
            if max_score == min_score:
                for r in results:
                    r[f'{score_key}_norm'] = 1.0
            else:
                for r in results:
                    r[f'{score_key}_norm'] = (r[score_key] - min_score) / (max_score - min_score)
            return results
        
        # Нормализуем semantic (1 - distance)
        for r in semantic_results:
            r['semantic_score'] = 1 - r.get('distance', 0)
        semantic_results = normalize_scores(semantic_results, 'semantic_score')
        
        # Нормализуем BM25
        bm25_results = normalize_scores(bm25_results, 'bm25_score')
        
        # Объединяем по doc_id
        combined = {}
        
        for r in semantic_results:
            doc_id = r.get('id') or r.get('doc_id')
            combined[doc_id] = {
                'doc_id': doc_id,
                'text': r['text'],
                'metadata': r.get('metadata', {}),
                'semantic_norm': r.get('semantic_score_norm', 0),
                'bm25_norm': 0,
                'distance': r.get('distance', 1.0)
            }
        
        for r in bm25_results:
            doc_id = r['doc_id']
            if doc_id in combined:
                combined[doc_id]['bm25_norm'] = r.get('bm25_score_norm', 0)
            else:
                combined[doc_id] = {
                    'doc_id': doc_id,
                    'text': r['text'],
                    'metadata': {},
                    'semantic_norm': 0,
                    'bm25_norm': r.get('bm25_score_norm', 0),
                    'distance': 1.0
                }
        
        # Финальный score
        for doc_id in combined:
            combined[doc_id]['hybrid_score'] = (
                semantic_weight * combined[doc_id]['semantic_norm'] +
                bm25_weight * combined[doc_id]['bm25_norm']
            )
        
        # Сортировка по финальному score
        results = sorted(combined.values(), key=lambda x: x['hybrid_score'], reverse=True)
        
        return results
