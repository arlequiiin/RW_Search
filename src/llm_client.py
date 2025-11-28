"""
LLM клиент для работы с Ollama (llama3:8b)
"""
import ollama
from src.config import LLM_MODEL_NAME, LLM_MAX_TOKENS


class LLMClient:
    """
    Клиент для взаимодействия с локальной LLM через Ollama
    """

    def __init__(self, model_name: str = LLM_MODEL_NAME):
        self.model_name = model_name
        self._verify_model()

    def _verify_model(self):
        """Проверка доступности модели"""
        try:
            models = ollama.list()
            # Ollama возвращает словарь с ключом 'models', который содержит список моделей
            # Каждая модель - это словарь с разными ключами в зависимости от версии
            available_models = []
            for m in models.get('models', []):
                # Пытаемся получить имя модели из разных возможных ключей
                model_name = m.get('name') or m.get('model') or str(m)
                available_models.append(model_name)

            if self.model_name not in available_models:
                print(f"⚠️  Модель {self.model_name} не найдена в списке.")
                print(f"   Доступные модели: {available_models}")
        except Exception as e:
            print(f"⚠️  Предупреждение: не удалось проверить модель: {e}")

    def generate(
        self,
        prompt: str,
        system_prompt: str = None,
        max_tokens: int = LLM_MAX_TOKENS,
        temperature: float = 0.7
    ) -> str:
        """
        Генерация ответа от LLM

        Args:
            prompt: Пользовательский запрос
            system_prompt: Системная инструкция (опционально)
            max_tokens: Максимальное количество токенов в ответе
            temperature: Параметр случайности (0.0 - детерминированный, 1.0 - креативный)

        Returns:
            Ответ модели в виде строки
        """
        messages = []

        if system_prompt:
            messages.append({
                'role': 'system',
                'content': system_prompt
            })

        messages.append({
            'role': 'user',
            'content': prompt
        })

        try:
            response = ollama.chat(
                model=self.model_name,
                messages=messages,
                options={
                    'num_predict': max_tokens,
                    'temperature': temperature
                }
            )

            return response['message']['content']

        except Exception as e:
            error_msg = f"Ошибка при генерации ответа: {e}"
            print(f"❌ {error_msg}")
            return f"[ОШИБКА] {error_msg}"

    def generate_rag_answer(
        self,
        query: str,
        context: str,
        max_tokens: int = LLM_MAX_TOKENS
    ) -> str:
        """
        Генерация ответа в режиме RAG (с контекстом из базы знаний)

        Args:
            query: Вопрос пользователя
            context: Контекст из векторной базы (найденные документы)
            max_tokens: Максимальное количество токенов

        Returns:
            Ответ модели на основе контекста
        """
        system_prompt = """Ты — помощник по поиску информации в базе знаний инструкций.

ВАЖНЫЕ ПРАВИЛА:
1. Используй ТОЛЬКО информацию из предоставленного контекста
2. Если в контексте нет ответа на вопрос — честно скажи "В базе знаний нет информации по этому вопросу"
3. Не придумывай информацию, которой нет в контексте
4. Отвечай четко, структурированно, по делу
5. Если в контексте есть упоминания изображений в формате [[image: путь]] — обязательно упомяни об этом в ответе, например: "См. изображение для визуального примера" или "На изображении показано..."
6. Изображения из контекста будут автоматически показаны пользователю отдельно, но ты должен упомянуть их наличие в своем ответе
7. Отвечай на русском языке"""

        prompt = f"""КОНТЕКСТ ИЗ БАЗЫ ЗНАНИЙ:
{context}

---

ВОПРОС ПОЛЬЗОВАТЕЛЯ:
{query}

---

ОТВЕТ (используй только информацию из контекста выше):"""

        return self.generate(
            prompt=prompt,
            system_prompt=system_prompt,
            max_tokens=max_tokens,
            temperature=0.3  # Низкая температура для точности
        )


# Удобная функция для быстрого создания клиента
def get_llm_client(model_name: str = LLM_MODEL_NAME) -> LLMClient:
    """Создание и возврат LLM клиента"""
    return LLMClient(model_name=model_name)
