from abc import ABC, abstractmethod
from typing import List
from langchain_community.embeddings import OpenAIEmbeddings
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_core.vectorstores import VectorStore
from langchain_openai import ChatOpenAI
from langchain.schema import Document
from models import CompareResponse


class SimilarityService(ABC):
    """Интерфейс для сервиса сравнения документов."""

    @abstractmethod
    async def compare_texts(self, document_a: str, document_b: str) -> CompareResponse:
        """Сравнивает два текста по смыслу и возвращает оценку сходства."""
        pass


class LLMSimilarityService(SimilarityService):
    """
    Сервис для сравнения документов по смыслу с использованием LLM + RAG.
    """

    def __init__(
            self,
            llm_client: ChatOpenAI,
            embedding_model: OpenAIEmbeddings,
            vector_store: VectorStore
    ):
        """
        Инициализация зависимостей:
        - llm_client: экземпляр LangChain LLM (например, OpenAI)
        - embedding_model: модель эмбеддингов (например, OpenAIEmbeddings)
        - vector_store: FAISS / Chroma / Pinecone и т.д.
        """
        self.llm = llm_client
        self.embedder = embedding_model
        self.vector_store = vector_store

    # ----------- Вспомогательные методы -----------

    def _split_text(self, text: str) -> List[str]:
        """Разбивает длинный текст на перекрывающиеся фрагменты."""
        splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
        return splitter.split_text(text)

    def _build_vector_store(self, chunks: List[str]) -> VectorStore:
        """Создаёт векторное хранилище для набора фрагментов."""
        docs = [Document(page_content=chunk) for chunk in chunks]
        return self.vector_store.from_documents(docs, self.embedder)

    # ----------- Основная логика -----------

    async def compare_texts(self, document_a: str, document_b: str) -> CompareResponse:
        """
        Основная логика сравнения документов по смыслу:
        1. Разбивает тексты на фрагменты.
        2. Строит эмбеддинги.
        3. Выполняет cross-retrieval (A → B).
        4. Оценивает сходство через LLM.
        5. Агрегирует результаты.
        """

        # 1. Разбиваем документы
        chunks_a = self._split_text(document_a)
        chunks_b = self._split_text(document_b)

        if not chunks_a or not chunks_b:
            return CompareResponse(similarity_score=0.0, summary="Один из документов пуст.")

        # 2. Создаём векторное хранилище для документа B
        db_b = self._build_vector_store(chunks_b)

        # 3. Для каждого сегмента из A находим наиболее похожие из B
        retrieved_pairs = []
        for chunk in chunks_a:
            results = db_b.similarity_search(chunk, k=2)
            for res in results:
                retrieved_pairs.append((chunk, res.page_content))

        # 4. Прогоняем пары через LLM
        scores = []
        for (a_seg, b_seg) in retrieved_pairs[:10]:  # ограничим число пар для скорости
            prompt = f"""
            Сравни по смыслу два фрагмента текста. 
            Верни оценку от 0 до 1, где 1 — одинаковый смысл, 0 — совершенно разные.
            Также кратко объясни различия.
            ---
            Текст A:
            {a_seg}
            ---
            Текст B:
            {b_seg}
            ---
            Формат ответа:
            Оценка: <число>
            Пояснение: <строка>
            """

            response = await self.llm.ainvoke(prompt)
            text = response.content.strip()

            # Простая эвристика для извлечения числа
            score = 0.0
            for token in text.split():
                try:
                    val = float(token.replace(",", "."))
                    if 0 <= val <= 1:
                        score = val
                        break
                except ValueError:
                    continue

            scores.append(score)

        # 5. Агрегируем результаты
        avg_score = round(sum(scores) / len(scores), 3) if scores else 0.0
        summary = f"Средняя семантическая схожесть: {avg_score} (на основе {len(scores)} фрагментов)."

        return CompareResponse(
            similarity_score=avg_score,
            summary=summary
        )
