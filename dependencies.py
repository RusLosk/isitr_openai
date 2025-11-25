import os
import logging
from langchain_community.embeddings import OpenAIEmbeddings
from langchain_community.chat_models import ChatOpenAI

from services.similarity_service import LLMSimilarityService
from services.vector_store_manager import VectorStoreManager
from config import settings

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

_similarity_service = None
_vector_store_manager = None

def init_embeddings() -> OpenAIEmbeddings:
    return OpenAIEmbeddings(
        model=settings.EMBEDDING_MODEL,
        openai_api_key=settings.OPENAI_API_KEY
    )

def init_llm() -> ChatOpenAI:
    return ChatOpenAI(
        model=settings.LLM_MODEL,
        temperature=0,
        openai_api_key=settings.OPENAI_API_KEY
    )

def init_vector_store_manager() -> VectorStoreManager:
    global _vector_store_manager
    if _vector_store_manager is None:
        embeddings = init_embeddings()
        _vector_store_manager = VectorStoreManager(embeddings)

        # Пытаемся загрузить существующее хранилище, иначе создаем новое
        if not _vector_store_manager.load_vector_store():
            logger.info("Created new vector store")

    return _vector_store_manager

def init_similarity_service() -> LLMSimilarityService:
    global _similarity_service
    if _similarity_service is None:
        llm = init_llm()
        embeddings = init_embeddings()
        vector_store_manager = init_vector_store_manager()

        _similarity_service = LLMSimilarityService(
            llm_client=llm,
            embedding_model=embeddings,
            vector_store_manager=vector_store_manager
        )

        logger.info("Similarity service initialized successfully")

    return _similarity_service

def get_similarity_service() -> LLMSimilarityService:
    return init_similarity_service()

def get_vector_store_manager() -> VectorStoreManager:
    return init_vector_store_manager()
