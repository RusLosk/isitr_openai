from pydantic import BaseModel, Field
from typing import List, Dict, Any, Optional
from langchain.schema import Document

class SearchQuery(BaseModel):
    query: str = Field(..., description="Поисковый запрос")
    k: int = Field(5, description="Количество возвращаемых результатов")

class RAGQuery(BaseModel):
    question: str = Field(..., description="Вопрос для RAG системы")

class SearchResponse(BaseModel):
    documents: List[Dict[str, Any]]
    query: str

class RAGResponse(BaseModel):
    answer: str
    question: str
    sources: List[Dict[str, Any]]

class DocumentAddRequest(BaseModel):
    documents: List[Dict[str, Any]] = Field(..., description="Список документов для добавления")

    class Config:
        schema_extra = {
            "example": {
                "documents": [
                    {
                        "page_content": "Текст документа...",
                        "metadata": {"source": "file1.pdf", "page": 1}
                    }
                ]
            }
        }

class VectorStoreInfo(BaseModel):
    document_count: int
    vector_store_path: str
    embedding_model: str
    is_loaded: bool
