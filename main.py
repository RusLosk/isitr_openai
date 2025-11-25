from fastapi import FastAPI, Depends, HTTPException, status
from fastapi.middleware.cors import CORSMiddleware
from langchain.schema import Document
import logging

from services.similarity_service import LLMSimilarityService
from services.vector_store_manager import VectorStoreManager
from dependencies import get_similarity_service, get_vector_store_manager
from models import (
    SearchQuery, SearchResponse, RAGQuery, RAGResponse,
    DocumentAddRequest, VectorStoreInfo
)

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = FastAPI(
    title="RAG API with FAISS",
    description="API для семантического поиска и RAG с векторной БД FAISS",
    version="1.0.0"
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

@app.on_event("startup")
async def startup_event():
    try:
        similarity_service = get_similarity_service()
        vector_store_manager = get_vector_store_manager()

        logger.info("Application started successfully")
        logger.info(f"Vector store loaded: {vector_store_manager.vector_store is not None}")

    except Exception as e:
        logger.error(f"Error during startup: {e}")
        raise

@app.get("/")
async def root():
    return {
        "message": "RAG API with FAISS Vector Database",
        "status": "running"
    }

@app.get("/health")
async def health_check():
    similarity_service = get_similarity_service()
    store_info = similarity_service.get_store_info()

    return {
        "status": "healthy",
        "vector_store": store_info
    }

@app.post("/search", response_model=SearchResponse)
async def semantic_search(
    search_query: SearchQuery,
    similarity_service: LLMSimilarityService = Depends(get_similarity_service)
):
    try:
        documents = similarity_service.similarity_search(
            query=search_query.query,
            k=search_query.k
        )

        docs_data = []
        for doc in documents:
            docs_data.append({
                "page_content": doc.page_content,
                "metadata": doc.metadata,
                "type": "document"
            })

        return SearchResponse(
            documents=docs_data,
            query=search_query.query
        )

    except Exception as e:
        logger.error(f"Search error: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Search failed: {str(e)}"
        )

@app.post("/ask", response_model=RAGResponse)
async def ask_question(
    rag_query: RAGQuery,
    similarity_service: LLMSimilarityService = Depends(get_similarity_service)
):
    try:
        result = similarity_service.query_with_rag(rag_query.question)

        sources = []
        for doc in result.get("source_documents", []):
            sources.append({
                "content": doc.page_content[:500] + "..." if len(doc.page_content) > 500 else doc.page_content,
                "metadata": doc.metadata
            })

        return RAGResponse(
            answer=result["answer"],
            question=result["question"],
            sources=sources
        )

    except Exception as e:
        logger.error(f"RAG query error: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Question answering failed: {str(e)}"
        )

@app.post("/documents")
async def add_documents(
    document_request: DocumentAddRequest,
    similarity_service: LLMSimilarityService = Depends(get_similarity_service)
):
    try:
        # Конвертируем в Document objects
        documents = []
        for doc_data in document_request.documents:
            doc = Document(
                page_content=doc_data["page_content"],
                metadata=doc_data.get("metadata", {})
            )
            documents.append(doc)

        similarity_service.add_documents(documents)

        return {
            "message": f"Successfully added {len(documents)} documents",
            "total_documents": similarity_service.get_store_info()["document_count"]
        }

    except Exception as e:
        logger.error(f"Document addition error: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to add documents: {str(e)}"
        )

@app.get("/vector-store/info", response_model=VectorStoreInfo)
async def get_vector_store_info(
    similarity_service: LLMSimilarityService = Depends(get_similarity_service)
):
    return similarity_service.get_store_info()

@app.post("/vector-store/save")
async def save_vector_store(
    vector_store_manager: VectorStoreManager = Depends(get_vector_store_manager)
):
    try:
        vector_store_manager.save_vector_store()
        return {"message": "Vector store saved successfully"}
    except Exception as e:
        logger.error(f"Vector store save error: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to save vector store: {str(e)}"
        )

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
