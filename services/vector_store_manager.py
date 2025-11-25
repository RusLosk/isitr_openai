import os
import logging
from typing import List, Optional
from langchain_community.vectorstores import FAISS
from langchain_community.embeddings import OpenAIEmbeddings
from langchain.schema import Document
from config import settings

logger = logging.getLogger(__name__)

class VectorStoreManager:


    def __init__(self, embeddings: OpenAIEmbeddings):
        self.embeddings = embeddings
        self.vector_store: Optional[FAISS] = None
        self._ensure_vector_store_dir()

    def _ensure_vector_store_dir(self):

        os.makedirs(settings.VECTOR_STORE_PATH, exist_ok=True)

    def initialize_vector_store(self, documents: List[Document] = None):

        if documents:

            self.vector_store = FAISS.from_documents(documents, self.embeddings)
            logger.info(f"Initialized FAISS with {len(documents)} documents")
        else:

            sample_docs = [Document(page_content="Initial document", metadata={"source": "init"})]
            self.vector_store = FAISS.from_documents(sample_docs, self.embeddings)
            logger.info("Initialized empty FAISS store")

    def load_vector_store(self) -> bool:

        try:
            index_path = os.path.join(settings.VECTOR_STORE_PATH, "index.faiss")
            if os.path.exists(index_path):
                self.vector_store = FAISS.load_local(
                    settings.VECTOR_STORE_PATH,
                    self.embeddings,
                    allow_dangerous_deserialization=True
                )
                logger.info("FAISS vector store loaded successfully")
                return True
            else:
                logger.warning("FAISS index not found, initializing new store")
                self.initialize_vector_store()
                return False
        except Exception as e:
            logger.error(f"Error loading vector store: {e}")
            self.initialize_vector_store()
            return False

    def save_vector_store(self):

        if self.vector_store:
            self.vector_store.save_local(settings.VECTOR_STORE_PATH)
            logger.info(f"Vector store saved to {settings.VECTOR_STORE_PATH}")

    def add_documents(self, documents: List[Document]):

        if self.vector_store:
            self.vector_store.add_documents(documents)
            logger.info(f"Added {len(documents)} documents to vector store")

    def similarity_search(self, query: str, k: int = 5, **kwargs) -> List[Document]:

        if not self.vector_store:
            raise ValueError("Vector store not initialized")

        return self.vector_store.similarity_search(query, k=k, **kwargs)

    def similarity_search_with_scores(self, query: str, k: int = 5) -> List[tuple]:

        if not self.vector_store:
            raise ValueError("Vector store not initialized")

        return self.vector_store.similarity_search_with_score(query, k=k)

    def get_document_count(self) -> int:

        if self.vector_store and hasattr(self.vector_store, 'docstore'):
            return len(self.vector_store.docstore._dict)
        return 0

    def delete_documents(self, ids: List[str]):

        if self.vector_store:
            self.vector_store.delete(ids)
            logger.info(f"Deleted {len(ids)} documents from vector store")
