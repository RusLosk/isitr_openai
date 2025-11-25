import os
from dotenv import load_dotenv

load_dotenv()

class Settings:
    # OpenAI
    OPENAI_API_KEY: str = os.getenv("OPENAI_API_KEY", "")
    EMBEDDING_MODEL: str = os.getenv("EMBEDDING_MODEL", "text-embedding-3-large")
    LLM_MODEL: str = os.getenv("LLM_MODEL", "gpt-4o-mini")

    # FAISS
    VECTOR_STORE_PATH: str = os.getenv("VECTOR_STORE_PATH", "./data/vector_store")

    # App
    DEBUG: bool = os.getenv("DEBUG", "False").lower() == "true"

settings = Settings()
