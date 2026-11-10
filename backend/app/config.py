"""Application settings, loaded from environment variables (or a .env file)."""

from functools import lru_cache

from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_file=".env", extra="ignore")

    # Storage
    database_url: str = "sqlite:///./data/app.db"
    chroma_path: str = "./data/chroma"
    upload_dir: str = "./data/uploads"
    redis_url: str | None = None
    eval_dir: str = "../eval"

    # Auth
    jwt_secret: str = "change-me-in-production"
    jwt_expire_minutes: int = 60
    admin_username: str | None = None
    admin_password: str | None = None

    # Ingestion
    max_upload_mb: int = 20
    chunk_size: int = 800
    chunk_overlap: int = 120

    # Embeddings: "sentence-transformers" (default) or "hash" (offline, for tests/demos)
    embedding_backend: str = "sentence-transformers"
    embedding_model: str = "google/embeddinggemma-2"

    # Retrieval
    top_k: int = 5
    candidate_k: int = 20

    # LLM: "openai", "ollama" or "fake" (deterministic, offline)
    llm_provider: str = "openai"
    openai_api_key: str | None = None
    openai_model: str = "gpt-4o-mini"
    ollama_base_url: str = "http://localhost:11434"
    ollama_model: str = "llama3.2"

    # Observability
    cloudwatch_enabled: bool = False
    cloudwatch_namespace: str = "EnterpriseKnowledgeAssistant"
    aws_region: str = "us-east-1"


@lru_cache
def get_settings() -> Settings:
    return Settings()
