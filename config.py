from typing import Optional
from pydantic_settings import BaseSettings


class Settings(BaseSettings):
    # Core
    openai_api_key: str
    pinecone_api_key: str
    pinecone_index_name: str = "parking-assistant"
    openai_model: str = "gpt-4o-mini"
    embedding_model: str = "text-embedding-3-small"
    top_k: int = 3

    # Stage 2: admin notification via email (all optional — falls back to file log)
    smtp_host: Optional[str] = None
    smtp_port: int = 465
    smtp_user: Optional[str] = None
    smtp_password: Optional[str] = None
    admin_email: Optional[str] = None

    class Config:
        env_file = ".env"


settings = Settings()
