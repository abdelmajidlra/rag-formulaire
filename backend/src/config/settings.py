from functools import lru_cache
from typing import List, Optional

from pydantic import BaseSettings, Field


class Settings(BaseSettings):
    """Application settings loaded from environment or .env files."""

    api_host: str = Field("0.0.0.0", alias="API_HOST")
    api_port: int = Field(8000, alias="API_PORT")
    allowed_origins: List[str] = Field(default_factory=lambda: ["*"] , alias="ALLOWED_ORIGINS")

    rag_index_path: Optional[str] = Field(None, alias="RAG_INDEX_PATH")
    rag_data_path: Optional[str] = Field(None, alias="RAG_DATA_PATH")
    llm_model_path: Optional[str] = Field(None, alias="LLM_MODEL_PATH")

    enable_auth: bool = Field(False, alias="ENABLE_AUTH")
    azure_ad_tenant_id: Optional[str] = Field(None, alias="AZURE_AD_TENANT_ID")
    azure_ad_client_id: Optional[str] = Field(None, alias="AZURE_AD_CLIENT_ID")
    azure_ad_api_audience: Optional[str] = Field(None, alias="AZURE_AD_API_AUDIENCE")

    class Config:
        env_file = [".env", "../.env", "../../.env"]
        env_file_encoding = "utf-8"
        case_sensitive = False


@lru_cache()
def get_settings() -> Settings:
    return Settings()
