"""Runtime settings, read from environment variables or a local .env file."""
from functools import lru_cache
from pathlib import Path
from typing import Literal

from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        env_prefix="EARSHOT_",
        env_file=Path(__file__).resolve().parent.parent / ".env",
        extra="ignore",
    )

    llm_api_key: str = ""
    llm_base_url: str = "https://api.groq.com/openai/v1"
    llm_model: str = "llama-3.3-70b-versatile"
    llm_price_input_per_m: float = 0.0
    llm_price_output_per_m: float = 0.0
    llm_timeout_s: float = 8.0

    embedder: Literal["hashing", "minilm"] = "hashing"
    reranker: Literal["none", "lexical", "cross-encoder"] = "lexical"
    whisper_model: str = "tiny.en"

    # A smart-glasses caption area: short lines, few of them.
    hud_columns: int = 32
    hud_rows: int = 2

    retrieve_k: int = 12
    rerank_k: int = 4
    api_token: str = ""


@lru_cache
def get_settings() -> Settings:
    return Settings()
