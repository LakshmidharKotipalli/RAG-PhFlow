"""
Centralized application settings.

All values are read from the .env file at the project root via pydantic-settings.
Edit .env to change any setting; restart the API server after changes.
"""

from pathlib import Path

from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_file=".env", env_file_encoding="utf-8", extra="ignore")

    # Embedding
    embedding_model: str = "BAAI/bge-base-en-v1.5"
    embedding_device: str = "cuda"
    embedding_batch_size: int = 64

    # FAISS
    faiss_index_dir: str = "faiss_index"

    # LLM server (llama.cpp or ollama; scheme is added automatically if missing)
    llama_cpp_url: str = "http://localhost:8080"
    llm_backend: str = "llamacpp"  # "llamacpp" or "ollama"
    ollama_model: str = "qwen3.5"
    llama_cpp_temperature: float = 0.1
    llama_cpp_num_ctx: int = 65536
    llama_cpp_max_tokens: int = 768
    llama_cpp_repeat_penalty: float = 1.1
    llama_cpp_repeat_last_n: int = 256
    llama_cpp_frequency_penalty: float = 0.1  # penalise tokens proportional to their frequency
    llama_cpp_top_p: float = 0.9              # nucleus sampling threshold
    llama_cpp_query_rewrite_tokens: int = 128 # token budget for query rewrite (one sentence)
    # Sessions
    session_max_messages: int = 100

    # Semantic cache (disabled by default — enable after feedback loop is in place)
    cache_enabled: bool = False
    cache_similarity_threshold: float = 0.95  # cosine similarity for cache hit
    cache_max_size: int = 512
    cache_ttl_seconds: int = 3600             # 1 hour

    # Retrieval
    initial_retrieval_k: int = 10
    retrieval_top_n: int = 5    # how many top RRF results to send to the LLM
    prefetch_multiplier: int = 2
    enable_query_rewrite: bool = True
    redis_url: str | None = None
    cors_origins: list[str] = ["*"]
    scope_part_numbers: list[int] = [210, 211]

    # Chunking
    chunk_max_tokens: int = 512

    # Data paths (relative to project root)
    xml_path: str = "data/ecfr-title-21/ECFR-title21.xml"
    versions_path: str = "data/ecfr-title-21/versions-title-21.json"

    # System prompt
    system_prompt_path: str = "prompts/system_prompt.txt"

    @property
    def llama_cpp_base_url(self) -> str:
        url = self.llama_cpp_url
        if not url.startswith(("http://", "https://")):
            url = "http://" + url
        return url.rstrip("/")

    @property
    def xml_file(self) -> Path:
        return Path(self.xml_path)

    @property
    def versions_file(self) -> Path:
        return Path(self.versions_path)


settings = Settings()
