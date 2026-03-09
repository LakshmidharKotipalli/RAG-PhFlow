from pathlib import Path

from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_file=".env", env_file_encoding="utf-8", extra="ignore")

    # Embedding
    embedding_model: str = "NeuML/pubmedbert-base-embeddings"
    embedding_device: str = "cpu"
    embedding_batch_size: int = 64

    # Qdrant
    qdrant_url: str = "http://localhost:6333"
    collection_name: str = "ecfr_title21"

    # llama.cpp server (scheme is added automatically if missing)
    llama_cpp_url: str = "http://localhost:8080"
    llama_cpp_temperature: float = 0.1
    llama_cpp_num_ctx: int = 65536   # -c / --ctx-size when launching server
    llama_cpp_max_tokens: int = 2048

    # Reranker
    reranker_model: str = "Qwen/Qwen3-Reranker-0.6B"

    # Retrieval
    initial_retrieval_k: int = 20
    reranker_top_n: int = 5
    reranker_score_threshold: float = 0.3  # min yes-probability to pass to LLM

    # Chunking
    chunk_target_tokens: int = 400
    chunk_max_tokens: int = 512
    # Data paths (relative to project root)
    xml_path: str = "data/ecfr-title-21/ECFR-title21.xml"
    versions_path: str = "data/ecfr-title-21/versions-title-21.json"

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
