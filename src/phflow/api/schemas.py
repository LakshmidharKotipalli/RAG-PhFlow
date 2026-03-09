"""
Pydantic request/response models for the FastAPI API.
"""

from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, Field


# --- Shared ---

class SourceResult(BaseModel):
    section_number: str
    section_title: str
    section_node: str
    part_number: int | None = None
    part_name: str = ""
    latest_amendment_date: str = ""
    score: float
    full_text: str


# --- Search ---

class SearchFilters(BaseModel):
    part_number: int | None = None
    chapter_number: str | None = None
    subpart_letter: str | None = None
    chunk_types: list[str] | None = None
    amended_after: str | None = None


class SearchRequest(BaseModel):
    query: str = Field(..., min_length=1, max_length=2000)
    filters: SearchFilters = Field(default_factory=SearchFilters)
    top_k: int = Field(default=10, ge=1, le=50)
    expand_to_parent: bool = True


class SearchResponse(BaseModel):
    results: list[SourceResult]
    total: int
    query_time_ms: float


# --- Chat ---

class ChatRequest(BaseModel):
    message: str = Field(..., min_length=1, max_length=5000)
    mode: Literal["qa", "compliance", "research"] = "qa"
    filters: SearchFilters = Field(default_factory=SearchFilters)
    stream: bool = False


class ChatMessage(BaseModel):
    role: Literal["user", "assistant"]
    content: str
    sources: list[SourceResult] | None = None


class ChatResponse(BaseModel):
    session_id: str
    answer: str
    sources: list[SourceResult]
    citations_footer: str
    mode: str


# --- Compliance ---

class ComplianceRequest(BaseModel):
    scenario: str = Field(..., min_length=10, max_length=10000)
    filters: SearchFilters = Field(default_factory=SearchFilters)


class ComplianceResponse(BaseModel):
    answer: str
    sources: list[SourceResult]
    citations_footer: str


# --- Parts listing ---

class PartInfo(BaseModel):
    part_number: int
    part_name: str
    chapter_number: str
    chapter_name: str


# --- Health ---

class ComponentHealth(BaseModel):
    status: Literal["ok", "error"]
    detail: str = ""


class HealthResponse(BaseModel):
    status: Literal["ok", "degraded", "error"]
    components: dict[str, ComponentHealth]
