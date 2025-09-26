"""
Data models and schemas for the search application.
"""
from dataclasses import dataclass
from typing import List, Optional, Dict

@dataclass
class SearchResult:
    """Enhanced search result with more metadata for a production environment."""
    page_id: int
    title: str
    text: str
    tables: List[Dict[str, List[List[str]]]]
    score: float
    semantic_score: float
    keyword_score: float
    position: int
    link: str
    last_updated: str
    chunk_id: str
    page_hierarchy: Optional[List[str]] = None