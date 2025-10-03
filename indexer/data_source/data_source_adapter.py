from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import List, Dict
import aiohttp

@dataclass
class PageContent:
    """Standardized page content structure"""
    page_id: str
    title: str
    body: str
    last_updated: str
    space_key: str
    space_name: str
    link: str
    hierarchy: List[str]
    metadata: Dict = None

class DataSourceAdapter(ABC):
    """
    Abstract base class for data source adapters.
    Implement this to support different platforms (Confluence, Notion, etc.)
    """

    def __init__(self, base_url: str, api_token: str):
        self.base_url = base_url
        self.api_token = api_token
        self._cache = {}

    @abstractmethod
    async def fetch_children(self, session: aiohttp.ClientSession, page_id: str) -> List[Dict]:
        """Fetch child pages for a given page ID"""
        pass

    @abstractmethod
    async def fetch_page_content(self, session: aiohttp.ClientSession, page_data: Dict) -> PageContent:
        """Extract and normalize page content"""
        pass

    @abstractmethod
    def get_headers(self) -> Dict[str, str]:
        """Return authentication headers"""
        pass