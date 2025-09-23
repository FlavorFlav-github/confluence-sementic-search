"""
Client for interacting with the Confluence API.
"""
import logging
import requests
from typing import Dict, List
from config import settings

logger = logging.getLogger(__name__)

class ConfluenceClient:
    """
    A client to fetch data from the Confluence API.

    Handles authentication, request headers, and error handling.
    """
    def __init__(self, base_url: str, api_token: str):
        self.base_url = base_url
        self.headers = {"Authorization": f"Bearer {api_token}"}

    def fetch_children(self, page_id: int, limit: int = 100) -> List[Dict]:
        """
        Fetches child pages for a given page ID with expanded details.

        Args:
            page_id: The ID of the parent page.
            limit: The maximum number of child pages to return.

        Returns:
            A list of dictionaries, where each dictionary represents a page.
        """
        url = f"{self.base_url}/content/{page_id}/child/page"
        params = {
            "limit": limit,
            "expand": "body.storage,version,ancestors,space,metadata.labels"
        }
        try:
            response = requests.get(url, headers=self.headers, params=params, timeout=30)
            response.raise_for_status()  # Raise HTTPError for bad responses
            return response.json().get("results", [])
        except requests.exceptions.RequestException as e:
            logger.error(f"Error fetching children for page {page_id}: {e}")
            return []

def build_page_hierarchy(page_data: Dict) -> List[str]:
    """
    Constructs a human-readable hierarchical path for a Confluence page.

    Args:
        page_data: The dictionary containing page information, including ancestors.

    Returns:
        A list of strings representing the page's hierarchical path.
    """
    hierarchy = [ancestor['title'] for ancestor in page_data.get('ancestors', [])]
    hierarchy.append(page_data.get('title', 'N/A'))
    return hierarchy