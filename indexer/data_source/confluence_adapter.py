from typing import List, Dict, Optional
import asyncio
import aiohttp

from config.logging_config import logger
from indexer.data_source.data_source_adapter import DataSourceAdapter, PageContent


class ConfluenceAdapter(DataSourceAdapter):
    """Confluence-specific implementation"""

    def get_headers(self) -> Dict[str, str]:
        return {"Authorization": f"Bearer {self.api_token}"}

    async def fetch_children(self, session: aiohttp.ClientSession, page_id: str) -> List[Dict]:
        """Fetch child pages from Confluence API"""
        url = f"{self.base_url}/rest/api/content/{page_id}/child/page"
        params = {
            "limit": 100,
            "expand": "body.storage,version,ancestors,space,metadata.labels"
        }

        children = await self._make_request(session, url, params=params)
        return children.get("results", []) if children else []

    async def fetch_page_content(self, session: aiohttp.ClientSession, page_data: Dict) -> PageContent:
        """Extract content from Confluence page data"""
        page_id = str(page_data["id"])
        title = page_data["title"]
        body = page_data.get("body", {}).get("storage", {}).get("value", "")
        last_updated = page_data["version"]["when"]

        space_uri = page_data.get("_expandable", {}).get("container", "")
        space_key = space_uri.split("/")[-1] if space_uri else "unknown"
        space_name = await self._fetch_space_name(session, space_key, space_uri)

        link = f"{self.base_url}/spaces/{space_key}/pages/{page_id}"
        hierarchy = self._build_hierarchy(page_data)

        return PageContent(
            page_id=page_id,
            title=title,
            body=body,
            last_updated=last_updated,
            space_key=space_key,
            space_name=space_name,
            link=link,
            hierarchy=hierarchy,
            metadata=page_data.get("metadata", {})
        )

    async def _fetch_space_name(self, session: aiohttp.ClientSession, space_key: str, space_uri: str) -> str:
        """Fetch space name with caching"""
        if space_key in self._cache:
            return self._cache[space_key]

        url = f"{self.base_url}{space_uri}"
        data = await self._make_request(session, url)
        space_name = data.get("name", space_key) if data else space_key
        self._cache[space_key] = space_name
        return space_name

    @staticmethod
    def _build_hierarchy(page_data: Dict) -> List[str]:
        """Build page hierarchy from ancestors"""
        hierarchy = []
        if 'ancestors' in page_data:
            for ancestor in page_data['ancestors']:
                hierarchy.append(ancestor['title'])
        hierarchy.append(page_data['title'])
        return hierarchy

    async def _make_request(self, session: aiohttp.ClientSession, url: str,
                            params: Dict = None, max_retries: int = 5) -> Optional[Dict]:
        """Make HTTP request with retry logic and throttling handling"""
        headers = self.get_headers()

        for attempt in range(max_retries):
            try:
                logger.debug(f"Requesting {url}, attempt {attempt+1}")
                async with session.get(url, params=params, headers=headers, timeout=30) as response:
                    # Handle rate limiting
                    if response.status == 429:
                        retry_after = response.headers.get('X-Retry-After') or response.headers.get('Retry-After')

                        if retry_after:
                            wait_time = int(retry_after)
                        else:
                            # Exponential backoff if no retry-after header
                            wait_time = min(2 ** attempt, 60)

                        logger.warning(
                            f"Rate limited. Waiting {wait_time}s before retry (attempt {attempt + 1}/{max_retries})")
                        await asyncio.sleep(wait_time)
                        continue

                    # Handle other errors
                    if response.status >= 500:
                        wait_time = min(2 ** attempt, 30)
                        logger.warning(
                            f"Server error {response.status}. Retrying in {wait_time}s (attempt {attempt + 1}/{max_retries})")
                        await asyncio.sleep(wait_time)
                        continue

                    response.raise_for_status()
                    return await response.json()

            except asyncio.TimeoutError:
                wait_time = min(2 ** attempt, 30)
                logger.warning(f"Request timeout. Retrying in {wait_time}s (attempt {attempt + 1}/{max_retries})")
                await asyncio.sleep(wait_time)
                continue
            except Exception as e:
                logger.error(f"Error making request to {url}: {e}")
                if attempt == max_retries - 1:
                    return None
                await asyncio.sleep(2 ** attempt)

        logger.error(f"Max retries exceeded for {url}")
        return None

