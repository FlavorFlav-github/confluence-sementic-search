from typing import List, Dict, Optional
import asyncio
import aiohttp
from datetime import datetime

from config.logging_config import logger
from indexer.data_source.data_source_adapter import DataSourceAdapter, PageContent


class NotionAdapter(DataSourceAdapter):
    """Notion-specific implementation"""

    def get_headers(self) -> Dict[str, str]:
        return {
            "Authorization": f"Bearer {self.api_token}",
            "Notion-Version": "2022-06-28",
            "Content-Type": "application/json"
        }

    async def fetch_children(self, session: aiohttp.ClientSession, page_id: str) -> List[Dict]:
        """Fetch child pages from Notion API"""
        url = f"{self.base_url}/v1/blocks/{page_id}/children"
        params = {"page_size": 100}

        data = await self._make_request(session, url, params=params)
        if not data:
            return []

        # Filter for child pages only
        children = []
        for block in data.get("results", []):
            if block.get("type") == "child_page":
                # Fetch full page details
                page_data = await self._fetch_page_details(session, block["id"])
                if page_data:
                    children.append(page_data)

        return children

    async def _fetch_page_details(self, session: aiohttp.ClientSession, page_id: str) -> Optional[Dict]:
        """Fetch full page details including content"""
        url = f"{self.base_url}/v1/pages/{page_id}"
        return await self._make_request(session, url)

    async def fetch_page_content(self, session: aiohttp.ClientSession, page_data: Dict) -> PageContent:
        """Extract content from Notion page data"""
        page_id = page_data["id"]

        # Extract title from properties
        title_prop = page_data.get("properties", {}).get("title", {})
        title = "".join([t.get("plain_text", "") for t in title_prop.get("title", [])])

        # Fetch page content blocks
        body = await self._fetch_page_blocks(session, page_id)

        last_updated = page_data.get("last_edited_time", datetime.utcnow().isoformat())

        # Notion doesn't have spaces, use parent as space
        parent = page_data.get("parent", {})
        space_key = parent.get("database_id") or parent.get("page_id") or "root"
        space_name = self._cache.get(space_key, "Notion Workspace")

        link = page_data.get("url", f"https://notion.so/{page_id.replace('-', '')}")

        return PageContent(
            page_id=page_id,
            title=title or "Untitled",
            body=body,
            last_updated=last_updated,
            space_key=space_key,
            space_name=space_name,
            link=link,
            hierarchy=[title or "Untitled"],
            metadata=page_data.get("properties", {})
        )

    async def _fetch_page_blocks(self, session: aiohttp.ClientSession, page_id: str) -> str:
        """Fetch and convert page blocks to text"""
        url = f"{self.base_url}/v1/blocks/{page_id}/children"
        data = await self._make_request(session, url)

        if not data:
            return ""

        text_parts = []
        for block in data.get("results", []):
            block_text = self._extract_block_text(block)
            if block_text:
                text_parts.append(block_text)

        return "\n\n".join(text_parts)

    @staticmethod
    def _extract_block_text(block: Dict) -> str:
        """Extract text from a Notion block"""
        block_type = block.get("type")
        if not block_type:
            return ""

        block_data = block.get(block_type, {})
        rich_text = block_data.get("rich_text", [])

        return "".join([t.get("plain_text", "") for t in rich_text])

    async def _make_request(self, session: aiohttp.ClientSession, url: str,
                            params: Dict = None, max_retries: int = 5) -> Optional[Dict]:
        """Make HTTP request with retry logic and throttling handling"""
        headers = self.get_headers()

        for attempt in range(max_retries):
            try:
                async with session.get(url, params=params, headers=headers, timeout=30) as response:
                    # Handle rate limiting
                    if response.status == 429:
                        retry_after = response.headers.get('Retry-After')

                        if retry_after:
                            wait_time = int(retry_after)
                        else:
                            wait_time = min(2 ** attempt, 60)

                        logger.warning(
                            f"Notion rate limited. Waiting {wait_time}s (attempt {attempt + 1}/{max_retries})")
                        await asyncio.sleep(wait_time)
                        continue

                    if response.status >= 500:
                        wait_time = min(2 ** attempt, 30)
                        logger.warning(f"Server error {response.status}. Retrying in {wait_time}s")
                        await asyncio.sleep(wait_time)
                        continue

                    response.raise_for_status()
                    return await response.json()

            except asyncio.TimeoutError:
                wait_time = min(2 ** attempt, 30)
                logger.warning(f"Request timeout. Retrying in {wait_time}s")
                await asyncio.sleep(wait_time)
            except Exception as e:
                logger.error(f"Error making request: {e}")
                if attempt == max_retries - 1:
                    return None
                await asyncio.sleep(2 ** attempt)

        return None