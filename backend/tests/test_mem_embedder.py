from __future__ import annotations

import unittest
from unittest.mock import AsyncMock, patch

from mem.embedder import MemEmbedder


class MemEmbedderTests(unittest.IsolatedAsyncioTestCase):
    async def test_remote_request_includes_configured_dimensions(self) -> None:
        response = AsyncMock()
        response.raise_for_status = lambda: None
        response.json = lambda: {"data": [{"embedding": [0.1, 0.2]}]}
        client = AsyncMock()
        client.post.return_value = response
        context = AsyncMock()
        context.__aenter__.return_value = client

        embedder = MemEmbedder(
            provider="openai_compatible",
            model="text-embedding-v4",
            api_key="test-key",
            base_url="https://example.test/v1",
            dimensions=1536,
        )
        with patch("mem.embedder.httpx.AsyncClient", return_value=context):
            await embedder.embed_query("测试")

        body = client.post.await_args.kwargs["json"]
        self.assertEqual(body["dimensions"], 1536)


if __name__ == "__main__":
    unittest.main()
