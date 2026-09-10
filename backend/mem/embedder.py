"""向量嵌入封装 — 统一 embed / embed_query 接口

支持:
  - openai / openai_compatible: 远程 OpenAI 兼容 /embeddings API
  - local: sentence-transformers 本地模型 (可选依赖)
"""

from __future__ import annotations

import logging
import os
import re
from typing import Any

import httpx

logger = logging.getLogger(__name__)

_HAS_SENTENCE_TRANSFORMERS = False
try:
    from sentence_transformers import SentenceTransformer  # type: ignore

    _HAS_SENTENCE_TRANSFORMERS = True
except ImportError:
    pass

_ENV_VAR_RE = re.compile(r"^\$\{([A-Za-z_][A-Za-z0-9_]*)\}$")


def _resolve_env(value: str) -> str:
    m = _ENV_VAR_RE.match(value.strip())
    if m:
        return os.getenv(m.group(1), "")
    return value


class MemEmbedder:
    """Embedding 统一接口，按 docs/memory-system-refactor.md §9 阶段一。"""

    def __init__(
        self,
        provider: str = "openai",
        model: str = "text-embedding-3-small",
        api_key: str = "",
        dimensions: int = 1536,
        base_url: str = "",
        batch_size: int = 32,
        timeout: float = 60.0,
    ):
        self.provider = provider.lower().strip()
        self.model = model.strip()
        self.api_key = _resolve_env(api_key) if api_key else ""
        self.dimensions = dimensions
        self.base_url = (base_url or "").rstrip("/")
        self.batch_size = batch_size
        self.timeout = timeout

        self._local_model: Any = None

        if self.provider == "local":
            self._init_local()
        else:
            if not self.api_key:
                self.api_key = os.getenv("OPENAI_API_KEY", "")
            if not self.base_url:
                self.base_url = "https://api.openai.com/v1"

        logger.info(
            "MemEmbedder initialized: provider=%s model=%s dim=%d",
            self.provider,
            self.model,
            self.dimensions,
        )

    def _init_local(self) -> None:
        if not _HAS_SENTENCE_TRANSFORMERS:
            raise RuntimeError(
                "provider='local' requires sentence-transformers: "
                "pip install sentence-transformers"
            )
        model_name = self.model or "paraphrase-multilingual-MiniLM-L12-v2"
        self._local_model = SentenceTransformer(model_name)
        self.dimensions = self._local_model.get_sentence_embedding_dimension()
        logger.info("Local embedding model loaded: %s (dim=%d)", model_name, self.dimensions)

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    async def embed(self, texts: list[str]) -> list[list[float]]:
        """Embed a batch of texts. Returns one vector per text."""
        if not texts:
            return []

        results: list[list[float]] = []
        for i in range(0, len(texts), self.batch_size):
            batch = texts[i : i + self.batch_size]
            vecs = await self._embed_batch(batch)
            results.extend(vecs)
        return results

    async def embed_query(self, text: str) -> list[float]:
        """Embed a single query string."""
        vecs = await self._embed_batch([text])
        return vecs[0]

    # ------------------------------------------------------------------
    # Internal
    # ------------------------------------------------------------------

    async def _embed_batch(self, texts: list[str]) -> list[list[float]]:
        if self.provider == "local":
            return self._embed_local(texts)
        return await self._embed_remote(texts)

    def _embed_local(self, texts: list[str]) -> list[list[float]]:
        if self._local_model is None:
            raise RuntimeError("Local embedding model not initialized")
        arr = self._local_model.encode(texts, normalize_embeddings=True)
        if hasattr(arr, "tolist"):
            return [arr[i].tolist() for i in range(len(texts))]
        return [list(v) for v in arr]

    async def _embed_remote(self, texts: list[str]) -> list[list[float]]:
        endpoint = self._normalize_endpoint(self.base_url)
        headers = {
            "Content-Type": "application/json",
            "Authorization": f"Bearer {self.api_key}",
        }
        body: dict[str, Any] = {
            "model": self.model,
            "input": texts[0] if len(texts) == 1 else texts,
            "dimensions": self.dimensions,
        }

        async with httpx.AsyncClient(timeout=self.timeout) as client:
            resp = await client.post(endpoint, json=body, headers=headers)
            resp.raise_for_status()
            data = resp.json()

        out: list[list[float]] = []
        for item in data.get("data", []):
            emb = item.get("embedding")
            if emb is not None:
                out.append(list(emb))
        if len(out) != len(texts):
            logger.warning(
                "Embedding count mismatch: expected %d, got %d", len(texts), len(out)
            )
        return out

    @staticmethod
    def _normalize_endpoint(base_url: str) -> str:
        stripped = base_url.rstrip("/")
        if stripped.endswith("/embeddings"):
            return stripped
        return f"{stripped}/embeddings"

    # ------------------------------------------------------------------
    # Factory
    # ------------------------------------------------------------------

    @classmethod
    def from_config(cls, config: dict[str, Any]) -> MemEmbedder:
        """Build from the `mem.embedding` config dict."""
        return cls(
            provider=config.get("provider", "openai"),
            model=config.get("model", "text-embedding-3-small"),
            api_key=config.get("api_key", ""),
            dimensions=config.get("dimensions", 1536),
            base_url=config.get("base_url", ""),
            batch_size=config.get("batch_size", 32),
            timeout=config.get("timeout", 60.0),
        )
