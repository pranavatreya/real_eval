from typing import Dict, Any, Tuple
import json
import os

from sqlitedict import SqliteDict
from openai import OpenAI, OpenAIError


class OpenAIClient:
    def __init__(self, api_key: str, cache_dir: str):
        self._client = OpenAI(api_key=api_key)

        assert os.path.isdir(cache_dir)
        cache_path: str = os.path.join(cache_dir, "openai.sqlite")
        self._cache = SqliteDict(cache_path, autocommit=True)

    def _make_cache_key(self, request: Dict[str, Any]) -> str:
        """Create a stable, hashable cache key from the request."""
        return json.dumps(request, sort_keys=True)

    def run_inference(self, model: str, messages: list[Dict[str, str]], **kwargs) -> Tuple[Dict[str, Any], bool]:
        """Make a chat request with caching."""
        request = {
            "model": model,
            "messages": messages,
            **kwargs,
        }
        key = self._make_cache_key(request)

        if key in self._cache:
            return self._cache[key], True

        # Try OpenAI call; do NOT cache errors
        try:
            response = self._client.chat.completions.create(**request).model_dump(mode="json")
        except OpenAIError as e:
            raise RuntimeError(f"OpenAI request failed: {e}")

        self._cache[key] = response
        return response, False
