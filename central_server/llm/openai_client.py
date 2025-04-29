from typing import Dict, Any, List, Tuple, Type
from base64 import b64encode
import json
import os

from sqlitedict import SqliteDict
from openai import OpenAI, OpenAIError
from pydantic import BaseModel


class OpenAIClient:
    def __init__(self, api_key: str, cache_dir: str):
        self._client = OpenAI(api_key=api_key)

        assert os.path.isdir(cache_dir)
        cache_path: str = os.path.join(cache_dir, "openai.sqlite")
        self._cache = SqliteDict(cache_path, autocommit=True)

    def _make_cache_key(self, request: Dict[str, Any]) -> str:
        """Create a stable, hashable cache key from the request."""
        return json.dumps(request, sort_keys=True)

    def run_inference(
        self, model: str, messages: list[Dict[str, str]], **kwargs
    ) -> Tuple[Dict[str, Any], bool]:
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
            response = self._client.chat.completions.create(**request).model_dump(
                mode="json"
            )
        except OpenAIError as e:
            raise RuntimeError(f"OpenAI request failed: {e}")

        self._cache[key] = response
        return response, False

    def run_image_inference(
        self, model: str, image_paths: List[str], text: str, **kwargs
    ) -> Tuple[Dict[str, Any], bool]:
        """Make a multimodal request with multiple images and a text prompt."""
        content_blocks = [{"type": "text", "text": text}]

        for path in image_paths:
            with open(path, "rb") as f:
                encoded_image = b64encode(f.read()).decode("utf-8")

            image_block = {
                "type": "image_url",
                "image_url": {"url": f"data:image/png;base64,{encoded_image}"},
            }
            content_blocks.append(image_block)

        messages = [{"role": "user", "content": content_blocks}]
        return self.run_inference(model=model, messages=messages, **kwargs)

    def run_image_inference_structured(
        self,
        model: str,
        image_paths: List[str],
        text: str,
        expected_structure: Type[BaseModel],
        force_recompute: bool = False,
        **kwargs,
    ) -> Tuple[Any, bool]:
        """Make a multimodal request with a structured outputs"""
        content_blocks = [{"type": "text", "text": text}]
        for path in image_paths:
            with open(path, "rb") as f:
                encoded_image = b64encode(f.read()).decode("utf-8")
            image_block = {
                "type": "image_url",
                "image_url": {"url": f"data:image/png;base64,{encoded_image}"},
            }
            content_blocks.append(image_block)

        messages = [{"role": "user", "content": content_blocks}]

        key = self._make_cache_key(
            {
                "model": model,
                "messages": messages,
                "expected_structure": expected_structure.__name__,
                **kwargs,
            }
        )

        if key in self._cache and not force_recompute:
            raw_response = self._cache[key]
            try:
                parsed = expected_structure(
                    **raw_response["choices"][0]["message"]["parsed"]
                )
                return parsed, True
            except Exception as e:
                print(
                    f"Cached response parsing failed for structured output: {e}. Will recompute fresh."
                )

        try:
            completion = self._client.beta.chat.completions.parse(
                model=model,
                messages=messages,
                response_format=expected_structure,
                **kwargs,
            )
            parsed = completion.choices[0].message.parsed

            # Save to cache after successful fresh call
            self._cache[key] = completion.model_dump(mode="json")
        except OpenAIError as e:
            raise RuntimeError(f"OpenAI structured request failed: {e}")
        except Exception as e:
            raise RuntimeError(f"Fresh OpenAI structured response parsing failed: {e}")

        return parsed, False
