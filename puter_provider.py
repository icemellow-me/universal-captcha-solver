# -*- coding: utf-8 -*-
"""
PuterProvider - Puter.js REST API implementation for hcaptcha-challenger.

Replaces GeminiProvider with Puter's free AI API (z-ai/glm models).
Uses the /drivers/call endpoint with the user's auth token.
"""
import asyncio
import base64
import json
from pathlib import Path
from typing import List, Type, TypeVar

import aiohttp
from loguru import logger
from pydantic import BaseModel

ResponseT = TypeVar("ResponseT", bound=BaseModel)

# Default models mapping (Puter free models)
DEFAULT_CHAT_MODEL = "z-ai/glm-5.2"
DEFAULT_VISION_MODEL = "z-ai/glm-4.6v-flash"


def extract_first_json_block(text: str) -> dict | None:
    """Extract the first JSON code block from text."""
    import re
    pattern = r"```json\s*([\s\S]*?)```"
    matches = re.findall(pattern, text)
    if matches:
        try:
            return json.loads(matches[0])
        except json.JSONDecodeError:
            pass
    # Also try raw JSON parse
    try:
        return json.loads(text)
    except (json.JSONDecodeError, ValueError):
        pass
    return None


class PuterProvider:
    """
    Puter-based chat provider implementation.

    Uses Puter's /drivers/call REST API endpoint to access free AI models
    (GLM 5.2 for text, GLM 4.6V Flash for vision) without any API key costs.
    """

    def __init__(self, auth_token: str, model: str = DEFAULT_VISION_MODEL):
        """
        Initialize the Puter provider.

        Args:
            auth_token: Puter JWT auth token.
            model: Model name to use (e.g., "z-ai/glm-4.6v-flash").
        """
        self._auth_token = auth_token
        self._model = model
        self._api_origin = "https://api.puter.com"
        self._session: aiohttp.ClientSession | None = None
        self._last_response: dict | None = None

    @property
    def last_response(self) -> dict | None:
        """Get the last response for debugging."""
        return self._last_response

    async def _get_session(self) -> aiohttp.ClientSession:
        """Lazy-initialize the aiohttp session."""
        if self._session is None or self._session.closed:
            self._session = aiohttp.ClientSession(
                timeout=aiohttp.ClientTimeout(total=60)
            )
        return self._session

    async def _encode_images(self, images: List[Path]) -> List[str]:
        """Encode image files as base64 data URIs."""
        data_uris = []
        for img_path in images:
            path = Path(img_path)
            if not path.exists():
                logger.warning(f"Image not found: {path}")
                continue
            suffix = path.suffix.lower().lstrip('.')
            mime_map = {"png": "image/png", "jpg": "image/jpeg", "jpeg": "image/jpeg", "webp": "image/webp", "gif": "image/gif"}
            mime = mime_map.get(suffix, "image/png")
            b64 = base64.b64encode(path.read_bytes()).decode()
            data_uris.append(f"data:{mime};base64,{b64}")
        return data_uris

    async def generate_with_images(
        self,
        *,
        images: List[Path],
        response_schema: Type[ResponseT],
        user_prompt: str | None = None,
        description: str | None = None,
        **kwargs,
    ) -> ResponseT:
        """
        Generate content with image inputs via Puter API.

        Args:
            images: List of image file paths.
            response_schema: Pydantic model class for structured output.
            user_prompt: User-provided prompt/instructions.
            description: System instruction/description for the model.
            **kwargs: Additional options.

        Returns:
            Parsed response matching the response_schema type.
        """
        # Build messages
        content_parts = []

        # Build a concrete example instead of abstract field descriptions
        # This prevents the model from echoing the schema back
        schema = response_schema.model_json_schema()
        properties = schema.get("properties", {})
        
        # Build example values based on field types
        example_fields = {}
        for fname, finfo in properties.items():
            ftype = finfo.get("type", "string")
            enum_vals = finfo.get("enum")
            if enum_vals:
                example_fields[fname] = enum_vals[0]
            elif ftype == "string":
                example_fields[fname] = "example_value"
            elif ftype == "integer" or ftype == "number":
                example_fields[fname] = 1
            elif ftype == "boolean":
                example_fields[fname] = True
            else:
                example_fields[fname] = "value"

        example_json = json.dumps(example_fields, ensure_ascii=False)
        json_instruction = (
            f"\n\nIMPORTANT: You MUST respond with ONLY a valid JSON object. No markdown, no code blocks, no explanation. "
            f"\nThe JSON object must have exactly these fields: {list(properties.keys())}"
            f"\nExample format: {example_json}"
            f"\nNow give your actual answer in this exact JSON format:"
        )
        full_prompt = (user_prompt or "") + json_instruction

        content_parts.append({"type": "text", "text": full_prompt})

        # Add images as base64 data URIs
        data_uris = await self._encode_images(images)
        for uri in data_uris:
            content_parts.append({
                "type": "image_url",
                "image_url": {"url": uri}
            })

        messages = []

        # System message (description)
        if description:
            messages.append({
                "role": "system",
                "content": description + "\n\nYou MUST respond with valid JSON only. No markdown, no explanation, just the JSON object."
            })

        # User message with images + prompt
        messages.append({
            "role": "user",
            "content": content_parts
        })

        # Call Puter API
        payload = {
            "interface": "puter-chat-completion",
            "driver": "ai-chat",
            "method": "complete",
            "args": {
                "messages": messages,
                "model": self._model,
                "temperature": kwargs.get("temperature", 0.1),
                "max_tokens": kwargs.get("max_tokens", 4096),
            },
            "auth_token": self._auth_token,
        }

        session = await self._get_session()
        url = f"{self._api_origin}/drivers/call"

        for attempt in range(3):
            try:
                async with session.post(
                    url,
                    json=payload,
                    headers={"Content-Type": "text/plain;actually=json"},
                ) as resp:
                    if resp.status != 200:
                        error_text = await resp.text()
                        logger.warning(f"Puter API error {resp.status}: {error_text[:300]}")
                        if attempt < 2:
                            await asyncio.sleep(3)
                            continue
                        raise ValueError(f"Puter API returned {resp.status}: {error_text[:300]}")

                    data = await resp.json()

                if not data.get("success"):
                    raise ValueError(f"Puter API failed: {data}")

                self._last_response = data
                result = data.get("result", {})
                message = result.get("message", {})
                text = message.get("content", "")

                if not text:
                    raise ValueError(f"Empty response from Puter: {data}")

                # Try to parse JSON from response
                json_data = extract_first_json_block(text)
                if json_data:
                    # Detect schema-echo: model returned its own schema definition instead of data
                    # This happens when the model copies the example/schema structure literally
                    if isinstance(json_data, dict) and "properties" in json_data and "required" in json_data:
                        # But check if the model put the actual answer INSIDE properties
                        inner = json_data.get("properties", {})
                        if isinstance(inner, dict) and "answer" in inner and not isinstance(inner["answer"], dict):
                            # The model gave us the answer nested inside properties - extract it
                            logger.info(f"Puter schema-echo with answer inside properties, extracting: {inner}")
                            return response_schema(**inner)
                        logger.warning(f"Puter returned pure schema-echo (attempt {attempt+1}/3), retrying...")
                        if attempt < 2:
                            await asyncio.sleep(1)
                            continue
                    return response_schema(**json_data)

                # Fallback: try direct parse
                try:
                    parsed = json.loads(text)
                    # Same schema-echo check
                    if isinstance(parsed, dict) and "properties" in parsed and "required" in parsed:
                        # Check if answer is inside properties (same as above)
                        inner = parsed.get("properties", {})
                        if isinstance(inner, dict) and "answer" in inner and not isinstance(inner["answer"], dict):
                            logger.info(f"Puter schema-echo (fallback) with answer inside properties, extracting: {inner}")
                            return response_schema(**inner)
                        logger.warning(f"Puter returned pure schema-echo fallback (attempt {attempt+1}/3), retrying...")
                        if attempt < 2:
                            await asyncio.sleep(1)
                            continue
                    return response_schema(**parsed)
                except (json.JSONDecodeError, Exception):
                    pass

                raise ValueError(f"Failed to parse Puter response as {response_schema.__name__}: {text[:500]}")

            except (aiohttp.ClientError, asyncio.TimeoutError) as e:
                logger.warning(f"Puter request attempt {attempt+1}/3 failed: {e}")
                if attempt < 2:
                    await asyncio.sleep(3)
                else:
                    raise

    async def close(self):
        """Close the aiohttp session."""
        if self._session and not self._session.closed:
            await self._session.close()

    def cache_response(self, path: Path) -> None:
        """Cache the last response to a file."""
        if not self._last_response:
            return
        try:
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_text(
                json.dumps(self._last_response, indent=2, ensure_ascii=False),
                encoding="utf-8",
            )
        except Exception as e:
            logger.warning(f"Failed to cache response: {e}")


class NvidiaProvider:
    """
    NVIDIA NIM API provider as fallback.

    Uses NVIDIA's free inference endpoints for vision models.
    """

    def __init__(self, api_key: str, model: str = "meta/llama-3.2-90b-vision-instruct"):
        self._api_key = api_key
        self._model = model
        self._api_origin = "https://integrate.api.nvidia.com"
        self._session: aiohttp.ClientSession | None = None
        self._last_response: dict | None = None

    @property
    def last_response(self) -> dict | None:
        return self._last_response

    async def _get_session(self) -> aiohttp.ClientSession:
        if self._session is None or self._session.closed:
            self._session = aiohttp.ClientSession(
                timeout=aiohttp.ClientTimeout(total=60)
            )
        return self._session

    async def _encode_images(self, images: List[Path]) -> List[str]:
        data_uris = []
        for img_path in images:
            path = Path(img_path)
            if not path.exists():
                continue
            b64 = base64.b64encode(path.read_bytes()).decode()
            suffix = path.suffix.lower().lstrip('.')
            mime_map = {"png": "image/png", "jpg": "image/jpeg", "jpeg": "image/jpeg", "webp": "image/webp"}
            mime = mime_map.get(suffix, "image/png")
            data_uris.append(f"data:{mime};base64,{b64}")
        return data_uris

    async def generate_with_images(
        self,
        *,
        images: List[Path],
        response_schema: Type[ResponseT],
        user_prompt: str | None = None,
        description: str | None = None,
        **kwargs,
    ) -> ResponseT:
        """Generate content with images via NVIDIA NIM API."""
        content_parts = []
        # Build concrete example (same as PuterProvider)
        schema = response_schema.model_json_schema()
        properties = schema.get("properties", {})
        example_fields = {}
        for fname, finfo in properties.items():
            ftype = finfo.get("type", "string")
            enum_vals = finfo.get("enum")
            if enum_vals:
                example_fields[fname] = enum_vals[0]
            elif ftype == "string":
                example_fields[fname] = "example_value"
            elif ftype == "integer" or ftype == "number":
                example_fields[fname] = 1
            elif ftype == "boolean":
                example_fields[fname] = True
            else:
                example_fields[fname] = "value"
        example_json = json.dumps(example_fields, ensure_ascii=False)
        json_instruction = (
            f"\n\nIMPORTANT: You MUST respond with ONLY a valid JSON object. No markdown, no code blocks, no explanation. "
            f"\nThe JSON object must have exactly these fields: {list(properties.keys())}"
            f"\nExample format: {example_json}"
            f"\nNow give your actual answer in this exact JSON format:"
        )
        full_prompt = (user_prompt or "") + json_instruction
        content_parts.append({"type": "text", "text": full_prompt})

        data_uris = await self._encode_images(images)
        for uri in data_uris:
            content_parts.append({"type": "image_url", "image_url": {"url": uri}})

        messages = []
        if description:
            messages.append({"role": "system", "content": description + "\n\nRespond with valid JSON only."})
        messages.append({"role": "user", "content": content_parts})

        payload = {
            "model": self._model,
            "messages": messages,
            "temperature": kwargs.get("temperature", 0.1),
            "max_tokens": kwargs.get("max_tokens", 4096),
        }

        session = await self._get_session()
        url = f"{self._api_origin}/v1/chat/completions"

        for attempt in range(3):
            try:
                async with session.post(
                    url,
                    json=payload,
                    headers={
                        "Content-Type": "application/json",
                        "Authorization": f"Bearer {self._api_key}",
                        "Accept": "application/json",
                    },
                ) as resp:
                    if resp.status != 200:
                        error_text = await resp.text()
                        logger.warning(f"NVIDIA API error {resp.status}: {error_text[:300]}")
                        if attempt < 2:
                            await asyncio.sleep(3)
                            continue
                        raise ValueError(f"NVIDIA API returned {resp.status}: {error_text[:300]}")

                    data = await resp.json()

                self._last_response = data
                text = data.get("choices", [{}])[0].get("message", {}).get("content", "")

                if not text:
                    raise ValueError(f"Empty response from NVIDIA: {data}")

                json_data = extract_first_json_block(text)
                if json_data:
                    return response_schema(**json_data)

                try:
                    return response_schema(**json.loads(text))
                except (json.JSONDecodeError, Exception):
                    pass

                raise ValueError(f"Failed to parse NVIDIA response as {response_schema.__name__}: {text[:500]}")

            except (aiohttp.ClientError, asyncio.TimeoutError) as e:
                logger.warning(f"NVIDIA request attempt {attempt+1}/3 failed: {e}")
                if attempt < 2:
                    await asyncio.sleep(3)
                else:
                    raise

    async def close(self):
        if self._session and not self._session.closed:
            await self._session.close()

    def cache_response(self, path: Path) -> None:
        if not self._last_response:
            return
        try:
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_text(
                json.dumps(self._last_response, indent=2, ensure_ascii=False),
                encoding="utf-8",
            )
        except Exception as e:
            logger.warning(f"Failed to cache response: {e}")


def create_provider(
    *,
    puter_token: str | None = None,
    nvidia_api_key: str | None = None,
    model: str = "auto",
    vision_model: str = "auto",
) -> PuterProvider | NvidiaProvider:
    """
    Factory function to create the best available provider.

    Priority: Puter (free, unlimited) > NVIDIA (free tier with key)
    """
    # Auto-select models
    if vision_model == "auto":
        puter_vision = "z-ai/glm-4.6v-flash"
        nvidia_vision = "meta/llama-3.2-90b-vision-instruct"
    else:
        puter_vision = vision_model
        nvidia_vision = vision_model

    if puter_token:
        logger.info(f"Using PuterProvider with model: {puter_vision}")
        return PuterProvider(auth_token=puter_token, model=puter_vision)

    if nvidia_api_key:
        logger.info(f"Using NvidiaProvider with model: {nvidia_vision}")
        return NvidiaProvider(api_key=nvidia_api_key, model=nvidia_vision)

    raise ValueError("No Puter token or NVIDIA API key provided. Set PUTER_TOKEN or NVIDIA_API_KEY environment variable.")
