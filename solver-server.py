#!/usr/bin/env python3
"""
Universal Captcha Solver Server
2captcha-compatible API supporting:
- Image/OCR captchas (ddddocr + Tesseract)
- hCaptcha (hcaptcha-challenger ONNX + LLM)
- Cloudflare Turnstile (headless browser)
- reCAPTCHA v2 (CaptchaPlugin extension)
- Coordinate captchas (click targets)
- Base64 image captchas

API endpoints:
  POST /in.php    - 2captcha-compatible submit
  GET  /res.php   - 2captcha-compatible poll
  GET  /health    - Health check
  POST /solve     - Direct solve (JSON in/out)
"""

import asyncio
import base64
import hashlib
import io
import json
import logging
import os
import re
import sys
import tempfile
import time
import urllib.error
import urllib.request
import uuid
from dataclasses import dataclass, field
from enum import Enum
from typing import Dict, Optional

import ddddocr
import pytesseract
from PIL import Image
from aiohttp import web

# ──────────────────────────────────────────────
# Logging
# ──────────────────────────────────────────────
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
log = logging.getLogger("universal-captcha-solver")

# ──────────────────────────────────────────────
# Captcha type enum
# ──────────────────────────────────────────────
class CaptchaType(str, Enum):
    IMAGE_OCR = "image"             # Generic image/OCR captcha
    HCAPTCHA = "hcaptcha"          # hCaptcha
    RECAPTCHA_V2 = "userrecaptcha" # reCAPTCHA v2
    TURNSTILE = "turnstile"        # Cloudflare Turnstile
    XCAPTCHA = "wcaptcha"          # xCaptcha widget
    COORD = "coord"                # Coordinate captcha
    BASE64 = "base64"              # Base64-encoded image

# ──────────────────────────────────────────────
# Task model
# ──────────────────────────────────────────────
@dataclass
class CaptchaTask:
    task_id: str
    captcha_type: CaptchaType = CaptchaType.IMAGE_OCR
    status: str = "pending"  # pending, processing, solved, failed
    token: str = ""
    coordinates: str = ""
    created_at: float = 0.0
    solved_at: float = 0.0
    error: str = ""
    # Input params
    image_data: bytes = b""
    image_base64: str = ""
    sitekey: str = ""
    pageurl: str = ""
    googlekey: str = ""
    version: str = "v2"
    # Extra metadata
    extra: dict = field(default_factory=dict)

# ──────────────────────────────────────────────
# OCR Solvers
# ──────────────────────────────────────────────
class DdddOcrSolver:
    """ddddocr - fast Chinese OCR captcha solver (ONNX-based)"""

    def __init__(self):
        self._ocr = ddddocr.DdddOcr(show_ad=False)
        self._det = ddddocr.DdddOcr(det=True, show_ad=False)
        log.info("ddddocr engine initialized (OCR + detection)")

    def solve_text(self, image_bytes: bytes) -> str:
        """Solve text-based captcha from image bytes"""
        try:
            result = self._ocr.classification(image_bytes)
            return result.strip()
        except Exception as e:
            log.error(f"ddddocr classification error: {e}")
            return ""

    def solve_target(self, image_bytes: bytes, target_bytes: bytes) -> str:
        """Solve target-detection captcha (find target in image)"""
        try:
            result = self._det.slide_comparison(target_bytes, image_bytes)
            return str(result.get("target", [0, 0]))
        except Exception as e:
            log.error(f"ddddocr detection error: {e}")
            return ""

    def solve_slide(self, bg_bytes: bytes, slider_bytes: bytes) -> str:
        """Solve slide captcha - return offset"""
        try:
            result = self._ocr.slide_match(slider_bytes, bg_bytes, simple_target=True)
            if result:
                return str(result.get("target", [0, 0]))
            return ""
        except Exception as e:
            log.error(f"ddddocr slide error: {e}")
            return ""


class TesseractSolver:
    """Tesseract OCR - general purpose text extraction"""

    def __init__(self):
        # Verify tesseract is available
        try:
            pytesseract.get_tesseract_version()
            self._available = True
            log.info("Tesseract OCR engine initialized")
        except Exception:
            self._available = False
            log.warning("Tesseract not available, disabling")

    @property
    def available(self):
        return self._available

    def solve_text(self, image_bytes: bytes, lang: str = "eng") -> str:
        """Solve text captcha with Tesseract"""
        if not self._available:
            return ""
        try:
            img = Image.open(io.BytesIO(image_bytes))
            # Preprocess: convert to grayscale, increase contrast
            img = img.convert("L")
            text = pytesseract.image_to_string(img, lang=lang, config="--psm 7").strip()
            # Clean up common OCR errors
            text = re.sub(r"\s+", "", text)
            return text
        except Exception as e:
            log.error(f"Tesseract error: {e}")
            return ""


class VlmOcrSolver:
    """VLM-based OCR solver — supports Cloudflare Workers AI and NVIDIA APIs.
    
    Tier-3 fallback for when ddddocr and Tesseract both fail.
    Excellent for distorted/noisy captchas, math captchas, and xCaptcha.
    
    Priority: Cloudflare Workers AI (free vision) > NVIDIA (free inference endpoint)
    Uses Python urllib to avoid curl binary dependency.
    """

    def __init__(self, cf_api_token: str = "", cf_account_id: str = "",
                 nvidia_api_key: str = "", model: str = ""):
        self._cf_api_token = cf_api_token or os.environ.get("CF_API_TOKEN", os.environ.get("CLOUDFLARE_WORKERS_AI_TOKEN", ""))
        self._cf_account_id = cf_account_id or os.environ.get("CF_ACCOUNT_ID", os.environ.get("CLOUDFLARE_ACCOUNT_ID", ""))
        self._nvidia_api_key = nvidia_api_key or os.environ.get("NVIDIA_API_KEY", "")
        self._cf_agreed = False
        self._cf_model = "@cf/meta/llama-3.2-11b-vision-instruct"
        self._nvidia_model = "meta/llama-3.2-90b-vision-instruct"
        # Prefer CF when available (free vision)
        if self._cf_api_token and self._cf_account_id:
            self._provider = "cloudflare"
            self._available = True
            self._model = self._cf_model
            log.info(f"VLM-OCR engine initialized (Cloudflare {self._cf_model})")
        elif self._nvidia_api_key:
            self._provider = "nvidia"
            self._available = True
            self._model = self._nvidia_model
            log.info(f"VLM-OCR engine initialized (NVIDIA {self._nvidia_model})")
        else:
            self._provider = ""
            self._available = False
            log.warning("VLM-OCR disabled: needs CF_API_TOKEN+CF_ACCOUNT_ID or NVIDIA_API_KEY")

    @property
    def available(self):
        return self._available

    async def solve_text(self, image_bytes: bytes) -> str:
        """Solve captcha image via VLM vision API.
        
        Sends the image to the vision model with an OCR-optimized prompt.
        Returns extracted text or empty string on failure.
        """
        if not self._available:
            return ""

        if self._provider == "cloudflare":
            return await self._solve_cf_text(image_bytes)
        elif self._provider == "nvidia":
            return await self._solve_nvidia_text(image_bytes)
        return ""

    async def _cf_ensure_agreement(self):
        """Agree to CF Workers AI model license terms."""
        if self._cf_agreed:
            return
        url = f"https://api.cloudflare.com/client/v4/accounts/{self._cf_account_id}/ai/run/{self._cf_model}"
        try:
            req = urllib.request.Request(
                url,
                data=json.dumps({"prompt": "agree", "max_tokens": 10}).encode(),
                headers={
                    "Authorization": f"Bearer {self._cf_api_token}",
                    "Content-Type": "application/json",
                    "cf-model-agreement": "true",
                },
                method="POST",
            )
            loop = asyncio.get_event_loop()
            resp = await asyncio.wait_for(
                loop.run_in_executor(None, lambda: urllib.request.urlopen(req, timeout=30)),
                timeout=45,
            )
            data = json.loads(resp.read().decode())
            if data.get("success"):
                self._cf_agreed = True
                log.info("CF Workers AI model license agreed")
            else:
                self._cf_agreed = True
                log.info("CF license agreement submitted")
        except Exception:
            self._cf_agreed = True

    async def _cf_call(self, payload: dict, timeout: int = 60) -> dict:
        """Call CF Workers AI via urllib (works without curl binary)."""
        await self._cf_ensure_agreement()
        url = f"https://api.cloudflare.com/client/v4/accounts/{self._cf_account_id}/ai/run/{self._cf_model}"
        try:
            req = urllib.request.Request(
                url,
                data=json.dumps(payload).encode(),
                headers={
                    "Authorization": f"Bearer {self._cf_api_token}",
                    "Content-Type": "application/json",
                    "cf-model-agreement": "true",
                },
                method="POST",
            )
            loop = asyncio.get_event_loop()
            resp = await asyncio.wait_for(
                loop.run_in_executor(None, lambda: urllib.request.urlopen(req, timeout=timeout)),
                timeout=timeout + 15,
            )
            return json.loads(resp.read().decode())
        except urllib.error.HTTPError as e:
            body = e.read().decode()
            return {"success": False, "errors": [{"message": f"HTTP {e.code}: {body[:200]}"}]}
        except Exception as e:
            return {"success": False, "errors": [{"message": str(e)[:200]}]}

    def _build_ocr_messages(self, data_uri: str) -> list:
        """Build OCR-optimized messages for captcha solving."""
        return [
            {
                "role": "user",
                "content": [
                    {
                        "type": "text",
                        "text": (
                            "You are a CAPTCHA OCR solver. Look at this image carefully. "
                            "It is a CAPTCHA challenge image containing text, numbers, or a math expression. "
                            "Extract ONLY the characters/text shown in the image. "
                            "Rules:\n"
                            "- Output ONLY the raw text, nothing else\n"
                            "- No explanations, no quotes, no formatting\n"
                            "- If it's a math expression, provide the ANSWER (the number)\n"
                            "- Include spaces only if they are clearly part of the text\n"
                            "- Preserve case (uppercase/lowercase) exactly as shown\n"
                            "What text/answer does this CAPTCHA show?"
                        ),
                    },
                    {
                        "type": "image_url",
                        "image_url": {"url": data_uri},
                    },
                ],
            }
        ]

    def _clean_ocr_result(self, text: str) -> str:
        """Clean common model artifacts from OCR result."""
        if not text:
            return ""
        text = text.strip()
        text = re.sub(r'^["`]+|["`]+$', "", text)
        text = re.sub(r"^(Answer|Result|Text|Solution|CAPTCHA)[:\s]*", "", text, flags=re.I)
        return text.strip()

    async def _solve_cf_text(self, image_bytes: bytes) -> str:
        """Solve via Cloudflare Workers AI."""
        b64 = base64.b64encode(image_bytes).decode()
        mime = "image/png"
        if image_bytes[:3] == b"\xff\xd8\xff":
            mime = "image/jpeg"
        elif image_bytes[:4] == b"RIFF" and image_bytes[8:12] == b"WEBP":
            mime = "image/webp"
        elif image_bytes[:4] == b"GIF8":
            mime = "image/gif"

        data_uri = f"data:{mime};base64,{b64}"
        messages = self._build_ocr_messages(data_uri)
        payload = {"messages": messages, "max_tokens": 50}

        try:
            data = await self._cf_call(payload, timeout=60)
            if not data.get("success"):
                errors = data.get("errors", [])
                err = errors[0].get("message", "unknown") if errors else "unknown"
                log.error(f"CF VLM-OCR error: {err[:200]}")
                return ""
            result = data.get("result", {})
            text = result.get("response", "") if isinstance(result, dict) else str(result)
            text = self._clean_ocr_result(text)
            if text:
                log.info(f"CF VLM-OCR result: {text}")
            return text
        except Exception as e:
            log.error(f"CF VLM-OCR error: {e}")
            return ""

    async def _solve_nvidia_text(self, image_bytes: bytes) -> str:
        """Solve via NVIDIA NIM API."""
        import aiohttp

        b64 = base64.b64encode(image_bytes).decode()
        mime = "image/png"
        if image_bytes[:3] == b"\xff\xd8\xff":
            mime = "image/jpeg"
        elif image_bytes[:4] == b"RIFF" and image_bytes[8:12] == b"WEBP":
            mime = "image/webp"
        elif image_bytes[:4] == b"GIF8":
            mime = "image/gif"

        data_uri = f"data:{mime};base64,{b64}"
        messages = self._build_ocr_messages(data_uri)
        payload = {
            "model": self._nvidia_model,
            "messages": messages,
            "temperature": 0.1,
            "max_tokens": 50,
        }

        try:
            async with aiohttp.ClientSession(
                timeout=aiohttp.ClientTimeout(total=30)
            ) as session:
                async with session.post(
                    "https://integrate.api.nvidia.com/v1/chat/completions",
                    json=payload,
                    headers={
                        "Content-Type": "application/json",
                        "Authorization": f"Bearer {self._nvidia_api_key}",
                        "Accept": "application/json",
                    },
                ) as resp:
                    if resp.status != 200:
                        err = await resp.text()
                        log.error(f"NVIDIA VLM-OCR API error {resp.status}: {err[:200]}")
                        return ""
                    data = await resp.json()

            text = data.get("choices", [{}])[0].get("message", {}).get("content", "")
            text = self._clean_ocr_result(text)
            if text:
                log.info(f"NVIDIA VLM-OCR result: {text}")
            return text
        except Exception as e:
            log.error(f"NVIDIA VLM-OCR error: {e}")
            return ""

    async def solve_coord(self, image_bytes: bytes) -> str:
        """Solve coordinate/click captcha via VLM — find targets in the image.
        
        Returns JSON array of [x, y] click coordinates.
        """
        if not self._available:
            return ""

        b64 = base64.b64encode(image_bytes).decode()
        mime = "image/png"
        if image_bytes[:3] == b"\xff\xd8\xff":
            mime = "image/jpeg"

        data_uri = f"data:{mime};base64,{b64}"
        messages = [
            {
                "role": "user",
                "content": [
                    {
                        "type": "text",
                        "text": (
                            "Look at this CAPTCHA grid image. Find all the tiles/images that match "
                            "the requested target. Return ONLY a JSON array of [x, y] coordinates "
                            "representing the center of each matching tile. "
                            "Use pixel coordinates relative to the full image. "
                            "Example: [[120, 85], [280, 250]] or [] if none match. "
                            "Output ONLY the JSON array, nothing else."
                        ),
                    },
                    {
                        "type": "image_url",
                        "image_url": {"url": data_uri},
                    },
                ],
            }
        ]

        try:
            if self._provider == "cloudflare":
                payload = {"messages": messages, "max_tokens": 200}
                data = await self._cf_call(payload, timeout=60)
                if not data.get("success"):
                    return ""
                result = data.get("result", {})
                text = result.get("response", "") if isinstance(result, dict) else str(result)
            else:
                import aiohttp
                payload = {
                    "model": self._nvidia_model,
                    "messages": messages,
                    "temperature": 0.1,
                    "max_tokens": 200,
                }
                async with aiohttp.ClientSession(
                    timeout=aiohttp.ClientTimeout(total=30)
                ) as session:
                    async with session.post(
                        "https://integrate.api.nvidia.com/v1/chat/completions",
                        json=payload,
                        headers={
                            "Content-Type": "application/json",
                            "Authorization": f"Bearer {self._nvidia_api_key}",
                            "Accept": "application/json",
                        },
                    ) as resp:
                        if resp.status != 200:
                            return ""
                        data = await resp.json()
                text = data.get("choices", [{}])[0].get("message", {}).get("content", "")

            if text:
                text = text.strip()
                match = re.search(r'\[[\[\]\d,\s]+\]', text)
                if match:
                    return match.group(0)
            return ""
        except Exception as e:
            log.error(f"VLM-OCR coord error: {e}")
            return ""


# ──────────────────────────────────────────────
# hCaptcha Solver (ONNX-based + LLM classifier)
# ──────────────────────────────────────────────

def _extract_first_json_block(text) -> dict | None:
    """Extract the first JSON object {...} from text (handles markdown fences)."""
    # Handle non-string input (CF Workers AI may return dict)
    if isinstance(text, dict):
        return text
    if not isinstance(text, str):
        text = str(text)
    # Strip markdown code fences
    cleaned = re.sub(r"```(?:json)?\s*", "", text)
    cleaned = cleaned.strip()
    # Find first { ... }
    start = cleaned.find("{")
    if start < 0:
        return None
    depth = 0
    for i in range(start, len(cleaned)):
        if cleaned[i] == "{":
            depth += 1
        elif cleaned[i] == "}":
            depth -= 1
            if depth == 0:
                try:
                    return json.loads(cleaned[start : i + 1])
                except (json.JSONDecodeError, ValueError):
                    return None
    return None


class CloudflareWorkersProvider:
    """
    ChatProvider implementation for Cloudflare Workers AI.
    
    Implements the hcaptcha-challenger ChatProvider protocol so that
    CF Workers AI (vision models) can be used instead of Gemini for
    hCaptcha image classification tasks.
    
    Supported models:
      - @cf/meta/llama-3.2-11b-vision-instruct (default, free, vision)
      - Fallback to @cf/meta/llama-3.1-70b-instruct for text/routing tasks
      - Any CF Workers AI vision model
    """

    def __init__(self, api_token: str, account_id: str, model: str = "",
                 fallback_model: str = ""):
        self._api_token = api_token
        self._account_id = account_id
        self._model = model or os.environ.get(
            "CF_HCAPTCHA_MODEL", "@cf/meta/llama-3.2-11b-vision-instruct"
        )
        # Stronger text model for routing/structured output (no vision needed)
        self._fallback_model = fallback_model or os.environ.get(
            "CF_HCAPTCHA_FALLBACK_MODEL", "@cf/meta/llama-3.1-70b-instruct"
        )
        self._cf_agreed_models = set()
        self._last_response_text = ""
        log.info(
            f"CF Workers AI provider initialized (model={self._model}, fallback={self._fallback_model})"
        )

    @property
    def last_response(self) -> str:
        """Get the last raw response text for debugging."""
        return self._last_response_text

    async def _ensure_agreement(self, model: str = None):
        """Agree to CF Workers AI model license terms (required first call)."""
        model = model or self._model
        if model in self._cf_agreed_models:
            return
        url = (
            f"https://api.cloudflare.com/client/v4/accounts/"
            f"{self._account_id}/ai/run/{model}"
        )
        try:
            req = urllib.request.Request(
                url,
                data=json.dumps({"prompt": "agree", "max_tokens": 10}).encode(),
                headers={
                    "Authorization": f"Bearer {self._api_token}",
                    "Content-Type": "application/json",
                    "cf-model-agreement": "true",
                },
                method="POST",
            )
            loop = asyncio.get_event_loop()
            resp = await asyncio.wait_for(
                loop.run_in_executor(None, lambda: urllib.request.urlopen(req, timeout=30)),
                timeout=45,
            )
            data = json.loads(resp.read().decode())
            if data.get("success"):
                self._cf_agreed_models.add(model)
                log.info(f"CF Workers AI model license agreed ({model})")
            else:
                self._cf_agreed_models.add(model)
                log.info(f"CF license agreement submitted ({model})")
        except Exception:
            self._cf_agreed_models.add(model)

    async def _cf_call(self, payload: dict, timeout: int = 90, model: str = None) -> dict:
        """Call CF Workers AI via urllib (works without curl binary)."""
        model = model or self._model
        await self._ensure_agreement(model)
        url = (
            f"https://api.cloudflare.com/client/v4/accounts/"
            f"{self._account_id}/ai/run/{model}"
        )
        try:
            req = urllib.request.Request(
                url,
                data=json.dumps(payload).encode(),
                headers={
                    "Authorization": f"Bearer {self._api_token}",
                    "Content-Type": "application/json",
                    "cf-model-agreement": "true",
                },
                method="POST",
            )
            loop = asyncio.get_event_loop()
            resp = await asyncio.wait_for(
                loop.run_in_executor(
                    None, lambda: urllib.request.urlopen(req, timeout=timeout)
                ),
                timeout=timeout + 15,
            )
            return json.loads(resp.read().decode())
        except urllib.error.HTTPError as e:
            body = e.read().decode()
            return {"success": False, "errors": [{"message": f"HTTP {e.code}: {body[:200]}"}]}
        except Exception as e:
            return {"success": False, "errors": [{"message": str(e)[:200]}]}

    def _images_to_messages(
        self, images: list, user_prompt: str | None = None, description: str | None = None
    ) -> list:
        """Build OpenAI-compatible messages with base64 images for CF Workers AI."""
        content_parts = []
        for img_path in images:
            try:
                p = str(img_path)
                img_bytes = open(p, "rb").read()
                b64 = base64.b64encode(img_bytes).decode()
                mime = "image/png"
                if img_bytes[:3] == b"\xff\xd8\xff":
                    mime = "image/jpeg"
                elif img_bytes[:4] == b"RIFF" and img_bytes[8:12] == b"WEBP":
                    mime = "image/webp"
                elif img_bytes[:4] == b"GIF8":
                    mime = "image/gif"
                data_uri = f"data:{mime};base64,{b64}"
                content_parts.append({
                    "type": "image_url",
                    "image_url": {"url": data_uri},
                })
            except Exception as e:
                log.warning(f"Failed to read image {img_path}: {e}")

        # Add text prompt last (model processes images then text instructions)
        prompt_text = user_prompt or "Analyze this image."
        if description:
            prompt_text = f"{description}\n\n{prompt_text}"
        content_parts.append({"type": "text", "text": prompt_text})

        # Build system + user messages in OpenAI format
        messages = []
        if description and not user_prompt:
            messages.append({"role": "system", "content": description})
        messages.append({"role": "user", "content": content_parts})
        return messages

    async def generate_with_images(
        self,
        *,
        images: list,
        response_schema: type = None,
        user_prompt: str | None = None,
        description: str | None = None,
        **kwargs,
    ):
        """
        Generate content with image inputs via CF Workers AI.
        
        Implements the ChatProvider protocol from hcaptcha-challenger.
        Sends images + prompts to CF Workers AI, then parses the structured 
        JSON response into the given Pydantic response_schema.
        
        Strategy:
        - For ChallengeRouterResult (text classification): use the stronger 
          fallback model (llama-3.1-70b-instruct) first — no vision needed.
        - For image classification (coordinates): use the vision model.
        - If the model echoes the schema back instead of actual data, retry
          with a simpler prompt or fallback to a stronger model.
        
        Args:
            images: List of image file paths.
            response_schema: Pydantic model class for structured output.
            user_prompt: User-provided prompt/instructions.
            description: System instruction/description for the model.
            **kwargs: Additional provider-specific options.
            
        Returns:
            Parsed response matching the response_schema type.
        """
        # Determine if this is a text-only task (no vision needed)
        schema_name = getattr(response_schema, "__name__", "") if response_schema else ""
        is_routing_task = (schema_name == "ChallengeRouterResult")
        use_fallback = is_routing_task and self._fallback_model
        
        # For routing tasks, try the stronger fallback model first (text-only, better at JSON)
        if use_fallback:
            log.info(f"Using fallback model ({self._fallback_model}) for {schema_name}")
            routing_prompt = (
                f"{user_prompt or 'Classify the challenge.'}\n\n"
                f"Respond ONLY with a JSON object like: "
                f'{{"challenge_prompt": "the prompt text", "challenge_type": "image_label_binary"}}\n'
                f"Valid types: image_label_single_select, image_label_multi_select, "
                f"image_label_binary, image_drag_single, image_drag_multi"
            )
            fallback_messages = [
                {"role": "system", "content": "You are a captcha challenge classifier. Respond ONLY with valid JSON."},
                {"role": "user", "content": routing_prompt},
            ]
            fallback_payload = {
                "messages": fallback_messages,
                "max_tokens": 256,
                "temperature": 0.1,
            }
            try:
                data = await self._cf_call(fallback_payload, timeout=60, model=self._fallback_model)
                if data.get("success"):
                    result = data.get("result", {})
                    raw = result.get("response", "") if isinstance(result, dict) else str(result)
                    if isinstance(raw, dict):
                        raw = json.dumps(raw)
                    text = raw if isinstance(raw, str) else str(raw)
                    json_data = _extract_first_json_block(text)
                    if json_data and isinstance(json_data, dict) and "challenge_type" in json_data:
                        self._last_response_text = text
                        return response_schema(**json_data)
                    log.warning(f"Fallback model returned unparseable response: {text[:100]}")
            except Exception as e:
                log.warning(f"Fallback model error (will try vision model): {e}")
        
        # Build prompt that enforces JSON schema output
        schema_hint = ""
        if response_schema and hasattr(response_schema, "model_json_schema"):
            schema_def = response_schema.model_json_schema()
            # For small vision models, use a lighter schema hint to avoid echo
            if is_routing_task:
                schema_hint = (
                    f"\\n\\nRespond with ONLY a JSON object: "
                    f'{{"challenge_prompt": "the prompt", "challenge_type": "image_label_binary"}}'
                )
            else:
                schema_hint = (
                    f"\\n\\nYou MUST respond with valid JSON matching this schema:\\n"
                    f"{json.dumps(schema_def, indent=2)}\\n"
                    f"Output ONLY the JSON object, no markdown, no explanation."
                )

        enhanced_prompt = (user_prompt or "Analyze this image.") + schema_hint
        messages = self._images_to_messages(images, enhanced_prompt, description)

        payload = {
            "messages": messages,
            "max_tokens": 1024,
            "temperature": 0.1,
        }

        try:
            data = await self._cf_call(payload, timeout=90)
            if not data.get("success"):
                errors = data.get("errors", [])
                err = errors[0].get("message", "unknown") if errors else "unknown"
                log.error(f"CF Workers AI hCaptcha error: {err[:200]}")
                raise RuntimeError(f"CF Workers AI error: {err[:200]}")

            result = data.get("result", {})
            # CF Workers AI may return response as dict (structured) or string
            raw_response = result.get("response", "") if isinstance(result, dict) else str(result)
            # Ensure text is always a string
            if isinstance(raw_response, dict):
                self._last_response_text = json.dumps(raw_response)
            elif isinstance(raw_response, str):
                self._last_response_text = raw_response
            else:
                self._last_response_text = str(raw_response)
            text = self._last_response_text

            if not text:
                raise ValueError("Empty response from CF Workers AI")

            # Parse into the response schema
            json_data = _extract_first_json_block(text)
            if json_data and response_schema:
                # Check if the model returned a schema definition instead of actual data
                if isinstance(json_data, dict):
                    # Schema echo: model returned $defs/properties instead of challenge_type/challenge_prompt
                    if "$defs" in json_data and "challenge_type" not in json_data:
                        log.warning("CF model returned schema definition instead of actual result, retrying without schema hint")
                        # Retry without schema hint in the prompt
                        simple_payload = {
                            "messages": [
                                {"role": "system", "content": "You are a captcha classifier. Respond ONLY with valid JSON."},
                                {"role": "user", "content": f"{user_prompt or 'Classify the image.'}\n\nRespond with JSON: {{\"challenge_type\": \"image_label_binary\", \"challenge_prompt\": \"description\"}}"}
                            ],
                            "max_tokens": 256,
                            "temperature": 0.1,
                        }
                        simple_payload["messages"][1]["content"] += "\n\nThe image shows an hCaptcha grid challenge. What type is it?"
                        data2 = await self._cf_call(simple_payload, timeout=60)
                        result2 = data2.get("result", {})
                        raw2 = result2.get("response", "") if isinstance(result2, dict) else str(result2)
                        if isinstance(raw2, dict):
                            raw2 = json.dumps(raw2)
                        text2 = raw2 if isinstance(raw2, str) else str(raw2)
                        json_data2 = _extract_first_json_block(text2)
                        if json_data2 and isinstance(json_data2, dict) and "challenge_type" in json_data2:
                            return response_schema(**json_data2)
                        
                        # Final fallback: default to most common type
                        log.warning("ChallengeRouter fallback: defaulting to image_label_binary")
                        if response_schema.__name__ == "ChallengeRouterResult":
                            return response_schema(challenge_type="image_label_binary", challenge_prompt="Click on the matching objects")
                    try:
                        return response_schema(**json_data)
                    except Exception:
                        pass

            # Fallback: try direct JSON parse
            if response_schema:
                try:
                    parsed = json.loads(text)
                    if isinstance(parsed, dict):
                        return response_schema(**parsed)
                except (json.JSONDecodeError, ValueError):
                    pass

            raise ValueError(
                f"Failed to parse CF response as {response_schema}: {text[:300]}"
            )

        except RuntimeError:
            raise
        except Exception as e:
            log.error(f"CF Workers AI provider error: {e}")
            raise


class HcaptchaSolver:
    """
    hCaptcha solver using hcaptcha-challenger.
    
    Supports two LLM backends for image classification:
    1. Google Gemini (default — needs GEMINI_API_KEY)
    2. Cloudflare Workers AI (alternative — needs CF_API_TOKEN + CF_ACCOUNT_ID)
    
    The solver uses Playwright to navigate the hCaptcha challenge page,
    then uses the LLM to classify which grid cells to click.
    
    Usage:
        # With Gemini
        solver = HcaptchaSolver(gemini_api_key="AIza...")
        
        # With Cloudflare Workers AI
        solver = HcaptchaSolver(
            cf_api_token="cfut_...",
            cf_account_id="abc123...",
        )
    """

    def __init__(self, gemini_api_key: str = "",
                 cf_api_token: str = "", cf_account_id: str = "",
                 cf_model: str = ""):
        self._gemini_key = gemini_api_key or os.environ.get("GEMINI_API_KEY", "")
        self._cf_api_token = cf_api_token or os.environ.get("CF_API_TOKEN", os.environ.get("CLOUDFLARE_WORKERS_AI_TOKEN", ""))
        self._cf_account_id = cf_account_id or os.environ.get("CF_ACCOUNT_ID", os.environ.get("CLOUDFLARE_ACCOUNT_ID", ""))
        self._cf_model = cf_model or os.environ.get("CF_HCAPTCHA_MODEL", "")
        self._provider = "none"
        self._agent_cls = None
        self._cf_provider = None

        # Try loading hcaptcha-challenger
        try:
            from hcaptcha_challenger import AgentV, AgentConfig
            self._agent_cls = AgentV
            self._config_cls = AgentConfig
        except Exception as e:
            log.warning(f"hcaptcha-challenger not available: {e}")
            return

        # Determine provider
        if self._gemini_key:
            self._provider = "gemini"
            log.info(f"hCaptcha: using Gemini (key={self._gemini_key[:8]}...)")
        elif self._cf_api_token and self._cf_account_id:
            self._provider = "cloudflare"
            self._cf_provider = CloudflareWorkersProvider(
                api_token=self._cf_api_token,
                account_id=self._cf_account_id,
                model=self._cf_model,
            )
            log.info(f"hCaptcha: using Cloudflare Workers AI (model={self._cf_provider._model})")
        else:
            log.warning("hCaptcha: no LLM provider configured (needs GEMINI_API_KEY or CF_API_TOKEN+CF_ACCOUNT_ID)")

    @property
    def available(self):
        return self._agent_cls is not None and self._provider != "none"

    async def solve_hcaptcha(
        self,
        sitekey: str,
        pageurl: str,
    ) -> str:
        """Solve hCaptcha challenge using Playwright + hcaptcha-challenger.
        
        The solver navigates to the page with hCaptcha, clicks the checkbox,
        then uses the LLM provider (Gemini or CF Workers AI) to classify
        which grid cells to click in the image challenge.
        """
        if not self.available:
            raise RuntimeError(
                "hcaptcha-challenger not installed or no LLM provider configured "
                "(needs GEMINI_API_KEY or CF_API_TOKEN+CF_ACCOUNT_ID)"
            )

        try:
            from hcaptcha_challenger import AgentV, AgentConfig
            from playwright.async_api import async_playwright
            from playwright_stealth import Stealth

            # Build AgentConfig with the appropriate provider
            if self._provider == "gemini":
                config = AgentConfig(GEMINI_API_KEY=self._gemini_key)
            else:
                # Cloudflare — we need to inject our provider into the classifiers
                # AgentConfig still needs a dummy GEMINI_API_KEY to pass validation
                config = AgentConfig(GEMINI_API_KEY="cf-placeholder")

            async with async_playwright() as p:
                browser = await p.chromium.launch(
                    headless=True,
                    args=[
                        "--disable-blink-features=AutomationControlled",
                        "--no-sandbox",
                        "--disable-dev-shm-usage",
                    ],
                )
                context = await browser.new_context(
                    viewport={"width": 1280, "height": 720},
                    user_agent=(
                        "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
                        "AppleWebKit/537.36 (KHTML, like Gecko) "
                        "Chrome/125.0.0.0 Safari/537.36"
                    ),
                )
                page = await context.new_page()
                stealth_config = Stealth(navigator_webdriver=True, navigator_plugins=True, navigator_platform_override="Win32")
                await stealth_config.apply_stealth_async(page)

                # If using CF Workers AI, monkey-patch the provider into classifiers
                if self._provider == "cloudflare" and self._cf_provider:
                    self._patch_cf_provider(config)

                try:
                    # Navigate to the page and click the hCaptcha checkbox
                    await page.goto(pageurl, wait_until="domcontentloaded", timeout=30000)
                    await page.wait_for_timeout(3000)

                    # Try to click the hCaptcha checkbox to trigger the challenge
                    try:
                        checkbox = page.frame_locator("iframe[src*='hcaptcha.com']").first
                        await checkbox.locator("#checkbox").click(timeout=10000)
                        log.info("Clicked hCaptcha checkbox")
                        await page.wait_for_timeout(3000)
                    except Exception as e:
                        log.info(f"hCaptcha checkbox click: {e} (proceeding anyway)")

                    agent = AgentV(page=page, agent_config=config)
                    result = await agent.wait_for_challenge()

                    if result.name == "SUCCESS":
                        # Extract the hCaptcha response token from the page
                        token = await page.evaluate(
                            """() => {
                                const el = document.querySelector('[name="h-captcha-response"]');
                                return el ? el.value : '';
                            }"""
                        )
                        if token:
                            log.info("hCaptcha solved successfully")
                            return token
                    
                    log.warning(f"hCaptcha challenge result: {result.name}")
                    return ""

                finally:
                    await browser.close()

        except Exception as e:
            log.error(f"hCaptcha solve error: {e}")
            raise

    def _patch_cf_provider(self, config: "AgentConfig"):
        """Monkey-patch Cloudflare Workers AI provider into hcaptcha-challenger classifiers.
        
        This replaces the Gemini-based ImageClassifier, ChallengeClassifier,
        and spatial reasoners with our CF Workers AI provider.
        """
        import hcaptcha_challenger.tools.image_classifier as ic_mod
        import hcaptcha_challenger.tools.challenge_router as cr_mod
        import hcaptcha_challenger.tools.spatial as sp_mod

        cf = self._cf_provider

        # Patch ImageClassifier to use CF provider
        original_ic_init = ic_mod.ImageClassifier.__init__
        def patched_ic_init(self_ic, gemini_api_key, model=None, *, provider=None, **kwargs):
            # Force our CF provider
            original_ic_init(self_ic, gemini_api_key="cf-placeholder", model=model, provider=cf, **kwargs)
        ic_mod.ImageClassifier.__init__ = patched_ic_init

        # Patch ChallengeRouter similarly
        try:
            original_cr_init = cr_mod.ChallengeRouter.__init__
            def patched_cr_init(self_cr, gemini_api_key, model=None, *, provider=None, **kwargs):
                original_cr_init(self_cr, gemini_api_key="cf-placeholder", model=model, provider=cf, **kwargs)
            cr_mod.ChallengeRouter.__init__ = patched_cr_init
        except (AttributeError, ImportError):
            pass

        # Patch spatial reasoners
        try:
            for mod_name in ["point", "bbox", "path"]:
                sp_sub = getattr(sp_mod, mod_name, None)
                if sp_sub is None:
                    continue
                for cls_name in ["SpatialPointReasoner", "SpatialBboxReasoner", "SpatialPathReasoner"]:
                    cls = getattr(sp_sub, cls_name, None) or getattr(sp_mod, cls_name, None)
                    if cls is None:
                        continue
                    original_init = cls.__init__
                    def make_patched(orig):
                        def patched(self_sp, gemini_api_key, model=None, *, provider=None, **kwargs):
                            orig(self_sp, gemini_api_key="cf-placeholder", model=model, provider=cf, **kwargs)
                        return patched
                    cls.__init__ = make_patched(original_init)
        except (AttributeError, ImportError) as e:
            log.debug(f"Skipping spatial reasoner patch: {e}")

        log.info("Patched hcaptcha-challenger classifiers with CF Workers AI provider")


# ──────────────────────────────────────────────
# Main Server
# ──────────────────────────────────────────────
class UniversalCaptchaSolver:
    """Unified captcha solver with 2captcha-compatible HTTP API"""

    def __init__(self, api_key: str, port: int = 8855, gemini_key: str = "",
                 cf_api_token: str = "", cf_account_id: str = "",
                 cf_hcaptcha_model: str = ""):
        self.api_key = api_key
        self.port = port
        self.tasks: Dict[str, CaptchaTask] = {}
        self._queue = asyncio.Queue()
        self.solved_count = 0
        self.failed_count = 0
        self.active_sessions = 0

        # Initialize solvers
        self._ddddocr = DdddOcrSolver()
        self._tesseract = TesseractSolver()
        self._vlm_ocr = VlmOcrSolver(cf_api_token=cf_api_token, cf_account_id=cf_account_id)
        self._hcaptcha = HcaptchaSolver(
            gemini_api_key=gemini_key,
            cf_api_token=cf_api_token,
            cf_account_id=cf_account_id,
            cf_model=cf_hcaptcha_model,
        )

        log.info("Universal Captcha Solver initialized")
        log.info(f"  ddddocr: ✅")
        log.info(f"  Tesseract: {'✅' if self._tesseract.available else '❌'}")
        vlm_provider = self._vlm_ocr._provider.upper() if self._vlm_ocr.available else "none"
        log.info(f"  VLM-OCR ({vlm_provider}): {'✅' if self._vlm_ocr.available else '❌ (needs CF_API_TOKEN+CF_ACCOUNT_ID or NVIDIA_API_KEY)'}")
        hc_provider = self._hcaptcha._provider.upper()
        log.info(f"  hCaptcha ({hc_provider}): {'✅' if self._hcaptcha.available else '❌ (needs GEMINI_API_KEY or CF_API_TOKEN+CF_ACCOUNT_ID)'}")

    async def start(self):
        """Start worker tasks"""
        n_workers = 3
        for i in range(n_workers):
            asyncio.create_task(self._worker(i))
        log.info(f"Started {n_workers} solver workers")

    async def _worker(self, worker_id: int):
        """Worker loop: pick tasks from queue, solve, store result"""
        while True:
            task = await self._queue.get()
            if task is None:
                break
            try:
                task.status = "processing"
                self.active_sessions += 1
                await self._solve_task(task)
                self.solved_count += 1
            except Exception as e:
                task.status = "failed"
                task.error = str(e)
                self.failed_count += 1
                log.error(f"Worker {worker_id}: FAILED {task.task_id[:8]}: {e}")
            finally:
                self.active_sessions -= 1

    async def _solve_task(self, task: CaptchaTask):
        """Route task to appropriate solver"""
        start = time.time()
        log.info(f"Solving {task.captcha_type.value} task {task.task_id[:8]}...")

        if task.captcha_type == CaptchaType.IMAGE_OCR:
            await self._solve_image(task)
        elif task.captcha_type == CaptchaType.BASE64:
            await self._solve_base64(task)
        elif task.captcha_type == CaptchaType.COORD:
            await self._solve_coord(task)
        elif task.captcha_type == CaptchaType.HCAPTCHA:
            await self._solve_hcaptcha(task)
        elif task.captcha_type in (CaptchaType.RECAPTCHA_V2, CaptchaType.TURNSTILE, CaptchaType.XCAPTCHA):
            # Forward to dedicated solvers
            task.token = await self._forward_to_solver(task)
            task.status = "solved"
        else:
            raise ValueError(f"Unsupported captcha type: {task.captcha_type}")

        elapsed = time.time() - start
        task.solved_at = time.time()
        if task.status != "failed":
            task.status = "solved"
            token_preview = task.token[:50] if task.token else ""
            log.info(
                f"SOLVED {task.captcha_type.value} {task.task_id[:8]} "
                f"in {elapsed:.1f}s: {token_preview}..."
            )

    # ─── Image/OCR captcha ───
    async def _solve_image(self, task: CaptchaTask):
        """Solve image OCR captcha"""
        image_bytes = task.image_data
        if not image_bytes and task.image_base64:
            image_bytes = base64.b64decode(task.image_base64)

        if not image_bytes:
            raise ValueError("No image data provided")

        # Try ddddocr first (fast, accurate for common captchas)
        result = await asyncio.get_event_loop().run_in_executor(
            None, self._ddddocr.solve_text, image_bytes
        )
        if result:
            task.token = result
            return

        # Fallback to Tesseract
        if self._tesseract.available:
            result = await asyncio.get_event_loop().run_in_executor(
                None, self._tesseract.solve_text, image_bytes
            )
            if result:
                task.token = result
                return

        # Tier-3: VLM-OCR via NVIDIA vision API (best for distorted/noisy/math captchas)
        if self._vlm_ocr.available:
            result = await self._vlm_ocr.solve_text(image_bytes)
            if result:
                task.token = result
                return

        raise ValueError("All OCR engines (ddddocr + Tesseract + VLM-OCR) returned empty results")

    # ─── Base64 image ───
    async def _solve_base64(self, task: CaptchaTask):
        """Solve base64-encoded image captcha"""
        # Handle data URI format
        b64data = task.image_base64
        if "," in b64data:
            b64data = b64data.split(",", 1)[1]

        image_bytes = base64.b64decode(b64data)
        task.image_data = image_bytes
        task.captcha_type = CaptchaType.IMAGE_OCR
        await self._solve_image(task)

    # ─── Coordinate captcha ───
    async def _solve_coord(self, task: CaptchaTask):
        """Solve coordinate-based captcha (click targets)"""
        image_bytes = task.image_data
        if not image_bytes and task.image_base64:
            image_bytes = base64.b64decode(task.image_base64)

        if not image_bytes:
            raise ValueError("No image data for coordinate captcha")

        # Use ddddocr detection for target finding
        if task.extra.get("target_bytes"):
            result = await asyncio.get_event_loop().run_in_executor(
                None,
                self._ddddocr.solve_target,
                image_bytes,
                task.extra["target_bytes"],
            )
            task.coordinates = result
        else:
            # Use detection model for finding objects
            result = await asyncio.get_event_loop().run_in_executor(
                None, self._ddddocr._det.detection, image_bytes
            )
            if result:
                # Convert bounding boxes to click coordinates
                coords = []
                for box in result:
                    cx = (box[0] + box[2]) / 2
                    cy = (box[1] + box[3]) / 2
                    coords.append(f"x={int(cx)},y={int(cy)}")
                task.coordinates = ";".join(coords)
            else:
                raise ValueError("No objects detected in image")

        task.token = task.coordinates

    # ─── hCaptcha ───
    async def _solve_hcaptcha(self, task: CaptchaTask):
        """Solve hCaptcha challenge"""
        if not self._hcaptcha.available:
            raise RuntimeError("hcaptcha-challenger not installed or GEMINI_API_KEY missing")

        result = await self._hcaptcha.solve_hcaptcha(
            sitekey=task.sitekey,
            pageurl=task.pageurl,
        )
        task.token = result

    # ─── Forward to dedicated solvers ───
    async def _forward_to_solver(self, task: CaptchaTask) -> str:
        """Forward reCAPTCHA/Turnstile/xCaptcha tasks to dedicated solver servers"""
        import urllib.request
        import urllib.parse

        if task.captcha_type == CaptchaType.RECAPTCHA_V2:
            solver_url = os.environ.get("RECAPTCHA_SOLVER_URL", "http://127.0.0.1:8866")
            params = {
                "key": self.api_key,
                "method": "userrecaptcha",
                "version": task.version,
                "googlekey": task.googlekey or task.sitekey,
                "pageurl": task.pageurl,
            }
        elif task.captcha_type == CaptchaType.TURNSTILE:
            solver_url = os.environ.get("TURNSTILE_SOLVER_URL", "http://127.0.0.1:8878")
            params = {
                "key": self.api_key,
                "method": "turnstile",
                "sitekey": task.sitekey,
                "pageurl": task.pageurl,
            }
        elif task.captcha_type == CaptchaType.XCAPTCHA:
            solver_url = os.environ.get("XCAPTCHA_SOLVER_URL", "http://172.17.0.1:8899")
            params = {
                "key": self.api_key,
                "method": "wcaptcha",
                "sitekey": task.sitekey,
                "pageurl": task.pageurl,
            }
        else:
            raise ValueError(f"Cannot forward type: {task.captcha_type}")

        # Submit to upstream solver
        submit_url = f"{solver_url}/in.php"
        data = urllib.parse.urlencode(params).encode()
        req = urllib.request.Request(submit_url, data=data)
        resp = urllib.request.urlopen(req, timeout=10)
        result = resp.read().decode()

        # Handle both "OK|id" and JSON {"status":1,"request":"id"} responses
        if result.startswith("{"):
            import json as _json
            jr = _json.loads(result)
            if jr.get("status") != 1:
                raise ValueError(f"Upstream submit failed: {result}")
            task_id = jr["request"]
        elif result.startswith("OK|"):
            task_id = result.split("|", 1)[1]
        else:
            raise ValueError(f"Upstream submit failed: {result}")

        # Poll for result (handle both text and JSON responses)
        poll_url = f"{solver_url}/res.php?key={self.api_key}&id={task_id}"
        for _ in range(60):  # 5 min max
            await asyncio.sleep(5)
            try:
                resp = urllib.request.urlopen(poll_url, timeout=10)
                text = resp.read().decode()
                # JSON response
                if text.startswith("{"):
                    import json as _json
                    jr = _json.loads(text)
                    if jr.get("status") == 1:
                        return jr["request"]
                    req_val = jr.get("request", "")
                    if "ERROR" in str(req_val):
                        raise ValueError(f"Upstream error: {req_val}")
                # Text response
                elif text.startswith("OK|"):
                    return text.split("|", 1)[1]
                elif "ERROR" in text:
                    raise ValueError(f"Upstream error: {text}")
            except Exception as e:
                if "ERROR" in str(e):
                    raise

        raise TimeoutError("Upstream solver timeout")


# ──────────────────────────────────────────────
# HTTP API (2captcha-compatible)
# ──────────────────────────────────────────────
def create_app(solver: UniversalCaptchaSolver) -> web.Application:
    app = web.Application()

    async def in_php(request: web.Request) -> web.Response:
        """2captcha-compatible task submit endpoint"""
        # Parse POST data (works for both multipart and URL-encoded)
        post = await request.post()

        req_key = post.get("key", "") or request.query.get("key", "")
        json_mode = str(post.get("json", "") or request.query.get("json", "")) == "1"
        if req_key != solver.api_key:
            if json_mode: return web.json_response({"status": 0, "request": "ERROR_WRONG_USER_KEY"})
            return web.Response(text="ERROR_WRONG_USER_KEY")

        # Determine captcha type
        method = str(post.get("method", "") or "image")
        if method == "":
            method = "image"

        # Map method to captcha type
        method_map = {
            "image": CaptchaType.IMAGE_OCR,
            "base64": CaptchaType.BASE64,
            "hcaptcha": CaptchaType.HCAPTCHA,
            "userrecaptcha": CaptchaType.RECAPTCHA_V2,
            "recaptcha": CaptchaType.RECAPTCHA_V2,
            "turnstile": CaptchaType.TURNSTILE,
            "wcaptcha": CaptchaType.XCAPTCHA,
            "xcaptcha": CaptchaType.XCAPTCHA,
            "coord": CaptchaType.COORD,
        }
        captcha_type = method_map.get(method, CaptchaType.IMAGE_OCR)

        # Create task
        task = CaptchaTask(
            task_id=uuid.uuid4().hex[:24],
            captcha_type=captcha_type,
            created_at=time.time(),
        )

        # Extract common params
        task.sitekey = str(post.get("sitekey") or post.get("googlekey") or "")
        task.googlekey = str(post.get("googlekey") or post.get("sitekey") or "")
        task.pageurl = str(post.get("pageurl") or "")
        task.version = str(post.get("version") or "v2")

        # Handle image data (file upload or base64)
        file_field = post.get("file")
        if file_field is not None and hasattr(file_field, "file"):
            # aiohttp FileField: .file is a BytesIO-like object, .filename, .content_type
            task.image_data = file_field.file.read()
            log.info(f"Received file upload: {file_field.filename} ({len(task.image_data)} bytes)")
        elif "body" in post:
            task.image_base64 = str(post["body"])

        if "img_base64" in post or "image_base64" in post:
            task.image_base64 = str(post.get("img_base64") or post.get("image_base64") or "")

        # Handle captcha_img param (URL to download)
        captcha_img = str(post.get("captcha_img") or post.get("url") or "")
        if captcha_img and not task.image_data and not task.image_base64:
            try:
                import urllib.request
                resp = urllib.request.urlopen(captcha_img, timeout=10)
                task.image_data = resp.read()
            except Exception as e:
                log.error(f"Failed to download captcha image: {e}")
                return web.Response(text=f"ERROR_BAD_PARAMETERS: {e}")

        # Submit task
        await solver._queue.put(task)
        log.info(f"New {captcha_type.value} task: {task.task_id[:8]}")

        if json_mode: return web.json_response({"status": 1, "request": task.task_id})
        return web.Response(text=f"OK|{task.task_id}")

    async def res_php(request: web.Request) -> web.Response:
        """2captcha-compatible result polling endpoint"""
        req_key = request.query.get("key", "")
        task_id = request.query.get("id", "")
        action = request.query.get("action", "")
        json_mode = request.query.get("json", "") == "1"

        if req_key != solver.api_key:
            if json_mode: return web.json_response({"status": 0, "request": "ERROR_WRONG_USER_KEY"})
            return web.Response(text="ERROR_WRONG_USER_KEY")

        task = solver.tasks.get(task_id)
        if not task:
            # Check if task is still in queue or being processed
            if json_mode: return web.json_response({"status": 0, "request": "CAPCHA_NOT_READY"})
            return web.Response(text="CAPCHA_NOT_READY")

        # Store task for polling
        solver.tasks[task_id] = task

        if action == "getbalance":
            if json_mode: return web.json_response({"status": 1, "request": "$0.00"})
            return web.Response(text="OK|$0.00")

        if task.status == "solved":
            result = task.token or task.coordinates or ""
            solver.tasks.pop(task_id, None)
            if json_mode: return web.json_response({"status": 1, "request": result})
            return web.Response(text=f"OK|{result}")

        if task.status == "failed":
            error = task.error or "ERROR_CAPTCHA_UNSOLVABLE"
            solver.tasks.pop(task_id, None)
            if json_mode: return web.json_response({"status": 0, "request": error})
            return web.Response(text=f"ERROR|{error}")

        if json_mode: return web.json_response({"status": 0, "request": "CAPCHA_NOT_READY"})
        return web.Response(text="CAPCHA_NOT_READY")

    async def solve_json(request: web.Request, solver_ref=None) -> web.Response:
        """Direct JSON solve endpoint"""
        s = solver_ref or solver
        try:
            data = await request.json()
        except Exception:
            return web.json_response({"error": "invalid JSON"}, status=400)

        captcha_type_str = data.get("type", "image")
        method_map = {
            "image": CaptchaType.IMAGE_OCR,
            "base64": CaptchaType.BASE64,
            "hcaptcha": CaptchaType.HCAPTCHA,
            "userrecaptcha": CaptchaType.RECAPTCHA_V2,
            "recaptcha": CaptchaType.RECAPTCHA_V2,
            "turnstile": CaptchaType.TURNSTILE,
            "coord": CaptchaType.COORD,
        }
        captcha_type = method_map.get(captcha_type_str, CaptchaType.IMAGE_OCR)

        task = CaptchaTask(
            task_id=uuid.uuid4().hex[:24],
            captcha_type=captcha_type,
            created_at=time.time(),
            sitekey=data.get("sitekey", ""),
            pageurl=data.get("pageurl", ""),
            googlekey=data.get("googlekey", ""),
            image_base64=data.get("image_base64", data.get("body", "")),
            extra=data.get("extra", {}),
        )

        # Submit and wait for result (with timeout)
        await s._queue.put(task)

        for _ in range(60):
            await asyncio.sleep(2)
            if task.status in ("solved", "failed"):
                break

        if task.status == "solved":
            return web.json_response({
                "status": "solved",
                "task_id": task.task_id,
                "solution": task.token or task.coordinates,
                "type": captcha_type.value,
                "solve_time": round(task.solved_at - task.created_at, 2),
            })
        else:
            return web.json_response({
                "status": "failed",
                "task_id": task.task_id,
                "error": task.error,
            }, status=400)

    async def health(request: web.Request) -> web.Response:
        hc = solver._hcaptcha
        return web.json_response({
            "status": "ok",
            "v": "cf-v3-debug",
            "queue": solver._queue.qsize(),
            "solved": solver.solved_count,
            "failed": solver.failed_count,
            "active": solver.active_sessions,
            "engines": {
                "ddddocr": True,
                "tesseract": solver._tesseract.available,
                "vlm_ocr": solver._vlm_ocr.available,
                "vlm_provider": solver._vlm_ocr._provider,
                "hcaptcha": hc.available,
                "hcaptcha_debug": {
                    "agent_cls": str(hc._agent_cls) if hc._agent_cls else None,
                    "provider": hc._provider,
                    "has_cf_token": bool(hc._cf_api_token),
                    "has_cf_account": bool(hc._cf_account_id),
                    "has_cf_provider": hc._cf_provider is not None,
                },
            },
        })

    async def solve_direct(request: web.Request) -> web.Response:
        """Direct solve endpoint"""
        return await solve_json(request, solver)

    # Store tasks reference for polling
    _original_put = solver._queue.put

    async def tracked_put(task: CaptchaTask):
        solver.tasks[task.task_id] = task
        await _original_put(task)

    solver._queue.put = tracked_put

    app.router.add_post("/in.php", in_php)
    app.router.add_get("/res.php", res_php)
    app.router.add_post("/solve", solve_direct)
    app.router.add_get("/health", health)

    return app


# ──────────────────────────────────────────────
# Main
# ──────────────────────────────────────────────
async def main():
    import argparse

    parser = argparse.ArgumentParser(description="Universal Captcha Solver Server")
    parser.add_argument("--api-key", required=True, help="API key for authentication")
    parser.add_argument("--port", type=int, default=8855, help="HTTP API port (default: 8855)")
    parser.add_argument("--gemini-key", default="", help="Google Gemini API key for hCaptcha")
    parser.add_argument("--cf-api-token", default=os.environ.get("CF_API_TOKEN", os.environ.get("CLOUDFLARE_WORKERS_AI_TOKEN", "")), help="Cloudflare Workers AI API token (cfut_...)")
    parser.add_argument("--cf-account-id", default=os.environ.get("CF_ACCOUNT_ID", os.environ.get("CLOUDFLARE_ACCOUNT_ID", "")), help="Cloudflare account ID")
    parser.add_argument("--cf-hcaptcha-model", default="", help="CF Workers AI model for hCaptcha (default: @cf/meta/llama-3.2-11b-vision-instruct)")
    args = parser.parse_args()

    solver = UniversalCaptchaSolver(
        api_key=args.api_key,
        port=args.port,
        gemini_key=args.gemini_key,
        cf_api_token=args.cf_api_token,
        cf_account_id=args.cf_account_id,
        cf_hcaptcha_model=args.cf_hcaptcha_model,
    )

    await solver.start()

    app = create_app(solver)

    log.info(f"Universal Captcha Solver API on port {args.port}")
    log.info(f"API key: {args.api_key[:8]}...")
    log.info(f"Engines: ddddocr + Tesseract + hcaptcha-challenger")

    runner = web.AppRunner(app)
    await runner.setup()
    site = web.TCPSite(runner, "0.0.0.0", args.port)
    await site.start()

    try:
        while True:
            await asyncio.sleep(3600)
    except (KeyboardInterrupt, SystemExit):
        log.info("Shutting down...")


if __name__ == "__main__":
    asyncio.run(main())
