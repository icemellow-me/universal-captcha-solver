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
    CF uses curl subprocess to bypass bot protection on direct HTTP.
    """

    def __init__(self, cf_api_token: str = "", cf_account_id: str = "",
                 nvidia_api_key: str = "", model: str = ""):
        self._cf_api_token = cf_api_token or os.environ.get("CF_API_TOKEN", "")
        self._cf_account_id = cf_account_id or os.environ.get("CF_ACCOUNT_ID", "")
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
        proc = await asyncio.create_subprocess_exec(
            "curl", "-s", "--max-time", "30", url,
            "-H", f"Authorization: Bearer {self._cf_api_token}",
            "-H", "Content-Type: application/json",
            "-d", '{"prompt":"agree","max_tokens":10}',
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE,
        )
        stdout, _ = await asyncio.wait_for(proc.communicate(), timeout=45)
        data = json.loads(stdout.decode())
        if data.get("success"):
            self._cf_agreed = True
            log.info("CF Workers AI model license agreed")
        else:
            errors = data.get("errors", [])
            msg = errors[0].get("message", "") if errors else ""
            if "agree" in msg.lower():
                self._cf_agreed = True  # agreement prompt = not yet blocked
                log.info("CF license agreement submitted")
            else:
                log.warning(f"CF agreement response: {msg[:100]}")
                self._cf_agreed = True

    async def _cf_call(self, payload: dict, timeout: int = 60) -> dict:
        """Call CF Workers AI via curl subprocess (bypasses bot protection)."""
        await self._cf_ensure_agreement()
        url = f"https://api.cloudflare.com/client/v4/accounts/{self._cf_account_id}/ai/run/{self._cf_model}"
        proc = await asyncio.create_subprocess_exec(
            "curl", "-s", "--max-time", str(timeout), url,
            "-H", f"Authorization: Bearer {self._cf_api_token}",
            "-H", "Content-Type: application/json",
            "-d", json.dumps(payload),
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE,
        )
        stdout, _ = await asyncio.wait_for(proc.communicate(), timeout=timeout + 15)
        return json.loads(stdout.decode())

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
# hCaptcha Solver (ONNX-based)
# ──────────────────────────────────────────────
class HcaptchaSolver:
    """
    hCaptcha solver using hcaptcha-challenger ONNX models.
    Handles image_label_binary challenges (click matching images).
    """

    def __init__(self, gemini_api_key: str = ""):
        self._api_key = gemini_api_key or os.environ.get("GEMINI_API_KEY", "")
        self._agent = None
        try:
            from hcaptcha_challenger import AgentV
            self._agent_cls = AgentV
            self._zero_shot_cls = None
            log.info(f"hCaptcha challenger loaded (gemini={'yes' if self._api_key else 'no'})")
        except Exception as e:
            log.warning(f"hcaptcha-challenger not available: {e}")
            self._agent_cls = None

    @property
    def available(self):
        return self._agent_cls is not None

    async def solve_hcaptcha(
        self,
        sitekey: str,
        pageurl: str,
        browser_context=None,
    ) -> str:
        """Solve hCaptcha using Playwright + hcaptcha-challenger"""
        if not self.available:
            raise RuntimeError("hcaptcha-challenger not installed")
        if not self._api_key:
            raise RuntimeError("GEMINI_API_KEY required for hcaptcha-challenger")

        try:
            from hcaptcha_challenger import AgentV
            from playwright.async_api import async_playwright

            agent = AgentV(
                api_key=self._api_key,
                sitekey=sitekey,
                pageurl=pageurl,
            )
            result = await agent.handle()
            return result or ""
        except Exception as e:
            log.error(f"hCaptcha agent error: {e}")
            raise


# ──────────────────────────────────────────────
# Main Server
# ──────────────────────────────────────────────
class UniversalCaptchaSolver:
    """Unified captcha solver with 2captcha-compatible HTTP API"""

    def __init__(self, api_key: str, port: int = 8855, gemini_key: str = "",
                 cf_api_token: str = "", cf_account_id: str = ""):
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
        self._hcaptcha = HcaptchaSolver(gemini_key)

        log.info("Universal Captcha Solver initialized")
        log.info(f"  ddddocr: ✅")
        log.info(f"  Tesseract: {'✅' if self._tesseract.available else '❌'}")
        vlm_provider = self._vlm_ocr._provider.upper() if self._vlm_ocr.available else "none"
        log.info(f"  VLM-OCR ({vlm_provider}): {'✅' if self._vlm_ocr.available else '❌ (needs CF_API_TOKEN+CF_ACCOUNT_ID or NVIDIA_API_KEY)'}")
        log.info(f"  hCaptcha: {'✅' if self._hcaptcha.available else '❌ (needs GEMINI_API_KEY)'}")

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
        return web.json_response({
            "status": "ok",
            "queue": solver._queue.qsize(),
            "solved": solver.solved_count,
            "failed": solver.failed_count,
            "active": solver.active_sessions,
            "engines": {
                "ddddocr": True,
                "tesseract": solver._tesseract.available,
                "vlm_ocr": solver._vlm_ocr.available,
                "vlm_provider": solver._vlm_ocr._provider,
                "hcaptcha": solver._hcaptcha.available,
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
    parser.add_argument("--cf-api-token", default="", help="Cloudflare Workers AI API token (cfut_...)")
    parser.add_argument("--cf-account-id", default="", help="Cloudflare account ID")
    args = parser.parse_args()

    solver = UniversalCaptchaSolver(
        api_key=args.api_key,
        port=args.port,
        gemini_key=args.gemini_key,
        cf_api_token=args.cf_api_token,
        cf_account_id=args.cf_account_id,
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
