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


# ──────────────────────────────────────────────
# hCaptcha Solver (Puter/NVIDIA/Gemini LLM-based)
# ──────────────────────────────────────────────
class HcaptchaSolver:
    """
    hCaptcha solver using hcaptcha-challenger + custom LLM provider.
    Supports Puter (free GLM), NVIDIA NIM, or Gemini backends.
    """

    def __init__(self, gemini_api_key: str = "", provider=None):
        self._api_key = gemini_api_key or os.environ.get("GEMINI_API_KEY", "")
        self._puter_token = os.environ.get("PUTER_TOKEN", "")
        self._nvidia_key = os.environ.get("NVIDIA_API_KEY", "")
        self._custom_provider = provider
        self._agent = None
        self._provider_instance = provider  # use external if given
        
        try:
            from hcaptcha_challenger import AgentV
            self._agent_cls = AgentV
            self._zero_shot_cls = None
            
            # Auto-create Puter provider if token available and no custom provider
            if not self._custom_provider and self._puter_token:
                try:
                    # Support both /app (Docker WORKDIR) and /opt paths
                    if '/app' not in sys.path:
                        sys.path.insert(0, '/app')
                    if '/opt' not in sys.path:
                        sys.path.insert(0, '/opt')
                    from puter_provider import PuterProvider
                    self._custom_provider = PuterProvider(auth_token=self._puter_token)
                    log.info("hCaptcha: PuterProvider auto-configured")
                except Exception as e:
                    log.warning(f"hCaptcha: PuterProvider init failed: {e}")
            
            backend = "custom" if self._custom_provider else ("gemini" if self._api_key else "none")
            log.info(f"hCaptcha challenger loaded (backend={backend})")
        except Exception as e:
            log.warning(f"hcaptcha-challenger not available: {e}")
            self._agent_cls = None

    @property
    def available(self):
        return self._agent_cls is not None and (bool(self._custom_provider) or bool(self._api_key))

    async def solve_hcaptcha(
        self,
        sitekey: str,
        pageurl: str,
        browser_context=None,
    ) -> str:
        """Solve hCaptcha using Playwright + hcaptcha-challenger with custom provider"""
        if not self.available:
            raise RuntimeError("hcaptcha-challenger not installed or no LLM provider configured")

        try:
            from hcaptcha_challenger import AgentV, AgentConfig
            from playwright.async_api import async_playwright

            # Build config with custom provider if available
            if self._custom_provider:
                config = AgentConfig(
                    GEMINI_API_KEY="unused",  # placeholder - we override the provider
                    sitekey=sitekey,
                    pageurl=pageurl,
                )
                
                # Create agent and inject custom provider into its tools
                agent = AgentV(config=config)
                
                # Override all tool providers with our PuterProvider
                provider = self._custom_provider
                
                from hcaptcha_challenger.tools.challenge_router import ChallengeRouter
                from hcaptcha_challenger.tools.image_classifier import ImageClassifier
                from hcaptcha_challenger.tools.spatial.point import SpatialPointReasoner
                from hcaptcha_challenger.tools.spatial.path import SpatialPathReasoner
                from hcaptcha_challenger.tools.spatial.bbox import SpatialBBoxReasoner
                
                # Re-init tools with our custom provider
                agent._challenge_router = ChallengeRouter(
                    gemini_api_key="unused",
                    model=config.CHALLENGE_CLASSIFIER_MODEL,
                    provider=provider,
                )
                agent._image_classifier = ImageClassifier(
                    gemini_api_key="unused",
                    model=config.IMAGE_CLASSIFIER_MODEL,
                    provider=provider,
                )
                agent._spatial_point_reasoner = SpatialPointReasoner(
                    gemini_api_key="unused",
                    model=config.SPATIAL_POINT_REASONER_MODEL,
                    provider=provider,
                )
                agent._spatial_path_reasoner = SpatialPathReasoner(
                    gemini_api_key="unused",
                    model=config.SPATIAL_PATH_REASONER_MODEL,
                    provider=provider,
                )
                
                result = await agent.handle()
            else:
                # Fallback to Gemini
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

    async def classify_images(
        self,
        image_paths: list,
        prompt: str,
        model: str = "z-ai/glm-4.6v-flash",
    ):
        """
        Direct image classification using Puter/NVIDIA vision API.
        Bypasses hcaptcha-challenger entirely for simple use cases.
        """
        if not self._custom_provider:
            raise RuntimeError("No LLM provider configured for image classification")
        
        from pathlib import Path
        from puter_provider import PuterProvider
        
        if isinstance(self._custom_provider, PuterProvider):
            from pydantic import BaseModel
            
            class ClassificationResult(BaseModel):
                answer: str
                confidence: float = 0.0
            
            try:
                result = await self._custom_provider.generate_with_images(
                    images=[Path(p) for p in image_paths],
                    response_schema=ClassificationResult,
                    user_prompt=prompt,
                )
                log.info(f"Classification result: answer={result.answer} conf={result.confidence}")
                return result.answer
            except Exception as e:
                # Log what the model actually returned for debugging
                if hasattr(self._custom_provider, '_last_response') and self._custom_provider._last_response:
                    raw = self._custom_provider._last_response.get('result',{}).get('message',{}).get('content','')
                    log.warning(f"Classification failed (raw LLM response): {raw[:300]}")
                log.error(f"Classification failed: {e}")
                # Retry once with simpler prompt
                try:
                    simple_prompt = f"Look at the image and answer: {prompt}\n\nRespond ONLY with a JSON object like {{\"answer\": \"your answer\", \"confidence\": 0.9}}"
                    result = await self._custom_provider.generate_with_images(
                        images=[Path(p) for p in image_paths],
                        response_schema=ClassificationResult,
                        user_prompt=simple_prompt,
                    )
                    log.info(f"Classification retry OK: answer={result.answer}")
                    return result.answer
                except Exception as e2:
                    log.error(f"Classification retry also failed: {e2}")
                    raise
        else:
            log.error(f"Unsupported provider type: {type(self._custom_provider).__name__}, module={type(self._custom_provider).__module__}")
            raise RuntimeError(f"Unsupported provider type: {type(self._custom_provider)}")


# ──────────────────────────────────────────────
# Main Server
# ──────────────────────────────────────────────
class UniversalCaptchaSolver:
    """Unified captcha solver with 2captcha-compatible HTTP API"""

    def __init__(self, api_key: str, port: int = 8855, gemini_key: str = ""):
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
        self._hcaptcha = HcaptchaSolver(gemini_key)

        log.info("Universal Captcha Solver initialized")
        log.info(f"  ddddocr: ✅")
        log.info(f"  Tesseract: {'✅' if self._tesseract.available else '❌'}")
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
        elif task.captcha_type in (CaptchaType.RECAPTCHA_V2, CaptchaType.TURNSTILE):
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

        raise ValueError("Both OCR engines returned empty results")

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
        """Forward reCAPTCHA/Turnstile tasks to dedicated solver servers"""
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
            solver_url = os.environ.get("TURNSTILE_SOLVER_URL", "http://127.0.0.1:8877")
            params = {
                "key": self.api_key,
                "method": "turnstile",
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

        if not result.startswith("OK|"):
            raise ValueError(f"Upstream submit failed: {result}")

        task_id = result.split("|", 1)[1]

        # Poll for result
        poll_url = f"{solver_url}/res.php?key={self.api_key}&id={task_id}"
        for _ in range(60):  # 5 min max
            await asyncio.sleep(5)
            try:
                resp = urllib.request.urlopen(poll_url, timeout=10)
                text = resp.read().decode()
                if text.startswith("OK|"):
                    return text.split("|", 1)[1]
                if "ERROR" in text:
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
        if req_key != solver.api_key:
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

        return web.Response(text=f"OK|{task.task_id}")

    async def res_php(request: web.Request) -> web.Response:
        """2captcha-compatible result polling endpoint"""
        req_key = request.query.get("key", "")
        task_id = request.query.get("id", "")
        action = request.query.get("action", "")

        if req_key != solver.api_key:
            return web.Response(text="ERROR_WRONG_USER_KEY")

        task = solver.tasks.get(task_id)
        if not task:
            # Check if task is still in queue or being processed
            return web.Response(text="CAPCHA_NOT_READY")

        # Store task for polling
        solver.tasks[task_id] = task

        if action == "getbalance":
            return web.Response(text="OK|$0.00")

        if task.status == "solved":
            result = task.token or task.coordinates or ""
            solver.tasks.pop(task_id, None)
            return web.Response(text=f"OK|{result}")

        if task.status == "failed":
            error = task.error or "ERROR_CAPTCHA_UNSOLVABLE"
            solver.tasks.pop(task_id, None)
            return web.Response(text=f"ERROR|{error}")

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
                "hcaptcha": solver._hcaptcha.available,
                "puter_vision": bool(solver._hcaptcha._custom_provider),
                "backend": "custom" if solver._hcaptcha._custom_provider else ("gemini" if solver._hcaptcha._api_key else "none"),
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

    async def classify_endpoint(request: web.Request) -> web.Response:
        """Direct image classification endpoint using Puter vision API"""
        try:
            data = await request.json()
        except Exception:
            return web.json_response({"error": "invalid JSON"}, status=400)
        
        req_key = data.get("key", "")
        if req_key != solver.api_key:
            return web.json_response({"error": "wrong key"}, status=403)
        
        if not solver._hcaptcha._custom_provider:
            return web.json_response({"error": "No vision provider configured"}, status=503)
        
        prompt = data.get("prompt", "Describe what you see in these images")
        images = data.get("images", [])  # list of base64-encoded images
        
        # Write base64 images to temp files
        import tempfile
        temp_paths = []
        try:
            for i, img_b64 in enumerate(images):
                if "," in img_b64:
                    img_b64 = img_b64.split(",", 1)[1]
                img_bytes = base64.b64decode(img_b64)
                f = tempfile.NamedTemporaryFile(suffix=".png", delete=False)
                f.write(img_bytes)
                f.close()
                temp_paths.append(f.name)
            
            result = await solver._hcaptcha.classify_images(
                image_paths=temp_paths,
                prompt=prompt,
            )
            return web.json_response({"result": result})
        except Exception as e:
            return web.json_response({"error": str(e)}, status=500)
        finally:
            for p in temp_paths:
                try:
                    os.unlink(p)
                except:
                    pass

    async def models_endpoint(request: web.Request) -> web.Response:
        """List available Puter AI models"""
        try:
            import urllib.request
            puter_token = os.environ.get("PUTER_TOKEN", "")
            if puter_token:
                req = urllib.request.Request(
                    "https://api.puter.com/puterai/chat/models/details",
                    headers={"Authorization": f"Bearer {puter_token}"}
                )
                resp = urllib.request.urlopen(req, timeout=10)
                models = json.loads(resp.read().decode())
                return web.json_response(models)
            return web.json_response({"error": "No Puter token"}, status=503)
        except Exception as e:
            return web.json_response({"error": str(e)}, status=500)

    app.router.add_post("/in.php", in_php)
    app.router.add_get("/res.php", res_php)
    app.router.add_post("/solve", solve_direct)
    app.router.add_post("/classify", classify_endpoint)
    app.router.add_get("/models", models_endpoint)
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
    parser.add_argument("--puter-token", default="", help="Puter JWT token for free AI API")
    parser.add_argument("--nvidia-key", default="", help="NVIDIA NIM API key")
    args = parser.parse_args()

    # Set env vars from CLI args (so HcaptchaSolver can read them)
    if args.puter_token:
        os.environ["PUTER_TOKEN"] = args.puter_token
    if args.nvidia_key:
        os.environ["NVIDIA_API_KEY"] = args.nvidia_key

    solver = UniversalCaptchaSolver(
        api_key=args.api_key,
        port=args.port,
        gemini_key=args.gemini_key,
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
