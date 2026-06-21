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
            solver_url = os.environ.get("TURNSTILE_SOLVER_URL", "http://127.0.0.1:8877")
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
    args = parser.parse_args()

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
