# Universal Captcha Solver

Self-hosted captcha solving API supporting multiple captcha types with a **2captcha-compatible HTTP interface**.

## Supported Captcha Types

| Type | Method | Engine | Speed |
|------|--------|--------|-------|
| **Image/OCR** | `image` | ddddocr (ONNX) + Tesseract | ~1s |
| **Base64 Image** | `base64` | ddddocr + Tesseract | ~1s |
| **hCaptcha** | `hcaptcha` | hcaptcha-challenger (ONNX + LLM) | ~30s |
| **reCAPTCHA v2** | `userrecaptcha` | CaptchaPlugin Extension | ~55s |
| **Cloudflare Turnstile** | `turnstile` | Headless Chromium | ~10s |
| **Coordinate** | `coord` | ddddocr Detection | ~2s |

## Quick Start

```bash
# Install
chmod +x install.sh && ./install.sh

# Start server
python3 solver-server.py --api-key YOUR_KEY --port 8855

# With hCaptcha support (requires Gemini API key)
python3 solver-server.py --api-key YOUR_KEY --port 8855 --gemini-key YOUR_GEMINI_KEY
```

## API Reference

### 2captcha-Compatible Endpoints

**Submit task:** `POST /in.php`

| Parameter | Type | Description |
|-----------|------|-------------|
| `key` | string | API key |
| `method` | string | `image`, `base64`, `userrecaptcha`, `turnstile`, `hcaptcha`, `coord` |
| `file` | file | Image file upload (for `image` method) |
| `body` | string | Base64-encoded image (for `base64` method) |
| `sitekey` | string | Site key (for reCAPTCHA/Turnstile/hCaptcha) |
| `googlekey` | string | Google site key (reCAPTCHA, alias for sitekey) |
| `pageurl` | string | Page URL (for reCAPTCHA/Turnstile/hCaptcha) |
| `version` | string | `v2` (default) or `v3` (reCAPTCHA) |

**Poll result:** `GET /res.php?key=YOUR_KEY&id=TASK_ID`

- Returns `OK|<solution>` when solved
- Returns `CAPCHA_NOT_READY` when still processing
- Returns `ERROR|<message>` on failure

### Direct JSON Endpoint

**POST /solve** — JSON in, JSON out

```json
{
  "type": "image",
  "image_base64": "<base64_encoded_image>"
}
```

```json
{
  "type": "turnstile",
  "sitekey": "0x4AAAAAAAaxsixy3iY0aOjP",
  "pageurl": "https://example.com"
}
```

Response:
```json
{
  "status": "solved",
  "task_id": "abc123...",
  "solution": "captchat_token_or_text",
  "type": "image",
  "solve_time": 1.33
}
```

### Health Check

`GET /health`

```json
{
  "status": "ok",
  "queue": 0,
  "solved": 42,
  "failed": 2,
  "active": 1,
  "engines": {
    "ddddocr": true,
    "tesseract": true,
    "hcaptcha": true
  }
}
```

## Architecture

```
                    ┌─────────────────────────┐
                    │  Universal Solver (8855) │
                    │  ┌─────────┐ ┌────────┐ │
  POST /in.php ──► │  │ ddddocr │ │Tessract│ │ ──► Image/OCR
                    │  └─────────┘ └────────┘ │
                    │  ┌────────────────────┐  │
  POST /solve  ──► │  │ hcaptcha-challenger │  │ ──► hCaptcha
                    │  └────────────────────┘  │
                    │  ┌─────────────────────┐ │
                    │  │ Forward to upstream  │ │ ──► Turnstile (8877)
                    │  │ solvers via HTTP     │ │ ──► reCAPTCHA (8866)
                    │  └─────────────────────┘ │
                    └─────────────────────────┘
```

## Examples

### Image OCR Captcha
```bash
# File upload
curl -X POST http://localhost:8855/in.php \
  -F "key=YOUR_KEY" \
  -F "method=image" \
  -F "file=@captcha.png"

# Base64
curl -X POST http://localhost:8855/in.php \
  -d "key=YOUR_KEY" \
  -d "method=base64" \
  --data-urlencode "body=$(base64 -w0 captcha.png)"
```

### Turnstile
```bash
curl -X POST http://localhost:8855/in.php \
  -F "key=YOUR_KEY" \
  -F "method=turnstile" \
  -F "sitekey=0x4AAAAAAAaxsixy3iY0aOjP" \
  -F "pageurl=https://example.com/page"
```

### reCAPTCHA v2
```bash
curl -X POST http://localhost:8855/in.php \
  -F "key=YOUR_KEY" \
  -F "method=userrecaptcha" \
  -F "googlekey=6Le-wvkS..." \
  -F "pageurl=https://example.com/page"
```

### Direct JSON Solve
```bash
curl -X POST http://localhost:8855/solve \
  -H "Content-Type: application/json" \
  -d '{"type":"image","image_base64":"..."}'
```

## Forwarding to Dedicated Solvers

The universal solver can forward reCAPTCHA and Turnstile requests to dedicated solver servers:

```bash
# Set these env vars before starting
export RECAPTCHA_SOLVER_URL=http://127.0.0.1:8866
export TURNSTILE_SOLVER_URL=http://127.0.0.1:8877

python3 solver-server.py --api-key YOUR_KEY --port 8855
```

## Restarting (Safe)

**Never use `pkill -f solver-server.py`** — it will kill ALL solver servers. Instead, use exact paths:

```bash
# Kill only universal solver
pkill -f "universal-captcha-solver/solver-server"

# Kill only turnstile solver  
pkill -f "turnstile-solver/solver-server"

# Kill only reCAPTCHA solver
pkill -f "recaptcha-v2-solver/recaptcha-playwright-server"
```

## License

MIT
