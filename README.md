# Universal Captcha Solver — Self-Hosted API

Self-hosted, 2captcha-compatible captcha solving server. Handles **image OCR**, **hCaptcha**, **reCAPTCHA v2**, and **Cloudflare Turnstile** — with built-in forwarding to dedicated solver backends.

## Architecture

```
                        ┌──────────────────────────────────────────┐
                        │   Universal Solver (:8855)                │
  POST /in.php ───────►│   ddddocr + Tesseract OCR                │
  GET  /res.php        │   Cloudflare Workers AI Vision (hCaptcha) │
  POST /solve          │   hcaptcha-challenger + CF dual-model     │
                        │   + upstream forwarder                   │
                        └──────┬──────────┬──────────┬───────────┘
                               │          │          │
                    ┌──────────┘          │          └──────────┐
                    ▼                     ▼                     ▼
          ┌─────────────────┐  ┌──────────────────┐  ┌──────────────────┐
          │ Turnstile :8878 │  │ reCAPTCHA :8866   │  │ xCaptcha :8899   │
          │ Playwright +     │  │ Playwright +      │  │ VLM + API leaks  │
          │ Headless Chrome  │  │ CaptchaPlugin ext │  │ (2captcha-compat)│
          └─────────────────┘  └──────────────────┘  └──────────────────┘
```

> **OCR Tier System:** ddddocr → Tesseract → **Cloudflare Workers AI Vision** (VLM fallback for hard captchas, math captchas, and xCaptcha)

> **hCaptcha Dual-Model Strategy:** Challenge routing uses the stronger **llama-3.1-70b-instruct** (text-only, better at structured JSON), image classification uses **llama-3.2-11b-vision-instruct** (needs to see images). Automatically falls back when the primary model fails or echoes schemas.

> **Chrome Extension** uses a **separate instance** on port **:8844** (with `json=1` support) that forwards to `:8833` (reCAPTCHA) and `:8822` (Turnstile). See [Extension-Specific Instance](#extension-specific-instance-json1).

---

## API Endpoints

### Health Check

```
GET /health
```

```json
{
  "status": "ok",
  "queue": 0,
  "solved": 42,
  "failed": 1,
  "active": 2,
  "engines": {
    "ddddocr": true,
    "tesseract": true,
    "hcaptcha": true,
    "hcaptcha_model": "@cf/meta/llama-3.2-11b-vision-instruct",
    "hcaptcha_fallback": "@cf/meta/llama-3.1-70b-instruct"
  }
}
```

---

### 2captcha-Compatible API

#### Submit Task — `POST /in.php`

All captcha types use the same endpoint. The `method` parameter determines the solving engine.

**Common Parameters:**

| Parameter | Required | Description |
|---|---|---|
| `key` | ✅ | API key (default: `801000...`) |
| `method` | ✅ | Captcha type (see below) |
| `json` | ⬜ | Set to `1` for JSON response (extension instances only) |
| `soft_id` | ⬜ | Client identifier |

**Response (plain text, default):**
```
OK|a1b2c3d4e5f6...
```

**Response (with `json=1`, extension instances only):**
```json
{"status": 1, "request": "a1b2c3d4e5f6..."}
```

---

#### 1. Image OCR — `method=image`

Solve a captcha image file upload. Returns the OCR text.

```bash
# File upload
curl -X POST http://YOUR_SERVER:8855/in.php \
  -F "key=YOUR_API_KEY" \
  -F "method=image" \
  -F "file=@captcha.png"
```

**Response:** `OK|task_id`

---

#### 2. Base64 Image OCR — `method=base64`

Solve a captcha image sent as base64 string. **The `body` parameter must be raw base64 — no `data:image/png;base64,` prefix.**

```bash
# From a base64 string
curl -X POST http://YOUR_SERVER:8855/in.php \
  -d "key=YOUR_API_KEY" \
  -d "method=base64" \
  -d "body=iVBORw0KGgoAAAANSUhEUgAAAMQAAABYCAMAAACd3Khh..."

# With json=1 (extension instance on :8844)
curl -X POST http://YOUR_SERVER:8844/in.php \
  -d "key=YOUR_API_KEY" \
  -d "method=base64" \
  -d "body=iVBORw0KGgoAAAANSUhEUgAAAMQAAABYCAMAAACd3Khh..." \
  -d "json=1"
```

**Python example:**

```python
import base64, requests, time

API_URL = "http://23.22.196.74:8855"
API_KEY = "8010000000ccojr5nrbg516w5jvw1wu9"

# Read image and encode to base64
with open("captcha.png", "rb") as f:
    b64 = base64.b64encode(f.read()).decode()

# Submit
resp = requests.post(f"{API_URL}/in.php", data={
    "key": API_KEY,
    "method": "base64",
    "body": b64,
})
task_id = resp.text.split("|", 1)[1]  # "OK|task_id"
print(f"Task ID: {task_id}")

# Poll for result
for _ in range(30):
    time.sleep(2)
    r = requests.get(f"{API_URL}/res.php", params={
        "key": API_KEY,
        "id": task_id,
    })
    if r.text == "CAPCHA_NOT_READY":
        continue
    if r.text.startswith("OK|"):
        print(f"Solution: {r.text[3:]}")
        break
    else:
        print(f"Error: {r.text}")
        break
```

**Direct `/solve` endpoint (synchronous, JSON in/out):**

```bash
curl -X POST http://YOUR_SERVER:8855/solve \
  -H "Content-Type: application/json" \
  -d '{
    "type": "base64",
    "image_base64": "iVBORw0KGgoAAAANSUhEUgAAAMQAAABYCAMAAACd3Khh..."
  }'
```

```json
{
  "status": "solved",
  "task_id": "b604d676...",
  "solution": "x7Kp2",
  "type": "image_ocr",
  "solve_time": 1.23
}
```

---

#### 3. Image URL — `method=image` + `captcha_img`

Download and OCR a captcha image from a URL.

```bash
curl -X POST http://YOUR_SERVER:8855/in.php \
  -d "key=YOUR_API_KEY" \
  -d "method=image" \
  -d "captcha_img=https://example.com/captcha.jpg"
```

---

#### 4. reCAPTCHA v2 — `method=userrecaptcha`

Solves reCAPTCHA v2 by forwarding to the dedicated Playwright + CaptchaPlugin backend.

| Parameter | Required | Description |
|---|---|---|
| `method` | ✅ | `userrecaptcha` |
| `googlekey` (or `sitekey`) | ✅ | reCAPTCHA site key from page |
| `pageurl` | ✅ | Full URL of the page with the captcha |

```bash
curl -X POST http://YOUR_SERVER:8855/in.php \
  -d "key=YOUR_API_KEY" \
  -d "method=userrecaptcha" \
  -d "googlekey=6Le-wvkSAAAAAPBMRTvw0Q4Mu3qRI2_9UIsghFot" \
  -d "pageurl=https://example.com/login"
```

**Response:** `OK|task_id` → poll `/res.php` for the token string.

---

#### 5. Cloudflare Turnstile — `method=turnstile`

Solves Cloudflare Turnstile challenges by forwarding to the dedicated Playwright backend.

| Parameter | Required | Description |
|---|---|---|
| `method` | ✅ | `turnstile` |
| `sitekey` | ✅ | Turnstile site key |
| `pageurl` | ✅ | Full URL of the page with the widget |

```bash
curl -X POST http://YOUR_SERVER:8855/in.php \
  -d "key=YOUR_API_KEY" \
  -d "method=turnstile" \
  -d "sitekey=0x4AAAAAAADnPIDsw" \
  -d "pageurl=https://example.com/protected"
```

---

#### 6. hCaptcha — `method=hcaptcha`

Solves hCaptcha challenges using **hcaptcha-challenger + Cloudflare Workers AI** (dual-model strategy).

**How it works:**
1. The **70b-instruct** (fallback) model classifies the challenge type (e.g., "image_label_binary")
2. The **11b-vision** model analyzes the challenge images and selects correct answers
3. If either model fails or echoes the schema, it auto-retries with a simplified prompt
4. Solve time: ~96-130 seconds per challenge

| Parameter | Required | Description |
|---|---|---|
| `method` | ✅ | `hcaptcha` |
| `sitekey` | ✅ | hCaptcha site key |
| `pageurl` | ✅ | Full URL of the page |

```bash
curl -X POST http://YOUR_SERVER:8855/in.php \
  -d "key=YOUR_API_KEY" \
  -d "method=hcaptcha" \
  -d "sitekey=2880e342-5e8b-4a0c-9c4f-5a253c6d1ee3" \
  -d "pageurl=https://accounts.hcaptcha.com/demo"
```

---

#### 7. Coordinate Captcha — `method=coord`

Returns click coordinates for image-based captchas (e.g. "click all traffic lights").

```bash
curl -X POST http://YOUR_SERVER:8855/in.php \
  -F "key=YOUR_API_KEY" \
  -F "method=coord" \
  -F "file=@grid_captcha.png"
```

**Response:** `OK|[[x1,y1],[x2,y2],...]` — click coordinates as JSON array.

---

### Poll for Result — `GET /res.php`

| Parameter | Required | Description |
|---|---|---|
| `key` | ✅ | API key |
| `id` | ✅ | Task ID from `/in.php` response |
| `json` | ⬜ | `1` for JSON response (extension instances only) |

```bash
# Plain text
curl "http://YOUR_SERVER:8855/res.php?key=YOUR_API_KEY&id=TASK_ID"
# → "CAPCHA_NOT_READY" (still solving)
# → "OK|solution_token_or_text" (done)
# → "ERROR|error_message" (failed)

# JSON mode (extension instance :8844)
curl "http://YOUR_SERVER:8844/res.php?key=YOUR_API_KEY&id=TASK_ID&json=1"
# → {"status": 0, "request": "CAPCHA_NOT_READY"}
# → {"status": 1, "request": "solution_token_or_text"}
# → {"status": 0, "request": "error_message"}
```

**Typical polling pattern:**

```python
import requests, time

for _ in range(60):  # max 2 minutes
    time.sleep(5)
    r = requests.get(f"{API_URL}/res.php", params={
        "key": API_KEY, "id": task_id
    })
    if "NOT_READY" in r.text:
        continue
    if r.text.startswith("OK|"):
        print(f"Result: {r.text[3:]}")
        break
    else:
        print(f"Failed: {r.text}")
        break
```

---

### Direct Solve — `POST /solve`

Synchronous endpoint — submits and waits for the result in one request (up to 120s).

```bash
# Image/base64
curl -X POST http://YOUR_SERVER:8855/solve \
  -H "Content-Type: application/json" \
  -d '{
    "type": "base64",
    "image_base64": "iVBORw0KGgo..."
  }'

# reCAPTCHA
curl -X POST http://YOUR_SERVER:8855/solve \
  -H "Content-Type: application/json" \
  -d '{
    "type": "userrecaptcha",
    "googlekey": "6Le-wvkSAAAAAPBMRTvw0Q4Mu3qRI2_9UIsghFot",
    "pageurl": "https://example.com/login"
  }'

# Turnstile
curl -X POST http://YOUR_SERVER:8855/solve \
  -H "Content-Type: application/json" \
  -d '{
    "type": "turnstile",
    "sitekey": "0x4AAAAAAADnPIDsw",
    "pageurl": "https://example.com/protected"
  }'

# hCaptcha
curl -X POST http://YOUR_SERVER:8855/solve \
  -H "Content-Type: application/json" \
  -d '{
    "type": "hcaptcha",
    "sitekey": "a5f74b19-9e45-40e0-b60d-7ba8d4e0",
    "pageurl": "https://example.com/page"
  }'
```

**Success response:**
```json
{
  "status": "solved",
  "task_id": "b604d6764e844408bc49a0fe",
  "solution": "03AGdBq26...",
  "type": "recaptcha_v2",
  "solve_time": 12.45
}
```

**Failure response:**
```json
{
  "status": "failed",
  "task_id": "b604d6764e844408bc49a0fe",
  "error": "timeout after 120s"
}
```

---

## Extension-Specific Instance (json=1)

The Chrome extension requires `json=1` support (2captcha spec). A **separate instance** runs on different ports to not modify the original servers:

| Service | Extension Port | Original Port |
|---|---|---|
| Universal Solver | **8844** | 8855 |
| reCAPTCHA v2 | **8833** | 8866 |
| Turnstile | **8822** | 8878 |

The extension-specific universal solver (`:8844`) forwards to `:8833` and `:8822` via the Docker gateway.

**Original ports (8855/8866/8878)** return plain-text responses only — `OK|id`, `CAPCHA_NOT_READY`, etc. Use these for scripts and non-extension integrations.

**Extension ports (8844/8833/8822)** support `json=1` — pass `&json=1` (GET) or `-d "json=1"` (POST) to get structured JSON responses:
- Submit: `{"status": 1, "request": "task_id"}` or `{"status": 0, "request": "ERROR_*"}`
- Poll: `{"status": 1, "request": "solution"}` or `{"status": 0, "request": "CAPCHA_NOT_READY"}`

---

## Error Codes

| Code | Meaning |
|---|---|
| `ERROR_WRONG_USER_KEY` | Invalid API key |
| `ERROR_WRONG_PARAMETER` | Missing or invalid parameter |
| `ERROR_BAD_PARAMETERS` | Malformed request body |
| `ERROR_NO_SUCH_TASK` | Task ID not found |
| `CAPCHA_NOT_READY` | Task still processing — keep polling |
| `ERROR_CAPTCHA_UNSOLVABLE` | Solver failed to produce a result |

---

## Docker Deployment

### Quick Start

```bash
# Build from source
docker build -t universal-captcha-solver .

# Run
docker run -d --name universal-captcha-solver \
  -p 8855:8855 \
  -e RECAPTCHA_SOLVER_URL=http://172.17.0.1:8866 \
  -e TURNSTILE_SOLVER_URL=http://172.17.0.1:8878 \
  universal-captcha-solver
```

### Environment Variables

| Variable | Default | Description |
|---|---|---|
| `API_KEY` | — | Authentication key (also set via `--api-key` flag) |
| `CF_API_TOKEN` | — | Cloudflare Workers AI token (cfut_... format) |
| `CF_ACCOUNT_ID` | — | Cloudflare account ID for Workers AI |
| `CF_HCAPTCHA_FALLBACK_MODEL` | `@cf/meta/llama-3.1-70b-instruct` | Stronger text-only model for challenge routing |
| `NVIDIA_API_KEY` | — | NVIDIA NIM API key (alternate VLM provider) |
| `RECAPTCHA_SOLVER_URL` | — | URL of reCAPTCHA v2 solver for forwarding |
| `TURNSTILE_SOLVER_URL` | — | URL of Turnstile solver for forwarding |
| `PORT` (via `--port` flag) | `8855` | Server listen port |

### CLI Flags

```
python3 solver-server.py --api-key YOUR_KEY --port 8855 \
  --cf-api-token cfut_xxx --cf-account-id xxx
```

---

## OCR Engines

The universal solver uses a **dual-engine approach** for image captchas:

1. **ddddocr** (primary) — Fast Chinese OCR library, great for alphanumeric captchas
2. **Tesseract** (fallback) — Classic OCR engine via pytesseract

Both engines run in parallel and results are combined for higher accuracy.

---

## Full API Flow Example (base64 image OCR)

```bash
SERVER="http://23.22.196.74:8855"
KEY="8010000000ccojr5nrbg516w5jvw1wu9"

# Step 1: Submit base64 image
TASK_ID=$(curl -s -X POST "$SERVER/in.php" \
  -d "key=$KEY" \
  -d "method=base64" \
  -d "body=iVBORw0KGgoAAAANSUhEUgAAAMQAAABYCAMAAACd3Khh..." \
  | cut -d'|' -f2)

echo "Submitted task: $TASK_ID"

# Step 2: Poll for result (every 3 seconds)
while true; do
  RESULT=$(curl -s "$SERVER/res.php?key=$KEY&id=$TASK_ID")
  if [[ "$RESULT" == "CAPCHA_NOT_READY" ]]; then
    echo "Still solving..."
    sleep 3
  elif [[ "$RESULT" == OK* ]]; then
    echo "Solved: ${RESULT#OK|}"
    break
  else
    echo "Error: $RESULT"
    break
  fi
done
```

**Same flow with `json=1` (extension instance on :8844):**

```bash
SERVER="http://23.22.196.74:8844"
KEY="8010000000ccojr5nrbg516w5jvw1wu9"

# Step 1: Submit + get JSON response
TASK_ID=$(curl -s -X POST "$SERVER/in.php" \
  -d "key=$KEY" \
  -d "method=base64" \
  -d "body=iVBORw0KGgoAAAANSUhEUgAAAMQAAABYCAMAAACd3Khh..." \
  -d "json=1" | python3 -c "import sys,json; print(json.load(sys.stdin)['request'])")

echo "Submitted task: $TASK_ID"

# Step 2: Poll with JSON responses
while true; do
  RESULT=$(curl -s "$SERVER/res.php?key=$KEY&id=$TASK_ID&json=1")
  STATUS=$(echo "$RESULT" | python3 -c "import sys,json; d=json.load(sys.stdin); print(d['status'])")
  REQUEST=$(echo "$RESULT" | python3 -c "import sys,json; d=json.load(sys.stdin); print(d['request'])")

  if [[ "$REQUEST" == "CAPCHA_NOT_READY" ]]; then
    echo "Still solving..."
    sleep 3
  elif [[ "$STATUS" == "1" ]]; then
    echo "Solved: $REQUEST"
    break
  else
    echo "Error: $REQUEST"
    break
  fi
done
```

---

## Repos

- **[universal-captcha-solver](https://github.com/icemellow-me/universal-captcha-solver)** — This server (OCR + forwarding hub)
- **[recaptcha-v2-solver](https://github.com/icemellow-me/recaptcha-v2-solver)** — reCAPTCHA v2 (Playwright + CaptchaPlugin)
- **[turnstile-solver](https://github.com/icemellow-me/turnstile-solver)** — Cloudflare Turnstile (Playwright)
- **[captcha-solver-extension](https://github.com/icemellow-me/captcha-solver-extension)** — Chrome extension (auto-detect + solve)

## License

MIT
