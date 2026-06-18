#!/bin/bash
# Install script for Universal Captcha Solver
# Handles: image/OCR, hCaptcha, Turnstile, reCAPTCHA v2, coordinate captchas
set -e

echo "=== Universal Captcha Solver Installer ==="

# Check Python
if ! command -v python3 &>/dev/null; then
    echo "❌ Python3 not found. Install: apt install python3"
    exit 1
fi

# Install system deps
echo "📦 Installing system dependencies..."
sudo apt-get update -qq
sudo apt-get install -yqq python3-pip tesseract-ocr xvfb 2>/dev/null || true

# Install Python packages
echo "📦 Installing Python packages..."
pip install --break-system-packages ddddocr pytesseract aiohttp pillow hcaptcha-challenger 2>/dev/null || \
  pip3 install --break-system-packages ddddocr pytesseract aiohttp pillow hcaptcha-challenger 2>/dev/null || \
  python3 -m pip install ddddocr pytesseract aiohttp pillow hcaptcha-challenger

# Verify
echo "🔍 Verifying installation..."
python3 -c "
import ddddocr; print('✅ ddddocr')
import pytesseract; print('✅ pytesseract')
import aiohttp; print('✅ aiohttp')
from PIL import Image; print('✅ pillow')
" || { echo "❌ Import check failed"; exit 1; }

echo ""
echo "=== Installation Complete ==="
echo ""
echo "Start the server:"
echo "  python3 solver-server.py --api-key YOUR_KEY --port 8855"
echo ""
echo "For hCaptcha support (optional):"
echo "  python3 solver-server.py --api-key YOUR_KEY --port 8855 --gemini-key YOUR_GEMINI_KEY"
echo ""
echo "API endpoints:"
echo "  POST /in.php     - Submit captcha (2captcha-compatible)"
echo "  GET  /res.php    - Poll for result (2captcha-compatible)"
echo "  POST /solve      - Direct JSON solve"
echo "  GET  /health     - Health check"
