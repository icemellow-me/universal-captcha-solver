#!/usr/bin/env python3
"""Test suite for Universal Captcha Solver"""
import asyncio
import base64
import io
import random

import aiohttp
from PIL import Image, ImageDraw, ImageFont


API_KEY = "8010000000ccojr5nrbg516w5jvw1wu9"
BASE_URL = "http://127.0.0.1:8855"


def generate_text_captcha(text: str = "ABCD1234") -> bytes:
    """Generate a test text captcha image"""
    img = Image.new("RGB", (200, 80), color=(255, 255, 255))
    draw = ImageDraw.Draw(img)
    # Draw some noise
    for _ in range(50):
        x1, y1 = random.randint(0, 200), random.randint(0, 80)
        x2, y2 = x1 + random.randint(5, 30), y1 + random.randint(5, 15)
        draw.ellipse([x1, y1, x2, y2], fill=(200, 200, 200))
    # Draw text
    try:
        font = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf", 40)
    except Exception:
        font = ImageFont.load_default()
    draw.text((20, 15), text, fill=(0, 0, 0), font=font)
    # Add lines
    for _ in range(5):
        draw.line(
            [random.randint(0, 200), random.randint(0, 80),
             random.randint(0, 200), random.randint(0, 80)],
            fill=(180, 180, 180), width=2
        )
    buf = io.BytesIO()
    img.save(buf, format="PNG")
    return buf.getvalue()


async def test_health():
    """Test health endpoint"""
    async with aiohttp.ClientSession() as session:
        async with session.get(f"{BASE_URL}/health") as resp:
            data = await resp.json()
            print(f"✅ Health: {data}")
            assert data["status"] == "ok"
            return data


async def test_image_ocr():
    """Test image OCR captcha solving"""
    test_text = "HELLO"
    img_bytes = generate_text_captcha(test_text)

    async with aiohttp.ClientSession() as session:
        data = aiohttp.FormData()
        data.add_field("key", API_KEY)
        data.add_field("method", "image")
        data.add_field("file", img_bytes, filename="captcha.png", content_type="image/png")

        async with session.post(f"{BASE_URL}/in.php", data=data) as resp:
            result = await resp.text()
            print(f"Submit: {result}")
            assert result.startswith("OK|")
            task_id = result.split("|", 1)[1]

        # Poll for result
        for i in range(10):
            await asyncio.sleep(2)
            async with session.get(f"{BASE_URL}/res.php?key={API_KEY}&id={task_id}") as resp:
                result = await resp.text()
                if result.startswith("OK|"):
                    token = result.split("|", 1)[1]
                    print(f"✅ Image OCR solved: '{token}' (expected ~'{test_text}')")
                    return True
                if "ERROR" in result:
                    print(f"❌ Error: {result}")
                    return False
                print(f"  [{i}] waiting...")

        print("❌ Timeout")
        return False


async def test_base64_ocr():
    """Test base64 image captcha solving"""
    img_bytes = generate_text_captcha("TEST42")
    b64 = base64.b64encode(img_bytes).decode()

    async with aiohttp.ClientSession() as session:
        data = aiohttp.FormData()
        data.add_field("key", API_KEY)
        data.add_field("method", "base64")
        data.add_field("body", b64)

        async with session.post(f"{BASE_URL}/in.php", data=data) as resp:
            result = await resp.text()
            print(f"Submit: {result}")
            task_id = result.split("|", 1)[1] if result.startswith("OK|") else ""

        if not task_id:
            print(f"❌ Submit failed: {result}")
            return False

        for i in range(10):
            await asyncio.sleep(2)
            async with session.get(f"{BASE_URL}/res.php?key={API_KEY}&id={task_id}") as resp:
                result = await resp.text()
                if result.startswith("OK|"):
                    token = result.split("|", 1)[1]
                    print(f"✅ Base64 OCR solved: '{token}'")
                    return True
                if "ERROR" in result:
                    print(f"❌ Error: {result}")
                    return False

        return False


async def test_direct_solve():
    """Test direct JSON solve endpoint"""
    img_bytes = generate_text_captcha("DIRECT")
    b64 = base64.b64encode(img_bytes).decode()

    async with aiohttp.ClientSession() as session:
        async with session.post(
            f"{BASE_URL}/solve",
            json={"type": "image", "image_base64": b64},
        ) as resp:
            data = await resp.json()
            print(f"✅ Direct solve: {data}")
            return data.get("status") == "solved"


async def test_forward_turnstile():
    """Test forwarding Turnstile to dedicated solver"""
    async with aiohttp.ClientSession() as session:
        data = aiohttp.FormData()
        data.add_field("key", API_KEY)
        data.add_field("method", "turnstile")
        data.add_field("sitekey", "0x4AAAAAAAaxsixy3iY0aOjP")
        data.add_field("pageurl", "https://peet.ws/turnstile-test/non-interactive.html")

        async with session.post(f"{BASE_URL}/in.php", data=data) as resp:
            result = await resp.text()
            print(f"Submit: {result}")
            if not result.startswith("OK|"):
                print(f"❌ Submit failed: {result}")
                return False
            task_id = result.split("|", 1)[1]

        for i in range(60):
            await asyncio.sleep(5)
            async with session.get(f"{BASE_URL}/res.php?key={API_KEY}&id={task_id}") as resp:
                result = await resp.text()
                if result.startswith("OK|"):
                    token = result.split("|", 1)[1]
                    print(f"✅ Turnstile forwarded & solved! Token ({len(token)} chars): {token[:40]}...")
                    return True
                if "ERROR" in result:
                    print(f"❌ Error: {result}")
                    return False
                if i % 5 == 0:
                    print(f"  [{i}] waiting...")

        print("❌ Timeout")
        return False


async def test_forward_recaptcha():
    """Test forwarding reCAPTCHA v2 to dedicated solver"""
    async with aiohttp.ClientSession() as session:
        data = aiohttp.FormData()
        data.add_field("key", API_KEY)
        data.add_field("method", "userrecaptcha")
        data.add_field("googlekey", "6Le-wvkSAAAAAPBMRTvw0Q4Muexq9GH0L6XecxIS")
        data.add_field("pageurl", "https://www.google.com/recaptcha/api2/demo")
        data.add_field("version", "v2")

        async with session.post(f"{BASE_URL}/in.php", data=data) as resp:
            result = await resp.text()
            print(f"Submit: {result}")
            if not result.startswith("OK|"):
                print(f"❌ Submit failed: {result}")
                return False
            task_id = result.split("|", 1)[1]

        for i in range(60):
            await asyncio.sleep(5)
            async with session.get(f"{BASE_URL}/res.php?key={API_KEY}&id={task_id}") as resp:
                result = await resp.text()
                if result.startswith("OK|"):
                    token = result.split("|", 1)[1]
                    print(f"✅ reCAPTCHA v2 forwarded & solved! Token ({len(token)} chars): {token[:40]}...")
                    return True
                if "ERROR" in result:
                    print(f"❌ Error: {result}")
                    return False
                if i % 5 == 0:
                    print(f"  [{i}] waiting...")

        print("❌ Timeout")
        return False


async def main():
    tests = [
        ("Health Check", test_health),
        ("Image OCR", test_image_ocr),
        ("Base64 OCR", test_base64_ocr),
        ("Direct JSON Solve", test_direct_solve),
        ("Forward Turnstile", test_forward_turnstile),
        ("Forward reCAPTCHA v2", test_forward_recaptcha),
    ]

    results = {}
    for name, test_fn in tests:
        print(f"\n{'='*50}")
        print(f"Running: {name}")
        print(f"{'='*50}")
        try:
            result = await test_fn()
            results[name] = "✅ PASS" if result else "❌ FAIL"
        except Exception as e:
            results[name] = f"❌ ERROR: {e}"
            import traceback
            traceback.print_exc()

    print(f"\n{'='*50}")
    print("RESULTS")
    print(f"{'='*50}")
    for name, result in results.items():
        print(f"  {name}: {result}")


if __name__ == "__main__":
    asyncio.run(main())
