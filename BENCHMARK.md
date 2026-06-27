# Captcha Solver Speed Benchmark

Results from benchmarking all self-hosted captcha solvers.

**Date:** 2026-06-27
**VPS:** 23.22.196.74
**API:** 2captcha-compatible (`/in.php` + `/res.php`)

## Results

### Run 1

| Solver | Status | Time | Submit | Solve |
|---|---|---|---|---|
| xCaptcha (direct) | ✅ Solved | 4.118s | 0.109s | 4.009s |
| Universal→xCaptcha | ✅ Solved | 6.143s | 0.106s | 6.037s |
| reCAPTCHA v2 | ✅ Solved | 80.292s | 0.122s | 80.17s |
| Turnstile | ❌ Failed | — | — | ERROR_CAPTCHA_UNSOLVABLE |

### Run 2

| Solver | Status | Time | Submit | Solve |
|---|---|---|---|---|
| xCaptcha (direct) | ✅ Solved | 4.113s | 0.092s | 4.021s |
| Universal→xCaptcha | ✅ Solved | 6.115s | 0.087s | 6.028s |
| reCAPTCHA v2 | ✅ Solved | 96.287s | 0.085s | 96.202s |
| Turnstile | ❌ Failed | — | — | ERROR_CAPTCHA_UNSOLVABLE |

### Run 3 (earlier, via Docker network)

| Solver | Status | Time |
|---|---|---|
| Universal→xCaptcha | ✅ Solved | 6.404s |
| xCaptcha (direct) | ✅ Solved | 10.263s |
| reCAPTCHA v2 | ✅ Solved | 72.272s |
| Turnstile | ❌ Failed | ERROR_CAPTCHA_UNSOLVABLE |

## Rankings

🥇 **xCaptcha (direct :8899)** — ~4s average  
Fast direct solver. VLM + API leaks for text/custom/empty types.

🥈 **Universal→xCaptcha (:8855)** — ~6s average  
Routes through the universal solver. ~2s overhead from forwarding.

🥉 **reCAPTCHA v2 (:8866)** — ~80-96s average  
Uses CaptchaPlugin WS addon. Slow due to real browser interaction.

❌ **Turnstile (:8877)** — UNSOLVABLE  
Cloudflare detects headless Playwright. Found Turnstile container div but iframe never loads. Needs better stealth patches (undetected-playwright or CDP approach).

## Solver Ports

- **:8899** — xCaptcha direct
- **:8855** — Universal solver (image OCR + forwards to all backends)
- **:8866** — reCAPTCHA v2 (Playwright + CaptchaPlugin WS)
- **:8877** — Turnstile (Playwright)
- **:8844** — Extension-specific universal instance (json=1 support)
- **:8833** — Extension reCAPTCHA relay
- **:8822** — Extension Turnstile relay

## Notes

- xCaptcha supports text (~70-80% VLM accuracy), custom (100% API leak), empty (100% API leak), dynamics (not yet supported)
- reCAPTCHA uses the `googlekey` param (not `sitekey`)
- Turnstile's "Found container (no iframe yet)" means Cloudflare blocked the headless browser before rendering the widget
