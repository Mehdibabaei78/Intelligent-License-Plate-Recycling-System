"""
================================================================================
  وب‌اپلیکیشن دی‌بلور پلاک — Blind Motion Deblurring (License Plate)
--------------------------------------------------------------------------------
  توسعه‌دهنده: مهدی بابایی
  ایمیل:       mehdibabaei1378@gmail.com
  © 2026 مهدی بابایی — تمام حقوق محفوظ است.
================================================================================

main.py
--------
اپلیکیشن FastAPI برای وب‌سایت دی‌بلور پلاک.

endpoints:
    GET   /                 → صفحه‌ی اصلی (SPA)
    GET   /api/health       → وضعیت سرویس و مدل‌ها
    GET   /api/samples      → فهرست تصاویر نمونه‌ی موجود برای دموی یک‌کلیکی
    GET   /api/samples/{id} → دانلود یک تصویر نمونه
    POST  /api/deblur       → دریافت تصویر + برگرداندن فقط خروجی JPEG (ساده)
    POST  /api/deblur/full  → دریافت تصویر + برگرداندن JSON کامل
                             (تصاویر input/fft/psf/output + زاویه و طول)
"""

import os
import time
from fastapi import FastAPI, File, UploadFile, HTTPException
from fastapi.responses import HTMLResponse, StreamingResponse, JSONResponse, FileResponse
from fastapi.staticfiles import StaticFiles
from starlette.middleware.cors import CORSMiddleware

from .inference import DeblurService

_BASE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
_STATIC_DIR = os.path.join(_BASE, "app", "static")
_SAMPLES_DIR = os.path.join(_BASE, "samples")

app = FastAPI(title="دی‌بلور پلاک — Blind Motion Deblurring", version="2.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# سرو کردن فایل‌های استاتیک (JS/CSS آفلاین در صورت نیاز)
app.mount("/static", StaticFiles(directory=_STATIC_DIR), name="static")

# بارگذاری مدل‌ها یک‌بار هنگام استارتاپ
print("[startup] در حال بارگذاری مدل‌های TensorFlow ...")
_deblurrer = DeblurService()
print("[startup] مدل‌ها با موفقیت بارگذاری شدند.")


# ---------------------------------------------------------------------------
@app.get("/", response_class=HTMLResponse)
def index():
    html_path = os.path.join(_STATIC_DIR, "index.html")
    with open(html_path, "r", encoding="utf-8") as f:
        return f.read()


@app.get("/api/health")
def health():
    return {"status": "ok", "models_loaded": True}


@app.get("/api/samples")
def list_samples():
    """فهرست تصاویر نمونه (اسم + id). برای دکمه‌های دموی یک‌کلیکی استفاده می‌شود."""
    samples = []
    if os.path.isdir(_SAMPLES_DIR):
        for i, name in enumerate(sorted(os.listdir(_SAMPLES_DIR))):
            if name.lower().endswith((".jpg", ".jpeg", ".png")):
                samples.append({"id": i, "name": name})
    return {"samples": samples}


@app.get("/api/samples/{sample_id}")
def get_sample(sample_id: int):
    """دانلود یک تصویر نمونه بر اساس id."""
    samples = []
    if os.path.isdir(_SAMPLES_DIR):
        samples = sorted(
            [n for n in os.listdir(_SAMPLES_DIR)
             if n.lower().endswith((".jpg", ".jpeg", ".png"))]
        )
    if sample_id < 0 or sample_id >= len(samples):
        raise HTTPException(status_code=404, detail="نمونه پیدا نشد.")
    path = os.path.join(_SAMPLES_DIR, samples[sample_id])
    return FileResponse(path, media_type="image/jpeg")


@app.post("/api/deblur")
async def deblur(file: UploadFile = File(...)):
    """دی‌بلور ساده: فقط تصویر خروجی JPEG برمی‌گرداند."""
    content = await file.read()
    try:
        out = _deblurrer.deblur_bytes(content)
        return StreamingResponse(
            iter([out]), media_type="image/jpeg",
            headers={"Content-Disposition": "inline; filename=deblurred.jpg"},
        )
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))


@app.post("/api/deblur/full")
async def deblur_full(file: UploadFile = File(...)):
    """دی‌بلور کامل: JSON شامل تصاویر data-URI و پارامترهای تشخیص‌داده‌شده."""
    content = await file.read()
    if len(content) == 0:
        raise HTTPException(status_code=400, detail="فایل خالی است.")
    if len(content) > 15 * 1024 * 1024:
        raise HTTPException(status_code=413, detail="حداکثر اندازه ۱۵ مگابایت است.")
    start = time.time()
    try:
        result = _deblurrer.deblur_full(content)
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    result["processing_ms"] = int((time.time() - start) * 1000)
    return JSONResponse(result)


# اگر فایل مستقیماً اجرا شد (python -m app.main یا uvicorn)
if __name__ == "__main__":
    import uvicorn
    uvicorn.run("app.main:app", host="0.0.0.0", port=8080, reload=False)
