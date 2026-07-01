# 📖 راهنمای نصب و اجرا | Setup & Run Guide

<div align="center">

# 🚗 وب‌اپلیکیشن دی‌بلور پلاک
### Blind Motion Deblurring Web App

**👨‍💻 توسعه‌دهنده: مهدی بابایی | Developer: Mehdi Babaei**
**📧 mehdibabaei1378@gmail.com**

</div>

---

<div dir="rtl">

## 🇮🇷 راهنمای فارسی

### پیش‌نیازها

| مورد | نسخه |
|------|------|
| Python | 3.8 تا 3.11 |
| TensorFlow | 2.12 (پیشنهادی) یا 2.18 |
| CUDA (اختیاری) | برای شتاب GPU |

### روش اول: اجرای آسان با محیط Conda (پیشنهادی) ⭐

اگر از قبل محیط `deblur-web` را دارید، ساده‌ترین راه:

```bash
# ۱. ورود به پوشه‌ی وب‌اپلیکیشن
cd webapp

# ۲. فعال‌سازی محیط مجازی
conda activate deblur-web

# ۳. اجرای سرور (روی پورت 8080)
python -m uvicorn app.main:app --host 0.0.0.0 --port 8080
```

سپس مرورگر را باز کنید: **http://127.0.0.1:8080**

#### راه دابل‌کلیک (ویندوز)
کافی است روی فایل **`webapp\run.bat`** دابل‌کلیک کنید. همه‌چیز خودکار انجام می‌شود:
- فعال‌سازی محیط `deblur-web`
- اجرای سرور روی پورت 8080
- باز شدن خودکار مرورگر

#### راه جایگزین با اسکریپت پایتون
```bash
cd webapp
python run.py
```
این اسکریپت خودش محیط `deblur-web` را پیدا کرده و سرور را اجرا می‌کند.

---

### روش دوم: نصب از صفر (بدون محیط کاندا)

```bash
# ۱. ساخت محیط مجازی
python -m venv venv
venv\Scripts\activate        # ویندوز
# source venv/bin/activate   # لینوکس/مک

# ۲. ورود به پوشه‌ی وب‌اپلیکیشن
cd webapp

# ۳. نصب وابستگی‌ها
pip install -r requirements.txt

# ۴. اجرای سرور
python -m uvicorn app.main:app --host 0.0.0.0 --port 8080
```

سپس: **http://127.0.0.1:8080**

---

### نکته‌ی مهم درباره‌ی نسخه‌ی TensorFlow

مدل‌های این پروژه (`angle_model.hdf5` و `length_model.hdf5`) با **Keras 2** ذخیره شده‌اند.

- ✅ **TensorFlow 2.12** (با Keras 2.12): بدون هیچ تنظیمی کار می‌کند.
- ⚠️ **TensorFlow 2.16+** (با Keras 3): باید پکیج `tf_keras` را نصب کنید و متغیر محیطی `TF_USE_LEGACY_KERAS=1` را تنظیم کنید:

```bash
pip install tf_keras
# سپس قبل از اجرا:
set TF_USE_LEGACY_KERAS=1    # ویندوز CMD
export TF_USE_LEGACY_KERAS=1 # لینوکس/مک
```

> کد پروژه این متغیر را به‌صورت خودکار در `app/inference.py` تنظیم می‌کند، پس معمولاً نیازی به کار دستی نیست.

---

### استفاده از وب‌سایت

1. **آپلود عکس**: عکس ماشین با پلاک بلور را بکشید و رها کنید (یا کلیک کنید)
2. **نمونه آماده**: روی یکی از تصاویر نمونه در بخش «نمونه‌های آماده» کلیک کنید
3. **دی‌بلور**: دکمه‌ی «دی‌بلور کن ✨» را بزنید
4. **مقایسه**: اسلایدر را با موس بکشید تا قبل/بعد را مقایسه کنید
5. **دانلود**: خروجی نهایی را با دکمه «⬇ دانلود خروجی» ذخیره کنید

---

### تغییر پورت سرور

اگر پورت 8080 اشغال است، می‌توانید پورت دیگری استفاده کنید:

```bash
python -m uvicorn app.main:app --host 0.0.0.0 --port 9000
# سپس به http://127.0.0.1:9000 بروید
```

---

### رفع اشکال

| مشکل | راه‌حل |
|------|--------|
| `ModuleNotFoundError` | مطمئن شوید محیط `deblur-web` فعال است |
| خطای لود مدل | از TensorFlow 2.12 استفاده کنید یا `TF_USE_LEGACY_KERAS=1` |
| پورت اشغال است | پورت دیگری مثل 9000 استفاده کنید |
| کندی پردازش | GPU/CUDA را فعال کنید |

</div>

---

## 🇬🇧 English Guide

### Prerequisites

| Requirement | Version |
|-------------|---------|
| Python | 3.8 to 3.11 |
| TensorFlow | 2.12 (recommended) or 2.18 |
| CUDA (optional) | For GPU acceleration |

### Method 1: Easy Run with Conda (Recommended) ⭐

If you already have the `deblur-web` environment:

```bash
# 1. Navigate to webapp folder
cd webapp

# 2. Activate the virtual environment
conda activate deblur-web

# 3. Start the server (on port 8080)
python -m uvicorn app.main:app --host 0.0.0.0 --port 8080
```

Then open your browser at: **http://127.0.0.1:8080**

#### Double-Click Method (Windows)
Simply double-click the file **`webapp\run.bat`**. Everything is automatic:
- Activates the `deblur-web` environment
- Starts the server on port 8080
- Opens the browser automatically

#### Alternative: Python Script
```bash
cd webapp
python run.py
```
This script auto-detects the `deblur-web` environment and starts the server.

---

### Method 2: Fresh Install (without Conda)

```bash
# 1. Create a virtual environment
python -m venv venv
venv\Scripts\activate        # Windows
# source venv/bin/activate   # Linux/macOS

# 2. Navigate to webapp folder
cd webapp

# 3. Install dependencies
pip install -r requirements.txt

# 4. Start the server
python -m uvicorn app.main:app --host 0.0.0.0 --port 8080
```

Then visit: **http://127.0.0.1:8080**

---

### Important: TensorFlow Version Note

The project's models (`angle_model.hdf5` and `length_model.hdf5`) were saved with **Keras 2**.

- ✅ **TensorFlow 2.12** (with Keras 2.12): Works out of the box.
- ⚠️ **TensorFlow 2.16+** (with Keras 3): Install `tf_keras` and set `TF_USE_LEGACY_KERAS=1`:

```bash
pip install tf_keras
# Then before running:
set TF_USE_LEGACY_KERAS=1    # Windows CMD
export TF_USE_LEGACY_KERAS=1 # Linux/macOS
```

> The codebase sets this variable automatically in `app/inference.py`, so manual setup is usually not needed.

---

### Using the Website

1. **Upload**: Drag & drop a car image with a blurred license plate (or click)
2. **Sample**: Click one of the sample images in the "Ready Samples" section
3. **Deblur**: Click the "Deblur ✨" button
4. **Compare**: Drag the slider to compare before/after
5. **Download**: Save the result with the "Download" button

---

### Change the Port

If port 8080 is busy, use another port:

```bash
python -m uvicorn app.main:app --host 0.0.0.0 --port 9000
# Then visit http://127.0.0.1:9000
```

---

### Troubleshooting

| Issue | Solution |
|-------|----------|
| `ModuleNotFoundError` | Ensure `deblur-web` environment is activated |
| Model load error | Use TensorFlow 2.12 or set `TF_USE_LEGACY_KERAS=1` |
| Port in use | Use a different port like 9000 |
| Slow processing | Enable GPU/CUDA |

---

<div align="center">

**© 2026 Mehdi Babaei — All Rights Reserved**
📧 mehdibabaei1378@gmail.com

</div>
