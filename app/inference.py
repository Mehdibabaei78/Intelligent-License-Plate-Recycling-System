"""
================================================================================
  وب‌اپلیکیشن دی‌بلور پلاک — لایه‌ی پردازش (Inference)
--------------------------------------------------------------------------------
  توسعه‌دهنده: مهدی بابایی
  ایمیل:       mehdibabaei1378@gmail.com
  © 2026 مهدی بابایی — تمام حقوق محفوظ است.
================================================================================

inference.py
-------------
لایه‌ی پردازش (inference) برای وب‌اپلیکیشن دی‌بلور پلاک.

این ماژول دقیقاً همان منطق اسکریپت اصلی پروژه (deblur_img.py) را پیاده می‌کند،
اما به شکل یک کلاس قابل استفاده‌ی مجدد که علاوه بر تصویر نهایی، تمام داده‌ی
میان‌فرآیندی (پیش‌بینی زاویه و طول بلور، طیف FFT، و تابع PSF) را هم برمی‌گرداند.

پایپ‌لاین:
    تصویر ورودی → خاکستری → اندازه 640×480
                → FFT → اندازه 224×224 → نرمال‌سازی
                → مدل زاویه و طول (CNN آموزش‌دیده)
                → ساخت PSF از زاویه/طول
                → Wiener Deconvolution
                → تصویر بازیابی‌شده
"""

import os

# ---------------------------------------------------------------------------
# سازگاری با مدل‌های قدیمی (.hdf5 ذخیره‌شده با Keras 2):
# مدل‌های این پروژه با Keras 2 ذخیره شده‌اند. در TensorFlow 2.16+ کِراس
# پیش‌فرض به Keras 3 تغییر کرده که با ساختار این مدل‌ها ناسازگار است.
# با فعال‌کردن TF_USE_LEGACY_KERAS، از بسته‌ی tf_keras (Keras 2) استفاده
# می‌شود که مدل‌ها را بدون خطا بارگذاری می‌کند.
# این متغیر باید قبل از import tensorflow تنظیم شود.
# ---------------------------------------------------------------------------
os.environ.setdefault("TF_USE_LEGACY_KERAS", "1")
# کم‌کردن حجم لاگ‌های TensorFlow
os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "3")

import io
import base64
import numpy as np
import cv2
from PIL import Image
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.image import img_to_array

# مسیر وزن‌های آموزش‌دیده (همان فایل‌هایی که در پوشه‌ی pretrained_models قرار دارند)
_BASE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
ANGLE_MODEL_PATH = os.path.join(_BASE, "pretrained_models", "angle_model.hdf5")
LENGTH_MODEL_PATH = os.path.join(_BASE, "pretrained_models", "length_model.hdf5")


# ----------------------------------------------------------------------------
# توابع اصلی الگوریتم (وام‌گرفته‌شده از deblur_img.py پروژه)
# ----------------------------------------------------------------------------
def build_psf(length, deblur_angle, size=200):
    """ساخت Point-Spread Function از طول و زاویه‌ی بلور حرکتی."""
    length = int(length)
    angle = (deblur_angle * np.pi) / 180.0

    psf = np.ones((1, length), np.float32)  # پایه‌ی خطی PSF
    costerm, sinterm = np.cos(angle), np.sin(angle)
    Ang = np.float32([[-costerm, sinterm, 0], [sinterm, costerm, 0]])
    size2 = size // 2
    Ang[:, 2] = (size2, size2) - np.dot(Ang[:, :2], ((length - 1) * 0.5, 0))
    psf = cv2.warpAffine(psf, Ang, (size, size), flags=cv2.INTER_CUBIC)
    return psf


def process(ip_image, length, deblur_angle):
    """Wiener Deconvolution برای بازیابی تصویر از روی PSF تخمین‌زده‌شده."""
    noise = 0.01
    size = 200
    length = int(length)
    angle = (deblur_angle * np.pi) / 180.0

    psf = np.ones((1, length), np.float32)
    costerm, sinterm = np.cos(angle), np.sin(angle)
    Ang = np.float32([[-costerm, sinterm, 0], [sinterm, costerm, 0]])
    size2 = size // 2
    Ang[:, 2] = (size2, size2) - np.dot(Ang[:, :2], ((length - 1) * 0.5, 0))
    psf = cv2.warpAffine(psf, Ang, (size, size), flags=cv2.INTER_CUBIC)

    gray = np.float32(ip_image) / 255.0
    gray_dft = cv2.dft(gray, flags=cv2.DFT_COMPLEX_OUTPUT)
    psf /= psf.sum()
    psf_mat = np.zeros_like(gray)
    psf_mat[:size, :size] = psf
    psf_dft = cv2.dft(psf_mat, flags=cv2.DFT_COMPLEX_OUTPUT)
    PSFsq = (psf_dft ** 2).sum(-1)
    imgPSF = psf_dft / (PSFsq + noise)[..., np.newaxis]  # فیلتر Wiener (H)
    gray_op = cv2.mulSpectrums(gray_dft, imgPSF, 0)
    gray_res = cv2.idft(gray_op, flags=cv2.DFT_SCALE | cv2.DFT_REAL_OUTPUT)
    gray_res = np.roll(gray_res, -size // 2, 0)
    gray_res = np.roll(gray_res, -size // 2, 1)
    return gray_res


def create_fft(img):
    """محاسبه‌ی طیف فرکانسی (مقدار FFT) برای ورودی مدل و نمایش بصری."""
    img = np.float32(img) / 255.0
    f = np.fft.fft2(img)
    fshift = np.fft.fftshift(f)
    mag_spec = 20 * np.log(np.abs(fshift) + 1e-8)  # +eps برای پایداری عددی
    mag_spec = np.asarray(mag_spec, dtype=np.uint8)
    return mag_spec


def _img_to_b64(img_bgr_or_gray, jpeg_quality=85):
    """تبدیل آرایه‌ی OpenCV به رشته‌ی Base64 (با پیشوند data URI)."""
    ok, buf = cv2.imencode(".jpg", img_bgr_or_gray, [int(cv2.IMWRITE_JPEG_QUALITY), jpeg_quality])
    if not ok:
        return ""
    return "data:image/jpeg;base64," + base64.b64encode(buf.tobytes()).decode("ascii")


def _normalize_gray(gray_float):
    """نرمال‌سازی تصویر خاکستری به بازه‌ی 0..255."""
    g = (gray_float * 255.0).astype(np.float32)
    if np.max(g) != np.min(g):
        g = (255.0 / (np.max(g) - np.min(g))) * (g - np.min(g))
    return g.astype(np.uint8)


# ----------------------------------------------------------------------------
# کلاس سرویس
# ----------------------------------------------------------------------------
class DeblurService:
    """لود یک‌باره‌ی مدل‌ها در ابتدا و ارائه‌ی اینفرنس کامل."""

    def __init__(self, angle_path=ANGLE_MODEL_PATH, length_path=LENGTH_MODEL_PATH):
        self.angle_model = load_model(angle_path, compile=False)
        self.length_model = load_model(length_path, compile=False)

    def _predict_angle_length(self, gray_640x480):
        """پیش‌بینی زاویه و طول بلور از روی طیف FFT."""
        fft_img = create_fft(gray_640x480)
        img_224 = cv2.resize(fft_img, (224, 224))
        x = np.expand_dims(img_to_array(img_224), axis=0) / 255.0

        # مدل زاویه: میانگین ۳ کلاس با بیشترین احتمال (دقیقاً مثل deblur_img.py)
        preds = self.angle_model.predict(x, verbose=0)
        angle_value = float(np.mean(np.argsort(preds[0])[-3:]))

        # مدل طول: خروجی رگرسیون
        length_value = float(self.length_model.predict(x, verbose=0)[0][0])
        return angle_value, length_value, fft_img

    def deblur_full(self, file_bytes: bytes) -> dict:
        """
        پردازش کامل تصویر و برگرداندن یک دیکشنری غنی:
          - input_b64, fft_b64, psf_b64, output_b64  (تصاویر با data-URI)
          - angle, length, processing_ms, shape
        """
        file_array = np.frombuffer(file_bytes, dtype=np.uint8)
        ip_image = cv2.imdecode(file_array, cv2.IMREAD_COLOR)
        if ip_image is None:
            raise ValueError("داده‌ی تصویر نامعتبر است (فایل خراب یا غیر تصویری).")

        # خاکستری + اندازه‌ی استاندارد (مثل اسکریپت اصلی)
        original_bgr = ip_image.copy()
        ip_gray = cv2.cvtColor(ip_image, cv2.COLOR_BGR2GRAY)
        ip_gray = cv2.resize(ip_gray, (640, 480))

        # پیش‌بینی پارامترها + طیف FFT
        angle_value, length_value, fft_img = self._predict_angle_length(ip_gray)

        # دی‌کانولوشن Wiener
        op_image = process(ip_gray, length_value, angle_value)
        op_image = _normalize_gray(op_image)

        # ساخت PSF برای نمایش
        psf = build_psf(length_value, angle_value, size=200)
        psf_view = cv2.normalize(psf, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
        psf_view = cv2.resize(psf_view, (200, 200), interpolation=cv2.INTER_NEAREST)
        psf_view = cv2.applyColorMap(psf_view, cv2.COLORMAP_INFERNO)

        # اندازه‌ی تصویر ورودی برای نمایش
        h, w = original_bgr.shape[:2]

        return {
            "input_b64": _img_to_b64(ip_gray),
            "fft_b64": _img_to_b64(fft_img),
            "psf_b64": _img_to_b64(psf_view),
            "output_b64": _img_to_b64(op_image, jpeg_quality=92),
            "angle": round(angle_value, 1),
            "length": round(length_value, 1),
            "shape": {"width": int(w), "height": int(h)},
        }

    def deblur_bytes(self, file_bytes: bytes) -> bytes:
        """پردازش و برگرداندن فقط بایت‌های تصویر JPEG نهایی (برای endpoint ساده)."""
        res = self.deblur_full(file_bytes)
        header, b64data = res["output_b64"].split(",", 1)
        return base64.b64decode(b64data)
