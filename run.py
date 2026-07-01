"""
================================================================================
  وب‌اپلیکیشن دی‌بلور پلاک — نقطه‌ی ورود اجرا
--------------------------------------------------------------------------------
  توسعه‌دهنده: مهدی بابایی
  ایمیل:       mehdibabaei1378@gmail.com
  © 2026 مهدی بابایی — تمام حقوق محفوظ است.
================================================================================

run.py
-------
نقطه‌ی ورود ساده برای اجرای وب‌اپلیکیشن.

روش اجرا (از داخل پوشه‌ی webapp):
    python run.py

این اسکریپت ترجیحاً از محیط مجازی Conda به نام «deblur-web» استفاده می‌کند
(اگر موجود باشد)؛ در غیر این‌صورت از همان پایتونی که با آن اجرا می‌شود استفاده
می‌کند. سپس مرورگر را روی http://127.0.0.1:8080 باز می‌کند.

برای توقف: Ctrl + C
"""

import os
import sys
import subprocess
import shutil
import webbrowser
import threading

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
WEBAPP_DIR = os.path.dirname(BASE_DIR)

# نام محیط مجازی Conda که وابستگی‌ها در آن نصب شده‌اند
CONDA_ENV = "deblur-web"


def find_conda_python(env_name):
    """یافتن مسیر python.exe محیط کاندا روی ویندوز."""
    candidates = [
        os.path.join(os.path.expanduser("~"), "anaconda3", "envs", env_name, "python.exe"),
        os.path.join(os.path.expanduser("~"), "miniconda3", "envs", env_name, "python.exe"),
        os.path.join("C:\\", "ProgramData", "anaconda3", "envs", env_name, "python.exe"),
        os.path.join("C:\\", "ProgramData", "miniconda3", "envs", env_name, "python.exe"),
    ]
    for c in candidates:
        if os.path.isfile(c):
            return c
    # تلاش با conda env list
    conda = shutil.which("conda")
    if conda:
        try:
            out = subprocess.check_output([conda, "env", "list"], text=True)
            for line in out.splitlines():
                parts = line.split()
                if len(parts) >= 2 and parts[0] == env_name:
                    return os.path.join(parts[-1], "python.exe")
        except Exception:
            pass
    return None


def main():
    os.chdir(BASE_DIR)

    python_exe = find_conda_python(CONDA_ENV)

    if python_exe:
        print(f"[run] از محیط مجازی «{CONDA_ENV}» استفاده می‌شود:")
        print(f"      {python_exe}")
        # اجرای uvicorn با همان پایتون
        cmd = [python_exe, "-m", "uvicorn", "app.main:app",
               "--host", "0.0.0.0", "--port", "8080"]
    else:
        # fallback: همان پایتون فعلی
        print(f"[run] محیط «{CONDA_ENV}» پیدا نشد؛ از پایتون فعلی استفاده می‌شود:")
        print(f"      {sys.executable}")
        print(f"      (اگر خطای مدل داد، ابتدا conda activate {CONDA_ENV} را بزن)")
        cmd = [sys.executable, "-m", "uvicorn", "app.main:app",
               "--host", "0.0.0.0", "--port", "8080"]

    # باز کردن خودکار مرورگر پس از ۳ ثانیه
    threading.Timer(3.0, lambda: webbrowser.open("http://127.0.0.1:8080")).start()

    print("=" * 60)
    print("  دی‌بلور پلاک در حال اجراست...")
    print("  آدرس: http://127.0.0.1:8080")
    print("  برای توقف: Ctrl + C")
    print("=" * 60)

    subprocess.call(cmd, cwd=BASE_DIR)


if __name__ == "__main__":
    main()
