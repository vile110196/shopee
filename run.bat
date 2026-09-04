@echo off
setlocal
chcp 65001 >nul
cd /d "%~dp0"
title SHOPEE ANALYTICS LAB - UIT

echo ===============================================================================
echo   SHOPEE ANALYTICS LAB - KHAI THAC DU LIEU VA TRUYEN THONG XA HOI
echo ===============================================================================
echo.

set "BOOTSTRAP_PY="
where py >nul 2>nul
if not errorlevel 1 (
    py -3.11 -c "import sys" >nul 2>nul
    if not errorlevel 1 set "BOOTSTRAP_PY=py -3.11"
)

if not defined BOOTSTRAP_PY (
    where python >nul 2>nul
    if not errorlevel 1 set "BOOTSTRAP_PY=python"
)

if not defined BOOTSTRAP_PY (
    echo [LOI] Khong tim thay Python 3.11 tro len trong PATH.
    pause
    exit /b 1
)

if not exist ".venv\Scripts\python.exe" (
    echo [1/4] Tao moi truong ao .venv...
    %BOOTSTRAP_PY% -m venv .venv
    if errorlevel 1 goto :fail
) else (
    echo [1/4] Da tim thay moi truong ao .venv.
)

set "VENV_PY=%CD%\.venv\Scripts\python.exe"

echo [2/4] Kiem tra thu vien Python...
"%VENV_PY%" -c "import flask, joblib, numpy, pandas, sklearn, scipy; assert sklearn.__version__ == '1.9.0'" >nul 2>nul
if errorlevel 1 (
    echo       Dang cai dat requirements.txt, lan dau co the mat vai phut...
    "%VENV_PY%" -m pip install --disable-pip-version-check -r requirements.txt
    if errorlevel 1 goto :fail
)

echo [3/4] Kiem tra ba bo du lieu Kaggle...
if not exist "data\provenance.json" (
    echo       Chua co du lieu; dang tai va kiem checksum cac ban version 1...
    "%VENV_PY%" import_public_data.py --download
    if errorlevel 1 goto :fail
)

echo       Kiem tra du lieu va artifact mo hinh...
"%VENV_PY%" -c "import app; print('      Du lieu va mo hinh hop le.')"
if errorlevel 1 goto :fail

echo [4/4] Khoi dong http://127.0.0.1:5000 ...
echo       Nhan Ctrl+C de dung may chu.
echo.
"%VENV_PY%" app.py
if errorlevel 1 goto :fail

exit /b 0

:fail
echo.
echo [LOI] Khoi dong that bai. Xem thong bao ngay phia tren.
pause
exit /b 1
