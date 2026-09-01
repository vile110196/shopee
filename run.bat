@echo off
chcp 65001 >nul
title HE THONG KHAI THAC DU LIEU E-COMMERCE SHOPEE - UIT

echo ===============================================================================
echo   TRUONG DAI HOC CONG NGHE THONG TIN - DHQG TP.HCM (UIT)
echo   DO AN: KHAI THAC DU LIEU & TRUYEN THONG XA HOI E-COMMERCE SHOPEE
echo   NHOM THUC HIEN:
echo   1. Tran Dinh Huy      - MSSV: 24730103 (Nhom truong)
echo   2. Le Thanh Truc Vi   - MSSV: 24730150 (NLP & Teencode)
echo   3. Vu Hoang Thien An  - MSSV: 24730155 (Gom cum & Luat ket hop)
echo   4. Duong Phuong Anh   - MSSV: 24730156 (Danh gia Mo hinh & UI)
echo ===============================================================================
echo.

:: Kiem tra Python
set PYTHON_CMD=python
where python >nul 2>nul
if %errorlevel% neq 0 (
    if exist "C:\Users\randy\AppData\Local\Programs\Python\Python311\python.exe" (
        set PYTHON_CMD="C:\Users\randy\AppData\Local\Programs\Python\Python311\python.exe"
    ) else if exist "C:\Python311\python.exe" (
        set PYTHON_CMD="C:\Python311\python.exe"
    ) else (
        echo [!] Khong tim thay Python tren he thong. Vui long cai dat Python 3.9 tro len.
        pause
        exit /b
    )
)

echo [1/3] Kiem tra du lieu va mo hinh Machine Learning...
if not exist "models\model_benchmarks.json" (
    echo [!] Chua co mo hinh da huan luyen. Dang tien hanh chay pipeline huan luyen...
    %PYTHON_CMD% model_training.py
)

echo [2/3] Khoi dong May chu Web Flask tai http://127.0.0.1:5000...
echo [3/3] He thong se tu dong mo trinh duyet Web trong giay lat...
echo.
echo ===============================================================================
echo   UNG DUNG DANG CHAY. DE DUNG MAY CHU, NHAN TO HOP PHIM: Ctrl + C
echo ===============================================================================
echo.

%PYTHON_CMD% app.py

pause
