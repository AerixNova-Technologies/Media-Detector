@echo off
setlocal
cd /d %~dp0

echo ──────────────────────────────────────────────────────────
echo        MEDIA DETECTOR - PORTABLE MODE (PRE-BUILT)
echo ──────────────────────────────────────────────────────────
echo.

set PY_EXE=package\python\python.exe
set PG_BIN=package\postgres\bin

:: Check if files exist
if not exist "%PY_EXE%" (
    echo [!] ERROR: Portable Python not found in \package\python.
    echo Please run BUILD_PORTABLE_PACKAGE.ps1 on your PC first!
    pause
    exit /b
)

if not exist "%PG_BIN%\psql.exe" (
    echo [!] ERROR: Portable PostgreSQL not found in \package\postgres.
    echo Please run BUILD_PORTABLE_PACKAGE.ps1 on your PC first!
    pause
    exit /b
)

:: Set Postgres in PATH for the session
set PATH=%~dp0%PG_BIN%;%PATH%

:: Set PYTHONPATH so "app" module can be found
set PYTHONPATH=%~dp0

:: Check if .env is missing and trigger setup if so
if not exist ".env" (
    echo [!] Configuration file ^(.env^) missing. Starting local setup...
    powershell -ExecutionPolicy Bypass -File .\install_windows.ps1
)

:: Start the application using LOCAL python
echo [OK] Starting Media Detector (Portable Mode)...
echo.
%PY_EXE% run.py dev

pause
