@echo off
REM ===========================================================================
REM  Projeto JOI - Server Start Launcher (double-click this file)
REM  Pure ASCII - works on all Windows systems
REM ===========================================================================

REM  Force UTF-8 codepage
chcp 65001 >nul 2>&1

REM  Change to script directory
cd /d "%~dp0"

REM  Set console title
title Projeto JOI - Server

REM  Set colors
color 0F

echo.
echo  ===============================================================
echo    Projeto JOI - Server Start Launcher
echo  ===============================================================
echo.
echo  This will start the uvicorn server on port 8001.
echo  The script verifies that the .venv is healthy first.
echo.
echo  Press any key to continue, or Ctrl+C to abort...
pause >nul

REM  Launch PowerShell with the run script
powershell.exe -NoProfile -ExecutionPolicy Bypass -File "%~dp0run.ps1" %*

REM  Keep window open if server crashes
echo.
echo  Server stopped. Press any key to close...
pause >nul
