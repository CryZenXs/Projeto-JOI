@echo off
REM ===========================================================================
REM  Projeto JOI - Windows Setup Launcher (double-click this file)
REM  Pure ASCII version - works on all Windows systems
REM ===========================================================================

REM  Force UTF-8 codepage to avoid garbled output from child processes
chcp 65001 >nul 2>&1

REM  Change to the directory of this script
cd /d "%~dp0"

REM  Set console title
title Projeto JOI - Setup

REM  Set colors (white on dark background)
color 0F

echo.
echo  ===============================================================
echo    Projeto JOI - Windows Setup Launcher
echo  ===============================================================
echo.
echo  This will launch the PowerShell setup script that:
echo    1. Detects/installs Python 3.12+, Git, Docker, Ollama
echo    2. Creates a virtual environment and installs dependencies
echo    3. Asks you for the Groq API key (get one free at
echo       https://console.groq.com)
echo    4. Generates the .env configuration file
echo    5. Runs the test suite to validate everything
echo    6. Optionally starts the development server
echo.
echo  Press any key to continue, or Ctrl+C to abort...
pause >nul

REM  Launch PowerShell with the setup script
REM  -ExecutionPolicy Bypass: allows this script to run without changing policy
REM  -NoExit: keep window open after script finishes (so user can read output)
REM  -File: run our setup script
powershell.exe -NoProfile -ExecutionPolicy Bypass -NoExit -File "%~dp0setup.ps1" %*

REM  If PowerShell exits, give the user a chance to read any error
echo.
echo  Setup script finished. Press any key to close this window...
pause >nul
