@echo off
title MazeMind
setlocal

REM ---------------------------------------------------------------------------
REM  MazeMind launcher - double-click to play.
REM
REM  A virtual environment cannot be copied between machines: .venv\pyvenv.cfg
REM  records an absolute path to the Python that built it, so a .venv carried
REM  over on a USB stick points at a folder that does not exist on the new PC.
REM  This script therefore treats .venv as disposable - it verifies the one it
REM  finds actually works, and rebuilds it from scratch when it does not.
REM ---------------------------------------------------------------------------

cd /d "%~dp0"

if not exist ".venv\Scripts\python.exe" goto :build

REM Import the real dependencies, not just start Python: a .venv copied from
REM another machine can still launch yet fail the moment it loads a package.
".venv\Scripts\python.exe" -c "import pygame, numpy, matplotlib" >nul 2>&1
if errorlevel 1 goto :rebuild
goto :run

:rebuild
echo.
echo   The existing .venv does not work on this PC. Rebuilding it...
echo.
rmdir /s /q ".venv" >nul 2>&1

:build
echo.
echo   Setting up MazeMind for the first time on this PC.
echo   This needs an internet connection and takes a minute or two.
echo.

REM Find any usable Python. "py" is the launcher shipped with python.org
REM installs; "python" covers everything else. The Microsoft Store stub named
REM python.exe reports no version, so the version check filters it out.
set "PYEXE="
py -3 --version >nul 2>&1
if not errorlevel 1 set "PYEXE=py -3"
if defined PYEXE goto :havepython

python --version >nul 2>&1
if not errorlevel 1 set "PYEXE=python"
if defined PYEXE goto :havepython

python3 --version >nul 2>&1
if not errorlevel 1 set "PYEXE=python3"
if defined PYEXE goto :havepython

goto :nopython

:havepython
echo   Using: %PYEXE%
%PYEXE% -m venv ".venv"
if errorlevel 1 goto :venvfailed

".venv\Scripts\python.exe" -m pip install --upgrade pip --quiet
".venv\Scripts\python.exe" -m pip install -r requirements.txt
if errorlevel 1 goto :pipfailed

echo.
echo   Setup complete.
echo.

:run
REM python.exe rather than pythonw.exe on purpose: pythonw hides all output, so
REM a crash would close silently with nothing to read. A console sits behind the
REM game instead and keeps any error on screen.
".venv\Scripts\python.exe" main.py
if errorlevel 1 goto :crashed
exit /b 0

:crashed
echo.
echo   MazeMind exited with an error. The message above says why.
echo.
pause
exit /b 1

:nopython
echo.
echo   No Python installation was found on this PC.
echo.
echo   Install Python 3.10 or newer from https://www.python.org/downloads/
echo   and TICK "Add python.exe to PATH" on the first screen of the installer.
echo.
echo   Then double-click this file again.
echo.
pause
exit /b 1

:venvfailed
echo.
echo   Could not create the virtual environment.
echo   On some systems the "venv" module ships separately - try:
echo       %PYEXE% -m pip install virtualenv
echo.
pause
exit /b 1

:pipfailed
echo.
echo   Could not install the dependencies listed in requirements.txt.
echo   Check that this PC has an internet connection, then try again.
echo.
pause
exit /b 1
