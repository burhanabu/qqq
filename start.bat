@echo off
setlocal
cd /d "%~dp0"
title AccentLive Pro

rem --- find a real Python 3.10-3.12 (the Microsoft Store shortcut does not count) ---
set "PY="

rem 1) plain "python" first (works even if the py launcher registry entry is broken)
python -c "import sys;sys.exit(0 if (3,10)<=sys.version_info[:2]<=(3,12) else 1)" >nul 2>nul
if not errorlevel 1 set "PY=python"

rem 2) py launcher, but only if it can really run Python
if not defined PY (
  for %%V in (3.11 3.12 3.10) do (
    if not defined PY (
      py -%%V -c "import sys" >nul 2>nul
      if not errorlevel 1 set "PY=py -%%V"
    )
  )
)

if not defined PY (
  echo.
  echo A usable Python was not found. Python 3.11 is needed.
  where winget >nul 2>nul
  if not errorlevel 1 (
    echo Installing Python 3.11 with winget ...
    winget install -e --id Python.Python.3.11 --accept-package-agreements --accept-source-agreements
    echo.
    echo Finished. CLOSE this window, then double-click start.bat again.
  ) else (
    echo Opening the Python download page.
    echo Download the Windows installer 64-bit, and on the first screen TICK: Add python.exe to PATH
    start "" https://www.python.org/downloads/release/python-3119/
    echo After installing, CLOSE this window and double-click start.bat again.
  )
  pause
  exit /b 1
)

echo Using: %PY%

rem remove a broken/half-made environment
if exist venv\Scripts\python.exe (
  venv\Scripts\python.exe -c "import sys" >nul 2>nul
  if errorlevel 1 (
    echo Old environment is broken - rebuilding ...
    rmdir /s /q venv
  )
)
if exist venv if not exist venv\Scripts\python.exe rmdir /s /q venv
if not exist venv\Scripts\python.exe (
  echo Creating environment ...
  %PY% -m venv venv
)
if not exist venv\Scripts\python.exe (
  echo Could not create the environment.
  pause
  exit /b 1
)

if not exist venv\.ready (
  echo Installing packages - first time only, takes a few minutes ...
  venv\Scripts\python.exe -m pip install --upgrade pip
  venv\Scripts\python.exe -m pip install -r requirements.txt
  if errorlevel 1 (
    echo.
    echo Install failed. Check your internet and run start.bat again.
    pause
    exit /b 1
  )
  echo ok> venv\.ready
)

venv\Scripts\python.exe app.py
pause
