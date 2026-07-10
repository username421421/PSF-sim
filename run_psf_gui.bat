@echo off
setlocal

cd /d "%~dp0"

set "APP=psf_gui.py"
set "PYTHON_EXE="
set "PYTHON_ARGS="

if not exist "%APP%" (
    echo Could not find "%APP%" in:
    echo   %CD%
    pause
    exit /b 1
)

if exist "%CD%\.venv\Scripts\python.exe" set "PYTHON_EXE=%CD%\.venv\Scripts\python.exe"
if not defined PYTHON_EXE if exist "%CD%\venv\Scripts\python.exe" set "PYTHON_EXE=%CD%\venv\Scripts\python.exe"
if not defined PYTHON_EXE if exist "%CD%\env\Scripts\python.exe" set "PYTHON_EXE=%CD%\env\Scripts\python.exe"

if defined PYTHON_EXE (
    "%PYTHON_EXE%" -c "import sys;sys.exit(0 if sys.version_info>=(3,10) else 1)" >nul 2>nul
    if errorlevel 1 set "PYTHON_EXE="
)

if not defined PYTHON_EXE (
    python -c "import sys;sys.exit(0 if sys.version_info>=(3,10) else 1)" >nul 2>nul
    if not errorlevel 1 set "PYTHON_EXE=python"
)

if not defined PYTHON_EXE for /f "tokens=2,*" %%A in ('reg query "HKCU\Software\Microsoft\Windows\CurrentVersion\App Paths\python.exe" /ve 2^>nul ^| find "REG_SZ"') do set "PYTHON_EXE=%%B"
if defined PYTHON_EXE (
    "%PYTHON_EXE%" -c "import sys;sys.exit(0 if sys.version_info>=(3,10) else 1)" >nul 2>nul
    if errorlevel 1 set "PYTHON_EXE="
)

if not defined PYTHON_EXE for /f "tokens=2,*" %%A in ('reg query "HKLM\Software\Microsoft\Windows\CurrentVersion\App Paths\python.exe" /ve 2^>nul ^| find "REG_SZ"') do set "PYTHON_EXE=%%B"
if defined PYTHON_EXE (
    "%PYTHON_EXE%" -c "import sys;sys.exit(0 if sys.version_info>=(3,10) else 1)" >nul 2>nul
    if errorlevel 1 set "PYTHON_EXE="
)

if not defined PYTHON_EXE (
    py -3 -c "import sys;sys.exit(0 if sys.version_info>=(3,10) else 1)" >nul 2>nul
    if not errorlevel 1 (
        set "PYTHON_EXE=py"
        set "PYTHON_ARGS=-3"
    )
)

if not defined PYTHON_EXE (
    echo Python was not found.
    echo Install Python 3.10 or newer, or create a local virtual environment in:
    echo   .venv\Scripts\python.exe
    if "%~1"=="" pause
    exit /b 1
)

echo Starting PSF Simulator...
"%PYTHON_EXE%" %PYTHON_ARGS% "%APP%" %*
set "EXIT_CODE=%ERRORLEVEL%"

if not "%EXIT_CODE%"=="0" (
    echo.
    echo The program exited with code %EXIT_CODE%.
    echo Make sure the required Python packages are installed for this environment.
    if "%~1"=="" pause
)

exit /b %EXIT_CODE%
