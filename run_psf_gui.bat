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

if exist "%CD%\.venv\Scripts\python.exe" (
    set "PYTHON_EXE=%CD%\.venv\Scripts\python.exe"
) else if exist "%CD%\venv\Scripts\python.exe" (
    set "PYTHON_EXE=%CD%\venv\Scripts\python.exe"
) else if exist "%CD%\env\Scripts\python.exe" (
    set "PYTHON_EXE=%CD%\env\Scripts\python.exe"
) else (
    where py >nul 2>nul
    if not errorlevel 1 (
        set "PYTHON_EXE=py"
        set "PYTHON_ARGS=-3"
    ) else (
        where python >nul 2>nul
        if not errorlevel 1 (
            set "PYTHON_EXE=python"
        )
    )
)

if not defined PYTHON_EXE (
    echo Python was not found.
    echo Install Python 3, or create a local virtual environment in:
    echo   .venv\Scripts\python.exe
    pause
    exit /b 1
)

echo Starting PSF Simulator...
%PYTHON_EXE% %PYTHON_ARGS% "%APP%" %*
set "EXIT_CODE=%ERRORLEVEL%"

if not "%EXIT_CODE%"=="0" (
    echo.
    echo The program exited with code %EXIT_CODE%.
    echo Make sure the required Python packages are installed for this environment.
    pause
)

exit /b %EXIT_CODE%
