@echo off
rem Lumen launcher for Windows.
rem
rem   scripts\run_lumen.bat [lumen options] [images...]      e.g.  scripts\run_lumen.bat --port 9000 cells.tif
rem
rem First run: creates .venv in the project folder (py -3 -m venv) and installs Lumen into it.
rem Later runs: just starts Lumen.  Needs Python 3.11 or newer (python.org installer with
rem "Add python.exe to PATH" ticked, or the Microsoft Store build).  Environment variables:
rem   LUMEN_VENV  virtual environment to use (default: <project>\.venv)
rem   LUMEN_HOME  Lumen's per-user folder for recent files, plugins and the sample cache
rem               (default: %USERPROFILE%\.lumen)
setlocal
set "ROOT=%~dp0.."
set "VENV=%ROOT%\.venv"
if defined LUMEN_VENV set "VENV=%LUMEN_VENV%"
set "PY=%VENV%\Scripts\python.exe"

if not exist "%PY%" (
    echo Creating a virtual environment in "%VENV%" ...
    py -3 -m venv "%VENV%" 2>nul || python -m venv "%VENV%"
    if not exist "%PY%" goto :nopython
)

"%PY%" -c "import lumen" >nul 2>&1
if errorlevel 1 (
    echo Installing Lumen into "%VENV%" ...
    "%PY%" -m pip install --upgrade pip
    "%PY%" -m pip install -e "%ROOT%" || goto :fail
)

"%PY%" -m lumen %*
exit /b %errorlevel%

:nopython
echo Python 3.11 or newer was not found.  Install it from https://www.python.org/downloads/windows/
echo (tick "Add python.exe to PATH") or from the Microsoft Store, then run this script again.
exit /b 1

:fail
echo Lumen could not be installed into "%VENV%".  See the messages above.
exit /b 1
