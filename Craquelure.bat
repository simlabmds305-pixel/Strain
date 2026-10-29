@echo off
rem Double-click to start the Craquelure app. Keep this file next to
rem craquelure_app.py, craquelure_extract.py and bucklow_classify.py.
rem Tip: right-click > Send to > Desktop (create shortcut) for a desktop icon.

cd /d "%~dp0"
set "PY="
for %%P in ("%USERPROFILE%\anaconda3\python.exe" "%USERPROFILE%\miniconda3\python.exe" "%LOCALAPPDATA%\anaconda3\python.exe" "C:\ProgramData\anaconda3\python.exe" "C:\ProgramData\miniconda3\python.exe") do (
    if not defined PY if exist %%P set "PY=%%~P"
)
if not defined PY (
    where python >nul 2>nul && set "PY=python"
)
if not defined PY (
    echo Could not find Python / Anaconda. Install Anaconda, or edit this file to point at python.exe.
    pause
    exit /b 1
)

rem first run: install anything missing (window-capable OpenCV, scikit-image, Pillow)
"%PY%" -c "import cv2, skimage, PIL; cv2.namedWindow('t'); cv2.destroyWindow('t')" >nul 2>nul
if errorlevel 1 (
    echo Installing the packages the app needs - this happens only once...
    "%PY%" -m pip uninstall -y opencv-python-headless opencv-contrib-python-headless >nul 2>nul
    "%PY%" -m pip install opencv-python numpy scikit-image pillow
)

set "PYW=%PY:python.exe=pythonw.exe%"
if exist "%PYW%" (
    start "" "%PYW%" "%~dp0craquelure_app.py"
) else (
    "%PY%" "%~dp0craquelure_app.py"
    if errorlevel 1 pause
)
