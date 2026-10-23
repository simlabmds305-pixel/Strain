@echo off
rem Trace the bead by hand on the first (dark, wet) frames of a set that was
rem already run in the Bead Shrinkage window. Separate from the main app.
set "PYW=C:\ProgramData\anaconda3\pythonw.exe"
if not exist "%PYW%" set "PYW=%USERPROFILE%\anaconda3\pythonw.exe"
if not exist "%PYW%" set "PYW=pythonw"
start "" "%PYW%" "%~dp0strain_tools\manual_trace.py"
