@echo off
rem Opens the Humidity Cycles window (before / after photos only; time-lapses
rem are run in Bead Shrinkage).
set "PYW=C:\ProgramData\anaconda3\pythonw.exe"
if not exist "%PYW%" set "PYW=%USERPROFILE%\anaconda3\pythonw.exe"
if not exist "%PYW%" set "PYW=pythonw"
start "" "%PYW%" "%~dp0strain_tools\humidity_gui.py"