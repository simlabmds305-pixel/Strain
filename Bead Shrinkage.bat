@echo off
rem Opens the Bead Shrinkage window. Double-click this, or the desktop shortcut.
rem Uses Anaconda's Python if it is there (plain "python" is not on this PC's PATH).
set "PYW=C:\ProgramData\anaconda3\pythonw.exe"
if not exist "%PYW%" set "PYW=%USERPROFILE%\anaconda3\pythonw.exe"
if not exist "%PYW%" set "PYW=pythonw"
start "" "%PYW%" "%~dp0strain_tools\gui.py"
