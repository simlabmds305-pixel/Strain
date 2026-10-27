@echo off
rem Every bead set side by side: comparative_master.csv, findings.txt and plots.
rem Reads each set's analysis\ folder; changes nothing. Separate from the main app.
set "PYW=C:\ProgramData\anaconda3\pythonw.exe"
if not exist "%PYW%" set "PYW=%USERPROFILE%\anaconda3\pythonw.exe"
if not exist "%PYW%" set "PYW=pythonw"
start "" "%PYW%" "%~dp0strain_tools\master_analysis.py"
