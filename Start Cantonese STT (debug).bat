@echo off
title Cantonese Speech to Text (debug)
cd /d "%~dp0app"
echo Starting the app - this window shows every message.
echo Close this window or press Ctrl+C to stop the app.
echo.
"..\python\python.exe" app.py
echo.
echo The app has stopped.
pause
