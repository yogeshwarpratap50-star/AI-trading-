@echo off
setlocal

cd /d "%~dp0"
powershell -ExecutionPolicy Bypass -File "%~dp0scripts\run_dashboard.ps1"
