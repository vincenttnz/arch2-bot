@echo off
title SenseNova AI Suite
echo [1/2] Starting AI Server (SenseNova)...
:: 'start' opens a new window for the server
start "SenseNova Server" uv run python sensenova.py

echo [2/2] Waiting for server to initialize...
timeout /t 5 /nobreak > nul

echo Launching Labeler...
uv run python ai_labeler.py

echo.
echo ========================================
echo Labeling Complete. Check the summary above.
echo ========================================
pause