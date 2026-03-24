@echo off
chcp 65001 >nul
title 家具 AI 工具箱

:: 杀掉占用 5000 端口的旧进程
for /f "tokens=5" %%a in ('netstat -ano ^| findstr ":5000 " ^| findstr "LISTENING"') do (
    taskkill /PID %%a /F >nul 2>&1
)

:: 切换到脚本所在目录
cd /d "%~dp0"

:: 启动
echo 正在启动家具 AI 工具箱...
python app.py

pause
