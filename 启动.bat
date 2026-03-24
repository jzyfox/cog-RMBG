@echo off
chcp 65001 >nul
title 家具 AI 工具箱
cd /d "%~dp0"

echo ================================================
echo   家具 AI 工具箱  -  启动中
echo ================================================
echo.

:: 检查 Python
python --version >nul 2>&1
if errorlevel 1 (
    echo [错误] 未找到 Python，请确认已安装并添加到系统 PATH。
    echo.
    pause
    exit /b 1
)

:: 用 Python 清理端口 5000（比 bat 解析更可靠）
echo [1/2] 检查并清理旧进程...
python -c "
import subprocess
r = subprocess.run('netstat -ano', capture_output=True, text=True, shell=True)
for line in r.stdout.splitlines():
    if ':5000 ' in line and 'LISTENING' in line:
        pid = line.split()[-1]
        subprocess.run(f'taskkill /F /PID {pid}', shell=True, capture_output=True)
        print(f'  已关闭旧进程 PID: {pid}')
"

echo [2/2] 启动服务...
echo.
echo     浏览器将自动打开：http://localhost:5000
echo     关闭此窗口即可停止服务
echo.
python app.py

echo.
echo 服务已停止。
pause
