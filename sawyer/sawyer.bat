@echo off
title Sawyer — Distributed MoE Inference Network
echo.
echo  Sawyer - Distributed MoE Inference Network
echo  The load is split. Friends help.
echo.
echo  What would you like to do?
echo.
echo    1  Chat          Start the chat client (web UI)
echo    2  Serve         Start a Sawyer inference node
echo    3  Ollama+Sawyer Use Ollama through the Sawyer network
echo    4  Chat+Serve    Chat locally and serve to the network
echo    5  Ollama+Serve  Ollama through Sawyer and serve to network
echo    6  All           Turn it all on
echo.
set /p choice="  Choose [1-6]: "

if "%choice%"=="1" (
    echo.
    echo  Starting Sawyer chat client...
    start /b python -m sawyer chat %*
    timeout /t 2 /nobreak >nul
    start http://localhost:8000
    echo  Browser opening at http://localhost:8000
    echo  Close this window to stop Sawyer.
    echo.
    pause >nul
    goto :eof
)
if "%choice%"=="2" (
    echo.
    echo  Starting Sawyer inference node...
    echo.
    python -m sawyer serve %*
    if errorlevel 1 (
        echo.
        echo  Sawyer exited with an error.
        echo  Make sure sawyer-core is installed: pip install sawyer-core
        echo.
        pause
    )
    goto :eof
)
if "%choice%"=="3" (
    echo.
    echo  Starting Ollama+Sawyer bridge...
    start /b python -m sawyer chat --ollama-bridge %*
    timeout /t 2 /nobreak >nul
    start http://localhost:8000
    echo  Browser opening at http://localhost:8000
    echo  Close this window to stop Sawyer.
    echo.
    pause >nul
    goto :eof
)
if "%choice%"=="4" (
    echo.
    echo  Starting chat client and inference node...
    start /b python -m sawyer serve %*
    timeout /t 3 /nobreak >nul
    start /b python -m sawyer chat %*
    timeout /t 2 /nobreak >nul
    start http://localhost:8000
    echo  Browser opening at http://localhost:8000
    echo  Close this window to stop Sawyer.
    echo.
    pause >nul
    goto :eof
)
if "%choice%"=="5" (
    echo.
    echo  Starting Ollama bridge and inference node...
    start /b python -m sawyer serve %*
    timeout /t 3 /nobreak >nul
    start /b python -m sawyer chat --ollama-bridge %*
    timeout /t 2 /nobreak >nul
    start http://localhost:8000
    echo  Browser opening at http://localhost:8000
    echo  Close this window to stop Sawyer.
    echo.
    pause >nul
    goto :eof
)
if "%choice%"=="6" (
    echo.
    echo  Turning it all on...
    start /b python -m sawyer serve %*
    timeout /t 3 /nobreak >nul
    start /b python -m sawyer chat %*
    timeout /t 2 /nobreak >nul
    start http://localhost:8000
    echo  Browser opening at http://localhost:8000
    echo  Close this window to stop Sawyer.
    echo.
    pause >nul
    goto :eof
)
echo.
echo  Invalid choice. Exiting.
pause