@echo off
REM ==============================================================================
REM 1-Click Launch Script for ROS 2 Edge Perception (Windows Docker Desktop)
REM ==============================================================================

echo ======================================================================
echo  Starting Commercial-Grade ROS 2 Edge Perception Pipeline
echo ======================================================================

where docker >nul 2>nul
if %ERRORLEVEL% neq 0 (
    echo [ERROR] Docker is not installed or not running. Please start Docker Desktop.
    pause
    exit /b 1
)

echo [INFO] Building and starting ROS 2 Perception Stack (CPU Mode)...
docker compose up --build perception-cpu

pause

