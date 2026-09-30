@echo off
setlocal

set "ROOT=%~dp0"
set "PATH=C:\msys64\mingw64\bin;C:\msys64\usr\bin;%PATH%"
set "APP=%ROOT%cpp_live\build-mingw64\projectile_live.exe"
set "CONFIG=%ROOT%cpp_live\config\pipeline_default.ini"

if not exist "%APP%" (
    echo ERROR: The C++ application was not found:
    echo %APP%
    echo Build it first with CMake.
    pause
    exit /b 1
)

if not exist "%CONFIG%" (
    echo ERROR: The configuration file was not found:
    echo %CONFIG%
    pause
    exit /b 1
)

cd /d "%ROOT%"
echo Starting Projectile Detection...
echo Configuration: %CONFIG%
echo.
"%APP%" "%CONFIG%"
set "EXIT_CODE=%ERRORLEVEL%"

echo.
echo Application exited with code %EXIT_CODE%.
pause
exit /b %EXIT_CODE%
