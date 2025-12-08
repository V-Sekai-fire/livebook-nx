@echo off
REM Build meshoptimizer DLL for Windows
REM Based on GitHub Actions workflow

echo Building meshoptimizer DLL for Windows...

REM Check if meshoptimizer directory exists
if not exist "meshoptimizer" (
    echo Cloning meshoptimizer repository...
    git clone https://github.com/zeux/meshoptimizer.git
    if errorlevel 1 (
        echo Error: Failed to clone meshoptimizer repository
        exit /b 1
    )
)

REM Create build directory
cd meshoptimizer
if not exist "build" mkdir build
cd build

REM Configure CMake
echo Configuring CMake...
cmake .. -DCMAKE_BUILD_TYPE=Release -DMESHOPT_BUILD_SHARED_LIBS=ON
if errorlevel 1 (
    echo Error: CMake configuration failed
    echo Make sure CMake is installed and in your PATH
    exit /b 1
)

REM Build
echo Building meshoptimizer...
cmake --build . --config Release
if errorlevel 1 (
    echo Error: Build failed
    exit /b 1
)

REM Copy DLL to blender-meshoptimizer root
cd ..\..
if exist "meshoptimizer\build\Release\meshoptimizer.dll" (
    copy "meshoptimizer\build\Release\meshoptimizer.dll" "meshoptimizer.dll"
    echo.
    echo Success! meshoptimizer.dll has been copied to thirdparty\blender-meshoptimizer\
) else (
    echo Error: meshoptimizer.dll not found in build output
    echo Expected location: meshoptimizer\build\Release\meshoptimizer.dll
    exit /b 1
)

echo.
echo Build complete! The DLL is ready to use.

