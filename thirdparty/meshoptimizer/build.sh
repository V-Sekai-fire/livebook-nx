#!/bin/bash
# Build script for meshoptimizer shared library

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

# Create build directory
mkdir -p build
cd build

# Configure with CMake
cmake .. -DCMAKE_BUILD_TYPE=Release

# Build
cmake --build . --config Release

echo "Build complete! Library should be in build/"
echo "Library path: $(find . -name 'libmeshoptimizer.*' -o -name 'meshoptimizer.*' | head -1)"

