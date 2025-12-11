cmake_minimum_required(VERSION 3.10)
project(meshoptimizer)

# Set C++ standard
set(CMAKE_CXX_STANDARD 11)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# Source files
set(SOURCES
    src/indexgenerator.cpp
    src/simplifier.cpp
    src/vertexfilter.cpp
    src/vertexcodec.cpp
    src/indexcodec.cpp
    src/vcacheoptimizer.cpp
    src/vfetchoptimizer.cpp
    src/clusterizer.cpp
    src/overdrawoptimizer.cpp
    src/spatialorder.cpp
    src/stripifier.cpp
    src/allocator.cpp
)

# Create shared library
add_library(meshoptimizer SHARED ${SOURCES})

# Include directories
target_include_directories(meshoptimizer PUBLIC
    ${CMAKE_CURRENT_SOURCE_DIR}/src
)

# Set output directory
set_target_properties(meshoptimizer PROPERTIES
    OUTPUT_NAME "meshoptimizer"
    VERSION 1.0.0
    SOVERSION 1
)

# Install
install(TARGETS meshoptimizer
    LIBRARY DESTINATION lib
    ARCHIVE DESTINATION lib
)

install(FILES src/meshoptimizer.h
    DESTINATION include
)

