cmake_minimum_required(VERSION 3.10)
project(PanoramicStitching)

# Set the C++ standard
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED True)

# Option to build the GPU implementation
option(BUILD_GPU "Build GPU implementation" ON)

# Add subdirectories for different implementations
add_subdirectory(src/serial)
add_subdirectory(src/opencv)
add_subdirectory(src/openmp)
if(BUILD_GPU)
    add_subdirectory(src/gpu)
endif()