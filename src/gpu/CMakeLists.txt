cmake_minimum_required(VERSION 3.10)
project(GPUStitching LANGUAGES CXX CUDA)

# Find the OpenCV package
find_package(OpenCV REQUIRED)
include_directories(${OpenCV_INCLUDE_DIRS})

# Create an executable for the GPU stitching implementation
add_executable(gpu_stitching main.cpp)

# Link the OpenCV libraries to the executable
target_link_libraries(gpu_stitching ${OpenCV_LIBS})
