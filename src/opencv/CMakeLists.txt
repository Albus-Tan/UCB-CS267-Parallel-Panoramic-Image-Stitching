cmake_minimum_required(VERSION 3.10)
project(OpenCVStitching)

# Find the OpenCV package
find_package(OpenCV REQUIRED)
include_directories(${OpenCV_INCLUDE_DIRS})

# Create an executable for the OpenCV stitching implementation
add_executable(opencv_impl main.cpp)

# Link the OpenCV libraries to the executable
target_link_libraries(opencv_impl ${OpenCV_LIBS})
