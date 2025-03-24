cmake_minimum_required(VERSION 3.10)
project(SerialStitching)

# Find the OpenCV package
find_package(OpenCV REQUIRED)
include_directories(${OpenCV_INCLUDE_DIRS})

# Create an executable for the serial stitching implementation
add_executable(serial_stitching main.cpp)

# Link the OpenCV libraries to the executable
target_link_libraries(serial_stitching ${OpenCV_LIBS})
