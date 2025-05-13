cmake_minimum_required(VERSION 3.10)
set(CMAKE_CUDA_ARCHITECTURES "native")  # Auto-detect GPU arch

project(GPUStitching LANGUAGES CXX CUDA)

enable_language(CUDA)

set(CUDA_SOURCES
    convolution.cu
    harris_matcher.cu
)

set(CPP_SOURCES
    main.cpp
)

add_executable(gpu_stitching ${CPP_SOURCES} ${CUDA_SOURCES})

# Create an executable for the GPU stitching implementation
# add_executable(gpu_stitching convolution.cu main.cpp)

# Link the OpenCV libraries to the executable
target_link_libraries(gpu_stitching PRIVATE ${OpenCV_LIBS} reader)



# cmake_minimum_required(VERSION 3.10)
# project(PanoramaCUDA LANGUAGES CXX CUDA)

# set(CMAKE_CXX_STANDARD 17)
# set(CMAKE_CUDA_STANDARD 14)

# # Find OpenCV
# find_package(OpenCV REQUIRED)

# # Enable CUDA separable compilation
# set(CMAKE_CUDA_SEPARABLE_COMPILATION ON)

# # Source files
# set(SOURCES
#     main.cpp
#     convolution.cu
#     reader.cpp
# )

# add_executable(${PROJECT_NAME} ${SOURCES})

# # Link OpenCV
# target_include_directories(${PROJECT_NAME} PRIVATE ${OpenCV_INCLUDE_DIRS})
# target_link_libraries(${PROJECT_NAME} PRIVATE ${OpenCV_LIBS})
