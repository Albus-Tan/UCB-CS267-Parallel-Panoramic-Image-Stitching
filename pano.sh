#!/bin/bash

# Get the directory where the script is located
SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" &> /dev/null && pwd )"
BUILD_DIR="${SCRIPT_DIR}/build"

# Define usage message
usage() {
    echo "Usage:"
    echo "  $0 build [--no-gpu] [--build-dir=<path>]"
    echo "      Build the project"
    echo ""
    echo "  $0 run <implementation> <image1> <image2> [<image3> ...] [options]"
    echo "      Run an implementation with image files"
    echo "      implementation: serial, openmp, gpu, opencv"
    echo ""
    echo "Options for 'build':"
    echo "  --no-gpu               Build without GPU support"
    echo "  --build-dir=<path>     Specify build directory (default: ./build)"
    echo ""
    echo "Options for 'run':"
    echo "  --build-dir=<path>     Specify build directory (default: ./build)"
    echo "  --dir <directory>      Use all images in the specified directory"
    echo "  --out <filename>       Specify the output filename (default: result.jpg)"
    echo ""
    echo "Examples:"
    echo "  $0 build"
    echo "  $0 build --no-gpu"
    echo "  $0 run openmp images/mountain/mountain1.jpg images/mountain/mountain2.jpg"
    echo "  $0 run serial --dir images/campus/ --out campus_panorama.jpg"
    exit 1
}

# Check if a command is provided
if [ $# -lt 1 ]; then
    usage
fi

# Parse command
COMMAND=$1
shift

case $COMMAND in
    build)
        # Parse build options
        NO_GPU=false
        
        while [[ $1 =~ ^-- ]]; do
            case $1 in
                --no-gpu)
                    NO_GPU=true
                    shift
                    ;;
                --build-dir=*)
                    BUILD_DIR="${1#*=}"
                    shift
                    ;;
                *)
                    echo "Unknown option for build command: $1"
                    usage
                    ;;
            esac
        done
        
        echo "=== Building project in $BUILD_DIR ==="
        
        # Create build directory if it doesn't exist
        mkdir -p "$BUILD_DIR"
        
        # Enter build directory
        cd "$BUILD_DIR" || { echo "Failed to enter build directory"; exit 1; }
        
        # Run CMake
        echo "Running CMake..."
        if [ "$NO_GPU" = true ]; then
            CMAKE_ARGS="-DBUILD_GPU=OFF"
            echo "Building without GPU support"
        else
            CMAKE_ARGS=""
        fi
        
        cmake $CMAKE_ARGS .. || { echo "CMake failed"; exit 1; }
        
        # Run Make
        echo "Running Make..."
        make -j$(nproc) || { echo "Make failed"; exit 1; }
        
        echo "=== Build completed successfully ==="
        ;;
        
    run)
        # Check if at least implementation is provided
        if [ $# -lt 1 ]; then
            echo "Error: Missing implementation"
            usage
        fi
        
        # Parse run options
        while [[ $1 =~ ^-- ]]; do
            case $1 in
                --build-dir=*)
                    BUILD_DIR="${1#*=}"
                    shift
                    ;;
                *)
                    # Other options like --dir and --out will be passed directly to the executable
                    break
                    ;;
            esac
        done
        
        # Extract implementation type
        IMPL=$1
        shift
        
        # Map implementation to executable name
        case $IMPL in
            serial)
                EXEC_NAME="serial_stitching"
                ;;
            openmp)
                EXEC_NAME="openmp_stitching"
                ;;
            gpu)
                EXEC_NAME="gpu_stitching"
                ;;
            opencv)
                EXEC_NAME="opencv_impl"
                ;;
            *)
                echo "Unknown implementation: $IMPL"
                echo "Supported implementations: serial, openmp, gpu, opencv"
                exit 1
                ;;
        esac
        
        # Check if there are enough arguments after processing options and implementation
        if [ $# -lt 1 ] && [[ ! "$*" =~ "--dir" ]]; then
            echo "Error: No image files specified and --dir option not used"
            usage
        fi
        
        # Look for the executable in different possible locations
        # First try directly in build directory
        if [ -f "${BUILD_DIR}/${EXEC_NAME}" ]; then
            EXEC="${BUILD_DIR}/${EXEC_NAME}"
        # Then try in the implementation-specific subdirectory
        elif [ -f "${BUILD_DIR}/src/${IMPL}/${EXEC_NAME}" ]; then
            EXEC="${BUILD_DIR}/src/${IMPL}/${EXEC_NAME}"
        # Try current directory
        elif [ -f "./${EXEC_NAME}" ]; then
            EXEC="./${EXEC_NAME}"
        else
            echo "Executable not found: ${EXEC_NAME}"
            echo "Checked in:"
            echo "  ${BUILD_DIR}/${EXEC_NAME}"
            echo "  ${BUILD_DIR}/src/${IMPL}/${EXEC_NAME}"
            echo "  ./${EXEC_NAME}"
            echo ""
            echo "Try building the project first:"
            echo "  $0 build"
            exit 1
        fi
        
        # Run the executable with all remaining arguments (including --dir, --out, and image files)
        echo "Running $IMPL implementation using $EXEC..."
        "$EXEC" "$@"
        
        # Check the exit status
        if [ $? -eq 0 ]; then
            echo "Stitching completed successfully!"
        else
            echo "Stitching failed with error code $?"
        fi
        ;;
        
    help)
        usage
        ;;
        
    *)
        echo "Unknown command: $COMMAND"
        usage
        ;;
esac
