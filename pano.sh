#!/bin/bash

# Get the directory where the script is located
SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" &> /dev/null && pwd )"
BUILD_DIR="${SCRIPT_DIR}/build"

# Function to find and validate executable
find_executable() {
    local IMPL=$1
    local EXEC_NAME=$2
    
    # Look for the executable in different possible locations
    # First try directly in build directory
    if [ -f "${BUILD_DIR}/${EXEC_NAME}" ]; then
        echo "${BUILD_DIR}/${EXEC_NAME}"
    # Then try in the implementation-specific subdirectory
    elif [ -f "${BUILD_DIR}/src/${IMPL}/${EXEC_NAME}" ]; then
        echo "${BUILD_DIR}/src/${IMPL}/${EXEC_NAME}"
    # Try current directory
    elif [ -f "./${EXEC_NAME}" ]; then
        echo "./${EXEC_NAME}"
    else
        echo "Executable not found: ${EXEC_NAME}"
        echo "Checked in:"
        echo "  ${BUILD_DIR}/${EXEC_NAME}"
        echo "  ${BUILD_DIR}/src/${IMPL}/${EXEC_NAME}"
        echo "  ./${EXEC_NAME}"
        echo ""
        echo "Try building the project first:"
        echo "  $0 build"
        return 1
    fi
}

# Function to get executable name from implementation
get_exec_name() {
    local IMPL=$1
    case $IMPL in
        serial)
            echo "serial_stitching"
            ;;
        openmp)
            echo "openmp_stitching"
            ;;
        gpu)
            echo "gpu_stitching"
            ;;
        opencv)
            echo "opencv_impl"
            ;;
        *)
            echo "Unknown implementation: $IMPL"
            echo "Supported implementations: serial, openmp, gpu, opencv"
            return 1
            ;;
    esac
}

# Function to process common run/perf arguments
process_common_args() {
    local IMPL=$1
    shift
    
    # Parse options
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
    
    # Check if there are enough arguments after processing options and implementation
    if [ $# -lt 1 ] && [[ ! "$*" =~ "--dir" ]]; then
        echo "Error: No image files specified and --dir option not used"
        usage
        return 1
    fi
    
    # Get executable name
    local EXEC_NAME
    EXEC_NAME=$(get_exec_name "$IMPL") || return 1
    
    # Find executable
    local EXEC
    EXEC=$(find_executable "$IMPL" "$EXEC_NAME") || return 1
    
    echo "$EXEC"
}

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
    echo "  $0 perf <implementation> <image1> <image2> [<image3> ...] [options]"
    echo "      Run performance profiling on an implementation with image files"
    echo "      implementation: serial, openmp, gpu, opencv"
    echo ""
    echo "Options for 'build':"
    echo "  --no-gpu               Build without GPU support"
    echo "  --build-dir=<path>     Specify build directory (default: ./build)"
    echo ""
    echo "Options for 'run' and 'perf':"
    echo "  --build-dir=<path>     Specify build directory (default: ./build)"
    echo "  --dir <directory>      Use all images in the specified directory"
    echo "  --out <filename>       Specify the output filename (default: result.jpg)"
    echo ""
    echo "Examples:"
    echo "  $0 build"
    echo "  $0 build --no-gpu"
    echo "  $0 run openmp images/mountain/mountain1.jpg images/mountain/mountain2.jpg"
    echo "  $0 run serial --dir images/campus/ --out campus_panorama.jpg"
    echo "  $0 perf openmp images/mountain/mountain1.jpg images/mountain/mountain2.jpg"
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
        
        # Extract implementation type
        IMPL=$1
        shift
        
        # Process common arguments and get executable path
        EXEC=$(process_common_args "$IMPL" "$@") || exit 1
        
        # Run the executable with all remaining arguments
        echo "Running $IMPL implementation using $EXEC..."
        "$EXEC" "$@"
        
        # Check the exit status
        if [ $? -eq 0 ]; then
            echo "Stitching completed successfully!"
        else
            echo "Stitching failed with error code $?"
        fi
        ;;
        
    perf)
        # Check if at least implementation is provided
        if [ $# -lt 1 ]; then
            echo "Error: Missing implementation"
            usage
        fi
        
        # Extract implementation type
        IMPL=$1
        shift
        
        # Process common arguments and get executable path
        EXEC=$(process_common_args "$IMPL" "$@") || exit 1
        
        # Run perf with the executable
        echo "Running performance profiling on $IMPL implementation using $EXEC..."
        perf record -g "$EXEC" "$@"
        
        # Generate perf report
        echo "Generating performance report..."
        perf report --stdio > "${IMPL}_perf_report.txt"
        
        # Check the exit status
        if [ $? -eq 0 ]; then
            echo "Performance profiling completed successfully!"
            echo "Performance report saved to ${IMPL}_perf_report.txt"
        else
            echo "Performance profiling failed with error code $?"
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
