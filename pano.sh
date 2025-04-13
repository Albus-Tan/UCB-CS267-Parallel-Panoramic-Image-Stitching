#!/bin/bash

# Check if at least two arguments are provided
if [ $# -lt 2 ]; then
    echo "Usage: $0 <implementation> <image1> <image2> [<image3> ...]"
    echo "  implementation: serial, openmp, gpu, opencv"
    echo "  image1, image2, ...: Image files to stitch together"
    exit 1
fi

# Get the directory where the script is located
SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" &> /dev/null && pwd )"
# Default build directory location relative to script location
BUILD_DIR="${SCRIPT_DIR}/build"

# Check if BUILD_DIR environment variable is set
if [ ! -z "$BUILD_DIR_ENV" ]; then
    BUILD_DIR="$BUILD_DIR_ENV"
fi

# Extract implementation type from first argument
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
    echo "You can set the BUILD_DIR_ENV environment variable to specify the build directory:"
    echo "  BUILD_DIR_ENV=/path/to/build ./pano.sh ..."
    exit 1
fi

# Run the executable with all remaining arguments (the image files)
echo "Running $IMPL implementation with ${#@} images using $EXEC..."
"$EXEC" "$@"

# Check the exit status
if [ $? -eq 0 ]; then
    echo "Stitching completed successfully!"
else
    echo "Stitching failed with error code $?"
fi
