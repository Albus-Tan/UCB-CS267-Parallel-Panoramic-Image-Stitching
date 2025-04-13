# UCB-CS267-Parallel-Panoramic-Image-Stitching


This project implements various panoramic image stitching approaches. The implementations are organized into separate folders under `src/` (e.g., `serial`, `opencv`, `openmp`, and `gpu`), and a sample set of images is located in the `images/` folder.

## Installation

On Ubuntu:

```
# Install OpenCV
sudo apt-get update
sudo apt-get install libopencv-dev
# Install CMake
sudo apt-get install cmake
```


## Build

```
mkdir build
cd build
cmake ..
make
```

To build without GPU, run `cmake -DBUILD_GPU=OFF ..` instead.

After the build completes, you will find the following executables:

- `serial_stitching` in the `serial` directory
- `opencv_impl` in the `opencv` directory
- `openmp_stitching` in the `openmp` directory
- `gpu_stitching` in the `gpu` directory

## Run

### Using the pano.sh Script (Recommended)

We provide a convenient shell script that makes it easy to run any implementation:

```
./pano.sh <implementation> <image1> <image2> [<image3> ...]
```

Where `<implementation>` is one of: `serial`, `openmp`, `gpu`, or `opencv`.

Examples:

```bash
# Stitch two images using the OpenMP implementation
./pano.sh openmp images/mountain/mountain1.jpg images/mountain/mountain2.jpg

# Stitch multiple images using the serial implementation
./pano.sh serial images/campus/campus1.jpg images/campus/campus2.jpg images/campus/campus3.jpg

# Use the GPU implementation
./pano.sh gpu images/city/city1.jpg images/city/city2.jpg
```

The script will automatically find the correct executable regardless of your current working directory. If you're running from a different directory and the script can't find the executables, you can specify the build directory:

```bash
BUILD_DIR_ENV=/path/to/build ./pano.sh openmp images/mountain/mountain1.jpg images/mountain/mountain2.jpg
```

### Running Directly (Alternative)

Alternatively, you can run the executables directly from the build directory:

```
./src/opencv/opencv_impl ../images/mountain/mountain1.jpg ../images/mountain/mountain2.jpg
```

Or use all files under a directory as image sources:

```
./src/opencv/opencv_impl --dir ../images/mountain/
```

Note: The `--dir` option also works with the pano.sh script:

```
./pano.sh opencv --dir images/mountain/
```