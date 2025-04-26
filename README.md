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

### Using the pano.sh Script (Recommended)

The project includes a convenient shell script that handles building and running:

```bash
# Build the project with default settings
./pano.sh build

# Build without GPU support
./pano.sh build --no-gpu

# Build with a custom build directory
./pano.sh build --build-dir=/path/to/build
```

### Manual Build

Alternatively, you can build manually:

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

The pano.sh script provides a unified interface to run any implementation:

```bash
./pano.sh run <implementation> <image1> <image2> [<image3> ...] [options]
```

Where `<implementation>` is one of: `serial`, `openmp`, `gpu`, or `opencv`.

Available options:
- `--dir <directory>`: Use all images in the specified directory
- `--out <filename>`: Specify the output filename (default: result.jpg)

Examples:

```bash
# Stitch two images using the OpenMP implementation
./pano.sh run openmp images/mountain/mountain1.jpg images/mountain/mountain2.jpg

# Stitch multiple images using the serial implementation
./pano.sh run serial images/campus/campus1.jpg images/campus/campus2.jpg images/campus/campus3.jpg

# Use the GPU implementation with a custom build directory
./pano.sh run --build-dir=/path/to/build gpu images/city/city1.jpg images/city/city2.jpg

# Specify an output filename
./pano.sh run openmp --out panorama.jpg images/mountain/mountain1.jpg images/mountain/mountain2.jpg

# Use all images in a directory and specify output
./pano.sh run serial --dir images/campus/ --out campus_panorama.jpg
```

The script will automatically find the correct executable regardless of your current working directory.

### Running Directly (Alternative)

Alternatively, you can run the executables directly from the build directory:

```
./src/opencv/opencv_impl ../images/mountain/mountain1.jpg ../images/mountain/mountain2.jpg
```

Or use all files under a directory as image sources:

```
./src/opencv/opencv_impl --dir ../images/mountain/
```

You can also specify the output filename:

```
./src/opencv/opencv_impl --out panorama.jpg ../images/mountain/mountain1.jpg ../images/mountain/mountain2.jpg
```

Note: All these options also work with the pano.sh script:

```
./pano.sh run opencv --dir images/mountain/ --out mountain_panorama.jpg
```


## Evaluate Panorama Quality

You can evaluate the quality of a generated panorama by comparing it with a reference image using the `eval` command:

```bash
./pano.sh eval <generated_panorama> <reference_panorama>
```

This command will run the evaluation script to compare the generated panorama with a reference image and provide quality metrics.

Example:
```bash
# Evaluate a generated panorama against a reference image
./pano.sh eval result.jpg images/oilseed-ref.jpg
```

## Performance Profiling

The pano.sh script also provides a performance profiling interface using Linux's perf tool:

```bash
./pano.sh perf <implementation> <image1> <image2> [<image3> ...] [options]
```

This command works similarly to the `run` command but uses perf to collect performance data. It supports the same options as the `run` command.

Examples:

```bash
# Profile the OpenMP implementation with two images
./pano.sh perf openmp images/mountain/mountain1.jpg images/mountain/mountain2.jpg

# Profile with a custom build directory
./pano.sh perf --build-dir=/path/to/build gpu images/city/city1.jpg images/city/city2.jpg

# Profile using all images in a directory
./pano.sh perf serial --dir images/campus/
```

The profiling results will be saved to a file named `<implementation>_perf_report.txt` in the current directory. This report includes:
- Function-level performance metrics
- Call graphs
- Hot spots in the code
- CPU usage statistics
