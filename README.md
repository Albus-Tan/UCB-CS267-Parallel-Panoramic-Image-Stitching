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
- `opencv_stitching` in the `opencv` directory
- `openmp_stitching` in the `openmp` directory
- `gpu_stitching` in the `gpu` directory

## Run

Under `/build` directory:

```
./src/opencv/opencv_impl ../images/mountain/mountain1.jpg ../images/mountain/mountain2.jpg
```

Or use all files under a directory as image sources:

```
./src/opencv/opencv_impl --dir ../images/mountain/
```