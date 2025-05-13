#include <cuda_runtime.h>
#include <device_launch_parameters.h>
#include <opencv2/core.hpp>
#include <opencv2/imgproc.hpp>
#include <iostream>
#include <iomanip>
#include <vector>
#include <chrono>
#include <thrust/device_vector.h>
#include <thrust/host_vector.h>
#include <thrust/scan.h>
#include "harris_detector.cuh"
#include "convolution.cuh"

// CUDA kernel: Compute Harris response
__global__ void computeHarrisResponseKernel(
    const double* gradXX, const double* gradYY, const double* gradXY,
    double* harrisResp, int width, int height, double k) {
    
    // Calculate current pixel position for this thread
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;
    
    // Check boundaries
    if (x < width && y < height) {
        int idx = y * width + x;
        double xx = gradXX[idx];
        double yy = gradYY[idx];
        double xy = gradXY[idx];
        
        // Calculate Harris response
        double det = xx * yy - xy * xy;
        double trace = xx + yy;
        harrisResp[idx] = det - k * trace * trace;
    }
}

// CUDA kernel: Find local maxima (non-maximum suppression)
__global__ void findLocalMaximaKernel(
    const double* harrisResp, 
    int* candidateKeypoints, 
    int* keypointCount,
    int width, int height, 
    double threshold, 
    int neighborhood) {
    
    // Calculate current pixel position for this thread
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;
    
    // Check boundaries (considering neighborhood size)
    int halfSize = neighborhood / 2;
    if (x >= halfSize && x < width - halfSize && y >= halfSize && y < height - halfSize) {
        int idx = y * width + x;
        double resp = harrisResp[idx];
        
        // Skip if response is below threshold
        if (resp <= threshold) {
            return;
        }
        
        // Check if this is a local maximum
        bool isLocalMax = true;
        for (int dy = -halfSize; dy <= halfSize && isLocalMax; dy++) {
            for (int dx = -halfSize; dx <= halfSize; dx++) {
                // Skip the center point
                if (dx == 0 && dy == 0) continue;
                
                int nx = x + dx;
                int ny = y + dy;
                int nidx = ny * width + nx;
                
                if (harrisResp[nidx] >= resp) {
                    isLocalMax = false;
                    break;
                }
            }
        }
        
        // If it's a local maximum, record this keypoint
        if (isLocalMax) {
            int position = atomicAdd(keypointCount, 1);
            candidateKeypoints[position] = idx;
        }
    }
}

/**
 * GPU-accelerated Harris corner detector
 */
std::vector<cv::KeyPoint> gpuHarrisCornerDetectorDetect(
    const cv::Mat &image,
    double k,
    double nmsThresh,
    int nmsNeighborhood) {
    
    // Start timing
    auto start = std::chrono::high_resolution_clock::now();
    
    // Convert to grayscale
    cv::Mat gray;
    if (image.channels() == 3) {
        cv::cvtColor(image, gray, cv::COLOR_BGR2GRAY);
    } else {
        gray = image.clone();
    }
    gray.convertTo(gray, CV_64F);
    
    int width = gray.cols;
    int height = gray.rows;
    
    // Define Sobel and Gaussian convolution kernels
    std::vector<std::vector<double>> sobelXKernel = {
        {-1, 0, 1},
        {-2, 0, 2},
        {-1, 0, 1}
    };
    
    std::vector<std::vector<double>> sobelYKernel = {
        {-1, -2, -1},
        { 0,  0,  0},
        { 1,  2,  1}
    };
    
    // Use 5x5 Gaussian kernel for smoothing
    std::vector<std::vector<double>> gaussianKernel(5, std::vector<double>(5, 0));
    double sigma = 1.0;
    double sum = 0.0;
    int halfSize = 2;
    for (int y = -halfSize; y <= halfSize; y++) {
        for (int x = -halfSize; x <= halfSize; x++) {
            double value = exp(-(x*x + y*y) / (2 * sigma * sigma));
            gaussianKernel[y+halfSize][x+halfSize] = value;
            sum += value;
        }
    }
    // Normalize the Gaussian kernel
    for (auto &row : gaussianKernel) {
        for (auto &val : row) {
            val /= sum;
        }
    }
    
    // Calculate gradients
    cv::Mat gradX, gradY;
    convolveCUDA(gray, gradX, sobelXKernel);
    convolveCUDA(gray, gradY, sobelYKernel);
    
    // Calculate gradient products
    cv::Mat gradXX = gradX.mul(gradX);
    cv::Mat gradYY = gradY.mul(gradY);
    cv::Mat gradXY = gradX.mul(gradY);
    
    // Apply Gaussian smoothing to gradient products
    convolveCUDA(gradXX, gradXX, gaussianKernel);
    convolveCUDA(gradYY, gradYY, gaussianKernel);
    convolveCUDA(gradXY, gradXY, gaussianKernel);
    
    // Allocate GPU memory
    double *d_gradXX, *d_gradYY, *d_gradXY, *d_harrisResp;
    cudaMalloc(&d_gradXX, width * height * sizeof(double));
    cudaMalloc(&d_gradYY, width * height * sizeof(double));
    cudaMalloc(&d_gradXY, width * height * sizeof(double));
    cudaMalloc(&d_harrisResp, width * height * sizeof(double));
    
    // Copy data to GPU
    cudaMemcpy(d_gradXX, gradXX.ptr<double>(), width * height * sizeof(double), cudaMemcpyHostToDevice);
    cudaMemcpy(d_gradYY, gradYY.ptr<double>(), width * height * sizeof(double), cudaMemcpyHostToDevice);
    cudaMemcpy(d_gradXY, gradXY.ptr<double>(), width * height * sizeof(double), cudaMemcpyHostToDevice);
    
    // Setup CUDA kernel configuration
    dim3 blockSize(16, 16);
    dim3 gridSize((width + blockSize.x - 1) / blockSize.x, (height + blockSize.y - 1) / blockSize.y);
    
    // Call Harris response computation kernel
    computeHarrisResponseKernel<<<gridSize, blockSize>>>(
        d_gradXX, d_gradYY, d_gradXY, d_harrisResp, width, height, k);
    
    // Allocate memory for candidate keypoints and count
    int *d_candidateKeypoints, *d_keypointCount;
    cudaMalloc(&d_candidateKeypoints, width * height * sizeof(int));
    cudaMalloc(&d_keypointCount, sizeof(int));
    
    // Initialize keypoint counter to 0
    cudaMemset(d_keypointCount, 0, sizeof(int));
    
    // Call the kernel to find local maxima
    findLocalMaximaKernel<<<gridSize, blockSize>>>(
        d_harrisResp, d_candidateKeypoints, d_keypointCount, width, height, nmsThresh, nmsNeighborhood);
    
    // Get keypoint count
    int keypointCount;
    cudaMemcpy(&keypointCount, d_keypointCount, sizeof(int), cudaMemcpyDeviceToHost);
    
    // Limit the number of keypoints if too many are detected
    keypointCount = std::min(keypointCount, 10000);  // Limit to max 10000 keypoints
    
    // Copy candidate keypoint indices back to host
    std::vector<int> candidateKeypoints(keypointCount);
    cudaMemcpy(candidateKeypoints.data(), d_candidateKeypoints, keypointCount * sizeof(int), cudaMemcpyDeviceToHost);
    
    // Create OpenCV keypoint objects
    std::vector<cv::KeyPoint> keypoints;
    keypoints.reserve(keypointCount);
    
    // Create keypoints from indices
    for (int i = 0; i < keypointCount; i++) {
        int idx = candidateKeypoints[i];
        int y = idx / width;
        int x = idx % width;
        keypoints.push_back(cv::KeyPoint(static_cast<float>(x), static_cast<float>(y), 1.0f));
    }
    
    // Free GPU memory
    cudaFree(d_gradXX);
    cudaFree(d_gradYY);
    cudaFree(d_gradXY);
    cudaFree(d_harrisResp);
    cudaFree(d_candidateKeypoints);
    cudaFree(d_keypointCount);
    
    // Output timing results
    auto end = std::chrono::high_resolution_clock::now();
    auto duration = std::chrono::duration<double, std::milli>(end - start).count();
    std::cout << "GPU Harris Corner Detection: " << std::fixed << std::setprecision(3) << duration << " ms, Found " << keypoints.size() << " keypoints" << std::endl;
    
    return keypoints;
} 