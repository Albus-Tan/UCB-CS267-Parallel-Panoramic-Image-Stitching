#include <cuda_runtime.h>
#include <device_launch_parameters.h>
#include <opencv2/core.hpp>
#include <iostream>
#include <vector>
#include <limits>
#include "harris_matcher.cuh"

// GPU kernel to compute SSD (Sum of Squared Differences) between patches
__global__ void computeSSDKernel(
    const unsigned char* img1, const unsigned char* img2,
    int img1Width, int img2Width, int channels,
    int patchSize, 
    const int* keypoints1X, const int* keypoints1Y,
    const int* keypoints2X, const int* keypoints2Y,
    int numKeypoints1, int numKeypoints2,
    unsigned long long* ssdMatrix,
    int border) {
    
    // Calculate global thread ID
    int idx1 = blockIdx.x * blockDim.x + threadIdx.x;
    int idx2 = blockIdx.y * blockDim.y + threadIdx.y;
    
    // Check bounds
    if (idx1 >= numKeypoints1 || idx2 >= numKeypoints2)
        return;
    
    // Get keypoint coordinates
    int x1 = keypoints1X[idx1];
    int y1 = keypoints1Y[idx1];
    int x2 = keypoints2X[idx2];
    int y2 = keypoints2Y[idx2];
    
    // Check if keypoints are too close to the border
    if (x1 < border || y1 < border || x2 < border || y2 < border)
        return;
    
    // Compute SSD between patches
    unsigned long long ssd = 0;
    for (int dy = -border; dy <= border; dy++) {
        for (int dx = -border; dx <= border; dx++) {
            for (int c = 0; c < channels; c++) {
                int idx1 = ((y1 + dy) * img1Width + (x1 + dx)) * channels + c;
                int idx2 = ((y2 + dy) * img2Width + (x2 + dx)) * channels + c;
                
                int diff = static_cast<int>(img1[idx1]) - static_cast<int>(img2[idx2]);
                ssd += static_cast<unsigned long long>(diff * diff);
            }
        }
    }
    
    // Store SSD value
    ssdMatrix[idx1 * numKeypoints2 + idx2] = ssd;
}

// Kernel to find best matches for each keypoint
__global__ void findBestMatchesKernel(
    const unsigned long long* ssdMatrix,
    int numKeypoints1, int numKeypoints2,
    int* bestMatchIndices,
    unsigned long long* bestMatchSSD) {
    
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    
    if (idx >= numKeypoints1)
        return;
    
    unsigned long long minSSD = ULLONG_MAX;
    int bestIdx = -1;
    
    for (int j = 0; j < numKeypoints2; j++) {
        unsigned long long ssd = ssdMatrix[idx * numKeypoints2 + j];
        if (ssd < minSSD) {
            minSSD = ssd;
            bestIdx = j;
        }
    }
    
    bestMatchIndices[idx] = bestIdx;
    bestMatchSSD[idx] = minSSD;
}

// Main function for GPU-accelerated Harris corner matching
std::vector<cv::DMatch> gpuHarrisMatchKeyPoints(
    const std::vector<cv::KeyPoint> &keypointsL,
    const std::vector<cv::KeyPoint> &keypointsR,
    const cv::Mat &image1, const cv::Mat &image2,
    int patchSize, double maxSSDThresh, int offset) {
    
    // Define constants and prepare data
    int numKeypointsL = keypointsL.size();
    int numKeypointsR = keypointsR.size();
    int border = patchSize / 2;
    int channels = image1.channels();
    
    if (numKeypointsL == 0 || numKeypointsR == 0) {
        return std::vector<cv::DMatch>();
    }
    
    // Prepare keypoint coordinates for GPU
    std::vector<int> keypointsLX(numKeypointsL), keypointsLY(numKeypointsL);
    std::vector<int> keypointsRX(numKeypointsR), keypointsRY(numKeypointsR);
    
    for (int i = 0; i < numKeypointsL; i++) {
        keypointsLX[i] = static_cast<int>(keypointsL[i].pt.x);
        keypointsLY[i] = static_cast<int>(keypointsL[i].pt.y);
    }
    
    for (int i = 0; i < numKeypointsR; i++) {
        keypointsRX[i] = static_cast<int>(keypointsR[i].pt.x);
        keypointsRY[i] = static_cast<int>(keypointsR[i].pt.y);
    }
    
    // Allocate GPU memory
    unsigned char *d_img1, *d_img2;
    int *d_keypointsLX, *d_keypointsLY, *d_keypointsRX, *d_keypointsRY;
    unsigned long long *d_ssdMatrix;
    int *d_bestMatchIndices;
    unsigned long long *d_bestMatchSSD;
    
    cudaMalloc(&d_img1, image1.total() * channels);
    cudaMalloc(&d_img2, image2.total() * channels);
    cudaMalloc(&d_keypointsLX, numKeypointsL * sizeof(int));
    cudaMalloc(&d_keypointsLY, numKeypointsL * sizeof(int));
    cudaMalloc(&d_keypointsRX, numKeypointsR * sizeof(int));
    cudaMalloc(&d_keypointsRY, numKeypointsR * sizeof(int));
    cudaMalloc(&d_ssdMatrix, numKeypointsL * numKeypointsR * sizeof(unsigned long long));
    cudaMalloc(&d_bestMatchIndices, numKeypointsL * sizeof(int));
    cudaMalloc(&d_bestMatchSSD, numKeypointsL * sizeof(unsigned long long));
    
    // Copy data to GPU
    cudaMemcpy(d_img1, image1.data, image1.total() * channels, cudaMemcpyHostToDevice);
    cudaMemcpy(d_img2, image2.data, image2.total() * channels, cudaMemcpyHostToDevice);
    cudaMemcpy(d_keypointsLX, keypointsLX.data(), numKeypointsL * sizeof(int), cudaMemcpyHostToDevice);
    cudaMemcpy(d_keypointsLY, keypointsLY.data(), numKeypointsL * sizeof(int), cudaMemcpyHostToDevice);
    cudaMemcpy(d_keypointsRX, keypointsRX.data(), numKeypointsR * sizeof(int), cudaMemcpyHostToDevice);
    cudaMemcpy(d_keypointsRY, keypointsRY.data(), numKeypointsR * sizeof(int), cudaMemcpyHostToDevice);
    
    // Launch the SSD computation kernel
    dim3 blockSize(16, 16);
    dim3 gridSize((numKeypointsL + blockSize.x - 1) / blockSize.x, 
                  (numKeypointsR + blockSize.y - 1) / blockSize.y);
    
    computeSSDKernel<<<gridSize, blockSize>>>(
        d_img1, d_img2,
        image1.cols, image2.cols, channels,
        patchSize,
        d_keypointsLX, d_keypointsLY,
        d_keypointsRX, d_keypointsRY,
        numKeypointsL, numKeypointsR,
        d_ssdMatrix,
        border
    );
    
    // Launch the best match finding kernel
    dim3 findBlockSize(256);
    dim3 findGridSize((numKeypointsL + findBlockSize.x - 1) / findBlockSize.x);
    
    findBestMatchesKernel<<<findGridSize, findBlockSize>>>(
        d_ssdMatrix,
        numKeypointsL, numKeypointsR,
        d_bestMatchIndices,
        d_bestMatchSSD
    );
    
    // Copy results back to host
    std::vector<int> bestMatchIndices(numKeypointsL);
    std::vector<unsigned long long> bestMatchSSD(numKeypointsL);
    
    cudaMemcpy(bestMatchIndices.data(), d_bestMatchIndices, numKeypointsL * sizeof(int), cudaMemcpyDeviceToHost);
    cudaMemcpy(bestMatchSSD.data(), d_bestMatchSSD, numKeypointsL * sizeof(unsigned long long), cudaMemcpyDeviceToHost);
    
    // Free GPU memory
    cudaFree(d_img1);
    cudaFree(d_img2);
    cudaFree(d_keypointsLX);
    cudaFree(d_keypointsLY);
    cudaFree(d_keypointsRX);
    cudaFree(d_keypointsRY);
    cudaFree(d_ssdMatrix);
    cudaFree(d_bestMatchIndices);
    cudaFree(d_bestMatchSSD);
    
    // Create matches
    std::vector<cv::DMatch> matches;
    for (int i = 0; i < numKeypointsL; i++) {
        if (bestMatchIndices[i] >= 0 && bestMatchSSD[i] < maxSSDThresh) {
            matches.push_back(cv::DMatch(i + offset, bestMatchIndices[i], static_cast<float>(bestMatchSSD[i])));
        }
    }
    
    return matches;
} 