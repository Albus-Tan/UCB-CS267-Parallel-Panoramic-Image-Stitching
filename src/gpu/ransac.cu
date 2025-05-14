#include <cuda_runtime.h>
#include <device_launch_parameters.h>
#include <curand_kernel.h>
#include <opencv2/core.hpp>
#include <opencv2/calib3d.hpp>
#include <iostream>
#include <vector>
#include <limits>
#include <algorithm>
#include <ctime>  // for time()
#include "ransac.cuh"

// Constants for GPU computation
constexpr int MAX_MATCHES = 4096;  // Increased max matches

// Structure to hold point correspondences
struct PointPair {
    float x1, y1;
    float x2, y2;
};

// Simple, robust homography computation (adapted from OpenCV)
__device__ bool computeHomographyFromPoints(
    const PointPair* points,
    float* H) {
    
    float h[9] = {0}; // homography elements
    
    // Center the points to improve numerical stability
    float cx1 = 0, cy1 = 0, cx2 = 0, cy2 = 0;
    for (int i = 0; i < 4; i++) {
        cx1 += points[i].x1;
        cy1 += points[i].y1;
        cx2 += points[i].x2;
        cy2 += points[i].y2;
    }
    cx1 /= 4;
    cy1 /= 4;
    cx2 /= 4;
    cy2 /= 4;
    
    // Calculate scale to normalize coordinates
    float scale1 = 0, scale2 = 0;
    for (int i = 0; i < 4; i++) {
        float dx1 = points[i].x1 - cx1;
        float dy1 = points[i].y1 - cy1;
        float dx2 = points[i].x2 - cx2;
        float dy2 = points[i].y2 - cy2;
        scale1 += sqrtf(dx1 * dx1 + dy1 * dy1);
        scale2 += sqrtf(dx2 * dx2 + dy2 * dy2);
    }
    
    if (scale1 < 1e-10f || scale2 < 1e-10f)
        return false;
    
    scale1 = (4.0f * sqrtf(2.0f)) / scale1;
    scale2 = (4.0f * sqrtf(2.0f)) / scale2;
    
    // Create normalized points
    float nx1[4], ny1[4], nx2[4], ny2[4];
    for (int i = 0; i < 4; i++) {
        nx1[i] = (points[i].x1 - cx1) * scale1;
        ny1[i] = (points[i].y1 - cy1) * scale1;
        nx2[i] = (points[i].x2 - cx2) * scale2;
        ny2[i] = (points[i].y2 - cy2) * scale2;
    }
    
    // Set up linear system A*h = 0
    float A[8*9] = {0};
    
    for (int i = 0; i < 4; i++) {
        int j = i * 2;
        
        A[j*9+0] = nx1[i];
        A[j*9+1] = ny1[i];
        A[j*9+2] = 1.0f;
        A[j*9+3] = 0.0f;
        A[j*9+4] = 0.0f;
        A[j*9+5] = 0.0f;
        A[j*9+6] = -nx2[i] * nx1[i];
        A[j*9+7] = -nx2[i] * ny1[i];
        A[j*9+8] = -nx2[i];
        
        j++;
        
        A[j*9+0] = 0.0f;
        A[j*9+1] = 0.0f;
        A[j*9+2] = 0.0f;
        A[j*9+3] = nx1[i];
        A[j*9+4] = ny1[i];
        A[j*9+5] = 1.0f;
        A[j*9+6] = -ny2[i] * nx1[i];
        A[j*9+7] = -ny2[i] * ny1[i];
        A[j*9+8] = -ny2[i];
    }
    
    // Using a simplified Gaussian elimination to get an approximate solution
    int i, j, k;
    for (i = 0; i < 8; i++) {
        // Find pivot
        int pi = i;
        float piv = fabsf(A[i*9+i]);
        
        for (j = i + 1; j < 8; j++) {
            float v = fabsf(A[j*9+i]);
            if (v > piv) {
                pi = j;
                piv = v;
            }
        }
        
        // Check for singularity
        if (piv < 1e-6f)
            return false;
        
        // Swap rows
        if (pi != i) {
            for (j = i; j < 9; j++) {
                float t = A[i*9+j];
                A[i*9+j] = A[pi*9+j];
                A[pi*9+j] = t;
            }
        }
        
        // Normalize row
        float inv_piv = 1.0f / A[i*9+i];
        for (j = i; j < 9; j++)
            A[i*9+j] *= inv_piv;
        
        // Eliminate
        for (k = 0; k < 8; k++) {
            if (k != i) {
                float factor = A[k*9+i];
                for (j = i; j < 9; j++)
                    A[k*9+j] -= factor * A[i*9+j];
            }
        }
    }
    
    // Extract solution (last column)
    for (i = 0; i < 8; i++)
        h[i] = -A[i*9+8];
    h[8] = 1.0f;
    
    // Denormalize
    float T1[9] = {
        scale1, 0, -scale1 * cx1,
        0, scale1, -scale1 * cy1,
        0, 0, 1
    };
    
    float T2inv[9] = {
        1.0f/scale2, 0, cx2,
        0, 1.0f/scale2, cy2,
        0, 0, 1
    };
    
    // H = T2^-1 * H * T1
    float temp[9] = {0};
    
    // Multiply H * T1
    for (i = 0; i < 3; i++) {
        for (j = 0; j < 3; j++) {
            for (k = 0; k < 3; k++) {
                temp[i*3+j] += h[i*3+k] * T1[k*3+j];
            }
        }
    }
    
    // Multiply T2^-1 * (H * T1)
    for (i = 0; i < 3; i++) {
        for (j = 0; j < 3; j++) {
            H[i*3+j] = 0;
            for (k = 0; k < 3; k++) {
                H[i*3+j] += T2inv[i*3+k] * temp[k*3+j];
            }
        }
    }
    
    return true;
}

// Transform point using homography matrix
__device__ void transformPoint(
    float x, float y, 
    const float* H, 
    float& outX, float& outY) {
    
    float w = H[6] * x + H[7] * y + H[8];
    if (fabsf(w) < 1e-10f) {
        outX = 0.0f;
        outY = 0.0f;
        return;
    }
    
    float wx = (H[0] * x + H[1] * y + H[2]) / w;
    float wy = (H[3] * x + H[4] * y + H[5]) / w;
    
    outX = wx;
    outY = wy;
}

// Count inliers for the current homography
__device__ int countInliers(
    const PointPair* allPoints, 
    int numPoints, 
    const float* H, 
    float threshold) {
    
    int count = 0;
    
    for (int i = 0; i < numPoints; i++) {
        float transformedX, transformedY;
        transformPoint(allPoints[i].x1, allPoints[i].y1, H, transformedX, transformedY);
        
        float dx = transformedX - allPoints[i].x2;
        float dy = transformedY - allPoints[i].y2;
        float distance = sqrtf(dx * dx + dy * dy);
        
        if (distance < threshold) {
            count++;
        }
    }
    
    return count;
}

// RANSAC kernel - each thread performs one RANSAC iteration
__global__ void ransacKernel(
    const PointPair* points,
    int numPoints,
    float threshold,
    int* bestInlierCounts,
    float* bestHomographies,
    unsigned long long* seeds) {
    
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    
    // Initialize random number generator with different seed for each thread
    curandState state;
    curand_init(seeds[tid], 0, 0, &state);
    
    PointPair samples[4];
    float H[9] = {0};
    int bestInliers = 0;
    float bestH[9] = {0};
    
    // Each thread performs a complete RANSAC iteration
    // Sample unique 4 points randomly
    for (int attempt = 0; attempt < 5; attempt++) {
        bool valid = true;
        int idx[4] = {-1, -1, -1, -1};
        
        // Generate 4 unique random indices
        for (int i = 0; i < 4; i++) {
            bool unique;
            do {
                unique = true;
                idx[i] = (int)(curand_uniform(&state) * numPoints);
                if (idx[i] >= numPoints) idx[i] = numPoints - 1;
                
                // Check uniqueness
                for (int j = 0; j < i; j++) {
                    if (idx[i] == idx[j]) {
                        unique = false;
                        break;
                    }
                }
            } while (!unique);
            
            samples[i] = points[idx[i]];
        }
        
        // Check if the points are not degenerate (no 3 collinear)
        // Simple check: make sure no 3 points form a line
        for (int i = 0; i < 2; i++) {
            for (int j = i + 1; j < 3; j++) {
                for (int k = j + 1; k < 4; k++) {
                    float x1 = samples[i].x1, y1 = samples[i].y1;
                    float x2 = samples[j].x1, y2 = samples[j].y1;
                    float x3 = samples[k].x1, y3 = samples[k].y1;
                    
                    // Cross product to check collinearity
                    float cross = (x2 - x1) * (y3 - y1) - (y2 - y1) * (x3 - x1);
                    if (fabsf(cross) < 1e-3f) {
                        valid = false;
                        break;
                    }
                }
                if (!valid) break;
            }
            if (!valid) break;
        }
        
        if (!valid) continue;
        
        // Try to compute homography
        if (computeHomographyFromPoints(samples, H)) {
            // Count inliers
            int inliers = countInliers(points, numPoints, H, threshold);
            
            if (inliers > bestInliers) {
                bestInliers = inliers;
                for (int i = 0; i < 9; i++) {
                    bestH[i] = H[i];
                }
            }
        }
    }
    
    // Save results
    bestInlierCounts[tid] = bestInliers;
    for (int i = 0; i < 9; i++) {
        bestHomographies[tid * 9 + i] = bestH[i];
    }
}

// Constructor with options
GpuRansacHomographyCalculator::GpuRansacHomographyCalculator(const Options& options)
    : options_(options) {}

// Main compute homography function
cv::Mat GpuRansacHomographyCalculator::computeHomography(
    const std::vector<cv::KeyPoint>& keypoints1,
    const std::vector<cv::KeyPoint>& keypoints2,
    const std::vector<cv::DMatch>& matches) {
    
    // Safety check for empty inputs
    if (keypoints1.empty() || keypoints2.empty() || matches.empty()) {
        std::cerr << "Empty input data for homography computation" << std::endl;
        return cv::Mat();
    }
    
    int numMatches = std::min(static_cast<int>(matches.size()), MAX_MATCHES);
    if (numMatches < options_.numSamples_) {
        std::cerr << "Not enough matches for RANSAC (got " << numMatches << ", need at least " 
                  << options_.numSamples_ << ")" << std::endl;
        return cv::Mat();
    }
    
    // Prepare point pairs
    std::vector<PointPair> pointPairs(numMatches);
    for (int i = 0; i < numMatches; i++) {
        int queryIdx = matches[i].queryIdx;
        int trainIdx = matches[i].trainIdx;
        
        // Safety check for valid indices
        if (queryIdx < 0 || queryIdx >= static_cast<int>(keypoints1.size()) ||
            trainIdx < 0 || trainIdx >= static_cast<int>(keypoints2.size())) {
            std::cerr << "Invalid keypoint index in match" << std::endl;
            continue;
        }
        
        cv::Point2f pt1 = keypoints1[queryIdx].pt;
        cv::Point2f pt2 = keypoints2[trainIdx].pt;
        
        pointPairs[i].x1 = pt1.x;
        pointPairs[i].y1 = pt1.y;
        pointPairs[i].x2 = pt2.x;
        pointPairs[i].y2 = pt2.y;
    }
    
    // Create unique seeds for each thread to ensure good randomness
    std::vector<unsigned long long> seeds(options_.numIterations_);
    // Use simple time-based seeds instead of std::random_device and std::mt19937_64
    unsigned long long baseSeed = static_cast<unsigned long long>(time(nullptr));
    for (int i = 0; i < options_.numIterations_; i++) {
        seeds[i] = baseSeed + i * 17 + i * i * 31;  // Simple hash for varied seeds
    }
    
    try {
        // Allocate GPU memory
        PointPair* d_points = nullptr;
        int* d_bestInlierCounts = nullptr;
        float* d_bestHomographies = nullptr;
        unsigned long long* d_seeds = nullptr;
        
        cudaError_t err;
        
        // Allocate with error checking
        err = cudaMalloc(&d_points, numMatches * sizeof(PointPair));
        if (err != cudaSuccess) {
            std::cerr << "CUDA malloc failed for points: " << cudaGetErrorString(err) << std::endl;
            return cv::Mat();
        }
        
        err = cudaMalloc(&d_bestInlierCounts, options_.numIterations_ * sizeof(int));
        if (err != cudaSuccess) {
            std::cerr << "CUDA malloc failed for inlier counts: " << cudaGetErrorString(err) << std::endl;
            cudaFree(d_points);
            return cv::Mat();
        }
        
        err = cudaMalloc(&d_bestHomographies, options_.numIterations_ * 9 * sizeof(float));
        if (err != cudaSuccess) {
            std::cerr << "CUDA malloc failed for homographies: " << cudaGetErrorString(err) << std::endl;
            cudaFree(d_points);
            cudaFree(d_bestInlierCounts);
            return cv::Mat();
        }
        
        err = cudaMalloc(&d_seeds, options_.numIterations_ * sizeof(unsigned long long));
        if (err != cudaSuccess) {
            std::cerr << "CUDA malloc failed for seeds: " << cudaGetErrorString(err) << std::endl;
            cudaFree(d_points);
            cudaFree(d_bestInlierCounts);
            cudaFree(d_bestHomographies);
            return cv::Mat();
        }
        
        // Copy data to GPU
        err = cudaMemcpy(d_points, pointPairs.data(), numMatches * sizeof(PointPair), cudaMemcpyHostToDevice);
        if (err != cudaSuccess) {
            std::cerr << "CUDA memcpy failed for points: " << cudaGetErrorString(err) << std::endl;
            cudaFree(d_points);
            cudaFree(d_bestInlierCounts);
            cudaFree(d_bestHomographies);
            cudaFree(d_seeds);
            return cv::Mat();
        }
        
        err = cudaMemcpy(d_seeds, seeds.data(), options_.numIterations_ * sizeof(unsigned long long), cudaMemcpyHostToDevice);
        if (err != cudaSuccess) {
            std::cerr << "CUDA memcpy failed for seeds: " << cudaGetErrorString(err) << std::endl;
            cudaFree(d_points);
            cudaFree(d_bestInlierCounts);
            cudaFree(d_bestHomographies);
            cudaFree(d_seeds);
            return cv::Mat();
        }
        
        // Launch kernel
        dim3 blockSize(256);  // Use 256 threads per block
        dim3 gridSize((options_.numIterations_ + blockSize.x - 1) / blockSize.x);
        
        ransacKernel<<<gridSize, blockSize>>>(
            d_points,
            numMatches,
            static_cast<float>(options_.distanceThreshold_),
            d_bestInlierCounts,
            d_bestHomographies,
            d_seeds
        );
        
        // Synchronize to make sure kernel is done
        err = cudaDeviceSynchronize();
        if (err != cudaSuccess) {
            std::cerr << "CUDA kernel error: " << cudaGetErrorString(err) << std::endl;
            cudaFree(d_points);
            cudaFree(d_bestInlierCounts);
            cudaFree(d_bestHomographies);
            cudaFree(d_seeds);
            return cv::Mat();
        }
        
        // Copy results back to host
        std::vector<int> bestInlierCounts(options_.numIterations_);
        std::vector<float> bestHomographies(options_.numIterations_ * 9);
        
        err = cudaMemcpy(bestInlierCounts.data(), d_bestInlierCounts, options_.numIterations_ * sizeof(int), cudaMemcpyDeviceToHost);
        if (err != cudaSuccess) {
            std::cerr << "CUDA memcpy failed for inlier counts: " << cudaGetErrorString(err) << std::endl;
            cudaFree(d_points);
            cudaFree(d_bestInlierCounts);
            cudaFree(d_bestHomographies);
            cudaFree(d_seeds);
            return cv::Mat();
        }
        
        err = cudaMemcpy(bestHomographies.data(), d_bestHomographies, options_.numIterations_ * 9 * sizeof(float), cudaMemcpyDeviceToHost);
        if (err != cudaSuccess) {
            std::cerr << "CUDA memcpy failed for homographies: " << cudaGetErrorString(err) << std::endl;
            cudaFree(d_points);
            cudaFree(d_bestInlierCounts);
            cudaFree(d_bestHomographies);
            cudaFree(d_seeds);
            return cv::Mat();
        }
        
        // Free GPU memory
        cudaFree(d_points);
        cudaFree(d_bestInlierCounts);
        cudaFree(d_bestHomographies);
        cudaFree(d_seeds);
        
        // Find best result
        int bestIdx = 0;
        int maxInliers = 0;
        
        for (int i = 0; i < options_.numIterations_; i++) {
            if (bestInlierCounts[i] > maxInliers) {
                maxInliers = bestInlierCounts[i];
                bestIdx = i;
            }
        }
        
        std::cout << "Best RANSAC GPU run: " << maxInliers << " inliers out of " << numMatches << " matches" << std::endl;
        
        if (maxInliers < 10) {  // Need at least 10 inliers for a good homography
            std::cerr << "GPU RANSAC failed to find a good homography. Falling back to OpenCV." << std::endl;
            
            std::vector<cv::Point2f> srcPoints, dstPoints;
            for (const auto &match : matches) {
                srcPoints.push_back(keypoints1[match.queryIdx].pt);
                dstPoints.push_back(keypoints2[match.trainIdx].pt);
            }
            
            return cv::findHomography(srcPoints, dstPoints, cv::RANSAC, 
                                     options_.distanceThreshold_, cv::noArray(), 
                                     options_.numIterations_);
        }
        
        // Create OpenCV matrix
        cv::Mat H(3, 3, CV_64F);
        for (int i = 0; i < 3; i++) {
            for (int j = 0; j < 3; j++) {
                H.at<double>(i, j) = static_cast<double>(bestHomographies[bestIdx * 9 + i * 3 + j]);
            }
        }
        
        // Normalize homography
        H = H / H.at<double>(2, 2);
        
        // Refine homography using OpenCV functions with inliers
        std::vector<cv::Point2f> inlierPts1, inlierPts2;
        for (int i = 0; i < numMatches; i++) {
            const PointPair& pair = pointPairs[i];
            cv::Point2f pt1(pair.x1, pair.y1);
            cv::Point2f pt2(pair.x2, pair.y2);
            
            // Compute reprojection error
            cv::Mat pt1Mat = (cv::Mat_<double>(3, 1) << pt1.x, pt1.y, 1.0);
            cv::Mat pt2Transformed = H * pt1Mat;
            pt2Transformed /= pt2Transformed.at<double>(2, 0);
            cv::Point2f pt2Estimate(pt2Transformed.at<double>(0, 0), pt2Transformed.at<double>(1, 0));
            
            if (cv::norm(pt2Estimate - pt2) < options_.distanceThreshold_) {
                inlierPts1.push_back(pt1);
                inlierPts2.push_back(pt2);
            }
        }
        
        if (inlierPts1.size() >= 10) {
            H = cv::findHomography(inlierPts1, inlierPts2, 0);  // Use all inliers, no RANSAC here
        }
        
        return H;
    }
    catch (const std::exception& e) {
        std::cerr << "Exception in GPU RANSAC: " << e.what() << std::endl;
        return cv::Mat();
    }
} 