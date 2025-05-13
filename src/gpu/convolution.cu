#include <cuda_runtime.h>
#include <opencv2/core.hpp>
#include <vector>
#include "convolution.cuh"

__global__ void convolveKernel(const double* input, double* output, const double* kernel,
                               int rows, int cols, int ksize) {
    int k = ksize / 2;
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;

    if (x >= k && x < cols - k && y >= k && y < rows - k) {
        double sum = 0.0;
        for (int i = -k; i <= k; ++i) {
            for (int j = -k; j <= k; ++j) {
                int xi = x + j;
                int yi = y + i;
                sum += input[yi * cols + xi] * kernel[(i + k) * ksize + (j + k)];
            }
        }
        output[y * cols + x] = sum;
    }
}

void convolveCUDA(const cv::Mat& input, cv::Mat& output, const std::vector<std::vector<double>>& kernelVec) {
    int rows = input.rows;
    int cols = input.cols;
    int ksize = kernelVec.size();

    std::vector<double> kernelFlat(ksize * ksize);
    for (int i = 0; i < ksize; ++i)
        for (int j = 0; j < ksize; ++j)
            kernelFlat[i * ksize + j] = kernelVec[i][j];

    double *d_input, *d_output, *d_kernel;
    cudaMalloc(&d_input, sizeof(double) * rows * cols);
    cudaMalloc(&d_output, sizeof(double) * rows * cols);
    cudaMalloc(&d_kernel, sizeof(double) * ksize * ksize);

    cudaMemcpy(d_input, input.ptr<double>(), sizeof(double) * rows * cols, cudaMemcpyHostToDevice);
    cudaMemcpy(d_kernel, kernelFlat.data(), sizeof(double) * ksize * ksize, cudaMemcpyHostToDevice);

    dim3 threadsPerBlock(16, 16);
    dim3 numBlocks((cols + 15) / 16, (rows + 15) / 16);
    convolveKernel<<<numBlocks, threadsPerBlock>>>(d_input, d_output, d_kernel, rows, cols, ksize);
    cudaDeviceSynchronize();

    output.create(rows, cols, CV_64F);
    cudaMemcpy(output.ptr<double>(), d_output, sizeof(double) * rows * cols, cudaMemcpyDeviceToHost);

    cudaFree(d_input);
    cudaFree(d_output);
    cudaFree(d_kernel);
}
