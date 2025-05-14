#pragma once
#include <opencv2/core.hpp>
#include <vector>

std::vector<cv::KeyPoint> gpuHarrisCornerDetectorDetect(
    const cv::Mat &image,
    double k = 0.04,
    double nmsThresh = 1e6,
    int nmsNeighborhood = 3); 