#pragma once
#include <opencv2/core.hpp>
#include <vector>

std::vector<cv::DMatch> gpuHarrisMatchKeyPoints(
    const std::vector<cv::KeyPoint> &keypointsL,
    const std::vector<cv::KeyPoint> &keypointsR,
    const cv::Mat &image1, const cv::Mat &image2,
    int patchSize, double maxSSDThresh, int offset = 0);