#pragma once
#include <opencv2/core.hpp>
#include <vector>

void convolveCUDA(const cv::Mat& input, cv::Mat& output, const std::vector<std::vector<double>>& kernelVec);
