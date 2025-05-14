#pragma once
#include <opencv2/core.hpp>
#include <vector>

/**
 * GPU-accelerated RANSAC homography calculator
 */
class GpuRansacHomographyCalculator {
public:
    struct Options {
        int numIterations_ = 1000;         // Number of RANSAC iterations
        int numSamples_ = 4;               // Number of samples per RANSAC iteration
        double distanceThreshold_ = 3.0;   // RANSAC inlier distance threshold
    };

    /**
     * Constructor with options
     * @param options The RANSAC algorithm parameters
     */
    GpuRansacHomographyCalculator(const Options& options);

    /**
     * Compute homography matrix using GPU-accelerated RANSAC
     * @param keypoints1 First set of keypoints
     * @param keypoints2 Second set of keypoints
     * @param matches Matches between keypoints
     * @return Computed homography matrix
     */
    cv::Mat computeHomography(
        const std::vector<cv::KeyPoint>& keypoints1,
        const std::vector<cv::KeyPoint>& keypoints2,
        const std::vector<cv::DMatch>& matches);

private:
    Options options_;
}; 