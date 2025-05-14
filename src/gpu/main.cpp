#include <cassert>
#include <cmath>
#include <cstdlib>
#include <iostream>
#include <limits>
#include <random>
#include <string>
#include <vector>
#include <algorithm>
#include <filesystem>   // C++17
#include <chrono>       // For high-precision timing
#include <iomanip>      // For formatting output

#include <opencv2/core.hpp>
#include <opencv2/highgui.hpp>
#include <opencv2/imgcodecs.hpp>
#include <opencv2/imgproc.hpp>
#include <opencv2/calib3d.hpp>
#include <opencv2/features2d.hpp>
#include "convolution.cuh"  
#include "harris_matcher.cuh"
#include "ransac.cuh"
#include "harris_detector.cuh"

#include "reader.hpp"

// ---------- Parameter Structures ----------
struct PanoramicOptions {
    // Not used in this example; can be extended as needed.
};

struct HarrisCornerOptions {
    double k_ = 0.04;                  // Harris detector parameter
    double nmsThresh_ = 1e6;           // Harris response threshold
    int nmsNeighborhood_ = 3;          // Non-maximum suppression neighborhood size (must be odd)
    int patchSize_ = 5;                // Patch size used for matching
    double maxSSDThresh_ = 1e8;        // SSD matching threshold
};

struct RansacOptions {
    int numIterations_ = 1000;         // Number of RANSAC iterations
    int numSamples_ = 4;               // Number of samples per RANSAC iteration
    double distanceThreshold_ = 3.0;   // RANSAC inlier distance threshold
};

// ---------- Timing Utilities ----------
class Timer {
public:
    Timer() : start_(std::chrono::high_resolution_clock::now()) {}
    
    double elapsed() const {
        auto now = std::chrono::high_resolution_clock::now();
        return std::chrono::duration<double, std::milli>(now - start_).count();
    }
    
    void reset() {
        start_ = std::chrono::high_resolution_clock::now();
    }
    
private:
    std::chrono::time_point<std::chrono::high_resolution_clock> start_;
};

// ---------- Convolution Kernel Functions ----------
std::vector<std::vector<double>> getSobelXKernel() {
  return { {-1, 0, 1},
           {-2, 0, 2},
           {-1, 0, 1} };
}

std::vector<std::vector<double>> getSobelYKernel() {
  return { {-1, -2, -1},
           { 0,  0,  0},
           { 1,  2,  1} };
}

std::vector<std::vector<double>> getGaussianKernel(int kernelSize, double sigma) {
  std::vector<std::vector<double>> kernel(kernelSize, std::vector<double>(kernelSize));
  double sum = 0.0;
  int half = kernelSize / 2;
  for (int i = 0; i < kernelSize; ++i) {
    int x = i - half;
    for (int j = 0; j < kernelSize; ++j) {
      int y = j - half;
      kernel[i][j] = exp(-(x * x + y * y) / (2 * sigma * sigma));
      sum += kernel[i][j];
    }
  }
  for (auto &row : kernel) {
    for (auto &elem : row) {
      elem /= sum;
    }
  }
  return kernel;
}

/**
 * @brief Perform convolution on the input image (CV_64FC1) using the provided kernel.
 */
cv::Mat convolveSequential(const cv::Mat &input,
                           const std::vector<std::vector<double>> &kernel) {
  int kernelSize = kernel.size();
  assert(kernelSize % 2 == 1 && "Kernel size has to be odd");

  int k = kernelSize / 2;
  cv::Mat output(input.rows, input.cols, CV_64FC1, cv::Scalar(0));

  for (int y = k; y < input.rows - k; y++) {
    for (int x = k; x < input.cols - k; x++) {
      double sum = 0.0;
      for (int i = -k; i <= k; i++) {
        for (int j = -k; j <= k; j++) {
          sum += input.at<double>(y + i, x + j) * kernel[k + i][k + j];
        }
      }
      output.at<double>(y, x) = sum;
    }
  }
  return output;
}

// ---------- Harris Corner Detection ----------
std::vector<cv::KeyPoint> seqHarrisCornerDetectorDetect(const cv::Mat &image,
                              HarrisCornerOptions options) {
  Timer timer;
  
  cv::Mat gray;
  if (image.channels() == 3) {
    cv::cvtColor(image, gray, cv::COLOR_BGR2GRAY);
  } else {
    gray = image.clone();
  }
  gray.convertTo(gray, CV_64F);

  auto sobelXKernel = getSobelXKernel();
  auto sobelYKernel = getSobelYKernel();
  auto gaussianKernel = getGaussianKernel(5, 1.0);

  cv::Mat gradX, gradY;;
  convolveCUDA(gray, gradX, sobelXKernel);
  // cv::Mat gradY;
  convolveCUDA(gray, gradY, sobelYKernel);
  // cv::Mat gradX = convolveSequential(gray, sobelXKernel);
  // cv::Mat gradY = convolveSequential(gray, sobelYKernel);
  cv::Mat gradXX = gradX.mul(gradX);
  cv::Mat gradYY = gradY.mul(gradY);
  cv::Mat gradXY = gradX.mul(gradY);

  // gradXX = convolveSequential(gradXX, gaussianKernel);
  // gradYY = convolveSequential(gradYY, gaussianKernel);
  // gradXY = convolveSequential(gradXY, gaussianKernel);
  convolveCUDA(gradXX, gradXX, gaussianKernel);
  convolveCUDA(gradYY, gradYY, gaussianKernel);
  convolveCUDA(gradXY, gradXY, gaussianKernel);

  cv::Mat harrisResp(gray.size(), CV_64F, cv::Scalar(0));
  for (int y = 0; y < gray.rows; y++) {
    for (int x = 0; x < gray.cols; x++) {
      double xx = gradXX.at<double>(y, x);
      double yy = gradYY.at<double>(y, x);
      double xy = gradXY.at<double>(y, x);
      double det = xx * yy - xy * xy;
      double trace = xx + yy;
      harrisResp.at<double>(y, x) = det - options.k_ * trace * trace;
    }
  }

  std::vector<cv::KeyPoint> keypoints;
  int halfLen = options.nmsNeighborhood_ / 2;
  for (int y = halfLen; y < gray.rows - halfLen; y++) {
    for (int x = halfLen; x < gray.cols - halfLen; x++) {
      double resp = harrisResp.at<double>(y, x);
      if (resp <= options.nmsThresh_)
        continue;
      double max_resp = std::numeric_limits<double>::lowest();
      for (int i = -halfLen; i <= halfLen; i++) {
        for (int j = -halfLen; j <= halfLen; j++) {
          if (i == 0 && j == 0)
            continue;
          max_resp = std::max(max_resp, harrisResp.at<double>(y + i, x + j));
          if (max_resp > resp)
            goto skip;
        }
      }
      if (resp > max_resp) {
        keypoints.push_back(cv::KeyPoint(x, y, 1.f));
      }
    skip:
      ;
    }
  }
  
  double elapsed = timer.elapsed();
  std::cout << "Harris Corner Detection: " << std::fixed << std::setprecision(3) << elapsed << " ms" << std::endl;
  return keypoints;
}

// ---------- Harris Corner Matching (SSD) ----------
std::vector<cv::DMatch> seqHarrisMatchKeyPoints(
    const std::vector<cv::KeyPoint> &keypointsL,
    const std::vector<cv::KeyPoint> &keypointsR,
    const cv::Mat &image1, const cv::Mat &image2,
    const HarrisCornerOptions options, int offset = 0) {

  Timer timer;
  
  const int patchSize = options.patchSize_;
  const double maxSSDThresh = options.maxSSDThresh_;
  std::vector<cv::DMatch> matches;
  int border = patchSize / 2;

  for (size_t i = 0; i < keypointsL.size(); i++) {
    const auto &kp1 = keypointsL[i];
    cv::Point2f pos1 = kp1.pt;
    if (pos1.x < border || pos1.y < border ||
        pos1.x + border >= image1.cols || pos1.y + border >= image1.rows) {
      continue;
    }

    int bestMatchIndex = -1;
    uint64_t bestMatchSSD = std::numeric_limits<uint64_t>::max();
    for (size_t j = 0; j < keypointsR.size(); j++) {
      const auto &kp2 = keypointsR[j];
      cv::Point2f pos2 = kp2.pt;
      if (pos2.x < border || pos2.y < border ||
          pos2.x + border >= image2.cols || pos2.y + border >= image2.rows) {
        continue;
      }
      uint64_t ssd = 0;
      for (int y = -border; y <= border; y++) {
        for (int x = -border; x <= border; x++) {
          cv::Vec3b p1 = image1.at<cv::Vec3b>(static_cast<int>(pos1.y) + y, static_cast<int>(pos1.x) + x);
          cv::Vec3b p2 = image2.at<cv::Vec3b>(static_cast<int>(pos2.y) + y, static_cast<int>(pos2.x) + x);
          uint64_t diff = 0;
          for (int c = 0; c < 3; c++) {
            diff += (p1[c] - p2[c]) * (p1[c] - p2[c]);
          }
          ssd += diff;
        }
      }
      if (ssd < bestMatchSSD) {
        bestMatchSSD = ssd;
        bestMatchIndex = j;
      }
    }

    if (bestMatchSSD < maxSSDThresh) {
      matches.push_back(cv::DMatch(static_cast<int>(i + offset), bestMatchIndex, static_cast<float>(bestMatchSSD)));
    }
  }
  
  double elapsed = timer.elapsed();
  std::cout << "Harris Corner Matching: " << std::fixed << std::setprecision(3) << elapsed << " ms" << std::endl;
  return matches;
}

// ---------- RANSAC Homography Estimation ----------
class SeqRansacHomographyCalculator {
public:
  SeqRansacHomographyCalculator(RansacOptions options)
      : options_(options) {}

  cv::Mat computeHomography(const std::vector<cv::KeyPoint> &keypoints1,
                            const std::vector<cv::KeyPoint> &keypoints2,
                            const std::vector<cv::DMatch> &matches) {
    Timer timer;
    
    const int numIterations = options_.numIterations_;
    const int numSamples = options_.numSamples_;
    const double distanceThreshold = options_.distanceThreshold_;

    cv::Mat bestHomography;
    int bestInlierCount = 0;

    std::random_device rd;
    std::mt19937 rng(rd());

    for (int iter = 0; iter < numIterations; ++iter) {
      if (matches.size() < static_cast<size_t>(numSamples))
        break;
      std::vector<cv::DMatch> localMatches = matches;
      std::shuffle(localMatches.begin(), localMatches.end(), rng);

      std::vector<cv::Point2f> srcPoints, dstPoints;
      for (int j = 0; j < numSamples; j++) {
        srcPoints.push_back(keypoints1[localMatches[j].queryIdx].pt);
        dstPoints.push_back(keypoints2[localMatches[j].trainIdx].pt);
      }

      cv::Mat H = cv::findHomography(srcPoints, dstPoints);
      if (H.empty())
        continue;

      int inlierCount = 0;
      for (const auto &match : localMatches) {
        cv::Point2f pt1 = keypoints1[match.queryIdx].pt;
        cv::Point2f pt2 = keypoints2[match.trainIdx].pt;
        cv::Mat pt1Mat = (cv::Mat_<double>(3, 1) << pt1.x, pt1.y, 1.0);
        cv::Mat pt2Transformed = H * pt1Mat;
        pt2Transformed /= pt2Transformed.at<double>(2, 0);
        cv::Point2f pt2Estimate(pt2Transformed.at<double>(0, 0),
                                pt2Transformed.at<double>(1, 0));
        if (cv::norm(pt2Estimate - pt2) < distanceThreshold)
          inlierCount++;
      }
      if (inlierCount > bestInlierCount) {
        bestInlierCount = inlierCount;
        bestHomography = H;
      }
    }
    
    double elapsed = timer.elapsed();
    std::cout << "RANSAC Homography Estimation: " << std::fixed << std::setprecision(3) << elapsed << " ms" << std::endl;
    return bestHomography;
  }
private:
  RansacOptions options_;
};

// ---------- Stitching Two Images ----------
// Assume the left image is the base and the right image is transformed by the homography and stitched onto the left image.
cv::Mat stitchTwoImages(const cv::Mat &leftImage, const cv::Mat &rightImage,
                        HarrisCornerOptions harrisOpts, RansacOptions ransacOpts) {
  Timer timer;
  
  // 1. Corner Detection
  auto keypointsLeft = gpuHarrisCornerDetectorDetect(leftImage, harrisOpts.k_, harrisOpts.nmsThresh_, harrisOpts.nmsNeighborhood_);
  auto keypointsRight = gpuHarrisCornerDetectorDetect(rightImage, harrisOpts.k_, harrisOpts.nmsThresh_, harrisOpts.nmsNeighborhood_);

  // 2. Corner Matching: treat the right image as the one to be transformed and the left image as the base.
  Timer matchTimer;
  auto matches = gpuHarrisMatchKeyPoints(keypointsRight, keypointsLeft, rightImage, leftImage, 
                                        harrisOpts.patchSize_, harrisOpts.maxSSDThresh_);
  double matchElapsed = matchTimer.elapsed();
  std::cout << "Harris Corner Matching (GPU): " << std::fixed << std::setprecision(3) << matchElapsed << " ms" << std::endl;
  
  if (matches.empty()) {
    std::cerr << "Not enough matched corners for stitching!" << std::endl;
    return cv::Mat();
  }

  // 3. Use RANSAC to estimate the homography H such that H * (points from right image) approximates (points from left image).
  // Replace sequential RANSAC with GPU-accelerated RANSAC
  Timer ransacTimer;
  GpuRansacHomographyCalculator::Options gpuRansacOpts;
  gpuRansacOpts.numIterations_ = ransacOpts.numIterations_;
  gpuRansacOpts.numSamples_ = ransacOpts.numSamples_;
  gpuRansacOpts.distanceThreshold_ = ransacOpts.distanceThreshold_;
  
  GpuRansacHomographyCalculator ransac(gpuRansacOpts);
  cv::Mat H = ransac.computeHomography(keypointsRight, keypointsLeft, matches);
  double ransacElapsed = ransacTimer.elapsed();
  std::cout << "RANSAC Homography Estimation (GPU): " << std::fixed << std::setprecision(3) << ransacElapsed << " ms" << std::endl;
  
  if (H.empty()) {
    std::cerr << "GPU RANSAC failed, falling back to CPU implementation" << std::endl;
    Timer cpuRansacTimer;
    SeqRansacHomographyCalculator cpuRansac(ransacOpts);
    H = cpuRansac.computeHomography(keypointsRight, keypointsLeft, matches);
    double cpuRansacElapsed = cpuRansacTimer.elapsed();
    std::cout << "RANSAC Homography Estimation (CPU fallback): " << std::fixed << std::setprecision(3) << cpuRansacElapsed << " ms" << std::endl;
    
    if (H.empty()) {
      std::cerr << "RANSAC failed to estimate a homography matrix!" << std::endl;
      return cv::Mat();
    }
  }

  // 4. Compute the transformed boundaries of the right image and create a canvas that fits both images.
  std::vector<cv::Point2f> rightCorners = {
      cv::Point2f(0, 0),
      cv::Point2f(rightImage.cols, 0),
      cv::Point2f(rightImage.cols, rightImage.rows),
      cv::Point2f(0, rightImage.rows)
  };
  std::vector<cv::Point2f> warpedCorners;
  cv::perspectiveTransform(rightCorners, warpedCorners, H);

  std::vector<cv::Point2f> leftCorners = {
      cv::Point2f(0, 0),
      cv::Point2f(leftImage.cols, 0),
      cv::Point2f(leftImage.cols, leftImage.rows),
      cv::Point2f(0, leftImage.rows)
  };

  float minX = 0, minY = 0, maxX = static_cast<float>(leftImage.cols), maxY = static_cast<float>(leftImage.rows);
  for (const auto &pt : warpedCorners) {
    minX = std::min(minX, pt.x);
    minY = std::min(minY, pt.y);
    maxX = std::max(maxX, pt.x);
    maxY = std::max(maxY, pt.y);
  }
  for (const auto &pt : leftCorners) {
    minX = std::min(minX, pt.x);
    minY = std::min(minY, pt.y);
    maxX = std::max(maxX, pt.x);
    maxY = std::max(maxY, pt.y);
  }

  // Translation: apply a translation when there are negative coordinates.
  cv::Mat translation = (cv::Mat_<double>(3,3) << 1, 0, -minX,
                                                  0, 1, -minY,
                                                  0, 0, 1);
  cv::Size canvasSize(static_cast<int>(std::ceil(maxX - minX)), static_cast<int>(std::ceil(maxY - minY)));

  cv::Mat warpedRight;
  cv::warpPerspective(rightImage, warpedRight, translation * H, canvasSize);

  // Copy the left image onto the canvas at the appropriate position (translation only).
  cv::Mat canvas(canvasSize, leftImage.type(), cv::Scalar::all(0));
  cv::Mat roi = canvas(cv::Rect(-minX, -minY, leftImage.cols, leftImage.rows));
  leftImage.copyTo(roi);

  // Simple overlay fusion: for non-black pixels in warpedRight, overlay them onto the canvas.
  for (int y = 0; y < canvas.rows; y++) {
    for (int x = 0; x < canvas.cols; x++) {
      cv::Vec3b pixel = warpedRight.at<cv::Vec3b>(y, x);
      if (pixel != cv::Vec3b(0, 0, 0))
        canvas.at<cv::Vec3b>(y, x) = pixel;
    }
  }
  
  double elapsed = timer.elapsed();
  std::cout << "Image Stitching: " << std::fixed << std::setprecision(3) << elapsed << " ms" << std::endl;
  return canvas;
}

// ---------- Stitching All Images ----------
// Stitch the images in sequence: use the previous stitched image as the base for the next image.
cv::Mat stitchAllImages(const std::vector<cv::Mat> &images,
                        HarrisCornerOptions harrisOpts, RansacOptions ransacOpts) {
  Timer timer;
  
  if (images.empty()) return cv::Mat();
  cv::Mat panorama = images[0];
  for (size_t i = 1; i < images.size(); i++) {
    std::cout << "Stitching image " << i+1 << " of " << images.size() << "..." << std::endl;
    cv::Mat temp = stitchTwoImages(panorama, images[i], harrisOpts, ransacOpts);
    if (temp.empty()) {
      std::cerr << "Failed to stitch image " << i << "!" << std::endl;
      continue;
    }
    panorama = temp;
  }
  
  double elapsed = timer.elapsed();
  std::cout << "Total Stitching Process: " << std::fixed << std::setprecision(3) << elapsed << " ms" << std::endl;
  return panorama;
}

// ========================== Main Function ==========================
int main(int argc, char** argv) {
  Timer totalTimer;
  
  // Use the image reader to load images and determine the output file name.
  ImageReaderResult readerResult = readImagesFromArgs(argc, argv);
  if (readerResult.images.size() < 2) {
    std::cerr << "At least two images are required for stitching!" << std::endl;
    return -1;
  }

  // Set options (adjust as needed)
  HarrisCornerOptions harrisOpts;
  harrisOpts.nmsThresh_ = 1e6;
  harrisOpts.maxSSDThresh_ = 1e8;

  RansacOptions ransacOpts;
  ransacOpts.numIterations_ = 1000;
  ransacOpts.numSamples_ = 4;
  ransacOpts.distanceThreshold_ = 3.0;

  // Stitch all loaded images
  cv::Mat panorama = stitchAllImages(readerResult.images, harrisOpts, ransacOpts);
  if (panorama.empty()) {
    std::cerr << "Panoramic stitching failed!" << std::endl;
    return -1;
  }

  // Save the stitched result to the output file.
  cv::imwrite(readerResult.outputFile, panorama);
  std::cout << "Stitched result saved to " << readerResult.outputFile << std::endl;
  
  double totalElapsed = totalTimer.elapsed();
  std::cout << "\nTotal Execution Time: " << std::fixed << std::setprecision(3) << totalElapsed << " ms" << std::endl;

  return 0;
}
