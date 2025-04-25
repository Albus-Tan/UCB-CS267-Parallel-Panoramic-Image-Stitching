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

#ifdef _OPENMP
#include <omp.h>
#endif

#include "../reader/reader.hpp"

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
  
  #pragma omp parallel for reduction(+:sum) collapse(2)
  for (int i = 0; i < kernelSize; ++i) {
    for (int j = 0; j < kernelSize; ++j) {
      int x = i - half;
      int y = j - half;
      kernel[i][j] = exp(-(x * x + y * y) / (2 * sigma * sigma));
      sum += kernel[i][j];
    }
  }
  
  #pragma omp parallel for collapse(2)
  for (int i = 0; i < kernelSize; ++i) {
    for (int j = 0; j < kernelSize; ++j) {
      kernel[i][j] /= sum;
    }
  }
  
  return kernel;
}

/**
 * @brief Perform convolution on the input image (CV_64FC1) using the provided kernel, with OpenMP parallelization.
 */
cv::Mat convolveParallel(const cv::Mat &input,
                         const std::vector<std::vector<double>> &kernel) {
  int kernelSize = kernel.size();
  assert(kernelSize % 2 == 1 && "Kernel size has to be odd");

  int k = kernelSize / 2;
  cv::Mat output(input.rows, input.cols, CV_64FC1, cv::Scalar(0));

  #pragma omp parallel for collapse(2)
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
std::vector<cv::KeyPoint> ompHarrisCornerDetectorDetect(const cv::Mat &image,
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

  cv::Mat gradX = convolveParallel(gray, sobelXKernel);
  cv::Mat gradY = convolveParallel(gray, sobelYKernel);
  
  cv::Mat gradXX(gray.size(), CV_64F);
  cv::Mat gradYY(gray.size(), CV_64F);
  cv::Mat gradXY(gray.size(), CV_64F);
  
  #pragma omp parallel for collapse(2)
  for (int y = 0; y < gray.rows; y++) {
    for (int x = 0; x < gray.cols; x++) {
      double gx = gradX.at<double>(y, x);
      double gy = gradY.at<double>(y, x);
      gradXX.at<double>(y, x) = gx * gx;
      gradYY.at<double>(y, x) = gy * gy;
      gradXY.at<double>(y, x) = gx * gy;
    }
  }

  gradXX = convolveParallel(gradXX, gaussianKernel);
  gradYY = convolveParallel(gradYY, gaussianKernel);
  gradXY = convolveParallel(gradXY, gaussianKernel);

  cv::Mat harrisResp(gray.size(), CV_64F);
  
  #pragma omp parallel for collapse(2)
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

  // Non-maximum suppression - can't fully parallelize due to data dependencies
  // We'll use thread-local vectors and then merge
  std::vector<cv::KeyPoint> keypoints;
  int halfLen = options.nmsNeighborhood_ / 2;

  // First pass: Find local maxima in each region
  std::vector<std::vector<cv::KeyPoint>> threadLocalKeypoints;
  
  #pragma omp parallel
  {
    std::vector<cv::KeyPoint> localKeypoints;
    
    #pragma omp for schedule(dynamic)
    for (int y = halfLen; y < gray.rows - halfLen; y++) {
      for (int x = halfLen; x < gray.cols - halfLen; x++) {
        double resp = harrisResp.at<double>(y, x);
        if (resp <= options.nmsThresh_)
          continue;
        
        bool isLocalMax = true;
        for (int i = -halfLen; i <= halfLen && isLocalMax; i++) {
          for (int j = -halfLen; j <= halfLen; j++) {
            if (i == 0 && j == 0)
              continue;
            if (harrisResp.at<double>(y + i, x + j) > resp) {
              isLocalMax = false;
              break;
            }
          }
        }
        
        if (isLocalMax) {
          localKeypoints.push_back(cv::KeyPoint(x, y, 1.f));
        }
      }
    }
    
    #pragma omp critical
    {
      threadLocalKeypoints.push_back(std::move(localKeypoints));
    }
  }
  
  // Merge thread-local keypoints
  for (const auto& localKeypoints : threadLocalKeypoints) {
    keypoints.insert(keypoints.end(), localKeypoints.begin(), localKeypoints.end());
  }
  
  double elapsed = timer.elapsed();
  std::cout << "Harris Corner Detection (OpenMP): " << std::fixed << std::setprecision(3) << elapsed << " ms" << std::endl;
  return keypoints;
}

// ---------- Harris Corner Matching (SSD) ----------
std::vector<cv::DMatch> ompHarrisMatchKeyPoints(
    const std::vector<cv::KeyPoint>& keypointsL,
    const std::vector<cv::KeyPoint>& keypointsR,
    const cv::Mat& image1,
    const cv::Mat& image2,
    const HarrisCornerOptions options,
    int offset = 0)
{
    Timer timer;

    const int PS     = options.patchSize_;
    const int border = PS / 2;
    const uint64_t maxSSD = uint64_t(options.maxSSDThresh_);

    CV_Assert(image1.type() == CV_8UC3 && image2.type() == CV_8UC3);
    CV_Assert(image1.isContinuous() && image2.isContinuous());

    const int W1 = image1.cols, H1 = image1.rows;
    const int W2 = image2.cols, H2 = image2.rows;
    const uchar* d1 = image1.ptr<uchar>(0);
    const uchar* d2 = image2.ptr<uchar>(0);
    const int    s1 = W1 * 3, s2 = W2 * 3;

    std::vector<int> idxL, xL, yL;
    idxL.reserve(keypointsL.size());
    xL.reserve(idxL.capacity());
    yL.reserve(idxL.capacity());
    for (int i = 0; i < (int)keypointsL.size(); ++i) {
        int x = int(std::round(keypointsL[i].pt.x));
        int y = int(std::round(keypointsL[i].pt.y));
        if (x >= border && y >= border && x + border < W1 && y + border < H1) {
            idxL.push_back(i);
            xL.push_back(x);
            yL.push_back(y);
        }
    }

    std::vector<int> idxR, xR, yR;
    idxR.reserve(keypointsR.size());
    xR.reserve(idxR.capacity());
    yR.reserve(idxR.capacity());
    for (int j = 0; j < (int)keypointsR.size(); ++j) {
        int x = int(std::round(keypointsR[j].pt.x));
        int y = int(std::round(keypointsR[j].pt.y));
        if (x >= border && y >= border && x + border < W2 && y + border < H2) {
            idxR.push_back(j);
            xR.push_back(x);
            yR.push_back(y);
        }
    }

    int nthreads = omp_get_max_threads();
    std::vector<std::vector<cv::DMatch>> threadMatches(nthreads);

    #pragma omp parallel
    {
        int tid = omp_get_thread_num();
        auto& local = threadMatches[tid];

        #pragma omp for schedule(dynamic)
        for (int u = 0; u < (int)idxL.size(); ++u) {
            int i  = idxL[u];
            int x1 = xL[u], y1 = yL[u];

            uint64_t bestSSD = std::numeric_limits<uint64_t>::max();
            int      bestJ   = -1;

            for (int v = 0; v < (int)idxR.size(); ++v) {
                int j  = idxR[v];
                int x2 = xR[v], y2 = yR[v];

                uint64_t ssd = 0;
                bool pruned = false;

                for (int dy = -border; dy <= border; ++dy) {
                    const uchar* row1 = d1 + (y1 + dy) * s1 + (x1 - border) * 3;
                    const uchar* row2 = d2 + (y2 + dy) * s2 + (x2 - border) * 3;

                    #pragma omp simd reduction(+:ssd)
                    for (int dx = 0; dx < PS; ++dx) {
                        int idx3 = dx * 3;
                        int d0 = int(row1[idx3    ]) - int(row2[idx3    ]);
                        int d1 = int(row1[idx3 + 1]) - int(row2[idx3 + 1]);
                        int d2 = int(row1[idx3 + 2]) - int(row2[idx3 + 2]);
                        ssd += uint64_t(d0*d0 + d1*d1 + d2*d2);
                    }
                    if (ssd >= bestSSD) {
                        pruned = true;
                        break;
                    }
                }

                if (!pruned && ssd < bestSSD) {
                    bestSSD = ssd;
                    bestJ   = j;
                }
            }

            if (bestSSD < maxSSD) {
                local.emplace_back(i + offset, bestJ, static_cast<float>(bestSSD));
            }
        }
    }

    std::vector<cv::DMatch> matches;
    for (auto& vec : threadMatches) {
        matches.insert(matches.end(), vec.begin(), vec.end());
    }

    double elapsed = timer.elapsed();
    std::cout << "Harris Corner Matching (OpenMP): "
              << std::fixed << std::setprecision(3)
              << elapsed << " ms\n";
    return matches;
}

// ---------- RANSAC Homography Estimation ----------

class OmpRansacHomographyCalculator {
public:
  OmpRansacHomographyCalculator(RansacOptions options)
    : options_(options) {}

  cv::Mat computeHomography(
      const std::vector<cv::KeyPoint>& keypoints1,
      const std::vector<cv::KeyPoint>& keypoints2,
      const std::vector<cv::DMatch>& matches) {
    Timer timer;

    const int numIterations     = options_.numIterations_;
    const int numSamples        = options_.numSamples_;
    const double distThreshold  = options_.distanceThreshold_;
    const int M = (int)matches.size();

    if (M < numSamples) {
      std::cerr << "Not enough matches for RANSAC" << std::endl;
      return cv::Mat();
    }

    // Pre-allocate buffers
    std::vector<cv::DMatch> sampleMatches(numSamples);
    std::vector<cv::Point2f> srcPoints(numSamples);
    std::vector<cv::Point2f> dstPoints(numSamples);

    // Per-thread best inlier counts and homographies
    int T = omp_get_max_threads();
    std::vector<int> bestCounts(T, 0);
    std::vector<cv::Mat> bestH(T);

    // Random seed
    std::random_device rd;
    unsigned int seed0 = rd();

    #pragma omp parallel
    {
      int tid = omp_get_thread_num();
      int localBestCount = 0;
      cv::Mat localBestH;
      std::mt19937 rng(seed0 + tid);

      #pragma omp for schedule(dynamic)
      for (int iter = 0; iter < numIterations; ++iter) {
        // 1) Sample without replacement
        std::sample(
          matches.begin(), matches.end(),
          sampleMatches.begin(), sampleMatches.size(),
          rng);

        // 2) Fill src/dst point arrays
        for (int k = 0; k < numSamples; ++k) {
          const auto &m = sampleMatches[k];
          srcPoints[k] = keypoints1[m.queryIdx].pt;
          dstPoints[k] = keypoints2[m.trainIdx].pt;
        }

        // 3) Compute homography on this minimal sample
        cv::Mat H = cv::findHomography(srcPoints, dstPoints);
        if (H.empty()) continue;

        // 4) Count inliers in all matches
        int inlierCount = 0;
        double thr2 = distThreshold * distThreshold;
        for (int i = 0; i < M; ++i) {
          cv::Point2f p1 = keypoints1[matches[i].queryIdx].pt;
          cv::Point2f p2 = keypoints2[matches[i].trainIdx].pt;
          cv::Mat p1h = (cv::Mat_<double>(3,1) << p1.x, p1.y, 1.0);
          cv::Mat p2h = H * p1h;
          double w = p2h.at<double>(2,0);
          double dx = p2h.at<double>(0,0)/w - p2.x;
          double dy = p2h.at<double>(1,0)/w - p2.y;
          if (dx*dx + dy*dy < thr2) ++inlierCount;
        }

        // 5) Update local best
        if (inlierCount > localBestCount) {
          localBestCount = inlierCount;
          localBestH = H.clone();
        }
      }

      // Store per-thread best
      bestCounts[tid] = localBestCount;
      bestH[tid]      = localBestH;
    }

    // Serial reduction of thread results
    int bestId = std::max_element(bestCounts.begin(), bestCounts.end()) - bestCounts.begin();
    cv::Mat bestHomography = bestH[bestId];

    double elapsed = timer.elapsed();
    std::cout << "RANSAC Homography Estimation (OpenMP): "
              << std::fixed << std::setprecision(3)
              << elapsed << " ms\n";
    return bestHomography;
  }

private:
  RansacOptions options_;
};


// ---------- Stitching Two Images ----------
cv::Mat stitchTwoImages(const cv::Mat &leftImage, const cv::Mat &rightImage,
                        HarrisCornerOptions harrisOpts, RansacOptions ransacOpts) {
  Timer timer;
  
  // 1. Corner Detection
  auto keypointsLeft = ompHarrisCornerDetectorDetect(leftImage, harrisOpts);
  auto keypointsRight = ompHarrisCornerDetectorDetect(rightImage, harrisOpts);

  // 2. Corner Matching: treat the right image as the one to be transformed and the left image as the base.
  auto matches = ompHarrisMatchKeyPoints(keypointsRight, keypointsLeft, rightImage, leftImage, harrisOpts);
  if (matches.empty()) {
    std::cerr << "Not enough matched corners for stitching!" << std::endl;
    return cv::Mat();
  }

  // 3. Use RANSAC to estimate the homography H such that H * (points from right image) approximates (points from left image).
  OmpRansacHomographyCalculator ransac(ransacOpts);
  cv::Mat H = ransac.computeHomography(keypointsRight, keypointsLeft, matches);
  if (H.empty()) {
    std::cerr << "RANSAC failed to estimate a homography matrix!" << std::endl;
    return cv::Mat();
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

  // Overlay fusion with OpenMP parallelization
  #pragma omp parallel for collapse(2)
  for (int y = 0; y < canvas.rows; y++) {
    for (int x = 0; x < canvas.cols; x++) {
      cv::Vec3b pixel = warpedRight.at<cv::Vec3b>(y, x);
      if (pixel != cv::Vec3b(0, 0, 0))
        canvas.at<cv::Vec3b>(y, x) = pixel;
    }
  }
  
  double elapsed = timer.elapsed();
  std::cout << "Image Stitching (OpenMP): " << std::fixed << std::setprecision(3) << elapsed << " ms" << std::endl;
  return canvas;
}

// ---------- Stitching All Images ----------
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
  std::cout << "Total Stitching Process (OpenMP): " << std::fixed << std::setprecision(3) << elapsed << " ms" << std::endl;
  return panorama;
}

// ========================== Main Function ==========================
int main(int argc, char** argv) {
  Timer totalTimer;
  
  // Print OpenMP info
  #ifdef _OPENMP
  std::cout << "OpenMP Version: " << _OPENMP << std::endl;
  std::cout << "Number of Available Threads: " << omp_get_max_threads() << std::endl;
  #else
  std::cout << "OpenMP not available. Running in serial mode." << std::endl;
  #endif
  
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
  std::cout << "\nTotal Execution Time (OpenMP): " << std::fixed << std::setprecision(3) << totalElapsed << " ms" << std::endl;

  return 0;
}
