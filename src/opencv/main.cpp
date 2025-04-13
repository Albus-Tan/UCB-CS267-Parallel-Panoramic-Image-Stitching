#include <iostream>
#include <vector>
#include <string>
#include <chrono>
#include <iomanip>
#include <opencv2/opencv.hpp>
#include <opencv2/stitching.hpp>
#include "reader.hpp"

// Use std::filesystem if available; fallback to experimental if necessary.
#if __cplusplus >= 201703L
#include <filesystem>
namespace fs = std::filesystem;
#else
#include <experimental/filesystem>
namespace fs = std::experimental::filesystem;
#endif

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

int main(int argc, char** argv) {
    Timer totalTimer;
    
    Timer timer;
    std::cout << "Reading input images..." << std::endl;
    ImageReaderResult readerRes = readImagesFromArgs(argc, argv);
    std::vector<cv::Mat>& images = readerRes.images;
    std::string outputFile = readerRes.outputFile;
    double readingTime = timer.elapsed();
    std::cout << "Reading input images: " << std::fixed << std::setprecision(3) << readingTime << " ms" << std::endl;

    // Ensure that at least two images have been loaded.
    if (images.size() < 2) {
        std::cerr << "Error: Need at least two valid images to perform stitching." << std::endl;
        return -1;
    }
    
    // Create a stitcher instance (using PANORAMA mode).
    timer.reset();
    std::cout << "Creating stitcher and performing stitching..." << std::endl;
    cv::Ptr<cv::Stitcher> stitcher = cv::Stitcher::create(cv::Stitcher::PANORAMA);
    cv::Mat panorama;
    cv::Stitcher::Status status = stitcher->stitch(images, panorama);
    double stitchingTime = timer.elapsed();
    std::cout << "Stitching process: " << std::fixed << std::setprecision(3) << stitchingTime << " ms" << std::endl;
    
    if (status != cv::Stitcher::OK) {
        std::cerr << "Error during stitching, error code: " << int(status) << std::endl;
        return -1;
    }
    
    // Save the resulting panorama.
    timer.reset();
    std::cout << "Saving panorama..." << std::endl;
    if (!cv::imwrite(outputFile, panorama)) {
        std::cerr << "Error: Could not save the panorama image to " << outputFile << std::endl;
        return -1;
    }
    double savingTime = timer.elapsed();
    std::cout << "Saving panorama: " << std::fixed << std::setprecision(3) << savingTime << " ms" << std::endl;
    
    std::cout << "Panorama successfully saved to " << outputFile << std::endl;
    
    double totalElapsed = totalTimer.elapsed();
    std::cout << "\nTotal Execution Time (OpenCV): " << std::fixed << std::setprecision(3) << totalElapsed << " ms" << std::endl;
    
    return 0;
}
