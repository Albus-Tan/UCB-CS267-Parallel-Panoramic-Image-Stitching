#include <iostream>
#include <vector>
#include <string>
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

int main(int argc, char** argv) {
    ImageReaderResult readerRes = readImagesFromArgs(argc, argv);
    std::vector<cv::Mat>& images = readerRes.images;
    std::string outputFile = readerRes.outputFile;

    // Ensure that at least two images have been loaded.
    if (images.size() < 2) {
        std::cerr << "Error: Need at least two valid images to perform stitching." << std::endl;
        return -1;
    }
    
    // Create a stitcher instance (using PANORAMA mode).
    cv::Ptr<cv::Stitcher> stitcher = cv::Stitcher::create(cv::Stitcher::PANORAMA);
    cv::Mat panorama;
    cv::Stitcher::Status status = stitcher->stitch(images, panorama);
    
    if (status != cv::Stitcher::OK) {
        std::cerr << "Error during stitching, error code: " << int(status) << std::endl;
        return -1;
    }
    
    // Save the resulting panorama.
    if (!cv::imwrite(outputFile, panorama)) {
        std::cerr << "Error: Could not save the panorama image to " << outputFile << std::endl;
        return -1;
    }
    
    std::cout << "Panorama successfully saved to " << outputFile << std::endl;
    return 0;
}
