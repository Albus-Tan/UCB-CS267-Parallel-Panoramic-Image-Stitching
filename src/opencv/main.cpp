#include <iostream>
#include <vector>
#include <string>
#include <opencv2/opencv.hpp>
#include <opencv2/stitching.hpp>

// Use std::filesystem if available; fallback to experimental if necessary.
#if __cplusplus >= 201703L
#include <filesystem>
namespace fs = std::filesystem;
#else
#include <experimental/filesystem>
namespace fs = std::experimental::filesystem;
#endif

int main(int argc, char** argv) {
    // Usage message if not enough arguments are provided.
    if (argc < 2) {
        std::cout << "Usage: " << argv[0] << " [--dir directory_name] [--out output_file_name] [image1 image2 ...]" << std::endl;
        return -1;
    }
    
    // Default output file name.
    std::string outputFile = "result.jpg";
    std::vector<std::string> fileNames;
    std::string dirName;
    
    // Parse command-line arguments.
    for (int i = 1; i < argc; i++) {
        std::string arg = argv[i];
        if (arg == "--dir") {
            if (i + 1 < argc) {
                dirName = argv[++i];
            } else {
                std::cerr << "Error: --dir requires a directory name" << std::endl;
                return -1;
            }
        } else if (arg == "--out") {
            if (i + 1 < argc) {
                outputFile = argv[++i];
            } else {
                std::cerr << "Error: --out requires an output file name" << std::endl;
                return -1;
            }
        } else {
            // Otherwise, assume the argument is an image file name.
            fileNames.push_back(arg);
        }
    }
    
    std::vector<cv::Mat> images;
    
    // If a directory is specified, load all images from that directory.
    if (!dirName.empty()) {
        if (!fs::exists(dirName) || !fs::is_directory(dirName)) {
            std::cerr << "Error: " << dirName << " is not a valid directory." << std::endl;
            return -1;
        }
        // Iterate over files in the directory.
        for (const auto& entry : fs::directory_iterator(dirName)) {
            if (entry.is_regular_file()) {
                std::string filePath = entry.path().string();
                cv::Mat img = cv::imread(filePath);
                if (img.empty()) {
                    std::cerr << "Warning: Unable to open image file: " << filePath << std::endl;
                    continue;
                }
                images.push_back(img);
            }
        }
    } else {
        // Otherwise, use the provided file names.
        for (const auto& fileName : fileNames) {
            cv::Mat img = cv::imread(fileName);
            if (img.empty()) {
                std::cerr << "Warning: Unable to open image file: " << fileName << std::endl;
                continue;
            }
            images.push_back(img);
        }
    }
    
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
