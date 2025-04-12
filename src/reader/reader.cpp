// reader.cpp
// This file contains functions/classes for loading images from command-line arguments.

#include "reader.hpp"

#include <iostream>
#include <filesystem>  // C++17
#include <vector>
#include <string>
#include <opencv2/opencv.hpp>

namespace fs = std::filesystem;

ImageReaderResult readImagesFromArgs(int argc, char** argv) {
    ImageReaderResult result;
    result.outputFile = "result.jpg";  // Default output file name

    std::vector<std::string> fileNames;
    std::string dirName;

    // Basic argument check
    if (argc < 2) {
        std::cerr << "Usage: " << argv[0] 
                  << " [--dir directory_name] [--out output_file_name] [image1 image2 ...]" 
                  << std::endl;
        std::exit(-1);
    }

    // Parse command-line arguments
    for (int i = 1; i < argc; i++) {
        std::string arg(argv[i]);
        if (arg == "--dir") {
            if (i + 1 < argc) {
                dirName = argv[++i];
            } else {
                std::cerr << "Error: --dir requires a directory name" << std::endl;
                std::exit(-1);
            }
        } else if (arg == "--out") {
            if (i + 1 < argc) {
                result.outputFile = argv[++i];
            } else {
                std::cerr << "Error: --out requires an output file name" << std::endl;
                std::exit(-1);
            }
        } else {
            // If it's neither --dir nor --out, treat it as an image filename
            fileNames.push_back(arg);
        }
    }

    // Load images from directory if specified
    if (!dirName.empty()) {
        if (!fs::exists(dirName) || !fs::is_directory(dirName)) {
            std::cerr << "Error: " << dirName << " is not a valid directory." << std::endl;
            std::exit(-1);
        }
        for (const auto& entry : fs::directory_iterator(dirName)) {
            if (entry.is_regular_file()) {
                std::string filePath = entry.path().string();
                cv::Mat img = cv::imread(filePath);
                if (img.empty()) {
                    std::cerr << "Warning: Unable to open image file: " << filePath << std::endl;
                    continue;
                }
                result.images.push_back(img);
            }
        }
    } else {
        // Otherwise, load images from the list of filenames
        for (const auto& fileName : fileNames) {
            cv::Mat img = cv::imread(fileName);
            if (img.empty()) {
                std::cerr << "Warning: Unable to open image file: " << fileName << std::endl;
                continue;
            }
            result.images.push_back(img);
        }
    }

    return result;
}
