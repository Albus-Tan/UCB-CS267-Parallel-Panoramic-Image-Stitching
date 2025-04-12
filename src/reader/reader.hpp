#pragma once

#include <string>
#include <vector>
#include <opencv2/core.hpp>

// A structure to hold the read images and output file name
struct ImageReaderResult {
    std::vector<cv::Mat> images;
    std::string outputFile;
};

// Function that parses command-line arguments and reads the images.
// See reader.cpp for the function implementation.
ImageReaderResult readImagesFromArgs(int argc, char** argv);
