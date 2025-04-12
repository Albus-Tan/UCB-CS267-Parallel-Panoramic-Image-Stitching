#include <opencv2/opencv.hpp>
#include <opencv2/imgproc.hpp>
#include <opencv2/highgui.hpp>
#include <iostream>
#include <fstream>
#include <sstream>
#include <vector>
#include <string>
#include <cmath>
#include <ctime>
#include <limits>
#include "reader.hpp"

using namespace cv;
using namespace std;

// --------------------------------------------------------------------------
// Structure to store BRIEF features: keypoints and binary descriptors.
struct BriefResult {
    vector<KeyPoint> keypoints;
    Mat descriptors; // Binary descriptors stored as CV_8U: each row is one descriptor.
};

// --------------------------------------------------------------------------
// Util class encapsulates the core functions as implemented in the repo.
class Util {
public:
    // Convert the image to floating point with range [0,1].
    void convertImg2Float(Mat &image) {
        image.convertTo(image, CV_32F, 1.0 / 255);
    }

    void generateTestPattern(Point* &compareA, Point* &compareB, int patchWidth = 9, int nbits = 256, int seed = 42) {
        compareA = new Point[nbits];
        compareB = new Point[nbits];
        cv::RNG rng(seed); // Fixed seed for reproducibility.
        for (int i = 0; i < nbits; ++i) {
            compareA[i] = Point(rng.uniform(0, patchWidth), rng.uniform(0, patchWidth));
            compareB[i] = Point(rng.uniform(0, patchWidth), rng.uniform(0, patchWidth));
        }
    }    

    // Manual BRIEF feature extraction.
    // Uses FAST to detect keypoints on the grayscale image and then computes a binary
    // descriptor for each keypoint based on the test pattern (compareA, compareB).
    BriefResult BriefLite(const Mat &image_color, const Point* compareA, const Point* compareB) {
        BriefResult result;
    
        // Convert to grayscale if needed
        Mat gray;
        if (image_color.channels() == 3) {
            cvtColor(image_color, gray, COLOR_BGR2GRAY);
        } else {
            gray = image_color.clone();  // already grayscale
        }
    
        // Detect keypoints using FAST
        vector<KeyPoint> keypoints;
        FAST(gray, keypoints, 20, true); // threshold=20, nonmaxSuppression=true
        result.keypoints = keypoints;
    
        int n = 256;
        int descriptorLength = (n + 7) / 8;
        result.descriptors = Mat::zeros((int)keypoints.size(), descriptorLength, CV_8U);
    
        // For each keypoint, compute binary descriptor
        for (size_t i = 0; i < keypoints.size(); i++) {
            KeyPoint kp = keypoints[i];
            uchar* descPtr = result.descriptors.ptr<uchar>((int)i);
            for (int j = 0; j < n; j++) {
                int bit = j % 8;
                int byteIndex = j / 8;
                int xA = cvRound(kp.pt.x + compareA[j].x);
                int yA = cvRound(kp.pt.y + compareA[j].y);
                int xB = cvRound(kp.pt.x + compareB[j].x);
                int yB = cvRound(kp.pt.y + compareB[j].y);
                uchar intensityA = 0, intensityB = 0;
                if (xA >= 0 && xA < gray.cols && yA >= 0 && yA < gray.rows)
                    intensityA = gray.at<uchar>(yA, xA);
                if (xB >= 0 && xB < gray.cols && yB >= 0 && yB < gray.rows)
                    intensityB = gray.at<uchar>(yB, xB);
                if (intensityA < intensityB) {
                    descPtr[byteIndex] |= (1 << bit);
                }
            }
        }
    
        return result;
    }
    

    // Helper: Compute Hamming distance between two binary descriptors.
    int hammingDistance(const uchar* a, const uchar* b, int len) {
        int dist = 0;
        for (int i = 0; i < len; i++) {
            uchar v = a[i] ^ b[i];
            // Count set bits.
            while (v) {
                dist += v & 1;
                v >>= 1;
            }
        }
        return dist;
    }

    // Perform brute-force matching between two sets of BRIEF descriptors.
    vector<DMatch> briefMatch(const BriefResult &br1, const BriefResult &br2) {
        vector<DMatch> matches;
        int descriptorLength = br1.descriptors.cols;
        for (int i = 0; i < br1.descriptors.rows; i++) {
            const uchar* desc1 = br1.descriptors.ptr<uchar>(i);
            int bestDist = INT_MAX;
            int bestIdx = -1;
            for (int j = 0; j < br2.descriptors.rows; j++) {
                const uchar* desc2 = br2.descriptors.ptr<uchar>(j);
                int dist = hammingDistance(desc1, desc2, descriptorLength);
                if (dist < bestDist) {
                    bestDist = dist;
                    bestIdx = j;
                }
            }
            // Use a threshold to filter poor matches.
            if (bestIdx != -1 && bestDist < 64) {
                DMatch match;
                match.queryIdx = i;
                match.trainIdx = bestIdx;
                match.distance = (float)bestDist;
                matches.push_back(match);
            }
        }
        return matches;
    }

    // Compute homography between two images based on their BRIEF results.
    Mat computeHomography(const BriefResult &br1, const BriefResult &br2, int idx) {
        vector<DMatch> matches = briefMatch(br1, br2);
        if (matches.size() < 4) {
            if (idx != -1) {
                cerr << "Not enough matches for homography between image " 
                     << idx << " and " << (idx + 1) << endl;
            } else {
                cerr << "Not enough matches for homography." << endl;
            }
            return Mat();
        }
    
        vector<Point2f> pts1, pts2;
        for (const auto &m : matches) {
            pts1.push_back(br1.keypoints[m.queryIdx].pt);
            pts2.push_back(br2.keypoints[m.trainIdx].pt);
        }
        return findHomography(pts1, pts2, RANSAC);
    }
    
    // Compute warped corners for an image given a homography.
    vector<Point2d> getWarpCorners(const Mat &image, const Mat &H) {
        vector<Point2d> corners;
        corners.push_back(Point2d(0, 0));
        corners.push_back(Point2d(image.cols, 0));
        corners.push_back(Point2d(0, image.rows));
        corners.push_back(Point2d(image.cols, image.rows));
        vector<Point2d> warped;
        perspectiveTransform(corners, warped, H);
        return warped;
    }

    // Return a translation matrix that shifts coordinates by (shiftX, shiftY).
    Mat getTranslationMatrix(double shiftX, double shiftY) {
        Mat T = Mat::eye(3, 3, CV_32F);
        T.at<float>(0, 2) = (float)shiftX;
        T.at<float>(1, 2) = (float)shiftY;
        return T;
    }

    // Stitch the images into a panorama.
    // For each image, use its corresponding homography to warp into the final panorama coordinate system.
    // The fusion here is a simple pixel-wise averaging in overlapping regions.
    void stitch(const vector<Mat> &images, const vector<Mat> &homographies,
                int panoWidth, int panoHeight, const std::string &outputFile) {
        Mat pano = Mat::zeros(panoHeight, panoWidth, CV_32FC3);
        Mat weight = Mat::zeros(panoHeight, panoWidth, CV_32FC1);
        for (size_t i = 0; i < images.size(); i++) {
            Mat warped;
            warpPerspective(images[i], warped, homographies[i], Size(panoWidth, panoHeight));
            Mat warped_f;
            if (warped.channels() == 1)
                cvtColor(warped, warped_f, COLOR_GRAY2BGR);
            else
                warped.convertTo(warped_f, CV_32FC3);
            // Build a mask: nonzero pixels are valid.
            Mat mask = Mat::zeros(warped.size(), CV_32FC1);
            for (int y = 0; y < warped.rows; y++) {
                for (int x = 0; x < warped.cols; x++) {
                    Vec3f pix = warped_f.at<Vec3f>(y, x);
                    if (pix[0] != 0 || pix[1] != 0 || pix[2] != 0)
                        mask.at<float>(y, x) = 1.0f;
                }
            }
            // Accumulate pixel values and weight.
            for (int y = 0; y < panoHeight; y++) {
                for (int x = 0; x < panoWidth; x++) {
                    float mVal = mask.at<float>(y, x);
                    if (mVal > 0) {
                        pano.at<Vec3f>(y, x) += warped_f.at<Vec3f>(y, x);
                        weight.at<float>(y, x) += 1.0f;
                    }
                }
            }
        }
        // Normalize to get average.
        for (int y = 0; y < panoHeight; y++) {
            for (int x = 0; x < panoWidth; x++) {
                float w = weight.at<float>(y, x);
                if (w > 0)
                    pano.at<Vec3f>(y, x) /= w;
            }
        }
        Mat pano8;
        pano.convertTo(pano8, CV_8UC3, 255.0);
        imwrite(outputFile, pano8);
        imshow("Panorama", pano8);
        waitKey(0);
    }

    // Return elapsed time (in seconds) since the given start clock.
    double get_time_elapsed(clock_t start) {
        return ((double)(clock() - start)) / CLOCKS_PER_SEC;
    }

    // Print timing details (placeholder implementation).
    void printTiming() {
        cout << "Timing details: (not implemented)" << endl;
    }
};

// --------------------------------------------------------------------------
// Main function: sequential panorama stitching.
// The process follows these steps:
//    1. Read images and convert to float.
//    2. Compute BRIEF descriptors (using test pattern).
//    3. Compute pairwise homographies (accumulate transform).
//    4. Calibrate using center image as reference.
//    5. Compute panorama bounds and adjust translation.
//    6. Stitch images into panorama.
extern const int num_images = 6;
int main(int argc, char** argv) {
    clock_t total_start = clock();
    clock_t IO_start = clock();

    Util util;

    // ----------------------------------------------------------
    // Phase 1: Load images and convert them to float.
    // ----------------------------------------------------------
    ImageReaderResult readerRes = readImagesFromArgs(argc, argv);
    vector<Mat> images = readerRes.images;
    string outputFile = readerRes.outputFile;
    
    if (images.size() < 2) {
        cerr << "Error: Need at least two valid images to perform stitching." << endl;
        return -1;
    }
    
    // Convert to float
    for (auto &img : images) {
        util.convertImg2Float(img);
    }
    
    double IO_elapsed = util.get_time_elapsed(IO_start);

    // ----------------------------------------------------------
    // Phase 2: Compute BRIEF descriptors.
    // ----------------------------------------------------------
    Point* compareA = nullptr;
    Point* compareB = nullptr;
    util.generateTestPattern(compareA, compareB);
    
    vector<BriefResult> brief_results;
    brief_results.reserve(num_images);
    for (int i = 0; i < num_images; i++) {
        BriefResult res = util.BriefLite(images[i], compareA, compareB);
        brief_results.push_back(res);
    }

    // ----------------------------------------------------------
    // Phase 3: Compute homographies between consecutive images.
    // Accumulate transformations relative to image 0.
    // ----------------------------------------------------------
    clock_t homography_start = clock();
    vector<Mat> homographies;
    homographies.reserve(num_images);
    Mat identity = Mat::eye(3, 3, CV_32F);
    homographies.push_back(identity);
    for (int i = 1; i < num_images; i++) {
        Mat H = util.computeHomography(brief_results[i-1], brief_results[i], i - 1);

        if (H.empty()) {
            cerr << "Error: Homography estimation failed between images " << i-1 << " and " << i << endl;
            return -1;
        }
        H = homographies[i-1] * H;
        homographies.push_back(H);
    }
    cout << "Computed homographies" << endl;

    // ----------------------------------------------------------
    // Phase 4: Center image calibration.
    // Use the center image as reference.
    // ----------------------------------------------------------
    int center_idx = (num_images - 1) / 2;
    Mat center_inv = homographies[center_idx].inv();
    for (int i = 0; i < num_images; i++) {
        homographies[i] = center_inv * homographies[i];
    }

    // ----------------------------------------------------------
    // Phase 5: Compute panorama bounds using warped corners.
    // ----------------------------------------------------------
    double xMin = numeric_limits<double>::max();
    double yMin = numeric_limits<double>::max();
    double xMax = numeric_limits<double>::lowest();
    double yMax = numeric_limits<double>::lowest();
    for (int i = 0; i < num_images; i++) {
        vector<Point2d> warpedCorners = util.getWarpCorners(images[i], homographies[i]);
        for (size_t j = 0; j < warpedCorners.size(); j++) {
            xMin = min(xMin, warpedCorners[j].x);
            xMax = max(xMax, warpedCorners[j].x);
            yMin = min(yMin, warpedCorners[j].y);
            yMax = max(yMax, warpedCorners[j].y);
        }
    }
    double shiftX = -xMin;
    double shiftY = -yMin;
    Mat transM = util.getTranslationMatrix(shiftX, shiftY);
    int panoWidth = cvRound(xMax - xMin);
    int panoHeight = cvRound(yMax - yMin);
    for (int i = 0; i < num_images; i++) {
        homographies[i] = transM * homographies[i];
        homographies[i] = homographies[i] / homographies[i].at<float>(2,2);
    }
    cout << "Adjusted panorama using center image reference" << endl;
    double compute_homography_elapsed = util.get_time_elapsed(homography_start);

    // ----------------------------------------------------------
    // Phase 6: Stitch images using warp and simple averaging fusion.
    // ----------------------------------------------------------
    util.stitch(images, homographies, panoWidth, panoHeight, outputFile);

    double total_time_elapsed = util.get_time_elapsed(total_start);
    printf("Total Time: %.2f seconds\n", total_time_elapsed);
    printf("IO Time: %.2f seconds\n", IO_elapsed);
    printf("Compute Homography Time: %.2f seconds\n", compute_homography_elapsed);
    util.printTiming();

    // Cleanup allocated test pattern arrays.
    if(compareA) delete[] compareA;
    if(compareB) delete[] compareB;

    return 0;
}
