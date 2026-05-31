/*******************************************************
 * SuperPoint ONNX Inference for Loop Closure
 *
 * Runs SuperPoint feature extraction via ONNX Runtime (GPU/CPU),
 * then performs mutual nearest-neighbor matching with ratio test.
 * Replaces BRIEF-based matching in fisheye loop closure pipeline.
 *******************************************************/

#pragma once

#include <string>
#include <vector>
#include <memory>
#include <opencv2/opencv.hpp>
#include <onnxruntime_cxx_api.h>

struct SuperPointFeatures {
    std::vector<cv::KeyPoint> keypoints;   // detected keypoints in original image coords
    cv::Mat descriptors;                    // (N, 256) float32 descriptors
    std::vector<float> scores;             // detection scores
};

class SuperPointONNX {
public:
    /**
     * @param model_path Path to superpoint.onnx
     * @param use_cuda Whether to use CUDA execution provider
     * @param max_keypoints Maximum number of keypoints to extract
     * @param input_size Resize longest edge to this before inference
     */
    SuperPointONNX(const std::string& model_path,
                   bool use_cuda = true,
                   int max_keypoints = 1024,
                   int input_size = 480);

    ~SuperPointONNX() = default;

    /**
     * Extract SuperPoint features from a grayscale image.
     * @param image Grayscale CV_8UC1 image (any size)
     * @return Detected keypoints with descriptors
     */
    SuperPointFeatures extract(const cv::Mat& image);

    /**
     * Match two sets of SuperPoint descriptors using mutual nearest neighbor.
     * @param desc0 (N, 256) descriptors from image 0
     * @param desc1 (M, 256) descriptors from image 1
     * @param ratio_threshold Lowe's ratio test threshold (0 = disabled)
     * @return Vector of matches (queryIdx into desc0, trainIdx into desc1)
     */
    static std::vector<cv::DMatch> matchDescriptors(
        const cv::Mat& desc0,
        const cv::Mat& desc1,
        float ratio_threshold = 0.9f);

private:
    // NMS on score map
    void nms(const cv::Mat& score_map, int radius,
             std::vector<cv::KeyPoint>& keypoints,
             std::vector<float>& scores);

    // Sample descriptors at keypoint locations via bilinear interpolation
    cv::Mat sampleDescriptors(const float* desc_data, int desc_h, int desc_w,
                              const std::vector<cv::KeyPoint>& keypoints,
                              int img_h, int img_w);

    Ort::Env env_;
    Ort::Session session_;
    Ort::AllocatorWithDefaultOptions allocator_;

    int max_keypoints_;
    int input_size_;
    int nms_radius_ = 4;
    float score_threshold_ = 0.005f;
};
