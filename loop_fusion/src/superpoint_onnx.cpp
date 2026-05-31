/*******************************************************
 * SuperPoint ONNX Inference for Loop Closure
 *******************************************************/

#include "superpoint_onnx.h"
#include <algorithm>
#include <numeric>
#include <cmath>

SuperPointONNX::SuperPointONNX(const std::string& model_path,
                               bool use_cuda,
                               int max_keypoints,
                               int input_size)
    : env_(ORT_LOGGING_LEVEL_WARNING, "superpoint"),
      session_(nullptr),
      max_keypoints_(max_keypoints),
      input_size_(input_size)
{
    Ort::SessionOptions session_options;
    session_options.SetIntraOpNumThreads(2);
    session_options.SetGraphOptimizationLevel(GraphOptimizationLevel::ORT_ENABLE_ALL);

    if (use_cuda) {
        try {
            OrtCUDAProviderOptions cuda_options;
            cuda_options.device_id = 0;
            cuda_options.arena_extend_strategy = 0;
            cuda_options.gpu_mem_limit = 2ULL * 1024 * 1024 * 1024;  // 2GB limit
            cuda_options.cudnn_conv_algo_search = OrtCudnnConvAlgoSearchExhaustive;
            session_options.AppendExecutionProvider_CUDA(cuda_options);
            printf("[SuperPointONNX] Using CUDA execution provider\n");
        } catch (const Ort::Exception& e) {
            printf("[SuperPointONNX] CUDA not available (%s), falling back to CPU\n", e.what());
        }
    }

    session_ = Ort::Session(env_, model_path.c_str(), session_options);
    printf("[SuperPointONNX] Model loaded: %s (inputs=%zu, outputs=%zu)\n",
           model_path.c_str(),
           session_.GetInputCount(),
           session_.GetOutputCount());
}

SuperPointFeatures SuperPointONNX::extract(const cv::Mat& image)
{
    SuperPointFeatures result;

    // Resize image (keep aspect ratio, pad to multiple of 8)
    cv::Mat gray;
    if (image.channels() == 3)
        cv::cvtColor(image, gray, cv::COLOR_BGR2GRAY);
    else
        gray = image;

    int orig_h = gray.rows;
    int orig_w = gray.cols;

    // Resize longest edge to input_size_
    float scale = static_cast<float>(input_size_) / std::max(orig_h, orig_w);
    int new_h = static_cast<int>(orig_h * scale);
    int new_w = static_cast<int>(orig_w * scale);
    // Ensure divisible by 8
    new_h = (new_h / 8) * 8;
    new_w = (new_w / 8) * 8;
    if (new_h == 0) new_h = 8;
    if (new_w == 0) new_w = 8;

    cv::Mat resized;
    cv::resize(gray, resized, cv::Size(new_w, new_h), 0, 0, cv::INTER_AREA);

    // Normalize to [0, 1] float32
    cv::Mat input_float;
    resized.convertTo(input_float, CV_32F, 1.0 / 255.0);

    // Create input tensor (1, 1, H, W)
    std::vector<int64_t> input_shape = {1, 1, new_h, new_w};
    size_t input_size = new_h * new_w;
    std::vector<float> input_data(input_size);
    std::memcpy(input_data.data(), input_float.data, input_size * sizeof(float));

    Ort::MemoryInfo mem_info = Ort::MemoryInfo::CreateCpu(OrtArenaAllocator, OrtMemTypeDefault);
    Ort::Value input_tensor = Ort::Value::CreateTensor<float>(
        mem_info, input_data.data(), input_size, input_shape.data(), input_shape.size());

    // Run inference
    const char* input_names[] = {"image"};
    const char* output_names[] = {"scores", "descriptors"};

    auto output_tensors = session_.Run(
        Ort::RunOptions{nullptr},
        input_names, &input_tensor, 1,
        output_names, 2);

    // Parse outputs
    // scores: (1, 1, H, W)
    float* scores_data = output_tensors[0].GetTensorMutableData<float>();
    auto scores_shape = output_tensors[0].GetTensorTypeAndShapeInfo().GetShape();
    int score_h = scores_shape[2];
    int score_w = scores_shape[3];

    // descriptors: (1, 256, H/8, W/8)
    float* desc_data = output_tensors[1].GetTensorMutableData<float>();
    auto desc_shape = output_tensors[1].GetTensorTypeAndShapeInfo().GetShape();
    int desc_h = desc_shape[2];
    int desc_w = desc_shape[3];

    // Build score map for NMS
    cv::Mat score_map(score_h, score_w, CV_32F, scores_data);

    // NMS + threshold + top-K
    std::vector<cv::KeyPoint> keypoints;
    std::vector<float> scores;
    nms(score_map, nms_radius_, keypoints, scores);

    if (keypoints.empty()) {
        return result;
    }

    // Sample descriptors at keypoint locations
    cv::Mat descriptors = sampleDescriptors(
        desc_data, desc_h, desc_w, keypoints, score_h, score_w);

    // Scale keypoints back to original image coordinates
    float scale_x = static_cast<float>(orig_w) / new_w;
    float scale_y = static_cast<float>(orig_h) / new_h;
    for (auto& kp : keypoints) {
        kp.pt.x *= scale_x;
        kp.pt.y *= scale_y;
    }

    result.keypoints = std::move(keypoints);
    result.descriptors = std::move(descriptors);
    result.scores = std::move(scores);
    return result;
}

void SuperPointONNX::nms(const cv::Mat& score_map, int radius,
                         std::vector<cv::KeyPoint>& keypoints,
                         std::vector<float>& scores)
{
    int h = score_map.rows;
    int w = score_map.cols;

    // Max pooling for NMS
    cv::Mat max_pool;
    int kernel = 2 * radius + 1;
    cv::dilate(score_map, max_pool, cv::getStructuringElement(
        cv::MORPH_RECT, cv::Size(kernel, kernel)));

    // Keep only local maxima above threshold
    cv::Mat is_max = (score_map == max_pool) & (score_map > score_threshold_);

    // Collect candidate keypoints
    struct Candidate {
        float score;
        int x, y;
    };
    std::vector<Candidate> candidates;
    candidates.reserve(2000);

    for (int y = radius; y < h - radius; y++) {
        const uchar* mask_row = is_max.ptr<uchar>(y);
        const float* score_row = score_map.ptr<float>(y);
        for (int x = radius; x < w - radius; x++) {
            if (mask_row[x]) {
                candidates.push_back({score_row[x], x, y});
            }
        }
    }

    // Sort by score descending
    std::sort(candidates.begin(), candidates.end(),
              [](const Candidate& a, const Candidate& b) { return a.score > b.score; });

    // Top-K
    int n = std::min(static_cast<int>(candidates.size()), max_keypoints_);
    keypoints.reserve(n);
    scores.reserve(n);

    for (int i = 0; i < n; i++) {
        keypoints.emplace_back(cv::Point2f(candidates[i].x, candidates[i].y),
                               1.0f, -1.0f, candidates[i].score);
        scores.push_back(candidates[i].score);
    }
}

cv::Mat SuperPointONNX::sampleDescriptors(const float* desc_data, int desc_h, int desc_w,
                                          const std::vector<cv::KeyPoint>& keypoints,
                                          int img_h, int img_w)
{
    // desc_data layout: (1, 256, desc_h, desc_w) = (256, desc_h, desc_w) for batch=1
    int n = keypoints.size();
    cv::Mat descriptors(n, 256, CV_32F);

    for (int i = 0; i < n; i++) {
        // Map keypoint from image coords to descriptor map coords
        float dx = keypoints[i].pt.x * desc_w / img_w;
        float dy = keypoints[i].pt.y * desc_h / img_h;

        // Bilinear interpolation
        int x0 = std::max(0, std::min(static_cast<int>(std::floor(dx)), desc_w - 1));
        int y0 = std::max(0, std::min(static_cast<int>(std::floor(dy)), desc_h - 1));
        int x1 = std::min(x0 + 1, desc_w - 1);
        int y1 = std::min(y0 + 1, desc_h - 1);

        float wx = dx - x0;
        float wy = dy - y0;

        float* desc_out = descriptors.ptr<float>(i);
        for (int c = 0; c < 256; c++) {
            // Layout: (256, desc_h, desc_w) — channel-first
            int offset_base = c * desc_h * desc_w;
            float v00 = desc_data[offset_base + y0 * desc_w + x0];
            float v01 = desc_data[offset_base + y0 * desc_w + x1];
            float v10 = desc_data[offset_base + y1 * desc_w + x0];
            float v11 = desc_data[offset_base + y1 * desc_w + x1];

            desc_out[c] = (1 - wy) * ((1 - wx) * v00 + wx * v01)
                        + wy * ((1 - wx) * v10 + wx * v11);
        }

        // L2 normalize the sampled descriptor
        float norm = 0;
        for (int c = 0; c < 256; c++)
            norm += desc_out[c] * desc_out[c];
        norm = std::sqrt(norm) + 1e-8f;
        for (int c = 0; c < 256; c++)
            desc_out[c] /= norm;
    }

    return descriptors;
}

std::vector<cv::DMatch> SuperPointONNX::matchDescriptors(
    const cv::Mat& desc0,
    const cv::Mat& desc1,
    float ratio_threshold)
{
    if (desc0.empty() || desc1.empty())
        return {};

    // Use BFMatcher with L2 norm and cross-check (mutual nearest neighbor)
    if (ratio_threshold <= 0 || ratio_threshold >= 1.0f) {
        // Simple mutual NN (cross-check)
        cv::BFMatcher matcher(cv::NORM_L2, true);  // crossCheck=true
        std::vector<cv::DMatch> matches;
        matcher.match(desc0, desc1, matches);
        return matches;
    }

    // Lowe's ratio test
    cv::BFMatcher matcher(cv::NORM_L2, false);
    std::vector<std::vector<cv::DMatch>> knn_matches;
    matcher.knnMatch(desc0, desc1, knn_matches, 2);

    std::vector<cv::DMatch> good_matches;
    good_matches.reserve(knn_matches.size());
    for (const auto& m : knn_matches) {
        if (m.size() == 2 && m[0].distance < ratio_threshold * m[1].distance) {
            good_matches.push_back(m[0]);
        }
    }

    // Cross-check: also verify reverse matches
    std::vector<std::vector<cv::DMatch>> knn_reverse;
    matcher.knnMatch(desc1, desc0, knn_reverse, 2);

    std::vector<bool> reverse_ok(desc1.rows, false);
    std::vector<int> reverse_match(desc1.rows, -1);
    for (const auto& m : knn_reverse) {
        if (m.size() == 2 && m[0].distance < ratio_threshold * m[1].distance) {
            reverse_ok[m[0].queryIdx] = true;
            reverse_match[m[0].queryIdx] = m[0].trainIdx;
        }
    }

    // Keep only mutual matches
    std::vector<cv::DMatch> mutual_matches;
    for (const auto& m : good_matches) {
        if (reverse_ok[m.trainIdx] && reverse_match[m.trainIdx] == m.queryIdx) {
            mutual_matches.push_back(m);
        }
    }

    return mutual_matches;
}
