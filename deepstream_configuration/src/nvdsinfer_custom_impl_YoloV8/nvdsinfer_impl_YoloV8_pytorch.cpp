#include <vector>
#include <iostream>
#include <algorithm>
#include "nvdsinfer_custom_impl.h"
#include <numeric>

static float IoU(const NvDsInferObjectDetectionInfo& a, const NvDsInferObjectDetectionInfo& b) {
    float areaA = a.width * a.height;
    float areaB = b.width * b.height;

    if (areaA <= 0 || areaB <= 0) return 0;

    float minX = std::max(a.left, b.left);
    float minY = std::max(a.top, b.top);
    float maxX = std::min(a.left + a.width, b.left + b.width);
    float maxY = std::min(a.top + a.height, b.top + b.height);

    float intersection = std::max(maxX - minX, 0.0f) * std::max(maxY - minY, 0.0f);

    return intersection / (areaA + areaB - intersection);
}

static std::vector<NvDsInferObjectDetectionInfo>
nonMaximumSuppression(const std::vector<NvDsInferObjectDetectionInfo>& detections, float iouThreshold) {
    std::vector<NvDsInferObjectDetectionInfo> outputDetections;

    std::vector<size_t> idxs(detections.size());
    std::iota(idxs.begin(), idxs.end(), 0);

    std::sort(idxs.begin(), idxs.end(), [&detections](size_t i1, size_t i2) {
        return detections[i1].detectionConfidence > detections[i2].detectionConfidence;
    });

    while (!idxs.empty()) {
        size_t top = idxs.front();
        outputDetections.push_back(detections[top]);

        idxs.erase(idxs.begin());
        idxs.erase(std::remove_if(idxs.begin(), idxs.end(), [&detections, &top, &iouThreshold](size_t i) {
            return IoU(detections[top], detections[i]) > iouThreshold;
        }), idxs.end());
    }

    return outputDetections;
}

static std::vector<NvDsInferObjectDetectionInfo>
decodeDetections(const float* detOutput, int numDets,
                 const NvDsInferNetworkInfo& networkInfo, float confThreshold) {
    std::vector<NvDsInferObjectDetectionInfo> detections;

    for (int i = 0; i < numDets; ++i) {
        const float* det = detOutput + i * 7; // 7 values per detection

        float x = det[0];
        float y = det[1];
        float w = det[2];
        float h = det[3];
        // Determine the class with the highest score
        int classId = std::max_element(det + 4, det + 7) - (det + 4);
        float score = det[4 + classId];

        if (score < confThreshold) {
            continue;
        }

        NvDsInferObjectDetectionInfo detection;
        detection.classId = classId;
        detection.detectionConfidence = score;
        detection.left = x * networkInfo.width;
        detection.top = y * networkInfo.height;
        detection.width = w * networkInfo.width;
        detection.height = h * networkInfo.height;

        detections.push_back(detection);
    }

    return detections;
}

extern "C" bool NvDsInferParseCustomYoloV8(
    std::vector<NvDsInferLayerInfo> const& outputLayersInfo,
    NvDsInferNetworkInfo const& networkInfo,
    NvDsInferParseDetectionParams const& detectionParams,
    std::vector<NvDsInferObjectDetectionInfo>& objectList) {

    if (outputLayersInfo.size() != 1 || std::string(outputLayersInfo[0].layerName) != "897") {
        std::cerr << "Expected 1 output layer named '897', but got different configuration" << std::endl;
        return false;
    }

    const NvDsInferLayerInfo& detLayer = outputLayersInfo[0]; // Detections
    if (detLayer.buffer == nullptr) {
        std::cerr << "Detection layer buffer is null" << std::endl;
        return false;
    }

    int numDets = 8400; // Number of detections
    const float* detOutput = static_cast<float*>(detLayer.buffer);

    float confThreshold = 0.3; // Set an appropriate confidence threshold
    auto detections = decodeDetections(detOutput, numDets, networkInfo, confThreshold);
    objectList = nonMaximumSuppression(detections, 0.45);

    return true;
}

CHECK_CUSTOM_PARSE_FUNC_PROTOTYPE(NvDsInferParseCustomYoloV8);