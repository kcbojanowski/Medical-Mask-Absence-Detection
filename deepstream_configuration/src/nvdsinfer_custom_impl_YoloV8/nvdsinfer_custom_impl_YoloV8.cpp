#include <vector>
#include <iostream>
#include <algorithm>
#include "nvdsinfer_custom_impl.h"
#include <numeric>
#include <vector>
#include <algorithm>

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
decodeDetections(const float* boxes, const float* scores, const int* classes, int numDets,
                 const NvDsInferNetworkInfo& networkInfo, const std::vector<float>& preclusterThreshold) {
    std::vector<NvDsInferObjectDetectionInfo> detections;

    for (int i = 0; i < numDets; ++i) {
        std::cout << "Processing detection " << i << std::endl;

        float score = scores[i];
        int classId = classes[i];
        std::cout << "Detection " << i << ": score=" << score << ", classId=" << classId << std::endl;

        std::cout << "preclusterThreshold size: " << preclusterThreshold.size() << std::endl;

        // Check if classId is valid
        if (classId < 0 || classId >= preclusterThreshold.size()) {
            std::cerr << "Invalid classId: " << classId << std::endl;
            continue;
        }

        if (score < preclusterThreshold[classId]) {
            continue;
        }

        // Ensure index is within bounds
        if (i * 4 + 3 >= 12) {
            std::cerr << "Index out of bounds for boxes array" << std::endl;
            break;
        }

        float x1 = boxes[i * 4 + 0];
        float y1 = boxes[i * 4 + 1];
        float x2 = boxes[i * 4 + 2];
        float y2 = boxes[i * 4 + 3];
        std::cout << "Coordinates: (" << x1 << ", " << y1 << ", " << x2 << ", " << y2 << ")" << std::endl;

        NvDsInferObjectDetectionInfo detection;
        detection.classId = classId;
        detection.detectionConfidence = score;
        detection.left = x1 * networkInfo.width;
        detection.top = y1 * networkInfo.height;
        detection.width = (x2 - x1) * networkInfo.width;
        detection.height = (y2 - y1) * networkInfo.height;

        detections.push_back(detection);
    }

    return detections;
}




extern "C" bool NvDsInferParseCustomYoloV8(
    std::vector<NvDsInferLayerInfo> const& outputLayersInfo,
    NvDsInferNetworkInfo const& networkInfo,
    NvDsInferParseDetectionParams const& detectionParams,
    std::vector<NvDsInferObjectDetectionInfo>& objectList) {

    std::cout << "preclusterThreshold size in NvDsInferParseCustomYoloV8: "
          << detectionParams.perClassPreclusterThreshold.size() << std::endl;

    // Validate the number of output layers
    if (outputLayersInfo.size() != 4) {
        std::cerr << "Expected 4 output layers, but got " << outputLayersInfo.size() << std::endl;
        return false;
    }

    // Validate each layer's buffer
    for (const auto& layer : outputLayersInfo) {
        if (layer.buffer == nullptr) {
            std::cerr << "One of the layer buffers is null" << std::endl;
            return false;
        }
    }

    // Extracting layer information
    const NvDsInferLayerInfo& numDetsLayer = outputLayersInfo[0]; // num_dets
    const NvDsInferLayerInfo& bboxesLayer = outputLayersInfo[1]; // bboxes
    const NvDsInferLayerInfo& scoresLayer = outputLayersInfo[2]; // scores
    const NvDsInferLayerInfo& labelsLayer = outputLayersInfo[3]; // labels

    // Safely cast buffer to the expected type
    int numDets = *static_cast<int*>(numDetsLayer.buffer);
    std::cout << "Number of detections: " << numDets << std::endl;

    // Validate numDets
    if (numDets < 0) {
        std::cerr << "Number of detections is negative: " << numDets << std::endl;
        return false;
    }

    const float* boxes = static_cast<float*>(bboxesLayer.buffer);
    const float* scores = static_cast<float*>(scoresLayer.buffer);
    const int* classes = static_cast<int*>(labelsLayer.buffer);

    // Decode detections
    auto detections = decodeDetections(
        boxes, scores, classes, numDets, networkInfo, detectionParams.perClassPreclusterThreshold);

    // Apply non-maximum suppression
    objectList = nonMaximumSuppression(detections, 0.45);

    return true;
}

CHECK_CUSTOM_PARSE_FUNC_PROTOTYPE(NvDsInferParseCustomYoloV8);
