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
        float score = scores[i];
        int classId = classes[i];

        if (score < preclusterThreshold[classId]) {
            continue;
        }

        float x1 = boxes[i * 4 + 0];
        float y1 = boxes[i * 4 + 1];
        float x2 = boxes[i * 4 + 2];
        float y2 = boxes[i * 4 + 3];

        NvDsInferObjectDetectionInfo detection;
        detection.classId = classId;
        detection.detectionConfidence = score;

        // Convert bbox coordinates from relative to absolute values
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

    if (outputLayersInfo.size() != 4) {
        std::cerr << "Expected 4 output layers, but got " << outputLayersInfo.size() << std::endl;
        return false;
    }

    const NvDsInferLayerInfo& numDetsLayer = outputLayersInfo[0]; // num_dets
    const NvDsInferLayerInfo& bboxesLayer = outputLayersInfo[1]; // bboxes
    const NvDsInferLayerInfo& scoresLayer = outputLayersInfo[2]; // scores
    const NvDsInferLayerInfo& labelsLayer = outputLayersInfo[3]; // labels

    int numDets = *static_cast<int*>(numDetsLayer.buffer);
    const float* boxes = static_cast<float*>(bboxesLayer.buffer);
    const float* scores = static_cast<float*>(scoresLayer.buffer);
    const int* classes = static_cast<int*>(labelsLayer.buffer);

    std::vector<NvDsInferObjectDetectionInfo> detections = decodeDetections(
        boxes, scores, classes, numDets, networkInfo, detectionParams.perClassPreclusterThreshold);

    objectList = nonMaximumSuppression(detections, 0.45);

    return true;
}

CHECK_CUSTOM_PARSE_FUNC_PROTOTYPE(NvDsInferParseCustomYoloV8);
