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
                 const NvDsInferNetworkInfo& networkInfo, float confThreshold) {
    std::vector<NvDsInferObjectDetectionInfo> detections;

    for (int i = 0; i < numDets; ++i) {
        float score = scores[i];
        if (score < confThreshold) {
            continue;
        }

        int classId = classes[i];

        // Convert from [x_center, y_center, width, height] to [left, top, right, bottom]
        float x_center = boxes[i * 4];
        float y_center = boxes[i * 4 + 1];
        float w = boxes[i * 4 + 2];
        float h = boxes[i * 4 + 3];
        float left = x_center - (w / 2);
        float top = y_center - (h / 2);

        NvDsInferObjectDetectionInfo detection;
        detection.classId = classId;
        detection.detectionConfidence = score;
        detection.left = left;
        detection.top = top;
        detection.width = w;
        detection.height = h;

        detections.push_back(detection);

//        std::cout << "Processed Detection: ClassID = " << detection.classId
//                  << ", Confidence = " << detection.detectionConfidence
//                  << ", Left = " << detection.left << ", Top = " << detection.top
//                  << ", Width = " << detection.width << ", Height = " << detection.height << std::endl;
    }
    return detections;
}




extern "C" bool NvDsInferParseCustomYoloV8(
    std::vector<NvDsInferLayerInfo> const& outputLayersInfo,
    NvDsInferNetworkInfo const& networkInfo,
    NvDsInferParseDetectionParams const& detectionParams,
    std::vector<NvDsInferObjectDetectionInfo>& objectList) {

    std::cout << "Number of output layers: " << outputLayersInfo.size() << std::endl;

    for (size_t i = 0; i < outputLayersInfo.size(); ++i) {
        const auto& layer = outputLayersInfo[i];
        std::cout << "Layer Index: " << i
                  << ", Layer Name: " << layer.layerName
                  << ", Layer Type: " << layer.dataType
                  << ", Layer Dims: " << layer.inferDims.numDims << std::endl;

        // Optionally, print dimensions details
        for (int j = 0; j < layer.inferDims.numDims; ++j) {
            std::cout << "Dim[" << j << "]: " << layer.inferDims.d[j] << std::endl;
        }
    }

    for (const auto& layer : outputLayersInfo) {
        if (layer.buffer == nullptr) {
            std::cerr << "One of the layer buffers is null" << std::endl;
            return false;
        }
    }

    const NvDsInferLayerInfo& numDetsLayer = outputLayersInfo[0]; // num_dets
    int numDets = *static_cast<int*>(numDetsLayer.buffer);

    if (numDets < 0) {
        std::cerr << "Number of detections is negative: " << numDets << std::endl;
        return false;
    }

    const float* boxes = static_cast<float*>(outputLayersInfo[1].buffer);
    const float* scores = static_cast<float*>(outputLayersInfo[2].buffer);
    const int* classes = static_cast<int*>(outputLayersInfo[3].buffer);

    std::cout << "scores: " << scores;
    std::cout << "classes: " << classes;

    float globalThreshold = 0.6;
    auto detections = decodeDetections(boxes, scores, classes, numDets, networkInfo, globalThreshold);
    objectList = nonMaximumSuppression(detections, 0.4);

    return true;
}

CHECK_CUSTOM_PARSE_FUNC_PROTOTYPE(NvDsInferParseCustomYoloV8);
