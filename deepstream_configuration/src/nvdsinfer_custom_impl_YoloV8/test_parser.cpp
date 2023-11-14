#include <iostream>
#include <vector>
#include "nvdsinfer_custom_impl.h"


constexpr int MAX_DETS = 3;
float bboxes[MAX_DETS * 4] = {
    0.1f, 0.2f, 0.4f, 0.5f, // First bbox
    0.3f, 0.3f, 0.6f, 0.6f, // Second bbox
    0.5f, 0.1f, 0.7f, 0.4f  // Third bbox
};
float scores[MAX_DETS] = {0.9f, 0.8f, 0.85f};
int classes[MAX_DETS] = {0, 1, 2};

// Declaration of custom parser function
extern "C" bool NvDsInferParseCustomYoloV8(
    std::vector<NvDsInferLayerInfo> const& outputLayersInfo,
    NvDsInferNetworkInfo const& networkInfo,
    NvDsInferParseDetectionParams const& detectionParams,
    std::vector<NvDsInferObjectDetectionInfo>& objectList);

void testParser() {
    std::cout << "Starting testParser" << std::endl;

    // Create layer info structures pointing to the test data
    NvDsInferLayerInfo numDetsLayer;
    numDetsLayer.buffer = const_cast<void*>(reinterpret_cast<const void*>(&MAX_DETS));

    NvDsInferLayerInfo bboxesLayer;
    bboxesLayer.buffer = bboxes;

    NvDsInferLayerInfo scoresLayer;
    scoresLayer.buffer = scores;

    NvDsInferLayerInfo classesLayer;
    classesLayer.buffer = classes;

    std::vector<NvDsInferLayerInfo> outputLayersInfo = {numDetsLayer, bboxesLayer, scoresLayer, classesLayer};

    NvDsInferNetworkInfo networkInfo = {640, 640, 3}; // Set appropriate network info
    NvDsInferParseDetectionParams detectionParams;
    detectionParams.perClassPreclusterThreshold = std::vector<float>(3, 0.3f); // Initialize with a default value for each class

    std::vector<NvDsInferObjectDetectionInfo> objectList;

    std::cout << "Calling NvDsInferParseCustomYoloV8" << std::endl;
    bool status = NvDsInferParseCustomYoloV8(outputLayersInfo, networkInfo, detectionParams, objectList);

    std::cout << "NvDsInferParseCustomYoloV8 returned " << status << std::endl;

    if (status) {
        std::cout << "Parser succeeded. Detected objects:" << std::endl;
        for (const auto& obj : objectList) {
            std::cout << "Class ID: " << obj.classId << ", Confidence: " << obj.detectionConfidence
                      << ", bbox(x, y, width, height): (" << obj.left << ", " << obj.top << ", "
                      << obj.width << ", " << obj.height << ")" << std::endl;
        }
    } else {
        std::cerr << "Parser failed." << std::endl;
    }
}

int main() {
    testParser();
    return 0;
}
