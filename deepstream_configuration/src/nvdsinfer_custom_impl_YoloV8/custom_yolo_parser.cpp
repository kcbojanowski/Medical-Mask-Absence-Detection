#include <cstring>
#include <iostream>
#include "nvdsinfer_custom_impl.h"
#include <cassert>

#define MIN(a,b) ((a) < (b) ? (a) : (b))
#define MAX(a,b) ((a) > (b) ? (a) : (b))
#define CLIP(a,min,max) (MAX(MIN(a, max), min))

extern "C"
bool NvDsInferParseModel (
         std::vector<NvDsInferLayerInfo> const &outputLayersInfo,
         NvDsInferNetworkInfo  const &networkInfo,
         NvDsInferParseDetectionParams const &detectionParams,
         std::vector<NvDsInferObjectDetectionInfo> &objectList);

extern "C"
bool NvDsInferParseModel (std::vector<NvDsInferLayerInfo> const &outputLayersInfo,
                                   NvDsInferNetworkInfo  const &networkInfo,
                                   NvDsInferParseDetectionParams const &detectionParams,
                                   std::vector<NvDsInferObjectDetectionInfo> &objectList) {

    std::cout << "Starting custom parser for face mask detection." << std::endl;
    auto layerFinder = [&outputLayersInfo](const std::string &name)
        -> const NvDsInferLayerInfo *{

        for (auto &layer : outputLayersInfo) {
            if (
              (layer.layerName && name == layer.layerName)) {
                return &layer;
            }
        }
        return nullptr;
    };

    const NvDsInferLayerInfo *scoreLayer = layerFinder("conf");
    const NvDsInferLayerInfo *classLayer = layerFinder("class_id");
    const NvDsInferLayerInfo *boxLayer = layerFinder("bbox");

    if (!scoreLayer || !classLayer || !boxLayer) {
        std::cerr << "ERROR: Some layers missing or unsupported data types in output tensors." << std::endl;
        return false;
    } else {
        std::cout << "Found all required layers: bbox, conf, class_id." << std::endl;
}


    unsigned int topK_elem_num = scoreLayer->inferDims.numElements;
    float* p_bboxes = (float*) boxLayer->buffer;
    float* p_scores = (float*) scoreLayer->buffer;
    unsigned int* p_classes = (unsigned int*) classLayer->buffer;

    const int out_class_size = detectionParams.numClassesConfigured;
    const float threshold = detectionParams.perClassThreshold[0];
    int p_keep_count = topK_elem_num;
    for (int i = 0; i < p_keep_count; i++) {
        if (p_scores[i] < threshold) continue;

        assert((int)p_classes[i] < out_class_size);

        NvDsInferObjectDetectionInfo object;
        object.classId = (int)p_classes[i];
        object.detectionConfidence = p_scores[i];
        object.left = CLIP((p_bboxes[4*i+0] - p_bboxes[4*i+2] / 2), 0, networkInfo.width - 1);
        object.top = CLIP((p_bboxes[4*i+1] - p_bboxes[4*i+3] / 2) , 0, networkInfo.height - 1);
        object.width = CLIP(p_bboxes[4*i+2], 0, networkInfo.width - 1);
        object.height = CLIP(p_bboxes[4*i+3], 0, networkInfo.height - 1);

        objectList.push_back(object);
    }
    return true;
}
/* Check that the custom function has been defined correctly */
CHECK_CUSTOM_PARSE_FUNC_PROTOTYPE(NvDsInferParseModel);