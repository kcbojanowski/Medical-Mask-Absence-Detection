# Configuration Directory README

This directory contains the configuration files for the DeepStream application. Below is a detailed explanation of the various properties used in the configuration file.

## Configuration Properties Overview

### Mandatory Properties
When engine files are not specified, certain properties are mandatory, depending on the model type:
- **ONNX Models:** `onnx-file` (if using an ONNX model directly)

For detectors, the following properties are mandatory:
- `model-engine-file`: Path to the TensorRT engine file.
- `num-detected-classes`: Number of classes that the model can detect.
- `output-blob-names`: Names of the output blobs.

### Optional Properties for Detectors
- `custom-lib-path`: Path to the custom library for model parsing.
- `parse-bbox-func-name`: Function name to parse bounding box information.
- `interval`: Interval for inference (default: `0`, i.e., every frame).
- `topk`: Maximum number of bounding boxes to keep per image after NMS.
- `nms-iou-threshold`: IOU threshold for non-maximum suppression.
- `pre-cluster-threshold`: Minimum score threshold for a box to be considered during NMS.

### Recommended Properties
- `batch-size`: Batch size for inference (default: `1`).
- `gpu-id`: Specifies the GPU device ID (default: `0`).

### Other Optional Properties
- `net-scale-factor`: Scaling factor for network input data normalization (default: `1`).
- `network-mode`: Precision mode (default: `0` i.e., FP32).
- `model-color-format`: Color format of the model (default: `0` i.e., RGB).
- `symmetric-padding`: Enables/disables symmetric padding (default: `0`).
- `network-input-order`: Defines order of input channels (default: `0`).
- `maintain-aspect-ratio`: Maintain aspect ratio during preprocessing (default: `0`).
- `process-mode`: Defines preprocessing mode (default: `1` i.e., primary).

**Note:** Values in the config file are overridden by values set through GObject properties.
