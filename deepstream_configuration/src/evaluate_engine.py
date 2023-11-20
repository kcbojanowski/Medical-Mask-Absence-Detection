from pathlib import Path
from prettytable import PrettyTable
import tensorrt as trt
import pycuda.driver as cuda
import pycuda.autoinit
import numpy as np
import cv2
import os
from typing import Tuple

class TensorRTEngine:
    def __init__(self, engine_path):
        self.engine_path = engine_path
        self.engine = self._load_engine(self.engine_path)
        self.context = self.engine.create_execution_context()
        self.stream = cuda.Stream()
        self.num_outputs = 3
        self.num_inputs = 1

    def _load_engine(self, engine_path):
        with open(engine_path, 'rb') as f, trt.Runtime(trt.Logger(trt.Logger.WARNING)) as runtime:
            return runtime.deserialize_cuda_engine(f.read())

    def _preprocess_image(self, image_path: str) -> np.ndarray:
        image = cv2.imread(image_path)
        image = cv2.resize(image, (640, 640))
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        image = image.transpose(2, 0, 1).astype(np.float32)  # Change to CHW format
        return np.ascontiguousarray(np.expand_dims(image, axis=0))  # Add batch dimension and ensure contiguous

    def infer(self, image_path: str):
        # Preprocess the image
        image = self._preprocess_image(image_path)

        # Allocate memory for inputs
        d_input = cuda.mem_alloc(1 * image.nbytes)
        bindings = [int(d_input)] + [None] * self.num_outputs

        # Allocate memory for outputs and setup bindings
        outputs = []
        for i in range(self.num_outputs):
            binding_idx = i + self.num_inputs
            binding_name = self.engine.get_tensor_name(binding_idx)
            dtype = trt.nptype(self.engine.get_tensor_dtype(binding_name))
            shape = tuple(self.context.get_tensor_shape(binding_name))
            size = trt.volume(shape) * np.dtype(dtype).itemsize

            if size <= 0:
                raise ValueError(f"Invalid memory size for output tensor: {size} bytes")

            d_output = cuda.mem_alloc(size)
            bindings[binding_idx] = int(d_output)
            outputs.append(np.empty(shape, dtype=dtype))

            # Debug: Print shape, dtype, and size
            print(f"Output Tensor: {binding_name}, Shape: {shape}, Dtype: {dtype}, Size: {size}")

        # Transfer input data to the GPU
        cuda.memcpy_htod(d_input, image)

        # Run inference using execute_v2
        self.context.execute_v2(bindings=bindings)

        # Transfer predictions back from the GPU using synchronous copy
        for i, output in enumerate(outputs):
            binding_idx = i + self.num_inputs

            # Corrected Memory Size Check
            allocated_size = output.nbytes
            expected_size = trt.volume(self.context.get_binding_shape(binding_idx)) * output.dtype.itemsize

            if allocated_size != expected_size:
                raise RuntimeError(f"Mismatch in output buffer size: Allocated={allocated_size}, Expected={expected_size}")

            cuda.memcpy_dtoh(output, bindings[binding_idx])

        # Extract bboxes, confs, and class_ids from outputs
        bboxes = outputs[0]
        confs = outputs[1]
        class_ids = outputs[2]

        return bboxes, confs, class_ids


# Define paths
DATASET_PATH = "../yolo-medical-mask-dataset/test/"
ENGINE_PATH = "../models/trt_engines/FaceMaskYolov8.engine"


def load_yolo_dataset(dataset_path):
    images = []
    labels = []
    for file in Path(dataset_path).glob("*.jpg"):
        images.append(str(file))
        label_file = file.with_suffix('.txt')
        labels.append(str(label_file))
    return images, labels


def parse_yolo_labels(label_path):
    ground_truth = []
    with open(label_path, 'r') as file:
        for line in file:
            class_id, x_center, y_center, width, height = map(float, line.split())
            ground_truth.append((class_id, x_center, y_center, width, height))
    return ground_truth


def preprocess_image(image_path, target_size=(640, 640)):
    image = cv2.imread(image_path)
    image = cv2.resize(image, target_size)
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    image = image.transpose(2, 0, 1).astype(np.float32)  # CHW format
    image = np.expand_dims(image, axis=0)  # Add batch dimension
    return image


def calculate_iou(box1, box2):
    """
    Calculate the Intersection over Union (IoU) of two bounding boxes.
    """

    x1, y1, w1, h1 = box1
    x2, y2, w2, h2 = box2

    x1_min, y1_min, x1_max, y1_max = x1 - w1 / 2, y1 - h1 / 2, x1 + w1 / 2, y1 + h1 / 2
    x2_min, y2_min, x2_max, y2_max = x2 - w2 / 2, y2 - h2 / 2, x2 + w2 / 2, y2 + h2 / 2

    # Calculate intersection area
    intersect_w = min(x1_max, x2_max) - max(x1_min, x2_min)
    intersect_h = min(y1_max, y2_max) - max(y1_min, y2_min)
    intersect_area = max(0, intersect_w) * max(0, intersect_h)

    # Calculate union area
    union_area = w1 * h1 + w2 * h2 - intersect_area

    # Compute the IoU
    iou = intersect_area / union_area if union_area != 0 else 0

    return iou


def calculate_accuracy_metrics(predictions, ground_truth, iou_threshold=0.5):
    """
    Calculate accuracy metrics like Precision, Recall, and mAP.
    """
    true_positives = 0
    detected = set()
    for pred_bbox in predictions:
        for idx, gt_bbox in enumerate(ground_truth):
            if idx in detected:
                continue
            iou = calculate_iou(pred_bbox, gt_bbox)
            if iou >= iou_threshold:
                true_positives += 1
                detected.add(idx)
                break

    precision = true_positives / len(predictions) if predictions else 0
    recall = true_positives / len(ground_truth) if ground_truth else 0

    # Simplified version for mAP, for demonstration purposes
    # In practice, mAP calculation is more involved
    mAP = (precision + recall) / 2 if precision and recall else 0

    return mAP, precision, recall


def main():
    engine = TensorRTEngine(ENGINE_PATH)
    images, labels = load_yolo_dataset(DATASET_PATH)

    # Prepare a table to display results for each image
    table = PrettyTable()
    table.field_names = ["Image", "mAP", "Precision", "Recall"]

    for image_path, label_path in zip(images, labels):
        ground_truth = parse_yolo_labels(label_path)

        # Run inference
        bboxes, scores, labels = engine.infer(image_path)

        # Convert bboxes from output format to (x, y, width, height)
        # Adjust this conversion based on your model's specific output format
        processed_bboxes = [(x, y, w, h) for x, y, w, h in bboxes]

        # Calculate accuracy metrics
        mAP, precision, recall = calculate_accuracy_metrics(processed_bboxes, ground_truth)

        # Add the results to the table
        table.add_row([os.path.basename(image_path), f"{mAP:.2f}", f"{precision:.2f}", f"{recall:.2f}"])

    print(table)


if __name__ == "__main__":
    main()
