import os
import cv2
import warnings
from collections import namedtuple
from pathlib import Path
from typing import List, Optional, Tuple, Union
from prettytable import PrettyTable
import numpy as np
import pycuda.autoinit  # noqa F401
import pycuda.driver as cuda
import tensorrt as trt
from numpy import ndarray

os.environ['CUDA_MODULE_LOADING'] = 'LAZY'
warnings.filterwarnings(action='ignore', category=DeprecationWarning)

DATASET_PATH = "../yolo-medical-mask-dataset/test/"
ENGINE_PATH = "../models/trt_engines/FaceMaskYolov8.engine"


class TRTEngine:
    def __init__(self, engine_path: Union[str, Path]) -> None:
        self.engine_path = Path(engine_path) if isinstance(engine_path, str) else engine_path
        self.stream = cuda.Stream(0)
        self.__init_engine()
        self.bindings = [0] * (self.num_inputs + self.num_outputs)
        self.__init_bindings()
        self.__warm_up()


    def preprocess(self, image):
        # Resize and reformat the image
        image = cv2.resize(image, (640, 640))
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        image = image.transpose(2, 0, 1).astype(np.float32)  # Change to CHW format
        return np.expand_dims(image, axis=0)  # Add batch dimension

    def infer(self, image):
        preprocessed_image = self.preprocess(image)
        bboxes, confs, class_ids = self.__call__(preprocessed_image)
        # Further processing can be done here
        return bboxes, confs, class_ids

    def __init_engine(self) -> None:
        logger = trt.Logger(trt.Logger.WARNING)
        trt.init_libnvinfer_plugins(logger, namespace='')
        with open(self.engine_path, 'rb') as f, trt.Runtime(logger) as runtime:
            self.model = runtime.deserialize_cuda_engine(f.read())
        self.context = self.model.create_execution_context()

        # Print binding names for debugging
        for i in range(self.model.num_bindings):
            print(f"Binding {i}: Name = {self.model.get_binding_name(i)}, Is Input = {self.model.binding_is_input(i)}")

        # Set input and output names
        self.input_name = 'input'
        self.output_names = ['bbox', 'conf', 'class_id']

        # Set number of inputs and outputs
        self.num_inputs = 1
        self.num_outputs = 3

    def __init_bindings(self) -> None:
        dynamic = False
        Tensor = namedtuple('Tensor', ('name', 'dtype', 'shape', 'cpu', 'gpu'))
        inp_info = []
        out_info = []
        out_ptrs = []

        # Handle input binding
        i = 0  # Input binding index
        name = self.input_name
        assert self.model.get_binding_name(i) == name
        dtype = trt.nptype(self.model.get_binding_dtype(i))
        shape = tuple(self.model.get_binding_shape(i))
        if -1 in shape:
            dynamic |= True
        if not dynamic:
            cpu = np.empty(shape, dtype)
            gpu = cuda.mem_alloc(cpu.nbytes)
            cuda.memcpy_htod_async(gpu, cpu, self.stream)
        else:
            cpu, gpu = np.empty(0), 0
        inp_info.append(Tensor(name, dtype, shape, cpu, gpu))

        # Handle output bindings
        for i, name in enumerate(self.output_names, start=1):  # Start from 1 as 0 is input
            assert self.model.get_binding_name(i) == name
            dtype = trt.nptype(self.model.get_binding_dtype(i))
            shape = tuple(self.model.get_binding_shape(i))
            if not dynamic:
                cpu = np.empty(shape, dtype=dtype)
                gpu = cuda.mem_alloc(cpu.nbytes)
                cuda.memcpy_htod_async(gpu, cpu, self.stream)
                out_ptrs.append(gpu)
            else:
                cpu, gpu = np.empty(0), 0
            out_info.append(Tensor(name, dtype, shape, cpu, gpu))

        self.is_dynamic = dynamic
        self.inp_info = inp_info
        self.out_info = out_info
        self.out_ptrs = out_ptrs

        # Update bindings with GPU addresses
        for i, tensor in enumerate(self.inp_info + self.out_info):
            self.bindings[i] = int(tensor.gpu)

    def __warm_up(self) -> None:
        if self.is_dynamic:
            print('Your engine has dynamic axes, please warm up by yourself!')
            return

        # Create a dummy image with the expected shape
        dummy_image = np.zeros(self.inp_info[0].shape, dtype=self.inp_info[0].dtype)

        # Warm-up with the dummy image
        for _ in range(10):
            self.__call__(dummy_image)

    def set_profiler(self, profiler: Optional[trt.IProfiler]) -> None:
        self.context.profiler = profiler \
            if profiler is not None else trt.Profiler()

    def __call__(self, preprocessed_image) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        expected_shape = self.inp_info[0].shape  # Assuming the first input info holds the shape
        assert preprocessed_image.shape == expected_shape, f"Expected shape {expected_shape}, but got {preprocessed_image.shape}"
        contiguous_input = np.ascontiguousarray(preprocessed_image)

        # Prepare GPU memory for input
        if self.is_dynamic:
            self.context.set_binding_shape(0, tuple(contiguous_input.shape))
            self.inp_info[0].gpu = cuda.mem_alloc(contiguous_input.nbytes)
        cuda.memcpy_htod_async(self.inp_info[0].gpu, contiguous_input, self.stream)
        self.bindings[0] = int(self.inp_info[0].gpu)

        # Allocate memory for outputs
        output_gpu_ptrs: List[int] = []
        outputs: List[ndarray] = []
        for i in range(self.num_outputs):
            j = i + self.num_inputs
            if self.is_dynamic:
                shape = tuple(self.context.get_binding_shape(j))
                dtype = self.out_info[i].dtype
                cpu = np.empty(shape, dtype=dtype)
                gpu = cuda.mem_alloc(cpu.nbytes)
            else:
                cpu = self.out_info[i].cpu
                gpu = self.out_info[i].gpu
            output_gpu_ptrs.append(gpu)
            self.bindings[j] = int(gpu)
            outputs.append(cpu)

        # Perform inference
        self.context.execute_async_v2(self.bindings, self.stream.handle)
        self.stream.synchronize()

        # Retrieve outputs from GPU
        for i, o in enumerate(output_gpu_ptrs):
            cuda.memcpy_dtoh_async(outputs[i], o, self.stream)

        # Assuming the outputs are ordered as bboxes, confs, and class_ids
        bboxes = outputs[0][0]  # Assuming bbox is the first output
        confs = outputs[1][0]  # Assuming conf is the second output
        class_ids = outputs[2][0]  # Assuming class_id is the third output

        return bboxes, confs, class_ids

def load_dataset(dataset_path: str) -> Tuple[List[str], List[str]]:
    images = []
    labels = []
    for image_path in Path(dataset_path).glob("*.jpg"):
        images.append(str(image_path))
        label_path = image_path.with_suffix('.txt')
        labels.append(str(label_path))

    print("-----------------------------------------")
    return images, labels

def parse_yolo_labels(label_path):
    ground_truth = []
    with open(label_path, 'r') as file:
        for line in file:
            class_id, x_center, y_center, width, height = map(float, line.split())
            if class_id == 0:
                class_id = 2  # Change class_id from 0 to 2
            ground_truth.append((class_id, x_center, y_center, width, height))
    return ground_truth

def calculate_iou(box1, box2):
    """
    Calculate the Intersection over Union (IoU) of two bounding boxes.
    """
    # print("box1:", box1)
    # print("box2:", box2)
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
    for pred in predictions:
        for idx, gt in enumerate(ground_truth):
            gt_class_id, gt_x_center, gt_y_center, gt_width, gt_height = gt
            pred_class_id, pred_x_center, pred_y_center, pred_width, pred_height = pred
            gt_bbox = (gt_x_center, gt_y_center, gt_width, gt_height)
            pred_bbox = (pred_x_center, pred_y_center, pred_width, pred_height)
            if idx in detected:
                continue
            iou = calculate_iou(pred_bbox, gt_bbox)
            if iou >= iou_threshold and pred_class_id == gt_class_id:
                true_positives += 1
                detected.add(idx)
                break

    precision = true_positives / len(predictions) if predictions else 0
    recall = true_positives / len(ground_truth) if ground_truth else 0

    # Simplified version for mAP, for demonstration purposes
    mAP = (precision + recall) / 2 if precision and recall else 0

    return mAP, precision, recall


def process_dataset(engine, dataset_path: str):
    images, label_files = load_dataset(dataset_path)
    total_precision, total_recall, total_mAP = 0, 0, 0

    for image_path, label_file in zip(images, label_files):
        image = cv2.imread(image_path)
        ground_truth = parse_yolo_labels(label_file)
        #print(ground_truth)
        predictions = engine.infer(image)
        mAP, precision, recall = calculate_accuracy_metrics(predictions, ground_truth)

        total_precision += precision
        total_recall += recall
        total_mAP += mAP

    num_images = len(images)
    avg_precision = total_precision / num_images
    avg_recall = total_recall / num_images
    avg_mAP = total_mAP / num_images

    print(f"Average Precision: {avg_precision:.2f}")
    print(f"Average Recall: {avg_recall:.2f}")
    print(f"Average mAP: {avg_mAP:.2f}")


engine = TRTEngine(ENGINE_PATH)
process_dataset(engine, DATASET_PATH)