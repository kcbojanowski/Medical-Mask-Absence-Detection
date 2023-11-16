import os
import cv2
import warnings
from collections import namedtuple
from pathlib import Path
from typing import List, Optional, Tuple, Union

import numpy as np
import pycuda.autoinit  # noqa F401
import pycuda.driver as cuda
import tensorrt as trt
from numpy import ndarray

os.environ['CUDA_MODULE_LOADING'] = 'LAZY'
warnings.filterwarnings(action='ignore', category=DeprecationWarning)


class TRTEngine:

    def __init__(self, weight: Union[str, Path]) -> None:
        self.weight = Path(weight) if isinstance(weight, str) else weight
        self.stream = cuda.Stream(0)
        self.__init_engine()
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
        num_dets, bboxes, scores, labels = self.__call__(preprocessed_image)

        # Check if num_dets is a scalar or a one-dimensional array
        if np.isscalar(num_dets):
            nums = int(num_dets)
        else:
            nums = int(num_dets[0])  # Assuming num_dets is a one-dimensional array

        # Process bboxes, scores, and labels based on nums
        bboxes = bboxes[:nums]
        scores = scores[:nums]
        labels = labels[:nums]

        return bboxes, scores, labels

    def __init_engine(self) -> None:
        logger = trt.Logger(trt.Logger.WARNING)
        trt.init_libnvinfer_plugins(logger, namespace='')
        with trt.Runtime(logger) as runtime:
            model = runtime.deserialize_cuda_engine(self.weight.read_bytes())

        context = model.create_execution_context()

        names = [model.get_binding_name(i) for i in range(model.num_bindings)]
        self.num_bindings = model.num_bindings
        self.bindings: List[int] = [0] * self.num_bindings
        num_inputs, num_outputs = 0, 0

        for i in range(model.num_bindings):
            if model.binding_is_input(i):
                num_inputs += 1
            else:
                num_outputs += 1

        self.num_inputs = num_inputs
        self.num_outputs = num_outputs
        self.model = model
        self.context = context
        self.input_names = names[:num_inputs]
        self.output_names = names[num_inputs:]

    def __init_bindings(self) -> None:
        dynamic = False
        Tensor = namedtuple('Tensor', ('name', 'dtype', 'shape', 'cpu', 'gpu'))
        inp_info = []
        out_info = []
        out_ptrs = []
        for i, name in enumerate(self.input_names):
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
        for i, name in enumerate(self.output_names):
            i += self.num_inputs
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

    def __warm_up(self) -> None:
        if self.is_dynamic:
            print('You engine has dynamic axes, please warm up by yourself !')
            return
        for _ in range(10):
            inputs = []
            for i in self.inp_info:
                inputs.append(i.cpu)
            self.__call__(inputs)

    def set_profiler(self, profiler: Optional[trt.IProfiler]) -> None:
        self.context.profiler = profiler \
            if profiler is not None else trt.Profiler()

    def __call__(self, *inputs) -> Tuple[ndarray, ndarray, ndarray, ndarray]:
        assert len(inputs) == self.num_inputs
        contiguous_inputs: List[ndarray] = [
            np.ascontiguousarray(i) for i in inputs
        ]

        # Prepare GPU memory for inputs
        for i in range(self.num_inputs):
            if self.is_dynamic:
                self.context.set_binding_shape(i, tuple(contiguous_inputs[i].shape))
                self.inp_info[i].gpu = cuda.mem_alloc(contiguous_inputs[i].nbytes)
            cuda.memcpy_htod_async(self.inp_info[i].gpu, contiguous_inputs[i], self.stream)
            self.bindings[i] = int(self.inp_info[i].gpu)

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

        # Unpack and process the output data
        num_dets = outputs[0][0][0]  # Assuming first output is num_dets
        bboxes = outputs[1][0]       # Assuming second output is bboxes
        scores = outputs[2][0]       # Assuming third output is scores
        labels = outputs[3][0]       # Assuming fourth output is labels

        # Truncate arrays based on num_dets
        bboxes = bboxes[:num_dets]
        scores = scores[:num_dets]
        labels = labels[:num_dets]

        return num_dets, bboxes, scores, labels

def process_video(engine, camera_index=0):
    cap = cv2.VideoCapture(camera_index)

    class_labels = ["with_mask", "without_mask", "incorrectly_worn_mask"]

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        # Run inference
        bboxes, scores, labels = engine.infer(frame)

        # Draw bounding boxes and labels
        for bbox, score, label in zip(bboxes, scores, labels):
            if score < 0.5:  # Confidence threshold
                continue
            x, y, w, h = bbox
            cv2.rectangle(frame, (int(x), int(y)), (int(x + w), int(y + h)), (0, 255, 0), 2)
            label_text = f"{class_labels[label]}: {score:.2f}"
            cv2.putText(frame, label_text, (int(x), int(y - 5)), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

        cv2.imshow("Frame", frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()


engine = TRTEngine("../models/trt_engines/train7.engine")
process_video(engine)