import torch
import time
import tensorrt as trt
import numpy as np
import common
import pycuda.autoinit
import pycuda.driver as cuda

# Load the pre-trained YOLOv8 model
TRT_PATH = "../deepstream_configuration/models/trt_engines/best.engine"
PYTORCH_PATH = "../deepstream_configuration/models/pytorch/best.pt"

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Load model
model = torch.load(PYTORCH_PATH)['model'].float().eval()

# Convert model to FP16 and move to device (GPU or CPU)
model = model.half().to(device)

input_shape = (1, 3, 640, 640)
num_iterations = 100

# PyTorch Inference with FP16 on GPU
total_time = 0.0
with torch.no_grad():
    for _ in range(num_iterations):
        # Generate input data and move to device
        input_data = torch.randn(input_shape).half().to(device)

        start_time = time.time()
        output_data = model(input_data)
        end_time = time.time()
        total_time += end_time - start_time

pytorch_fps = num_iterations / total_time
print(f"PyTorch FPS: {pytorch_fps:.2f}")

# TensorRT engine
# FP32
def load_engine(trt_runtime, engine_path):
    with open(engine_path, 'rb') as f:
        engine_data = f.read()
    engine = trt_runtime.deserialize_cuda_engine(engine_data)
    return engine


trt_runtime = trt.Runtime(trt.Logger(trt.Logger.WARNING))
engine = load_engine(trt_runtime, TRT_PATH)
context = engine.create_execution_context()

# Allocate buffers
inputs, outputs, bindings, stream = common.allocate_buffers(engine)

# Inference
total_time_trt = 0.0
for _ in range(num_iterations):
    # Prepare random input data
    random_input = np.random.random_sample(tuple(context.get_binding_shape(0))).astype(np.float32)

    # Perform inference
    start_time = time.time()
    trt_outputs = common.do_inference_v2(context, bindings=bindings, inputs=inputs, outputs=outputs, stream=stream)
    end_time = time.time()
    total_time_trt += end_time - start_time

trt_fps = num_iterations / total_time_trt
print(f"TensorRT FPS: {trt_fps:.2f}")