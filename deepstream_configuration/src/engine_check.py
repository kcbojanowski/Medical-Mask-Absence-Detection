import tensorrt as trt
import pycuda.driver as cuda
import pycuda.autoinit
import numpy as np

TRT_ENGINE_PATH = 'path_to_your_trt_engine_file.engine'

def load_engine(trt_runtime, engine_path):
    with open(engine_path, 'rb') as f:
        engine_data = f.read()
    engine = trt_runtime.deserialize_cuda_engine(engine_data)
    return engine

def allocate_buffers(engine):
    inputs = []
    outputs = []
    bindings = []
    stream = cuda.Stream()
    for binding in engine:
        size = trt.volume(engine.get_binding_shape(binding)) * engine.max_batch_size
        dtype = trt.nptype(engine.get_binding_dtype(binding))
        host_mem = cuda.pagelocked_empty(size, dtype)
        device_mem = cuda.mem_alloc(host_mem.nbytes)
        bindings.append(int(device_mem))
        if engine.binding_is_input(binding):
            inputs.append({'host': host_mem, 'device': device_mem})
        else:
            outputs.append({'host': host_mem, 'device': device_mem})
    return inputs, outputs, bindings, stream

def main():
    # Load TensorRT Engine
    trt_runtime = trt.Runtime(trt.Logger(trt.Logger.WARNING))
    engine = load_engine(trt_runtime, TRT_ENGINE_PATH)

    # Allocate Buffers for input and output
    inputs, outputs, bindings, stream = allocate_buffers(engine)

    # Create context for inference
    context = engine.create_execution_context()

    # Prepare dummy input data (Replace this with your actual input data)
    # Assuming input is a single image of shape (3, H, W)
    input_data = np.random.random(size=(3, 640, 640)).astype(np.float32)
    np.copyto(inputs[0]['host'], input_data.ravel())

    # Transfer input data to the GPU.
    cuda.memcpy_htod_async(inputs[0]['device'], inputs[0]['host'], stream)

    # Run inference
    context.execute_async(bindings=bindings, stream_handle=stream.handle)

    # Transfer predictions back from the GPU.
    cuda.memcpy_dtoh_async(outputs[0]['host'], outputs[0]['device'], stream)
    stream.synchronize()

    # Here we assume the output is a single vector of probabilities
    output_data = outputs[0]['host']
    print("Output Data:", output_data)

if __name__ == '__main__':
    main()
