#!/usr/bin/env python3
import os
import tensorrt as trt
TRT_LOGGER = trt.Logger()


def build_engine(onnx_file_path, engine_file_path=""):
    """Takes an ONNX file and creates a TensorRT engine to run inference with"""
    with trt.Builder(TRT_LOGGER) as builder, builder.create_network(
            1 << int(trt.NetworkDefinitionCreationFlag.EXPLICIT_BATCH)
    ) as network, builder.create_builder_config() as config, trt.OnnxParser(
        network, TRT_LOGGER
    ) as parser, trt.Runtime(TRT_LOGGER) as runtime:

        config.set_memory_pool_limit(trt.MemoryPoolType.WORKSPACE, 1 << 28)

        # Parse model file
        if not os.path.exists(onnx_file_path):
            print(f"ONNX file {onnx_file_path} not found.")
            exit(0)
        print(f"Loading ONNX file from path {onnx_file_path}...")
        with open(onnx_file_path, "rb") as model:
            print("Beginning ONNX file parsing")
            if not parser.parse(model.read()):
                print("ERROR: Failed to parse the ONNX file.")
                for error in range(parser.num_errors):
                    print(parser.get_error(error))
                return None
        print("Completed parsing of ONNX file")
        print(f"Building an engine from file {onnx_file_path}; this may take a while...")

        plan = builder.build_serialized_network(network, config)
        if plan is None:
            print("Failed to build the engine.")
            return None

        engine = runtime.deserialize_cuda_engine(plan)
        print("Completed creating Engine")
        if engine_file_path:
            with open(engine_file_path, "wb") as f:
                f.write(plan)
        return engine


def main():
    # File paths
    onnx_file_path = "models/YoloV8/yolov8_custom.onnx"
    engine_file_path = "yolov8_engine.trt"

    # Build a TensorRT engine.
    engine = build_engine(onnx_file_path, engine_file_path)
    if engine:
        print(f"Successfully created TensorRT engine: {engine_file_path}")
    else:
        print("Failed to create TensorRT engine.")


if __name__ == "__main__":
    main()
