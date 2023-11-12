import torch
from ultralytics import YOLO

# Load your model
model_path = '../../runs/weights/best.pt'
onnx_model_path = '../models/onnx'
model = YOLO(model_path)
model.eval()

# Create a dummy input tensor
batch_size, channels, height, width = 1, 3, 640, 640  # Adjust the size as per your model's requirement
dummy_input = torch.randn(batch_size, channels, height, width)

# Export the model
torch.onnx.export(model,
                  dummy_input,
                  f"{onnx_model_path}/yolov8.onnx",
                  export_params=True,       # store the trained parameter weights inside the model file
                  opset_version=11,
                  do_constant_folding=True, # whether to execute constant folding for optimization
                  input_names=['images'],
                  output_names=['num_dets', 'bboxes', 'scores', 'labels'],
                  dynamic_axes={'images': {0: 'batch_size'},               # variable length axes for input
                                'num_dets': {0: 'batch_size'},            # variable length axes for outputs
                                'bboxes': {0: 'batch_size'},
                                'scores': {0: 'batch_size'},
                                'labels': {0: 'batch_size'}})
