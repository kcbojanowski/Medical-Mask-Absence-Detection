import torch
import torch.onnx
import torch.nn as nn
from torch.autograd import Variable
import numpy as np
import onnx
import onnx_graphsurgeon as gs
import logging as logger
from typing import List
import os


class OnnxModifier:
    def __init__(self, onnx_path):
        self.graph = gs.import_onnx(onnx.load(onnx_path))
        self.tensor = self.graph.tensors()
        self.nodes = self.graph.nodes

    def get_modified_graph(self):
        return self.graph

    def remove_nodes(self, remove_node_list: List[str]):
        for remove_nd in remove_node_list:
            nodes_to_remove = [node for node in self.nodes if node.name == remove_nd]

            # Check if the node was found
            if nodes_to_remove:
                removed = nodes_to_remove[0]
                self.nodes.remove(removed)
            else:
                print(f"Node '{remove_nd}' not found and cannot be removed.")

        output_tensor = self.graph.outputs[0]

        self.graph.outputs.remove(output_tensor)

    def add_transpose_nodes(self, input_tensor, perm: List[int]):
        attrs = {"perm": perm}
        transpose_inputs = [input_tensor]
        if input_tensor.shape is not None:
            output_shape = [*input_tensor.shape[:-1]]
        else:
            output_shape = None

        transpose_outputs = [
            gs.Variable(name="%s_transpose_output" % input_tensor.name, dtype=input_tensor.dtype, shape=output_shape)]

        transpose_node = gs.Node(op="Transpose", name="%s_transpose" % input_tensor.name, inputs=transpose_inputs,
                                 outputs=transpose_outputs, attrs=attrs)

        self.nodes.append(transpose_node)

        return transpose_node.outputs[0]

    def add_reducemax_nodes(self, input_tensor, axes=2, keepdims=0):
        attrs = {}
        attrs["axes"] = [axes]
        attrs["keepdims"] = keepdims

        reducemax_inputs = [input_tensor]

        if input_tensor.shape is not None:
            output_shape = [*input_tensor.shape[:-1]]
        else:
            output_shape = None

        reducemax_outputs = [
            gs.Variable(name="%s_reducemax_output" % (input_tensor.name), dtype=input_tensor.dtype, shape=output_shape)]

        reducemax_node = gs.Node(op="ReduceMax", name="%s_reducemax" % (input_tensor.name), inputs=reducemax_inputs,
                                 outputs=reducemax_outputs, attrs=attrs)

        self.nodes.append(reducemax_node)

        return reducemax_node.outputs[0]

    def add_argmax_nodes(self, input_tensor, axis=2, keepdims=0):
        attrs = {}
        attrs["axis"] = axis
        attrs["keepdims"] = keepdims
        argmax_inputs = [input_tensor]

        # output_shape
        if input_tensor.shape is not None:
            output_shape = [*input_tensor.shape[:-1]]
        else:
            output_shape = None

        # argmax_outputs
        argmax_outputs = [
            gs.Variable(name="%s_argmax_output" % input_tensor.name, dtype=np.int64, shape=output_shape)]

        argmax_node = gs.Node(op="ArgMax", name="%s_argmax" % input_tensor.name, inputs=argmax_inputs,
                              outputs=argmax_outputs, attrs=attrs)

        self.nodes.append(argmax_node)

        return argmax_node.outputs[0]

    def rename_input_node(self, old_name: str, new_name: str):
        # Find and rename the input node
        for tensor in self.graph.inputs:
            if tensor.name == old_name:
                tensor.name = new_name
                print(f"Renamed input node from '{old_name}' to '{new_name}'.")
                break
        else:
            print(f"Input node '{old_name}' not found.")

    def add_yolov8_output(self, tensor, output_name):
        input_node = tensor.inputs[0]
        if output_name == "bbox":
            input_node.outputs = [
                gs.Variable(name=output_name).to_variable(shape=["batch", None, 4], dtype=np.float32)]

        elif output_name == "conf":
            input_node.outputs = [
                gs.Variable(name=output_name).to_variable(shape=["batch", None], dtype=np.float32)]

        elif output_name == "class_id":
            input_node.outputs = [
                gs.Variable(name=output_name).to_variable(shape=["batch", None], dtype=np.int32)]

        return input_node.outputs[0]

    def carve_output(self, output_list):
        self.graph.outputs = output_list

    def carve_output(self, output_list):
        self.graph.outputs = output_list


class Yolov8Modifier:
    # step 1: remove unused nodes
    # step 2: modify bbox node
    # step 3: modify conf, class_id nodes

    def __init__(self, onnx_load_path, modified_onnx_path):
        self.onnx_load_path = onnx_load_path
        self.onnx_save_path = modified_onnx_path

        self.remove_node_list = ["/model.22/Concat_10"]

        self.modify_tensors_list = []
        self._prepare_modify_tensors()

    def save_modified_onnx(self, graph):
        """Save the modified ONNX model to the specified path."""
        try:
            print(f"Saving modified ONNX model to {self.onnx_save_path}")

            # Export the graph to an ONNX ModelProto object
            model_proto = gs.export_onnx(graph)

            # Write the ModelProto object to a file
            with open(self.onnx_save_path, 'wb') as f:
                f.write(model_proto.SerializeToString())

            print("Modified ONNX model saved successfully.")
        except Exception as e:
            print(f"Failed to save ONNX model: {e}")

    def _prepare_modify_tensors(self):
        self.modify_tensors_list.append(("/model.22/Mul_2_output_0", "/model.22/Sigmoid_output_0"))

    def run(self):
        print("# ---- ONNX modification starts ---- #")
        dtype = np.float32

        onnx_md = OnnxModifier(self.onnx_load_path)
        graph = onnx_md.graph

        onnx_md.rename_input_node("input.1", "input")

        onnx_md.remove_nodes(self.remove_node_list)

        tensor_list = self.modify_tensors_list

        # ---- bbox ---- #
        bbox_out_tensor = onnx_md.add_transpose_nodes(onnx_md.tensor[tensor_list[0][0]], [0, 2, 1])

        # ---- conf & class-id ---- #
        sigmoid_tensor = onnx_md.tensor[tensor_list[0][1]]
        conf_out_tensor = onnx_md.add_reducemax_nodes(sigmoid_tensor, 1, 0)

        cls_id_out_tesnor = onnx_md.add_argmax_nodes(sigmoid_tensor, 1, 0)

        # ---- make last node ---- #
        bbox_last_output = onnx_md.add_yolov8_output(bbox_out_tensor, "bbox")
        conf_last_output = onnx_md.add_yolov8_output(conf_out_tensor, "conf")
        cls_id_last_output = onnx_md.add_yolov8_output(cls_id_out_tesnor, "class_id")

        # --- make output ---- #
        onnx_md.carve_output([bbox_last_output, conf_last_output, cls_id_last_output])

        modified_graph = onnx_md.get_modified_graph()
        # --- save onnx ---- #
        self.save_modified_onnx(modified_graph)


# Load your model
model_path = '../models/pytorch/best.pt'
onnx_model_path = '../models/onnx/best.onnx'
# onnx_model_path = '../models/onnx/FaceMaskYolov8.onnx'
# onnx_save_path = '../models/onnx/FaceMaskYolov8.onnx'

# dir_path = os.path.dirname(os.path.abspath(onnx_save_path))
# if not os.path.exists(dir_path):
#     print(f"Directory does not exist: {dir_path}")
# else:
#     print(f"Directory exists: {dir_path}")
#
# # Check if file already exists
# if os.path.isfile(onnx_save_path):
#     print(f"File already exists: {onnx_save_path}")
# else:
#     print(f"File does not exist and will be created: {onnx_save_path}")
# Load your trained model

model = torch.load(model_path)['model'].float().eval()

# Create a dummy input tensor
dummy_input = torch.randn(1, 3, 640, 640)

# Export the model to ONNX
torch.onnx.export(model, dummy_input, onnx_model_path)
onnx_model = onnx.load(onnx_model_path)
onnx.checker.check_model(onnx_model)

# modifier = Yolov8Modifier(onnx_model_path, onnx_save_path)
# modifier.run()


