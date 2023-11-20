#!/usr/bin/env python3

import sys
import gi
from gi.repository import GLib, Gst
from common.is_aarch_64 import is_aarch64
from common.bus_call import bus_call
import time
import pyds
import argparse

gi.require_version('Gst', '1.0')

PGIE_CLASS_ID_INCORRECT_MASK = 0
PGIE_CLASS_ID_WITH_MASK = 1
PGIE_CLASS_ID_WITHOUT_MASK = 2

# Bounding box options
bbox_border_color_0 = {"R": 1.0, "G": 1.0, "B": 0.0, "A": 1.0}
bbox_border_color_1 = {"R": 0.0, "G": 1.0, "B": 0.0, "A": 1.0}
bbox_border_color_2 = {"R": 1.0, "G": 0.0, "B": 0.0, "A": 1.0}
bbox_has_bg_color = True  # Bool for whether bounding box has background color

# Color of bbox background.
bbox_bg_color_0 = {"R": 1.0, "G": 1.0, "B": 0.0, "A": 0.1}
bbox_bg_color_1 = {"R": 0.0, "G": 1.0, "B": 0.0, "A": 0.1}
bbox_bg_color_2 = {"R": 1.0, "G": 0.0, "B": 0.0, "A": 0.1}

# Global variables to calculate FPS
global last_time, frame_count
global detected_labels
detected_labels = []
last_time = time.time()
frame_count = 0
fps_display_frequency = 2

labels_path = '../models/labels.txt'

def load_class_labels(label_path):
    with open(label_path, 'r') as f:
        labels = [line.strip() for line in f.readlines()]
    return labels


def osd_sink_pad_buffer_probe(pad, info, u_data):
    global last_time, frame_count
    frame_count += 1

    class_labels = load_class_labels(labels_path)

    obj_counter = {
        PGIE_CLASS_ID_INCORRECT_MASK: 0,
        PGIE_CLASS_ID_WITH_MASK: 0,
        PGIE_CLASS_ID_WITHOUT_MASK: 0,
    }

    gst_buffer = info.get_buffer()
    if not gst_buffer:
        print("Unable to get GstBuffer ")
        return

    # Retrieve batch metadata from the gst_buffer
    batch_meta = pyds.gst_buffer_get_nvds_batch_meta(hash(gst_buffer))
    if not batch_meta:
        print("Unable to get batch meta")
        return Gst.PadProbeReturn.OK

    l_frame = batch_meta.frame_meta_list
    if not l_frame:
        print("No frame meta in batch")
        return Gst.PadProbeReturn.OK

    while l_frame is not None:
        try:
            frame_meta = pyds.NvDsFrameMeta.cast(l_frame.data)
            if not frame_meta:
                print("Failed to cast to NvDsFrameMeta")
                l_frame = l_frame.next
                continue
        except StopIteration:
            break

        num_rects = frame_meta.num_obj_meta

        l_obj = frame_meta.obj_meta_list
        while l_obj is not None:
            try:
                obj_meta = pyds.NvDsObjectMeta.cast(l_obj.data)
                if not obj_meta:
                    print("Failed to cast to NvDsObjectMeta")
                    l_obj = l_obj.next
                    continue
            except StopIteration:
                break

            obj_counter[obj_meta.class_id] += 1
            rectparams = obj_meta.rect_params
            rectparams.border_width = 3
            rectparams.has_bg_color = 0

            # Choose color based on class ID
            if obj_meta.class_id == PGIE_CLASS_ID_INCORRECT_MASK:
                bbox_color = bbox_border_color_0
                bg_color = bbox_bg_color_0
            elif obj_meta.class_id == PGIE_CLASS_ID_WITH_MASK:
                bbox_color = bbox_border_color_1
                bg_color = bbox_bg_color_1
            elif obj_meta.class_id == PGIE_CLASS_ID_WITHOUT_MASK:
                bbox_color = bbox_border_color_2
                bg_color = bbox_bg_color_2
            else:
                bbox_color = {"R": 1.0, "G": 1.0, "B": 1.0, "A": 1.0}
                bg_color = {"R": 0.5, "G": 0.5, "B": 0.5, "A": 0.2}

            rectparams.border_color.set(bbox_color["R"], bbox_color["G"], bbox_color["B"], bbox_color["A"])
            rectparams.has_bg_color = bbox_has_bg_color

            if bbox_has_bg_color:
                rectparams.bg_color.set(bg_color["R"], bg_color["G"], bg_color["B"], bg_color["A"])

            label_name = class_labels[obj_meta.class_id] if obj_meta.class_id < len(class_labels) else "Unknown"
            confidence_percent = obj_meta.confidence * 100

            # Set text parameters with label and confidence
            obj_meta.text_params.display_text = f"{label_name} {confidence_percent:.2f}%"
            obj_meta.text_params.x_offset = int(rectparams.left)
            obj_meta.text_params.y_offset = max(int(rectparams.top) - 10, 0)
            obj_meta.text_params.font_params.font_name = "Serif"
            obj_meta.text_params.font_params.font_size = 12
            obj_meta.text_params.font_params.font_color.set(1.0, 1.0, 1.0, 1.0)
            obj_meta.text_params.set_bg_clr = 0

            try:
                l_obj = l_obj.next
            except StopIteration:
                break

        if frame_count % fps_display_frequency == 0:
            current_time = time.time()
            fps = fps_display_frequency / (current_time - last_time)
            last_time = current_time

            # Acquiring a display meta object
            display_meta = pyds.nvds_acquire_display_meta_from_pool(batch_meta)
            display_meta.num_labels = 1
            py_nvosd_text_params = display_meta.text_params[0]

            # Setting FPS text to be shown on screen
            py_nvosd_text_params.display_text = f"Number of Frame: {frame_count}, FPS: {fps:.2f}"
            py_nvosd_text_params.x_offset = 10
            py_nvosd_text_params.y_offset = 30  # Adjust the Y offset as needed
            py_nvosd_text_params.font_params.font_name = "Sans Serif"
            py_nvosd_text_params.font_params.font_size = 10
            py_nvosd_text_params.font_params.font_color.set(1.0, 1.0, 1.0, 1.0)
            py_nvosd_text_params.set_bg_clr = 1
            py_nvosd_text_params.text_bg_clr.set(0.0, 0.0, 0.0, 1.0)

            pyds.nvds_add_display_meta_to_frame(frame_meta, display_meta)

        # Acquiring a display meta object
        display_meta = pyds.nvds_acquire_display_meta_from_pool(batch_meta)
        display_meta.num_labels = 1
        py_nvosd_text_params = display_meta.text_params[0]

        # Setting display text to be shown on screen
        py_nvosd_text_params.display_text = f"Number of Objects={num_rects} Without_mask_count={obj_counter[PGIE_CLASS_ID_WITHOUT_MASK]} Incorrect_mask_count={obj_counter[PGIE_CLASS_ID_INCORRECT_MASK]}"
        py_nvosd_text_params.x_offset = 10
        py_nvosd_text_params.y_offset = 12
        py_nvosd_text_params.font_params.font_name = "Serif"
        py_nvosd_text_params.font_params.font_size = 10
        py_nvosd_text_params.font_params.font_color.set(1.0, 1.0, 1.0, 1.0)
        py_nvosd_text_params.set_bg_clr = 1
        py_nvosd_text_params.text_bg_clr.set(0.0, 0.0, 0.0, 1.0)
        pyds.nvds_add_display_meta_to_frame(frame_meta, display_meta)

        try:
            l_frame = l_frame.next
        except StopIteration:
            break

    return Gst.PadProbeReturn.OK


def parse_args():
    parser = argparse.ArgumentParser(description="DeepStream YOLO Dataset Processor")
    parser.add_argument("--device", type=str, default="/dev/video0", help="Path to the USB camera device (e.g., /dev/video0)")
    parser.add_argument("--framerate", type=str, default="30/1", help="Framerate for the video capture (e.g., '15/1')")
    parser.add_argument("--test", action="store_true", help="Run in test mode for dataset accuracy checking")
    return parser.parse_args()


def run_real_time_detection(args):

    # Initialize GStreamer
    Gst.init(None)

    print("Creating Pipeline \n ")
    pipeline = Gst.Pipeline()

    if not pipeline:
        sys.stderr.write(" Unable to create Pipeline \n")

    # Source element
    print("Creating Source \n ")
    source = Gst.ElementFactory.make("v4l2src", "usb-cam-source")
    if not source:
        sys.stderr.write(" Unable to create Source \n")

    caps_v4l2src = Gst.ElementFactory.make("capsfilter", "v4l2src_caps")
    if not caps_v4l2src:
        sys.stderr.write(" Unable to create v4l2src capsfilter \n")

    print("Creating Video Converter \n")

    # Videoconvert to make sure a superset of raw formats are supported
    vidconvsrc = Gst.ElementFactory.make("videoconvert", "convertor_src1")
    if not vidconvsrc:
        sys.stderr.write(" Unable to create videoconvert \n")

    # nvvideoconvert to convert incoming raw buffers to NVMM Mem (NvBufSurface API)
    nvvidconvsrc = Gst.ElementFactory.make("nvvideoconvert", "convertor_src2")
    if not nvvidconvsrc:
        sys.stderr.write(" Unable to create Nvvideoconvert \n")

    # Capsfilter for nvvidconvsrc to output RGBA format
    nvmm_caps = Gst.ElementFactory.make("capsfilter", "nvmm_caps")
    if not nvmm_caps:
        sys.stderr.write(" Unable to create capsfilter \n")

    # Nvstreammux instance to form batches from one or more sources.
    streammux = Gst.ElementFactory.make("nvstreammux", "Stream-muxer")
    if not streammux:
        sys.stderr.write(" Unable to create NvStreamMux \n")

    # Nvinfer to run inferencing on camera's output,
    # behaviour of inferencing is set through config file
    pgie = Gst.ElementFactory.make("nvinfer", "primary-inference")
    if not pgie:
        sys.stderr.write(" Unable to create pgie \n")

    # Convertor to convert from NV12 to RGBA as required by nvosd
    nvvidconv = Gst.ElementFactory.make("nvvideoconvert", "convertor")
    if not nvvidconv:
        sys.stderr.write(" Unable to create nvvidconv \n")

    # OSD to draw on the converted RGBA buffer
    nvosd = Gst.ElementFactory.make("nvdsosd", "onscreendisplay")

    if not nvosd:
        sys.stderr.write(" Unable to create nvosd \n")

    # Render the OSD output
    if is_aarch64():
        print("Creating nv3dsink \n")
        sink = Gst.ElementFactory.make("nv3dsink", "nv3d-sink")
        if not sink:
            sys.stderr.write(" Unable to create nv3dsink \n")
    else:
        print("Creating EGLSink \n")
        sink = Gst.ElementFactory.make("nveglglessink", "nvvideo-renderer")
        if not sink:
            sys.stderr.write(" Unable to create egl sink \n")

    print(f"Playing camera {args.device} with framerate {args.framerate}")
    caps_v4l2src.set_property('caps', Gst.Caps.from_string(f"video/x-raw, framerate={args.framerate}"))
    caps = Gst.caps_from_string("video/x-raw(memory:NVMM), format=(string)RGBA")
    nvmm_caps.set_property("caps", caps)
    source.set_property('device', args.device)
    streammux.set_property('gpu-id', 0)
    streammux.set_property('width', 1920)
    streammux.set_property('height', 1080)
    streammux.set_property('batch-size', 1)
    streammux.set_property('live-source', 1)
    streammux.set_property('batched-push-timeout', 4000000)
    pgie.set_property('config-file-path', "../configs/pgie_config.txt")
    # sync = false to avoid late frame drops at the display-sink
    sink.set_property('sync', 0)

    print("Adding elements to Pipeline \n")
    pipeline.add(source)
    pipeline.add(caps_v4l2src)
    pipeline.add(vidconvsrc)
    pipeline.add(nvvidconvsrc)
    pipeline.add(nvmm_caps)
    pipeline.add(streammux)
    pipeline.add(pgie)
    pipeline.add(nvvidconv)
    pipeline.add(nvosd)
    pipeline.add(sink)

    # v4l2src -> nvvideoconvert -> mux ->
    # nvinfer -> nvvideoconvert -> nvosd -> video-renderer
    print("Linking elements in the Pipeline \n")
    source.link(caps_v4l2src)
    caps_v4l2src.link(vidconvsrc)
    vidconvsrc.link(nvvidconvsrc)
    nvvidconvsrc.link(nvmm_caps)
    nvmm_caps.link(streammux)

    sinkpad = streammux.get_request_pad("sink_0")
    if not sinkpad:
        sys.stderr.write(" Unable to get the sink pad of streammux \n")
    srcpad = nvmm_caps.get_static_pad("src")
    if not srcpad:
        sys.stderr.write(" Unable to get source pad of caps_vidconvsrc \n")
    srcpad.link(sinkpad)
    streammux.link(pgie)
    pgie.link(nvvidconv)
    nvvidconv.link(nvosd)
    nvosd.link(sink)

    loop = GLib.MainLoop()
    bus = pipeline.get_bus()
    bus.add_signal_watch()
    bus.connect("message", bus_call, loop)

    osdsinkpad = nvosd.get_static_pad("sink")
    if not osdsinkpad:
        sys.stderr.write(" Unable to get sink pad of nvosd \n")

    osdsinkpad.add_probe(Gst.PadProbeType.BUFFER, osd_sink_pad_buffer_probe, 0)

    print("Starting pipeline \n")
    pipeline.set_state(Gst.State.PLAYING)
    try:
        loop.run()
    except:
        pass
    # cleanup
    pipeline.set_state(Gst.State.NULL)


def parse_labels(label_path):
    labels = []
    with open(label_path, 'r') as file:
        for line in file:
            components = line.strip().split()
            if len(components) == 5:
                class_id, x_center, y_center, width, height = map(float, components)
                labels.append({
                    "class_id": int(class_id),
                    "x_center": x_center,
                    "y_center": y_center,
                    "width": width,
                    "height": height
                })
    return labels


def test_osd_sink_pad_buffer_probe(pad, info, u_data):
    global detected_labels
    frame_detections = []

    # Get batch meta from the buffer
    batch_meta = pyds.gst_buffer_get_nvds_batch_meta(hash(info.get_buffer()))

    pyds.nvds_acquire_meta_lock(batch_meta)  # Pass the batch_meta object here

    try:
        l_frame = batch_meta.frame_meta_list
        while l_frame is not None:
            try:
                frame_meta = pyds.NvDsFrameMeta.cast(l_frame.data)
            except StopIteration:
                break

            l_obj = frame_meta.obj_meta_list
            while l_obj is not None:
                try:
                    obj_meta = pyds.NvDsObjectMeta.cast(l_obj.data)
                except StopIteration:
                    break

                bbox_info = {
                    "class_id": obj_meta.class_id,
                    "x_center": (obj_meta.rect_params.left + obj_meta.rect_params.width / 2) / frame_meta.source_frame_width,
                    "y_center": (obj_meta.rect_params.top + obj_meta.rect_params.height / 2) / frame_meta.source_frame_height,
                    "width": obj_meta.rect_params.width / frame_meta.source_frame_width,
                    "height": obj_meta.rect_params.height / frame_meta.source_frame_height
                }
                frame_detections.append(bbox_info)
                try:
                    l_obj = l_obj.next
                except StopIteration:
                    break

            try:
                l_frame = l_frame.next
            except StopIteration:
                break
    finally:
        pyds.nvds_release_meta_lock(batch_meta)  # Release the lock
    if frame_detections:
        detected_labels.append(frame_detections)

    return Gst.PadProbeReturn.OK

def iou(box1, box2):
    # Calculate the (x, y)-coordinates of the intersection rectangle
    xA = max(box1['x_center'] - box1['width'] / 2, box2['x_center'] - box2['width'] / 2)
    yA = max(box1['y_center'] - box1['height'] / 2, box2['y_center'] - box2['height'] / 2)
    xB = min(box1['x_center'] + box1['width'] / 2, box2['x_center'] + box2['width'] / 2)
    yB = min(box1['y_center'] + box1['height'] / 2, box2['y_center'] + box2['height'] / 2)

    # Compute the area of intersection rectangle
    interArea = max(0, xB - xA) * max(0, yB - yA)

    # Compute the area of both the prediction and ground-truth rectangles
    box1Area = box1['width'] * box1['height']
    box2Area = box2['width'] * box2['height']

    iou = interArea / float(box1Area + box2Area - interArea)

    return iou


def calculate_metrics(predicted_labels, ground_truth_labels):

    class_wise_data = {}

    for detected, ground_truth in zip(predicted_labels, ground_truth_labels):
        for gt in ground_truth:
            gt_class_id = int(gt['class_id'])
            if gt_class_id not in class_wise_data:
                class_wise_data[gt_class_id] = {'TP': 0, 'FP': 0, 'FN': 0, 'detected': []}

            matched = False
            for det in detected:
                if det['class_id'] == gt_class_id:
                    if iou(det, gt) >= 0.5:
                        matched = True
                        if det not in class_wise_data[gt_class_id]['detected']:
                            class_wise_data[gt_class_id]['TP'] += 1
                            class_wise_data[gt_class_id]['detected'].append(det)
                        break
            if not matched:
                class_wise_data[gt_class_id]['FN'] += 1

        for det in detected:
            if det['class_id'] not in class_wise_data:
                class_wise_data[det['class_id']] = {'TP': 0, 'FP': 0, 'FN': 0, 'detected': []}
            if det not in class_wise_data[det['class_id']]['detected']:
                class_wise_data[det['class_id']]['FP'] += 1

    total_TP, total_FP, total_FN = 0, 0, 0
    for class_id, data in class_wise_data.items():
        TP = data['TP']
        FP = data['FP']
        FN = data['FN']
        total_TP += TP
        total_FP += FP
        total_FN += FN

        precision = TP / (TP + FP) if TP + FP > 0 else 0
        recall = TP / (TP + FN) if TP + FN > 0 else 0
        data['precision'] = precision
        data['recall'] = recall

    all_precisions = [data['precision'] for data in class_wise_data.values()]
    all_recalls = [data['recall'] for data in class_wise_data.values()]
    mAP = sum(all_precisions) / len(all_precisions) if all_precisions else 0
    mean_recall = sum(all_recalls) / len(all_recalls) if all_recalls else 0

    # Calculate accuracy
    accuracy = total_TP / (total_TP + total_FP + total_FN) if total_TP + total_FP + total_FN > 0 else 0

    return mAP, mean_recall, accuracy


def run_dataset_test(args):
    global detected_labels
    detected_labels = []

    Gst.init(None)

    pipeline = Gst.Pipeline()

    if not pipeline:
        sys.stderr.write(" Unable to create Pipeline \n")

    # Source element for reading from the file
    source = Gst.ElementFactory.make("multifilesrc", "file-source")
    if not source:
        sys.stderr.write(" Unable to create Source \n")

    # Add a JPEG decoder
    jpegdec = Gst.ElementFactory.make("jpegdec", "jpeg-decoder")
    if not jpegdec:
        sys.stderr.write(" Unable to create JPEG Decoder \n")

    videoconvert = Gst.ElementFactory.make("videoconvert", "video-convert")
    if not videoconvert:
        sys.stderr.write(" Unable to create videoconvert \n")

    # Adding nvstreammux
    streammux = Gst.ElementFactory.make("nvstreammux", "Stream-muxer")
    if not streammux:
        sys.stderr.write(" Unable to create NvStreamMux \n")

    pgie = Gst.ElementFactory.make("nvinfer", "primary-inference")
    if not pgie:
        sys.stderr.write(" Unable to create pgie \n")

    # Convertor to convert from NV12 to RGBA as required by nvosd
    nvvidconv = Gst.ElementFactory.make("nvvideoconvert", "convertor")
    if not nvvidconv:
        sys.stderr.write(" Unable to create nvvidconv \n")

    # OSD to draw on the converted RGBA buffer
    nvosd = Gst.ElementFactory.make("nvdsosd", "onscreendisplay")

    if not nvosd:
        sys.stderr.write(" Unable to create nvosd \n")

    # Render the OSD output
    if is_aarch64():
        print("Creating nv3dsink \n")
        sink = Gst.ElementFactory.make("nv3dsink", "nv3d-sink")
        if not sink:
            sys.stderr.write(" Unable to create nv3dsink \n")
    else:
        print("Creating EGLSink \n")
        sink = Gst.ElementFactory.make("nveglglessink", "nvvideo-renderer")
        if not sink:
            sys.stderr.write(" Unable to create egl sink \n")

    source.set_property("location", "../yolo-medical-mask-dataset/test/images/image_%05d.jpg")
    source.set_property("caps", Gst.Caps.from_string("image/jpeg,framerate=1/1"))
    pgie.set_property('config-file-path', "../configs/pgie_config.txt")
    streammux.set_property('width', 640)
    streammux.set_property('height', 640)
    streammux.set_property('batch-size', 1)
    streammux.set_property('batched-push-timeout', 4000000)

    # Add elements to the pipeline
    pipeline.add(source)
    pipeline.add(jpegdec)
    pipeline.add(nvvidconv)
    pipeline.add(streammux)
    pipeline.add(pgie)
    pipeline.add(nvosd)
    pipeline.add(sink)

    # Link the elements together
    if not source.link(jpegdec):
        sys.stderr.write("Elements could not be linked: source -> jpegdec.\n")
        sys.exit(1)

    if not jpegdec.link(nvvidconv):
        sys.stderr.write("Elements could not be linked: jpegdec -> nvvidconv.\n")
        sys.exit(1)

    # Link nvvidconv to streammux
    srcpad = nvvidconv.get_static_pad("src")
    sinkpad = streammux.get_request_pad("sink_0")
    if not Gst.Pad.link(srcpad, sinkpad) == Gst.PadLinkReturn.OK:
        sys.stderr.write("Failed to link nvvidconv to streammux.\n")
        sys.exit(1)

    if not streammux.link(pgie):
        sys.stderr.write("Elements could not be linked: streammux -> pgie.\n")
        sys.exit(1)

    if not pgie.link(nvosd):
        sys.stderr.write("Elements could not be linked: pgie -> nvosd.\n")
        sys.exit(1)

    if not nvosd.link(sink):
        sys.stderr.write("Elements could not be linked: nvosd -> sink.\n")
        sys.exit(1)


    loop = GLib.MainLoop()
    bus = pipeline.get_bus()
    bus.add_signal_watch()
    bus.connect("message", bus_call, loop)

    osdsinkpad = nvosd.get_static_pad("sink")
    if not osdsinkpad:
        sys.stderr.write(" Unable to get sink pad of nvosd \n")
    else:
        osdsinkpad.add_probe(Gst.PadProbeType.BUFFER, test_osd_sink_pad_buffer_probe, 0)

    pipeline.set_state(Gst.State.PLAYING)
    try:
        loop = GLib.MainLoop()
        bus = pipeline.get_bus()
        bus.add_signal_watch()
        bus.connect("message", bus_call, loop)
        loop.run()
    except:
        pass
    finally:
        pipeline.set_state(Gst.State.NULL)

    pipeline.set_state(Gst.State.NULL)
    ground_truth_labels = [parse_labels(f"../yolo-medical-mask-dataset/test/labels/image_{i:05d}.txt") for i in
                           range(len(detected_labels))]
    mAP, mean_recall, accuracy = calculate_metrics(detected_labels, ground_truth_labels)
    print(f"mAP: {mAP}, Mean Recall: {mean_recall}, Accuracy: {accuracy}")


def main():
    args = parse_args()
    if args.test:
        # Code for testing accuracy with a dataset
        run_dataset_test(args)
    else:
        # Code for real-time detection
        run_real_time_detection(args)


if __name__ == '__main__':
    sys.exit(main())
