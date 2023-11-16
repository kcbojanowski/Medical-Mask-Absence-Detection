#!/usr/bin/env python3

import sys

sys.path.append('../')
import gi

gi.require_version('Gst', '1.0')
from gi.repository import GLib, Gst
from common.is_aarch_64 import is_aarch64
from common.bus_call import bus_call

import pyds

PGIE_CLASS_ID_INCORRECT_MASK = 0
PGIE_CLASS_ID_WITH_MASK = 1
PGIE_CLASS_ID_WITHOUT_MASK = 2

labels_path = '../models/labels.txt'

# Bounding box options
bbox_border_color_0 = {"R": 1.0, "G": 1.0, "B": 0.0, "A": 1.0}
bbox_border_color_1 = {"R": 0.0, "G": 1.0, "B": 0.0, "A": 1.0}
bbox_border_color_2 = {"R": 1.0, "G": 0.0, "B": 0.0, "A": 1.0}
bbox_has_bg_color = False  # Bool for whether bounding box has background color

# Color of bbox background.
bbox_bg_color_0 = {"R": 1.0, "G": 1.0, "B": 0.0, "A": 0.5}
bbox_bg_color_1 = {"R": 0.0, "G": 1.0, "B": 0.0, "A": 0.5}
bbox_bg_color_2 = {"R": 1.0, "G": 0.0, "B": 0.0, "A": 0.5}


def load_class_labels(labels_path):
    with open(labels_path, 'r') as f:
        labels = [line.strip() for line in f.readlines()]
    return labels


class_labels = load_class_labels(labels_path)


def osd_sink_pad_buffer_probe(pad, info, u_data):
    print("Entering osd_sink_pad_buffer_probe")
    frame_number = 0
    # Initializing object counter with 0.
    obj_counter = {
        PGIE_CLASS_ID_INCORRECT_MASK: 0,
        PGIE_CLASS_ID_WITH_MASK: 0,
        PGIE_CLASS_ID_WITHOUT_MASK: 0,
    }
    num_rects = 0

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

        frame_number = frame_meta.frame_num
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

            # Set color of bbox
            rectparams.border_color.set(bbox_color["R"], bbox_color["G"], bbox_color["B"], bbox_color["A"])
            # Set whether bbox has background color
            rectparams.has_bg_color = bbox_has_bg_color
            # If bbox has background color, set background color
            if bbox_has_bg_color:
                rectparams.bg_color.set(bg_color["R"], bg_color["G"], bg_color["B"], bg_color["A"])

            label_name = class_labels[obj_meta.class_id] if obj_meta.class_id < len(class_labels) else "Unknown"

            # Debug print for bounding box coordinates
            print(f"Bounding Box: Left {rectparams.left}, Top {rectparams.top}, Width {rectparams.width}, Height {rectparams.height}")

            # Set text parameters
            obj_meta.text_params.display_text = f"{label_name}"
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

        # Acquiring a display meta object
        display_meta = pyds.nvds_acquire_display_meta_from_pool(batch_meta)
        display_meta.num_labels = 1
        py_nvosd_text_params = display_meta.text_params[0]

        # Setting display text to be shown on screen
        py_nvosd_text_params.display_text = f"Frame Number={frame_number} Number of Objects={num_rects} Without_mask_count={obj_counter[PGIE_CLASS_ID_WITHOUT_MASK]} Incorrect_mask_count={obj_counter[PGIE_CLASS_ID_INCORRECT_MASK]}"
        py_nvosd_text_params.x_offset = 10
        py_nvosd_text_params.y_offset = 12
        py_nvosd_text_params.font_params.font_name = "Serif"
        py_nvosd_text_params.font_params.font_size = 10
        py_nvosd_text_params.font_params.font_color.set(1.0, 1.0, 1.0, 1.0)
        py_nvosd_text_params.set_bg_clr = 1
        py_nvosd_text_params.text_bg_clr.set(0.0, 0.0, 0.0, 1.0)

        print(pyds.get_string(py_nvosd_text_params.display_text))
        pyds.nvds_add_display_meta_to_frame(frame_meta, display_meta)

        try:
            l_frame = l_frame.next
        except StopIteration:
            break

    return Gst.PadProbeReturn.OK


def main(args):
    # Check input arguments
    if len(args) != 2:
        sys.stderr.write("usage: %s <v4l2-device-path>\n" % args[0])
        sys.exit(1)

    # Standard GStreamer initialization
    Gst.init(None)

    # Create gstreamer elements
    # Create Pipeline element that will form a connection of other elements
    print("Creating Pipeline \n ")
    pipeline = Gst.Pipeline()

    if not pipeline:
        sys.stderr.write(" Unable to create Pipeline \n")

    # Source element for reading from the file
    print("Creating Source \n ")
    source = Gst.ElementFactory.make("v4l2src", "usb-cam-source")
    if not source:
        sys.stderr.write(" Unable to create Source \n")

    caps_v4l2src = Gst.ElementFactory.make("capsfilter", "v4l2src_caps")
    if not caps_v4l2src:
        sys.stderr.write(" Unable to create v4l2src capsfilter \n")

    print("Creating Video Converter \n")

    # Adding videoconvert -> nvvideoconvert as not all
    # raw formats are supported by nvvideoconvert;
    # Say YUYV is unsupported - which is the common
    # raw format for many logi usb cams
    # In case we have a camera with raw format supported in
    # nvvideoconvert, GStreamer plugins' capability negotiation
    # shall be intelligent enough to reduce compute by
    # videoconvert doing passthrough (TODO we need to confirm this)

    # videoconvert to make sure a superset of raw formats are supported
    vidconvsrc = Gst.ElementFactory.make("videoconvert", "convertor_src1")
    if not vidconvsrc:
        sys.stderr.write(" Unable to create videoconvert \n")

    # nvvideoconvert to convert incoming raw buffers to NVMM Mem (NvBufSurface API)
    nvvidconvsrc = Gst.ElementFactory.make("nvvideoconvert", "convertor_src2")
    if not nvvidconvsrc:
        sys.stderr.write(" Unable to create Nvvideoconvert \n")

    # Create capsfilter for nvvidconvsrc to output RGBA format
    nvmm_caps = Gst.ElementFactory.make("capsfilter", "nvmm_caps")
    if not nvmm_caps:
        sys.stderr.write(" Unable to create capsfilter \n")

    # Create nvstreammux instance to form batches from one or more sources.
    streammux = Gst.ElementFactory.make("nvstreammux", "Stream-muxer")
    if not streammux:
        sys.stderr.write(" Unable to create NvStreamMux \n")

    # Use nvinfer to run inferencing on camera's output,
    # behaviour of inferencing is set through config file
    pgie = Gst.ElementFactory.make("nvinfer", "primary-inference")
    if not pgie:
        sys.stderr.write(" Unable to create pgie \n")

    # Use convertor to convert from NV12 to RGBA as required by nvosd
    nvvidconv = Gst.ElementFactory.make("nvvideoconvert", "convertor")
    if not nvvidconv:
        sys.stderr.write(" Unable to create nvvidconv \n")

    # Create OSD to draw on the converted RGBA buffer
    nvosd = Gst.ElementFactory.make("nvdsosd", "onscreendisplay")

    if not nvosd:
        sys.stderr.write(" Unable to create nvosd \n")

    # Finally render the osd output
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

    print("Playing cam %s " % args[1])
    caps_v4l2src.set_property('caps', Gst.Caps.from_string("video/x-raw, framerate=15/1"))
    caps = Gst.caps_from_string("video/x-raw(memory:NVMM), format=(string)RGBA")
    nvmm_caps.set_property("caps", caps)
    source.set_property('device', args[1])
    streammux.set_property('gpu-id', 0)
    streammux.set_property('width', 1920)
    streammux.set_property('height', 1080)
    streammux.set_property('batch-size', 1)
    streammux.set_property('live-source', 1)
    streammux.set_property('batched-push-timeout', 4000000)
    pgie.set_property('config-file-path', "../configs/pgie_config.txt")
    # Set sync = false to avoid late frame drops at the display-sink
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

    # create an event loop and feed gstreamer bus mesages to it
    loop = GLib.MainLoop()
    bus = pipeline.get_bus()
    bus.add_signal_watch()
    bus.connect("message", bus_call, loop)

    # Lets add probe to get informed of the meta data generated, we add probe to
    # the sink pad of the osd element, since by that time, the buffer would have
    # had got all the metadata.
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

if __name__ == '__main__':
    sys.exit(main(sys.argv))