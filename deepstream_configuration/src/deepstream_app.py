import sys
import os
import configparser
import math
import scipy
import numpy as np
from IPython.display import Video

import gi

gi.require_version('Gst', '1.0')
from gi.repository import GLib, Gst

import pyds


# Declare class label IDs
PGIE_CLASS_ID_INCORRECT_MASK = 0
PGIE_CLASS_ID_WITH_MASK = 1
PGIE_CLASS_ID_WITHOUT_MASK = 2
#%%
# Tiler properties and OSC properties
TILED_OUTPUT_WIDTH = 1920  # Tiler output width
TILED_OUTPUT_HEIGHT = 1080  # Tiler output height

# NvOSD options
OSD_PROCESS_MODE = 1
OSD_DISPLAY_TEXT= 1

## File paths ##
input_src = 'v4l2src device=/dev/video0'

output_file = "output.mp4"  # Output file location
pgie_config_file = "./configs/dslaunchpad_pgie_config.txt"  # Path to pgie config file
tracker_config_file = "./configs/dslaunchpad_tracker_config.txt"  # Path to tracker config file
sgie1_config_file = "./configs/dslaunchpad_sgie1_config.txt"  # Path to config file for first sgie

# Tracker options
enable_tracker = 1  # Enable/disable tracker and SGIEs. 0 for disable, 1 for enable

## Bounding box options ##
bbox_border_color_0 = {"R": 0.0, "G": 0.5, "B": 0.5, "A": 1.0}
bbox_border_color_1 = {"R": 0.0, "G": 1.0, "B": 0.0, "A": 1.0}
bbox_border_color_2 = {"R": 1.0, "G": 0.0, "B": 0.0, "A": 1.0}
bbox_has_bg_color = False # Bool for whether bounding box has background color

# Color of bbox background.
bbox_bg_color_0 = {"R": 0.0, "G": 0.5, "B": 0.5, "A": 0.2}
bbox_bg_color_1 = {"R": 0.0, "G": 1.0, "B": 0.0, "A": 0.2}
bbox_bg_color_2 = {"R": 1.0, "G": 0.0, "B": 0.0, "A": 0.2}

# Display text options, to be added to the frame.
text_x_offset = 10 # Offset in the x direction where string should appear
text_y_offset = 12 # Offset in the y direction where string should appear
text_font_name = "Serif" # Font name
text_font_size = 10 # Font size
text_font_color = {"R" : 1.0, "G": 1.0, "B": 1.0, "A": 1.0} # Color of text font. Set to white
text_set_bg_color = True # Bool for whether text box has background color
text_bg_color = {"R": 0.0, "G": 0.0, "B": 0.0, "A": 1.0} # Color of text box background. Set to black
#%%
def tiler_sink_pad_buffer_probe(pad, info, u_data):
    frame_number = 0
    # Initialize object counter with 0.
    obj_counter = {
        PGIE_CLASS_ID_INCORRECT_MASK: 0,
        PGIE_CLASS_ID_WITH_MASK: 0,
        PGIE_CLASS_ID_WITHOUT_MASK: 0
    }
    num_rects = 0
    gst_buffer = info.get_buffer()
    if not gst_buffer:
        print("Unable to get GstBuffer ")
        return Gst.PadProbeReturn.OK

    # Retrieve batch metadata from the gst_buffer
    batch_meta = pyds.gst_buffer_get_nvds_batch_meta(hash(gst_buffer))
    l_frame = batch_meta.frame_meta_list
    while l_frame is not None:
        try:
            frame_meta = pyds.NvDsFrameMeta.cast(l_frame.data)
        except StopIteration:
            break

        frame_number = frame_meta.frame_num
        num_rects = frame_meta.num_obj_meta
        l_obj = frame_meta.obj_meta_list
        while l_obj is not None:
            try:
                obj_meta = pyds.NvDsObjectMeta.cast(l_obj.data)
            except StopIteration:
                break
            obj_counter[obj_meta.class_id] += 1  # Increment object counter
            rectparams = obj_meta.rect_params  # Retrieve bounding box parameters

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
            else:  # Default color for any other class that might be added later
                bbox_color = {"R": 1.0, "G": 1.0, "B": 1.0, "A": 1.0}
                bg_color = {"R": 0.5, "G": 0.5, "B": 0.5, "A": 0.2}

            # Set color of bbox
            rectparams.border_color.set(bbox_color["R"], bbox_color["G"], bbox_color["B"], bbox_color["A"])
            # Set whether bbox has background color
            rectparams.has_bg_color = bbox_has_bg_color
            # If bbox has background color, set background color
            if bbox_has_bg_color:
                rectparams.bg_color.set(bg_color["R"], bg_color["G"], bg_color["B"], bg_color["A"])
            try:
                l_obj = l_obj.next
            except StopIteration:
                break

        # Acquiring a display meta object. The memory ownership remains in
        # the C code so downstream plugins can still access it. Otherwise
        # the garbage collector will claim it when this probe function exits.
        display_meta=pyds.nvds_acquire_display_meta_from_pool(batch_meta)
        display_meta.num_labels = 1
        py_nvosd_text_params = display_meta.text_params[0]
        # Setting display text to be shown on screen
        # Note that the pyds module allocates a buffer for the string, and the
        # memory will not be claimed by the garbage collector.
        # Reading the display_text field here will return the C address of the
        # allocated string. Use pyds.get_string() to get the string content.
        py_nvosd_text_params.display_text = "Frame Number={} Number of Objects={} Mask_Worn_Incorrectly_count={} Without_Mask_count={}".format(frame_number, num_rects, obj_counter[PGIE_CLASS_ID_INCORRECT_MASK], obj_counter[PGIE_CLASS_ID_WITHOUT_MASK])

        # Now set the offsets where the string should appear
        py_nvosd_text_params.x_offset = text_x_offset
        py_nvosd_text_params.y_offset = text_y_offset

        # Font, font-color and font-size
        py_nvosd_text_params.font_params.font_name = text_font_name
        py_nvosd_text_params.font_params.font_size = text_font_size
        # set(red, green, blue, alpha); set to White
        py_nvosd_text_params.font_params.font_color.set(text_font_color["R"], text_font_color["G"], text_font_color["B"], text_font_color["A"])

        # Text background color
        py_nvosd_text_params.set_bg_clr = text_set_bg_color
        # set(red, green, blue, alpha); set to Black
        py_nvosd_text_params.text_bg_clr.set(text_bg_color["R"], text_bg_color["G"], text_bg_color["B"], text_bg_color["A"])
        # Using pyds.get_string() to get display_text as string
        print(pyds.get_string(py_nvosd_text_params.display_text))
        # Add the display meta to the frame
        pyds.nvds_add_display_meta_to_frame(frame_meta, display_meta)

        try:
            l_frame = l_frame.next
        except StopIteration:
            break

    return Gst.PadProbeReturn.OK
#%%
def cb_newpad(decodebin, decoder_src_pad, data):
    print("In cb_newpad")
    caps = decoder_src_pad.get_current_caps()
    structure = caps.get_structure(0)
    name = structure.get_name()
    print("Pad name:", name)
    if 'video' in name:
        source_bin = data
        bin_ghost_pad = source_bin.get_static_pad('src')
        if not bin_ghost_pad.set_target(decoder_src_pad):
            sys.stderr.write("Failed to link decoder src pad to source bin ghost pad\n")

def decodebin_child_added(child_proxy, Object, name, user_data):
    print("Decodebin child added:", name)
    if 'source' in name:
        source_element = child_proxy.get_by_name('source')
        if source_element and source_element.find_property('drop-on-latency'):
            source_element.set_property('drop-on-latency', True)

def create_source_bin(index, uri):
    print("Creating source bin")
    bin_name = "source-bin-%02d" % index
    nbin = Gst.Bin.new(bin_name)
    if not nbin:
        sys.stderr.write("Unable to create source bin\n")

    if uri.startswith("v4l2://"):
        source = Gst.ElementFactory.make("v4l2src", "usb-source")
        if not source:
            sys.stderr.write("Unable to create V4L2 source element\n")
        device_path = uri.replace("v4l2://", "")
        source.set_property("device", device_path)
    elif uri.startswith("rtsp://"):
        source = Gst.ElementFactory.make("uridecodebin", "uri-decode-bin")
        if not source:
            sys.stderr.write("Unable to create uridecodebin for RTSP source\n")
        source.set_property("uri", uri)
        source.connect("pad-added", cb_newpad, nbin)
        source.connect("child-added", decodebin_child_added, nbin)
    else:
        sys.stderr.write("Unsupported URI format\n")
        return None

    Gst.Bin.add(nbin, source)

    if uri.startswith("v4l2://"):
        pad = source.get_static_pad("src")
        if not pad:
            sys.stderr.write("Failed to get src pad from V4L2 source\n")
            return None
        ghost_pad = Gst.GhostPad.new("src", pad)
        if not ghost_pad:
            sys.stderr.write("Failed to create ghost pad for V4L2 source\n")
            return None
        ghost_pad.set_active(True)
        nbin.add_pad(ghost_pad)
    else:
        bin_pad = Gst.GhostPad.new_no_target("src", Gst.PadDirection.SRC)
        if not bin_pad:
            sys.stderr.write("Failed to add ghost pad in source bin\n")
            return None
        nbin.add_pad(bin_pad)

    return nbin

# Example usage
# create_source_bin(0, "v4l2:///dev/video0")  # For a USB camera
# create_source_bin(1, "rtsp://user:pass@ip:port/path")  # For a RTSP stream
#%%
def bus_call(bus, message, loop):
    t = message.type
    if t == Gst.MessageType.EOS:
        sys.stdout.write("End-of-stream\n")
        loop.quit()
    elif t==Gst.MessageType.WARNING:
        err, debug = message.parse_warning()
        sys.stderr.write("Warning: %s: %s\n" % (err, debug))
    elif t == Gst.MessageType.ERROR:
        err, debug = message.parse_error()
        sys.stderr.write("Error: %s: %s\n" % (err, debug))
        loop.quit()
    return True
#%%
def build_and_run_pipeline():
    # Initialize GStreamer
    Gst.init(None)

    print("Creating Pipeline \n")
    pipeline = Gst.Pipeline()

    if not pipeline:
        sys.stderr.write("Unable to create Pipeline\n")

    print("Creating Source \n")
    source = Gst.ElementFactory.make("v4l2src", "usb-source")
    if not source:
        sys.stderr.write("Unable to create Source\n")
    else:
        source.set_property("device", "/dev/video0")

    # Set the capsfilter
    caps = Gst.ElementFactory.make("capsfilter", "filter")
    if not caps:
        sys.stderr.write("Unable to create caps filter\n")

    # This capsfilter will pass through the frames from the camera as they are
    caps.set_property("caps", Gst.Caps.from_string("video/x-raw, format=(string)I420"))

    print("Creating nvinfer for YOLOv8 \n")
    # Here you set up the inference engine for your custom YOLOv8 model.
    pgie = Gst.ElementFactory.make("nvinfer", "primary-inference")
    if not pgie:
        sys.stderr.write("Unable to create pgie\n")

    # Set the config file property to the path where your YOLOv8 TensorRT model config file is located.
    pgie.set_property('config-file-path', "yolov8_config_file.txt")

    print("Creating nvosd \n")
    # On-screen display for drawing bounding boxes.
    nvosd = Gst.ElementFactory.make("nvdsosd", "nv-onscreendisplay")
    if not nvosd:
        sys.stderr.write("Unable to create nvosd\n")

    print("Creating Video Converter \n")
    # Convert video to a format suitable for the encoder or display
    nvvidconv = Gst.ElementFactory.make("nvvideoconvert", "nv-video-converter")
    if not nvvidconv:
        sys.stderr.write("Unable to create nvvidconv\n")

    print("Creating Encoder \n")
    # Use an encoder and parser suitable for your use case (e.g., x264enc, omxh264enc for Raspberry Pi, etc.)
    encoder = Gst.ElementFactory.make("avenc_mpeg4", "encoder")
    if not encoder:
        sys.stderr.write("Unable to create encoder\n")

    print("Creating Parser \n")
    parser = Gst.ElementFactory.make("mpeg4videoparse", "parser")
    if not parser:
        sys.stderr.write("Unable to create parser\n")

    print("Creating Mux \n")
    # Use a container that suits your needs, mp4mux for .mp4, etc.
    mux = Gst.ElementFactory.make("mp4mux", "mux")
    if not mux:
        sys.stderr.write("Unable to create mux\n")

    print("Creating Sink \n")
    # Replace the filesink with an appsink or a network sink (tcpserversink/udpsink) to stream to a web application
    sink = Gst.ElementFactory.make("appsink", "sink")
    if not sink:
        sys.stderr.write("Unable to create sink\n")

    # If you're using appsink to capture frames directly in your application, configure it here.
    # If you're streaming via network, set the tcpserversink/udpsink properties (host, port, etc.)

    print("Playing video \n")
    pipeline.add(source)
    pipeline.add(caps)
    pipeline.add(pgie)
    pipeline.add(nvvidconv)
    pipeline.add(nvosd)
    pipeline.add(encoder)
    pipeline.add(parser)
    pipeline.add(mux)
    pipeline.add(sink)

    # Link the elements together
    source.link(caps)
    caps.link(pgie)
    pgie.link(nvvidconv)
    nvvidconv.link(nvosd)
    nvosd.link(encoder)
    encoder.link(parser)
    parser.link(mux)
    mux.link(sink)

    # Create an event loop and feed GStreamer bus messages to it
    loop = GLib.MainLoop()
    bus = pipeline.get_bus()
    bus.add_signal_watch()
    bus.connect("message", bus_call, loop)

    # Start the pipeline
    pipeline.set_state(Gst.State.PLAYING)
    try:
        loop.run()
    except:
        pass

    # Clean up
    pipeline.set_state(Gst.State.NULL)
#%%
output_file = "output2.mp4"
build_and_run_pipeline()
#%%
Video("output2.mp4")