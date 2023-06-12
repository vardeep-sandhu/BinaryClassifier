from torchvision.models import resnet50, ResNet50_Weights
import onnx
import torch
import os
import torch.nn as nn
import pycuda.driver as cuda
import pycuda.autoinit
import numpy as np
import tensorrt as trt
import argparse
import tensorrt as trt
import cv2


def allocate_buffers(engine, batch_size, data_type):
    """
    This is the function to allocate buffers for input and output in the device
    Args:
       engine : The path to the TensorRT engine. 
       batch_size : The batch size for execution time.
       data_type: The type of the data for input and output, for example trt.float32. 

    Output:
       h_input_1: Input in the host.
       d_input_1: Input in the device. 
       h_output_1: Output in the host. 
       d_output_1: Output in the device. 
       stream: CUDA stream.

    """

    # Determine dimensions and create page-locked memory buffers (which won't be swapped to disk) to hold host inputs/outputs.
    h_input_1 = cuda.pagelocked_empty(
        batch_size * trt.volume(engine.get_binding_shape(0)), dtype=trt.nptype(data_type))
    h_output = cuda.pagelocked_empty(
        batch_size * trt.volume(engine.get_binding_shape(1)), dtype=trt.nptype(data_type))
    # Allocate device memory for inputs and outputs.
    d_input_1 = cuda.mem_alloc(h_input_1.nbytes)

    d_output = cuda.mem_alloc(h_output.nbytes)
    # Create a stream in which to copy inputs/outputs and run inference.
    stream = cuda.Stream()
    return h_input_1, d_input_1, h_output, d_output, stream


def load_images_to_buffer(pics, pagelocked_buffer):
    preprocessed = np.asarray(pics).ravel()
    np.copyto(pagelocked_buffer, preprocessed)


def do_inference(engine, pics_1, h_input_1, d_input_1, h_output, d_output, stream, batch_size, height, width):
    """
    This is the function to run the inference
    Args:
       engine : Path to the TensorRT engine 
       pics_1 : Input images to the model.  
       h_input_1: Input in the host         
       d_input_1: Input in the device 
       h_output_1: Output in the host 
       d_output_1: Output in the device 
       stream: CUDA stream
       batch_size : Batch size for execution time
       height: Height of the output image
       width: Width of the output image

    Output:
       The list of output images

    """

    load_images_to_buffer(pics_1, h_input_1)

    with engine.create_execution_context() as context:
        # Transfer input data to the GPU.
        cuda.memcpy_htod_async(d_input_1, h_input_1, stream)

        # Run inference.
        context.execute(batch_size=1, bindings=[int(d_input_1), int(d_output)])

        # Transfer predictions back from the GPU.
        cuda.memcpy_dtoh_async(h_output, d_output, stream)
        # Synchronize the stream
        stream.synchronize()

        return h_output


def load_engine(trt_runtime, plan_path):
    with open(plan_path, 'rb') as f:
        engine_data = f.read()
    engine = trt_runtime.deserialize_cuda_engine(engine_data)
    return engine


def softmax(x):
    """Compute softmax values for each sets of scores in x."""
    e_x = np.exp(x - np.max(x))
    return e_x / e_x.sum()


if __name__ == "__main__":

    TRT_LOGGER = trt.Logger(trt.Logger.WARNING)
    trt_runtime = trt.Runtime(TRT_LOGGER)
    input_file_path = 'dataset/valid/0_y0920_x0572.png'
    serialized_plan_fp32 = "scripts/resnet50.plan"
    HEIGHT = 224
    WIDTH = 224

    image = cv2.imread(input_file_path)
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    image = cv2.resize(image, (256, 256)).astype(np.float32)
    # image = preprocess_input(image)

    engine = load_engine(trt_runtime, serialized_plan_fp32)
    h_input, d_input, h_output, d_output, stream = allocate_buffers(
        engine, 1, trt.float32)
    output = do_inference(engine, image, h_input, d_input,
                          h_output, d_output, stream, 1, HEIGHT, WIDTH)
    probabilities = softmax(output)
    print(probabilities)
