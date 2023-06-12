
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


def build_engine(onnx_path, shape=[1, 224, 224, 3]):
    """
    This is the function to create the TensorRT engine
    Args:
       onnx_path : Path to onnx_file. 
       shape : Shape of the input of the ONNX file. 
   """
    with trt.Builder(TRT_LOGGER) as builder, builder.create_network(1) as network, builder.create_builder_config() as config, trt.OnnxParser(network, TRT_LOGGER) as parser:
        config.max_workspace_size = (256 << 20)
        with open(onnx_path, 'rb') as model:
            parser.parse(model.read())
        network.get_input(0).shape = shape
        engine = builder.build_engine(network, config)
        return engine


def save_engine(engine, file_name):
    buf = engine.serialize()
    with open(file_name, 'wb') as f:
        f.write(buf)


def load_engine(trt_runtime, plan_path):
    with open(plan_path, 'rb') as f:
        engine_data = f.read()
    engine = trt_runtime.deserialize_cuda_engine(engine_data)
    return engine


def save_onnx(model, onnx_file_path):
    test_image = torch.rand((1, 3, 224, 224)).cuda()
    # ONNX_FILE_PATH = "resnet50.onnx"
    torch.onnx.export(model, test_image, onnx_file_path, input_names=[
                      "input"], output_names=["output"], export_params=True)

    onnx_model = onnx.load(ONNX_FILE_PATH)
    # check that the model converted fine
    onnx.checker.check_model(onnx_model)


def loading_model(model_path):

    model = resnet50(weights=ResNet50_Weights.DEFAULT)

    for param in model.parameters():
        param.requires_grad = False

    model.fc = nn.Linear(2048, 2)
    model.cuda()

    checkpoint = torch.load(model_path)
    model.load_state_dict(checkpoint['state_dict'])
    return model


# main()
if __name__ == "__main__":
    trained_model_path = "../runs/traversky/2023_03_05_21_30_33/model.pth"
    ONNX_FILE_PATH = 'bin_classifier.onnx'
    tensorrt_engine_path = "engine.trt"
    TRT_LOGGER = trt.Logger(trt.Logger.WARNING)

    model = loading_model(trained_model_path)
    save_onnx(model, ONNX_FILE_PATH)

    engine_name = "resnet50.plan"
    batch_size = 1

    shape = [batch_size, 3, 256, 256]
    engine = build_engine(ONNX_FILE_PATH, shape=shape)
    save_engine(engine, engine_name)
