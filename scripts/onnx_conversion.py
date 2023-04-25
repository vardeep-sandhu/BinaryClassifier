import torch
import torch.nn as nn
from torchvision.models import resnet50, ResNet50_Weights
import onnx

def main():
    model = resnet50(weights=ResNet50_Weights.DEFAULT)
    
    for param in model.parameters():
        param.requires_grad = False

    model.fc = nn.Linear(2048, 2)
    model.cuda()
    
    model_path = "runs/traversky/2023_03_05_21_30_33/model.pth"
    checkpoint = torch.load(model_path)
    model.load_state_dict(checkpoint['state_dict'])
    
    test_image = torch.rand((1, 3, 224, 224)).cuda()
    ONNX_FILE_PATH = "resnet50.onnx"
    torch.onnx.export(model, test_image, ONNX_FILE_PATH, input_names=["input"], output_names=["output"], export_params=True)
 
    onnx_model = onnx.load(ONNX_FILE_PATH)
    # check that the model converted fine
    onnx.checker.check_model(onnx_model)
 
    pass
    # load the torch mdoel from file



main()