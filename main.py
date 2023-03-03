import torch
from dataloader import ImageDataset 
import torch.nn as nn
from torchvision import  transforms
from torch.utils.data import DataLoader
from torchvision.models import resnet50, ResNet50_Weights

import argparse
import os
from test import test
from train import train
from focal_traversky_loss import FocalTverskyLoss
import utils

def get_parser():
    parser = argparse.ArgumentParser(description='PyTorch Point Cloud Semantic Segmentation')
    parser.add_argument('opts', help='see config/radar_scenes/radar_scenes_stratified.yaml for all options', default=None, nargs=argparse.REMAINDER)
    parser.add_argument('--config', type=str, default='config.yaml', help='config file')
    
    args = parser.parse_args()
    assert args.config is not None
    cfg = utils.load_cfg_from_cfg_file(args.config)
    if args.opts is not None:
        cfg = utils.merge_cfg_with_args(cfg, args.opts)
    return cfg

def main():
    args = get_parser()
    name = args.criterion
    args.save_path = os.path.join(args.save_path, name)

    if not os.path.exists(args.save_path):
        os.makedirs(args.save_path)
    
    logger = utils.get_logger(output=args.save_path)
    logger.info(f"The save path is: {args.save_path}")

    if not os.path.exists(args.save_path):
        os.makedirs(args.save_path)
    
    device = "cuda" if torch.cuda.is_available() else "cpu" 
    logger.info("=> creating model ...")
    transform = transforms.Compose([
                transforms.Resize((224, 224)),
                transforms.ToTensor(),
                transforms.Normalize((0.5,), (0.5,))
                ])
    # Splitting our dataset into train, val and test with ratio [75, 15, 15]

    train_data = ImageDataset(args.data_root, transform= transform, split="train")
    val_data = ImageDataset(args.data_root, transform= transform, split="val")
    test_data = ImageDataset(args.data_root, transform= transform, split="test")

    logger.info(f"The number of samples in train, val and test are: {len(train_data)}, {len(test_data)}, {len(val_data)}")

    train_loader = DataLoader(train_data, batch_size=args.batch_size, shuffle=True)
    val_loader = DataLoader(val_data, batch_size=args.batch_size_val, shuffle=False)
    test_loader = DataLoader(test_data, batch_size=args.batch_size_test, shuffle=False)

    # Getting a pretrained ResNet model and training only the FC last layer 
    logger.info("=> creating model ...")
    logger.info("Classes: {}".format(args.classes))
    
    model = resnet50(weights=ResNet50_Weights.DEFAULT)
    
    for param in model.parameters():
        param.requires_grad = False

    model.fc = nn.Linear(2048, args.classes)
    model.cuda()
    logger.info(model)
    logger.info('#Model parameters: {}'.format(sum([x.nelement() for x in model.parameters()])))


    # Defining the optimizer and the loss criterion
    optimizer = torch.optim.Adam(model.parameters(), lr = 0.001)
    
    if args.criterion == "traversky":
        criterion = FocalTverskyLoss()
    elif args.criterion == "cross_entropy":
        criterion = nn.CrossEntropyLoss()
    else:
        raise Exception("Not a valid loss function. Check your config.")
    logger.info(f'Criterion: {args.criterion}')

    if args.weights is not None:
        if os.path.isfile(args.weights):
            logger.info("=> loading weight '{}'".format(args.weights))
            checkpoint = torch.load(args.weights)
            model.load_state_dict(checkpoint['state_dict'])
            logger.info("=> loaded weight '{}'".format(args.weights))
        else:
            logger.info("=> no weight found at '{}'".format(args.weights))

    # Training the model and evaluating on the validation test
    if args.weights is None:
        train(model, train_loader, device, val_loader, optimizer, criterion, args, logger)
        # Saving model
        filename = os.path.join(args.save_path, 'model.pth')
        torch.save({'state_dict': model.state_dict(), 'optimizer': optimizer.state_dict()}, filename)
        logger.info('Saving checkpoint to: ' + filename)
        logger.info('==>Training done!')

    # Evaluating the model on the test set
    test(model, test_loader, device, criterion, logger)
    logger.info('==>Testing done!')

if __name__ == "__main__":
    main()