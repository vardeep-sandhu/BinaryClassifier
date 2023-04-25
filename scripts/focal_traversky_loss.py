import torch.nn.functional as F
import torch.nn as nn
import torch
#PyTorch
# These hyperparams should be adjusted cuz this loss has potential. 
# From the paper. 

ALPHA = 0.5
BETA = 0.5
GAMMA = 4/3

class FocalTverskyLoss(nn.Module):

    def __init__(self, weight=None, size_average=True):
        super(FocalTverskyLoss, self).__init__()

    def forward(self, inputs, targets, smooth=1, alpha=ALPHA, beta=BETA, gamma=GAMMA):
        
        inputs = torch.softmax(inputs, dim=1)[:, 1]

        #flatten label and prediction tensors
        inputs = inputs.view(-1)
        targets = targets.view(-1)
        
        #True Positives, False Positives & False Negatives
        TP = (inputs * targets).sum()

        FP = ((1-targets) * inputs).sum()
        FN = (targets * (1-inputs)).sum()
        Tversky = (TP + smooth) / (TP + alpha*FP + beta*FN + smooth)  
        FocalTversky = (1 - Tversky)**gamma
                       
        return FocalTversky