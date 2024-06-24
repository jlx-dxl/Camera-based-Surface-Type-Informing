#!/usr/bin/env python3

import torch
import torch.nn as nn
import torchvision.models as models
from torch.utils.data import DataLoader
import torch.optim as optim
from torch.optim.lr_scheduler import ExponentialLR
from tqdm import tqdm
import argparse
import wandb

from data.dataset import ImageDataset
from model import ResNet, Classifer
from util import *


def test_one_epoch(model, criterion, dataloader):
        
    model.eval()
    resnet = ResNet(freeze_layers=True)
    resnet.eval()
    
    total_loss = 0
    total_acc = 0

    with torch.no_grad():
        for inputs_res, inputs_glcm, labels, class_gt in tqdm(dataloader):
            inputs_res = inputs_res
            inputs_glcm = inputs_glcm
            labels = labels
            
            features_res = resnet(inputs_res)
            features_glcm = resnet(inputs_glcm)
            input = torch.cat((features_res, features_glcm), dim=1)
            outputs = model(input)
            outputs = outputs.squeeze(1)
            
            class_estimation = get_class_name(outputs)
            acc = torch.sum(class_gt == class_estimation).item() / class_gt.size(0)
            total_acc += acc
            
            loss = criterion(outputs, labels)
            # print(f"No:{i:3d}, Estimated: {class_estimation.item():1d}, Ground Truth: {class_gt.item():1d}")
            total_loss += loss.item() * input.size(0)
        
    return total_loss / len(dataloader.dataset), total_acc / dataloader.__len__()


def main():
        
    # 创建 DataLoader
    test_dataloader = DataLoader(ImageDataset(set_type='test'), batch_size=1, shuffle=True, num_workers=12)
    
    model = Classifer()
    
    criterion = nn.CrossEntropyLoss()  
    
    model.load_state_dict(torch.load(os.path.join('model','train0623-2','best.pth')))

    test_loss, test_acc = test_one_epoch(model, criterion, test_dataloader)
    
    print(f"Final test Loss: {test_loss:.4f}, Final test Accuracy: {test_acc*100:.4f}%")
    
    
if __name__ == "__main__":
    main()
    