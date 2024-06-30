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
import time

from data.dataset import ImageDataset
from model import ResNet18, Classifer18
from util import *

def record_time_once(last):
    now = time.time()
    diff = now - last
    # print(f"Time: {diff:.4f}")
    return now


def test_one_epoch(model, criterion, dataloader):
        
    model.eval()
    resnet18 = ResNet18(freeze_layers=True)
    resnet18.eval()
    
    total_loss = 0
    total_acc = 0

    with torch.no_grad():
        for inputs_res, inputs_glcm, labels, class_gt in tqdm(dataloader):
            
            start = time.time()
            
            features_res = resnet18(inputs_res)
            time1 = record_time_once(start)
            
            # glcm提取计算灰度图像的纹理特征
            gray_imgs = tensor_to_grayscale_list(inputs_glcm)
            batch_result = batch_glcm(gray_imgs)
            # 对灰度图像的纹理特征进行特征提取
            features_glcm = resnet18(batch_result)
            time2 = record_time_once(time1)
            
            input = torch.cat((features_res, features_glcm), dim=1)
            # print(f"input shape: {input.shape}")
            
            outputs = model(input)
            outputs = outputs.squeeze(1)
            time3 = record_time_once(time2)
            
            class_estimation = get_class_name(outputs)
            acc = torch.sum(class_gt == class_estimation).item() / class_gt.size(0)
            total_acc += acc
            
            loss = criterion(outputs, labels)
            # print(f"No:{i:3d}, Estimated: {class_estimation.item():1d}, Ground Truth: {class_gt.item():1d}")
            total_loss += loss.item() * input.size(0)
            time4 = record_time_once(time3)
        
    return total_loss / len(dataloader.dataset), total_acc / dataloader.__len__()


def main():
        
    # 创建 DataLoader
    test_dataloader = DataLoader(ImageDataset(set_type='test'), batch_size=1, shuffle=True, num_workers=12)
    
    model = Classifer18()
    
    criterion = nn.CrossEntropyLoss()  
    
    model.load_state_dict(torch.load(os.path.join('model','train0630-Res18-1','best.pth')))

    test_loss, test_acc = test_one_epoch(model, criterion, test_dataloader)
    
    print(f"Final test Loss: {test_loss:.4f}, Final test Accuracy: {test_acc*100:.4f}%")
    
    
if __name__ == "__main__":
    main()
    