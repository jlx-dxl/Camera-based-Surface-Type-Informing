#!/usr/bin/env python3

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.nn.init as init
import torchvision.models as models
from torchvision.models.resnet import ResNet50_Weights

class ResNet(nn.Module):
    def __init__(self, freeze_layers=True):
        super(ResNet, self).__init__()
        # 加载预训练的 ResNet50 模型
        base_model = models.resnet50(weights=ResNet50_Weights.DEFAULT)
        
        # 如果决定冻结预训练层的权重
        if freeze_layers:
            for param in base_model.parameters():
                param.requires_grad = False
        
        # 移除原有的全连接层
        base_model.fc = nn.Identity()
        
        self.resnet = base_model

    def forward(self, x):
        # 通过 ResNet 获取特征
        features = self.resnet(x)
        return features
    
class ChannelAdjuster(nn.Module):
    def __init__(self, input_channels):
        super(ChannelAdjuster, self).__init__()
        # 定义一个卷积层，输入通道为input_channels，输出通道为3
        # 这里的kernel_size设为1，使得这个转换不会改变空间维度
        self.conv = nn.Conv2d(in_channels=input_channels, out_channels=3, kernel_size=1, stride=1, padding=0)
    
    def forward(self, x):
        # 应用卷积层，x的形状应该是(B, C, W, H)
        x = self.conv(x)
        return x

class Classifer(nn.Module):
    def __init__(self, dropout_p=0.1, N=6):
        super(Classifer, self).__init__()
        self.fc1 = nn.Linear(4096, 1024)
        init.kaiming_normal_(self.fc1.weight, mode='fan_out', nonlinearity='relu')
        init.constant_(self.fc1.bias, 0)
        self.bn1 = nn.BatchNorm1d(1024)
        self.dropout1 = nn.Dropout(dropout_p)
        
        self.fc2 = nn.Linear(1024, 256)
        init.kaiming_normal_(self.fc2.weight, mode='fan_out', nonlinearity='relu')
        init.constant_(self.fc2.bias, 0)
        self.bn2 = nn.BatchNorm1d(256)
        self.dropout2 = nn.Dropout(dropout_p)
        
        self.fc3 = nn.Linear(256, 64)
        init.kaiming_normal_(self.fc3.weight, mode='fan_out', nonlinearity='relu')
        init.constant_(self.fc3.bias, 0)
        self.bn3 = nn.BatchNorm1d(64)
        self.dropout3 = nn.Dropout(dropout_p)
        self.fc4 = nn.Linear(64, N)

    def forward(self, x):
        x = F.relu(self.bn1(self.fc1(x)))
        x = self.dropout1(x)
        x = F.relu(self.bn2(self.fc2(x)))
        x = self.dropout2(x)
        x = F.relu(self.bn3(self.fc3(x)))
        x = self.dropout3(x)
        x = F.log_softmax(self.fc4(x), dim=1)
        return x

if __name__ == "__main__":
    
    # 实例化模型
    model = ResNet(freeze_layers=True)

    # 打印模型结构
    print(model)
