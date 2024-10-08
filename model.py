#!/usr/bin/env python3

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.nn.init as init
import torchvision.models as models
from torchvision.models.resnet import ResNet50_Weights, ResNet18_Weights
# import torch_tensorrt as trt

class ResNet50(nn.Module):
    def __init__(self, freeze_layers=True):
        """
        Initializes the ResNet50 model.

        Args:
            freeze_layers (bool): If True, freeze the pretrained layers.
        """
        super(ResNet50, self).__init__()
        # Load the pretrained ResNet50 model
        base_model = models.resnet50(weights=ResNet50_Weights.DEFAULT)
        
        # If freeze_layers is True, freeze the pretrained layers' weights
        if (freeze_layers):
            for param in base_model.parameters():
                param.requires_grad = False
        
        # Remove the original fully connected layer
        base_model.fc = nn.Identity()
        
        self.resnet = base_model

    def forward(self, x):
        """
        Forward pass to extract features from the input.

        Args:
            x (torch.Tensor): Input tensor.

        Returns:
            torch.Tensor: Extracted features.
        """
        features = self.resnet(x)
        return features
    

class ResNet18(nn.Module):
    def __init__(self, freeze_layers=True):
        """
        Initializes the ResNet18 model.

        Args:
            freeze_layers (bool): If True, freeze the pretrained layers.
        """
        super(ResNet18, self).__init__()
        # Load the pretrained ResNet18 model
        base_model = models.resnet18(weights=ResNet18_Weights.DEFAULT)
        
        # If freeze_layers is True, freeze the pretrained layers' weights
        if (freeze_layers):
            for param in base_model.parameters():
                param.requires_grad = False
        
        # Remove the original fully connected layer
        base_model.fc = nn.Identity()
        
        self.resnet = base_model

    def forward(self, x):
        """
        Forward pass to extract features from the input.

        Args:
            x (torch.Tensor): Input tensor.

        Returns:
            torch.Tensor: Extracted features.
        """
        features = self.resnet(x)
        return features


class VisionInform50(nn.Module):
    def __init__(self, dropout_p=0.1, N=6):
        """
        Initializes the Classifer50 model with fully connected layers.

        Args:
            dropout_p (float): Dropout probability.
            N (int): Number of output classes.
        """
        super(VisionInform50, self).__init__()

        self.resnet1 = ResNet50(freeze_layers=True)

        self.resnet2 = ResNet50(freeze_layers=False)

        self.fc0 = nn.Linear(4096, 2048)
        init.kaiming_normal_(self.fc0.weight, mode='fan_out', nonlinearity='relu')
        init.constant_(self.fc0.bias, 0)
        self.bn0 = nn.BatchNorm1d(2048)
        self.dropout0 = nn.Dropout(dropout_p)

        self.fc1 = nn.Linear(2048, 1024)
        init.kaiming_normal_(self.fc1.weight, mode='fan_out', nonlinearity='relu')
        init.constant_(self.fc1.bias, 0)
        self.bn1 = nn.BatchNorm1d(1024)
        self.dropout1 = nn.Dropout(dropout_p)
        
        self.fc2 = nn.Linear(1024, 512)
        init.kaiming_normal_(self.fc2.weight, mode='fan_out', nonlinearity='relu')
        init.constant_(self.fc2.bias, 0)
        self.bn2 = nn.BatchNorm1d(512)
        self.dropout2 = nn.Dropout(dropout_p)
        
        self.fc3 = nn.Linear(512, 256)
        init.kaiming_normal_(self.fc3.weight, mode='fan_out', nonlinearity='relu')
        init.constant_(self.fc3.bias, 0)
        self.bn3 = nn.BatchNorm1d(256)
        self.dropout3 = nn.Dropout(dropout_p)

        self.fc4 = nn.Linear(256, 128)
        init.kaiming_normal_(self.fc4.weight, mode='fan_out', nonlinearity='relu')
        init.constant_(self.fc4.bias, 0)
        self.bn4 = nn.BatchNorm1d(128)
        self.dropout4 = nn.Dropout(dropout_p)

        self.fc5 = nn.Linear(128, 64)
        init.kaiming_normal_(self.fc5.weight, mode='fan_out', nonlinearity='relu')
        init.constant_(self.fc5.bias, 0)
        self.bn5 = nn.BatchNorm1d(64)
        self.dropout5 = nn.Dropout(dropout_p)

        self.fc6 = nn.Linear(64, N)

    def latent(self, x, GLCM_x):
        """
        Forward pass through the fully connected layers.

        Args:
            x (torch.Tensor): Input tensor.

        Returns:
            torch.Tensor: Output tensor with log probabilities.
        """
        x = self.resnet1(x)
        GLCM_x = self.resnet2(GLCM_x)

        x = torch.cat((x, GLCM_x), dim=1)

        x = F.relu(self.bn0(self.fc0(x)))
        x = self.dropout0(x)
        x = F.relu(self.bn1(self.fc1(x)))
        x = self.dropout1(x)
        x = F.relu(self.bn2(self.fc2(x)))
        x = self.dropout2(x)
        x = F.relu(self.bn3(self.fc3(x)))
        x = self.dropout3(x)
        x = F.relu(self.bn4(self.fc4(x)))
        x = self.dropout4(x)
        x = F.relu(self.bn5(self.fc5(x)))
        x = self.dropout5(x)

        return x

    def forward(self, x, GLCM_x):
        """
        Forward pass through the fully connected layers.

        Args:
            x (torch.Tensor): Input tensor.
            GLCM_x (torch.Tensor): Input tensor with GLCM features.

        Returns:
            torch.Tensor: Output tensor with log probabilities.
        """
        x = self.latent(x, GLCM_x)
        x = F.log_softmax(self.fc6(x), dim=1)

        return x
    
class Classifer18(nn.Module):
    def __init__(self, dropout_p=0.1, N=6):
        """
        Initializes the Classifer18 model with fully connected layers.

        Args:
            dropout_p (float): Dropout probability.
            N (int): Number of output classes.
        """
        super(Classifer18, self).__init__()
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
        """
        Forward pass through the fully connected layers.

        Args:
            x (torch.Tensor): Input tensor.

        Returns:
            torch.Tensor: Output tensor with log probabilities.
        """
        x = F.relu(self.bn2(self.fc2(x)))
        x = self.dropout2(x)
        x = F.relu(self.bn3(self.fc3(x)))
        x = self.dropout3(x)
        x = F.log_softmax(self.fc4(x), dim=1)
        return x

if __name__ == "__main__":
    # Instantiate the ResNet18 model with frozen layers
    model = ResNet18(freeze_layers=True).cuda()
    model.eval()
    
    # Use TorchScript to accelerate the model
    example_input = torch.randn(1, 3, 224, 224).cuda()
    traced_model = torch.jit.trace(model, example_input)
    
    # Save the traced model
    traced_model.save("model/resnet18_traced.pt")
    
    print("TorchScript accelerated model has been saved")
    
    # # Use TensorRT to accelerate the model
    # # Convert the model to TorchScript
    # example_input = torch.randn(1, 3, 224, 224).cuda()
    # scripted_model = torch.jit.trace(model, example_input).eval().cuda()

    # # Convert to TensorRT engine
    # trt_model = trt.compile(scripted_model, inputs=[trt.Input(example_input.shape)], enabled_precisions={torch.float, torch.half})

    # # Save the TensorRT engine
    # with open("resnet18_trt.engine", "wb") as f:
    #     f.write(trt_model.engine.serialize())

    # print("TensorRT engine has been saved")
