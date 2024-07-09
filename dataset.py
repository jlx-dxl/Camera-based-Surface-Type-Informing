#!/usr/bin/env python3

import torch
from torch.utils.data import Dataset, DataLoader
from PIL import Image
import pandas as pd
import os
import torch
from torchvision import transforms
from torchvision.datasets import ImageFolder

from util import *

class ImageDataset(Dataset):
    def __init__(self, transform=None, set_type='train', if_resize=True):
        """
        Args:
            transform (callable, optional): Optional transform to be applied on a sample.
            set_type (string): Type of the dataset, should be 'train', 'test', or 'dev'.
            if_resize (bool): Whether to resize images to 224x224.
        """
        script_dir = os.path.dirname(os.path.abspath(__file__))
        if set_type == 'train':
            root_dir = os.path.join(script_dir, 'data/train/')
        elif set_type == 'test':
            root_dir = os.path.join(script_dir, 'data/test/')
        elif set_type == 'dev':
            root_dir = os.path.join(script_dir, 'data/dev/')
        else:   
            raise ValueError("Invalid set_type. Must be one of 'train', 'test', or 'dev'.")
            
        self.root_dir = root_dir
        self.transform = transform
        self.dataset = ImageFolder(root_dir)
        
        self.if_resize = if_resize  

        # If no specific transform is provided, use default transform (i.e., convert to tensor)
        if transform is None:
            if if_resize:
                self.transform_1 = transforms.Compose([
                    transforms.Resize((224, 224)),  # Optimal input size for ResNet
                    transforms.ToTensor(),
                ])
                self.transform_2 = transforms.Compose([
                    transforms.Resize((224, 224)),  # Optimal input size for ResNet
                    transforms.ToTensor(),
                ])
            else:
                self.transform = transforms.Compose([
                    transforms.ToTensor(),
                ])
        else:
            self.transform = transform
        
        # Get the number of classes
        self.num_classes = len(self.dataset.classes)
        self.class_to_idx = self.dataset.class_to_idx
        self.idx_to_class = {v: k for k, v in self.class_to_idx.items()}

    def get_class_name(self, one_hot_label):
        """
        Returns the class name given a one-hot encoded label.
        
        Args:
            one_hot_label (torch.Tensor): One-hot encoded label tensor.
        
        Returns:
            int: Index of the class.
        """
        label_idx = torch.argmax(one_hot_label).item()
        return label_idx
    
    def __len__(self):
        return len(self.dataset)
    
    def __getitem__(self, idx):
        """
        Returns the sample corresponding to the index.
        
        Args:
            idx (int): Index of the sample.
        
        Returns:
            tuple: (image, GLCM textures, one-hot label, class name) if resizing, else (image, one-hot label, class name).
        """
        image, label = self.dataset[idx]
        
        # Convert label to one-hot encoding
        one_hot_label = torch.zeros(self.num_classes)
        one_hot_label[label] = 1
        class_name = self.get_class_name(one_hot_label)
        
        if self.if_resize:
            image_1 = self.transform_1(image)
            image_2 = self.transform_2(image)
            
            # print(image_2.shape)
            
            gray_img = tensor_to_grayscale(image_2)
            
            glcm_textures = batch_glcm(gray_img)
            
            return image_1, glcm_textures, one_hot_label, class_name
        else:
            image = self.transform(image)
        return image, one_hot_label, class_name
    
    
if __name__ == "__main__":

    print("\ntrain:\n")

    # Create an instance of the ImageDataset class
    dataset = ImageDataset(set_type='train')
    
    # Create DataLoader
    dataloader = DataLoader(dataset, batch_size=20, shuffle=True, num_workers=12)

    # Use DataLoader
    for i, (images_1, images_2, gt_values, class_names) in enumerate(dataloader):
        # Perform your model training operations here
        print(f"{i:04d}:", images_1.shape, images_2.shape, gt_values.shape)
        if i == 0:
            print(gt_values, class_names)
        
    print("\ndev:\n")
    
    # Create an instance of the ImageDataset class
    dataset = ImageDataset(set_type='dev')
    
    # Create DataLoader
    dataloader = DataLoader(dataset, batch_size=20, shuffle=True, num_workers=12)

    # Use DataLoader
    for i, (images_1, images_2, gt_values, class_names) in enumerate(dataloader):
        # Perform your model training operations here
        print(f"{i:04d}:", images_1.shape, images_2.shape, gt_values.shape)
        if i == 0:
            print(gt_values, class_names)
        
    print("\ntest:\n")
    
    # Create an instance of the ImageDataset class
    dataset = ImageDataset(set_type='test')
    
    
    # Create DataLoader
    dataloader = DataLoader(dataset, batch_size=20, shuffle=True, num_workers=12)

    # Use DataLoader
    for i, (images_1, images_2, gt_values, class_names) in enumerate(dataloader):
        # Perform your model training operations here
        print(f"{i:04d}:", images_1.shape, images_2.shape, gt_values.shape)
        if i == 0:
            print(gt_values, class_names)
