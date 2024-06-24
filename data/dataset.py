#!/usr/bin/env python3

import torch
from torch.utils.data import Dataset, DataLoader
from PIL import Image
import pandas as pd
import os
import torch
from torchvision import transforms
from torchvision.datasets import ImageFolder


# class ImageDataset(Dataset):
#     def __init__(self, transform=None, set_type='train', if_resize=True):
#         script_dir = os.path.dirname(os.path.abspath(__file__))
#         if set_type == 'train':
#             csv_file_path = os.path.join(script_dir, 'index/train.csv')
#         else:
#             csv_file_path = os.path.join(script_dir, 'index/test.csv')
#         self.data = pd.read_csv(csv_file_path)
#         self.data = self.data.iloc[1:]  # 移除首行(表头)
#         self.if_resize = if_resize  

#         # 如果没有特定的转换被提供，就使用默认的转换（即转换到张量）
#         if transform is None:
#             if if_resize:
#                 self.transform_1 = transforms.Compose([
#                     transforms.Resize((224, 224)),  # 调整图像大小
#                     transforms.ToTensor(),  # 添加这行来转换图像到张量
#                 ])
#                 self.transform_2 = transforms.Compose([
#                     transforms.Resize((360, 640)),  # 调整图像大小
#                     transforms.ToTensor(),  # 添加这行来转换图像到张量
#                 ])
#             else:
#                 self.transform = transforms.Compose([
#                     transforms.ToTensor(),  # 添加这行来转换图像到张量
#                 ])
#         else:
#             self.transform = transform


#     def __len__(self):
#         return len(self.data)

#     def __getitem__(self, idx):
#         img_name = self.data.iloc[idx, 0]
#         img_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "Images", img_name)
#         gt_value = self.data.iloc[idx, 1]
        
#         img = Image.open(img_path).convert("RGB")
        
#         # 将 gt_value 转换为张量
#         gt_value = torch.tensor(gt_value, dtype=torch.float32)  # 使用 float32 类型，如果是分类任务，可能需要 torch.long


#         # 应用转换
#         if self.if_resize:
#             img_1 = self.transform_1(img)
#             img_2 = self.transform_2(img)
            
#             return img_1, img_2, gt_value
#         else:
#             img = self.transform(img)
#             return img, gt_value

class ImageDataset(Dataset):
    def __init__(self, transform=None, set_type='train', if_resize=True):
        """
        Args:
            root_dir (string): Directory with all the images, organized by class subfolders.
            transform (callable, optional): Optional transform to be applied on a sample.
        """
        script_dir = os.path.dirname(os.path.abspath(__file__))
        if set_type == 'train':
            root_dir = os.path.join(script_dir, 'train/')
        elif set_type == 'test':
            root_dir = os.path.join(script_dir, 'test/')
        elif set_type == 'dev':
            root_dir = os.path.join(script_dir, 'dev/')
        else:   
            raise ValueError("Invalid set_type. Must be one of 'train', 'test', or 'dev'.")
            
        self.root_dir = root_dir
        self.transform = transform
        self.dataset = ImageFolder(root_dir)
        
        self.if_resize = if_resize  

        # 如果没有特定的转换被提供，就使用默认的转换（即转换到张量）
        if transform is None:
            if if_resize:
                self.transform_1 = transforms.Compose([
                    transforms.Resize((224, 224)),  # ResNet的最佳输入大小
                    transforms.ToTensor(),
                ])
                self.transform_2 = transforms.Compose([
                    transforms.Resize((640, 640)),  # 
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
        label_idx = torch.argmax(one_hot_label).item()
        return label_idx
    
    def __len__(self):
        return len(self.dataset)
    
    def __getitem__(self, idx):
        image, label = self.dataset[idx]
        
        # Convert label to one-hot encoding
        one_hot_label = torch.zeros(self.num_classes)
        one_hot_label[label] = 1
        class_name = self.get_class_name(one_hot_label)
        
        if self.if_resize:
            image_1 = self.transform_1(image)
            image_2 = self.transform_2(image)
            return image_1, image_2, one_hot_label, class_name
        else:
            image = self.transform(image)
        return image, one_hot_label, class_name
    
    
if __name__ == "__main__":

    print("\ntrain:\n")

    # Create an instance of the ImageDataset class
    dataset = ImageDataset(set_type='train')
    
    # 创建 DataLoader
    dataloader = DataLoader(dataset, batch_size=20, shuffle=True, num_workers=12)

    # 使用 DataLoader
    for i,(images_1, images_2, gt_values, class_names) in enumerate(dataloader):
        # 这里可以进行你的模型训练等操作
        print(f"{i:04d}:", images_1.shape, images_2.shape, gt_values.shape)
        if i==0:
            print(gt_values, class_names)
        
    print("\ndev:\n")
    
    # Create an instance of the ImageDataset class
    dataset = ImageDataset(set_type='dev')
    
    # 创建 DataLoader
    dataloader = DataLoader(dataset, batch_size=20, shuffle=True, num_workers=12)

    # 使用 DataLoader
    for i,(images_1, images_2, gt_values, class_names) in enumerate(dataloader):
        # 这里可以进行你的模型训练等操作
        print(f"{i:04d}:", images_1.shape, images_2.shape, gt_values.shape)
        if i==0:
            print(gt_values, class_names)
        
    print("\ntest:\n")
    
    # Create an instance of the ImageDataset class
    dataset = ImageDataset(set_type='test')
    
    # 创建 DataLoader
    dataloader = DataLoader(dataset, batch_size=20, shuffle=True, num_workers=12)

    # 使用 DataLoader
    for i,(images_1, images_2, gt_values, class_names) in enumerate(dataloader):
        # 这里可以进行你的模型训练等操作
        print(f"{i:04d}:", images_1.shape, images_2.shape, gt_values.shape)
        if i==0:
            print(gt_values, class_names)
    
