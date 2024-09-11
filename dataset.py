import os
import torch
from torch.utils.data import Dataset
from torchvision import transforms
from torchvision.datasets import ImageFolder
import numpy as np
from util import fast_glcm_contrast, fast_glcm_homogeneity, fast_glcm_ASM


class VisionInformDataset(Dataset):
    def __init__(self, transform=None, set_type='train', if_GLCM=False):
        """
        Args:
            transform (callable, optional): Optional transform to be applied on a sample.
            set_type (string): Type of the dataset, should be 'train' or 'test'.
            if_GLCM (bool): If True, add GLCM features as extra channels.
        """
        script_dir = os.path.dirname(os.path.abspath(__file__))
        if set_type == 'train':
            root_dir = os.path.join(script_dir, 'data/train/')
        elif set_type == 'test':
            root_dir = os.path.join(script_dir, 'data/test/')
        else:
            raise ValueError("Invalid set_type. Must be one of 'train' or 'test'.")

        self.root_dir = root_dir
        self.transform = transform
        self.dataset = ImageFolder(root_dir)
        self.if_GLCM = if_GLCM

        if transform is None:
            self.transform = transforms.Compose([
                transforms.Resize((224, 224)),  # Resize to match input size of models like ResNet
                transforms.ToTensor(),
            ])
        else:
            self.transform = transform

        self.num_classes = len(self.dataset.classes)
        self.class_to_idx = self.dataset.class_to_idx
        self.idx_to_class = {v: k for k, v in self.class_to_idx.items()}

    def __len__(self):
        return len(self.dataset)

    def compute_glcm_features(self, gray_image):
        """
        计算灰度图像的 GLCM 特征 (对比度、同质性、能量)，并返回3个通道的特征图。
        """
        contrast = fast_glcm_contrast(gray_image, vmin=0, vmax=255, levels=16, ks=7)
        homogeneity = fast_glcm_homogeneity(gray_image, vmin=0, vmax=255, levels=16, ks=7)
        _, energy = fast_glcm_ASM(gray_image, vmin=0, vmax=255, levels=16, ks=7)

        # 将这些特征组合为 [3, H, W] 的张量
        glcm_features = np.stack([contrast, homogeneity, energy], axis=0)

        return torch.from_numpy(glcm_features).float()

    def __getitem__(self, idx):
        # 获取图像和类别标签
        img, class_idx = self.dataset[idx]

        # 应用图像转换
        img = self.transform(img)

        # 将类别标签转换为 one-hot 编码
        one_hot_label = torch.zeros(self.num_classes)
        one_hot_label[class_idx] = 1.0

        if self.if_GLCM:
            # 将图像转换为灰度图并计算 GLCM 特征
            img_gray = img.mean(dim=0).numpy().astype(np.uint8)  # 将 RGB 图像转换为灰度
            glcm_features = self.compute_glcm_features(img_gray)

            # 返回图像、GLCM特征、one-hot标签、class_idx
            return img, glcm_features, one_hot_label, class_idx
        else:
            # 只返回图像、one-hot标签、class_idx
            return img, one_hot_label, class_idx

