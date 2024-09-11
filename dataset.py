import os
import torch
from torch.utils.data import Dataset
from torchvision import transforms
from torchvision.datasets import ImageFolder
from skimage.feature import graycomatrix, graycoprops
import numpy as np
from PIL import Image

os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

class VisionInformDataset(Dataset):
    def __init__(self, transform=None, set_type='train', if_GLCM=False):
        """
        Args:
            transform (callable, optional): Optional transform to be applied on a sample.
            set_type (string): Type of the dataset, should be 'train', 'test', or 'dev'.
            if_GLCM (bool): if add GLCM features as extra channels.
        """
        script_dir = os.path.dirname(os.path.abspath(__file__))
        if set_type == 'train':
            root_dir = os.path.join(script_dir, 'data/train/')
        elif set_type == 'test':
            root_dir = os.path.join(script_dir, 'data/test/')
        else:
            raise ValueError("Invalid set_type. Must be one of 'train', 'test', or 'dev'.")

        self.root_dir = root_dir
        self.transform = transform
        self.dataset = ImageFolder(root_dir)
        self.if_GLCM = if_GLCM

        # If no specific transform is provided, use default transform (i.e., convert to tensor)
        if transform is None:
            self.transform = transforms.Compose([
                transforms.Resize((224, 224)),  # Optimal input size for ResNet
                transforms.ToTensor(),
            ])
        else:
            self.transform = transform

        # Get the number of classes
        self.num_classes = len(self.dataset.classes)
        self.class_to_idx = self.dataset.class_to_idx
        self.idx_to_class = {v: k for k, v in self.class_to_idx.items()}

    def __len__(self):
        return len(self.dataset)

    def compute_glcm_features(self, gray_image):
        """
        计算灰度图像的GLCM特征 (对比度、同质性、能量)，并返回3个通道的特征图。
        """
        glcm = graycomatrix(gray_image, distances=[1], angles=[0], levels=256, symmetric=True, normed=True)

        contrast = graycoprops(glcm, 'contrast').flatten()  # 对比度
        homogeneity = graycoprops(glcm, 'homogeneity').flatten()  # 同质性
        energy = graycoprops(glcm, 'energy').flatten()  # 能量

        # 将特征重塑为图像形状
        glcm_features = np.stack([contrast, homogeneity, energy], axis=0)
        return torch.from_numpy(glcm_features).float()

    def __getitem__(self, idx):
        # 获取图像和类别标签
        img, class_idx = self.dataset[idx]

        # 将图像应用转换
        img = self.transform(img)

        # 将类别标签转换为 one-hot 编码
        one_hot_label = torch.zeros(self.num_classes)
        one_hot_label[class_idx] = 1.0

        if self.if_GLCM:
            # 转换为灰度图像，并计算 GLCM 特征
            img_gray = img.mean(dim=0).numpy().astype(np.uint8)  # 将图像转换为灰度图像 (简化处理)
            glcm_features = self.compute_glcm_features(img_gray)

            # 返回图像、GLCM特征、one-hot标签、class_idx
            return img, glcm_features, one_hot_label, class_idx
        else:
            # 只返回图像、one-hot标签、class_idx
            return img, one_hot_label, class_idx
