#!/usr/bin/env python3
# coding: utf-8

import os
import wandb    
import torch
import numpy as np
from torch.utils.data import DataLoader
from data.dataset import ImageDataset
from tqdm import tqdm
import cv2

from torchvision.transforms import Grayscale, ToPILImage, ToTensor

def setup_wandb(args, name):
    # Implement this if you wish to use wandb in your experiments
    # TODO
    os.environ["WANDB_API_KEY"] = '28cbf19c5cd0619337ae4fb844d56992a283b007'
    wandb.init(project='Camera-based Friction Coefficient Estimation', config=args, name=name) 
    
def get_class_name(one_hot_labels):
    label_idices = []
    for one_hot_label in one_hot_labels:
        label_idx = torch.argmax(one_hot_label).item()
        label_idices.append(label_idx)
    return torch.tensor(np.array(label_idices))

def calculate_accuracy(tensor1, tensor2):
    # 计算相同元素的个数
    same_elements = torch.sum(tensor1 == tensor2).item()
    
    # 计算相同元素的百分比
    total_elements = tensor1.numel()
    percentage = same_elements / total_elements
    
    return percentage


def fast_glcm(img, vmin=0, vmax=255, levels=8, kernel_size=5, distance=1.0, angle=0.0):
    '''
    Parameters
    ----------
    img: array_like, shape=(h,w), dtype=np.uint8
        input image
    vmin: int
        minimum value of input image
    vmax: int
        maximum value of input image
    levels: int
        number of grey-levels of GLCM
    kernel_size: int
        Patch size to calculate GLCM around the target pixel
    distance: float
        pixel pair distance offsets [pixel] (1.0, 2.0, and etc.)
    angle: float
        pixel pair angles [degree] (0.0, 30.0, 45.0, 90.0, and etc.)

    Returns
    -------
    Grey-level co-occurrence matrix for each pixels
    shape = (levels, levels, h, w)
    '''

    mi, ma = vmin, vmax
    ks = kernel_size
    h,w = img.shape

    # digitize
    bins = np.linspace(mi, ma+1, levels+1)
    gl1 = np.digitize(img, bins) - 1

    # make shifted image
    dx = distance*np.cos(np.deg2rad(angle))
    dy = distance*np.sin(np.deg2rad(-angle))
    mat = np.array([[1.0,0.0,-dx], [0.0,1.0,-dy]], dtype=np.float32)
    gl2 = cv2.warpAffine(gl1, mat, (w,h), flags=cv2.INTER_NEAREST,
                         borderMode=cv2.BORDER_REPLICATE)

    # make glcm
    glcm = np.zeros((levels, levels, h, w), dtype=np.uint8)
    for i in range(levels):
        for j in range(levels):
            mask = ((gl1==i) & (gl2==j))
            glcm[i,j, mask] = 1

    kernel = np.ones((ks, ks), dtype=np.uint8)
    for i in range(levels):
        for j in range(levels):
            glcm[i,j] = cv2.filter2D(glcm[i,j], -1, kernel)

    glcm = glcm.astype(np.float32)
    return glcm


def fast_glcm_mean(img, vmin=0, vmax=255, levels=8, ks=5, distance=1.0, angle=0.0):
    '''
    calc glcm mean
    '''
    h,w = img.shape
    glcm = fast_glcm(img, vmin, vmax, levels, ks, distance, angle)
    mean = np.zeros((h,w), dtype=np.float32)
    for i in range(levels):
        for j in range(levels):
            mean += glcm[i,j] * i / (levels)**2

    return mean


def fast_glcm_std(img, vmin=0, vmax=255, levels=8, ks=5, distance=1.0, angle=0.0):
    '''
    calc glcm std
    '''
    h,w = img.shape
    glcm = fast_glcm(img, vmin, vmax, levels, ks, distance, angle)
    mean = np.zeros((h,w), dtype=np.float32)
    for i in range(levels):
        for j in range(levels):
            mean += glcm[i,j] * i / (levels)**2

    std2 = np.zeros((h,w), dtype=np.float32)
    for i in range(levels):
        for j in range(levels):
            std2 += (glcm[i,j] * i - mean)**2

    std = np.sqrt(std2)
    return std


def fast_glcm_contrast(img, vmin=0, vmax=255, levels=8, ks=5, distance=1.0, angle=0.0):
    '''
    calc glcm contrast
    '''
    h,w = img.shape
    glcm = fast_glcm(img, vmin, vmax, levels, ks, distance, angle)
    cont = np.zeros((h,w), dtype=np.float32)
    for i in range(levels):
        for j in range(levels):
            cont += glcm[i,j] * (i-j)**2

    return cont


def fast_glcm_dissimilarity(img, vmin=0, vmax=255, levels=8, ks=5, distance=1.0, angle=0.0):
    '''
    calc glcm dissimilarity
    '''
    h,w = img.shape
    glcm = fast_glcm(img, vmin, vmax, levels, ks, distance, angle)
    diss = np.zeros((h,w), dtype=np.float32)
    for i in range(levels):
        for j in range(levels):
            diss += glcm[i,j] * np.abs(i-j)

    return diss


def fast_glcm_homogeneity(img, vmin=0, vmax=255, levels=8, ks=5, distance=1.0, angle=0.0):
    '''
    calc glcm homogeneity
    '''
    h,w = img.shape
    glcm = fast_glcm(img, vmin, vmax, levels, ks, distance, angle)
    homo = np.zeros((h,w), dtype=np.float32)
    for i in range(levels):
        for j in range(levels):
            homo += glcm[i,j] / (1.+(i-j)**2)

    return homo


def fast_glcm_ASM(img, vmin=0, vmax=255, levels=8, ks=5, distance=1.0, angle=0.0):
    '''
    calc glcm asm, energy
    '''
    h,w = img.shape
    glcm = fast_glcm(img, vmin, vmax, levels, ks, distance, angle)
    asm = np.zeros((h,w), dtype=np.float32)
    for i in range(levels):
        for j in range(levels):
            asm  += glcm[i,j]**2

    ene = np.sqrt(asm)
    return asm, ene


def fast_glcm_max(img, vmin=0, vmax=255, levels=8, ks=5, distance=1.0, angle=0.0):
    '''
    calc glcm max
    '''
    glcm = fast_glcm(img, vmin, vmax, levels, ks, distance, angle)
    max_  = np.max(glcm, axis=(0,1))
    return max_


def fast_glcm_entropy(img, vmin=0, vmax=255, levels=8, ks=5, distance=1.0, angle=0.0):
    '''
    calc glcm entropy
    '''
    glcm = fast_glcm(img, vmin, vmax, levels, ks, distance, angle)
    pnorm = glcm / np.sum(glcm, axis=(0,1)) + 1./ks**2
    ent  = np.sum(-pnorm * np.log(pnorm), axis=(0,1))
    return ent

def display_img(img, name = '?'):
    # 将 glcm_mean 数据规范化到 0 到 255 范围
    img_normalized = cv2.normalize(img, None, alpha=0, beta=255, norm_type=cv2.NORM_MINMAX)
    # 将数据类型转换为 uint8
    img_normalized = img_normalized.astype(np.uint8)
    # 显示图像
    cv2.imshow(name, img_normalized)
    cv2.waitKey(0)  # 等待按键操作
    cv2.destroyAllWindows()  # 关闭显示的窗口
    

def tensor_to_grayscale_list(images_tensor):
    to_pil = ToPILImage()
    to_tensor = ToTensor()
    grayscale_images = []
    for img_tensor in images_tensor:
        pil_img = to_pil(img_tensor)
        pil_gray = pil_img.convert("L")
        gray_tensor = to_tensor(pil_gray)
        gray_np = gray_tensor.squeeze(0).numpy()
        grayscale_images.append(gray_np)
    return grayscale_images

def integrate_glcm(img):
    result = []
    result.append(fast_glcm_contrast(img, 0.0, 1.0, 8, 5))   # 对比度
    # result.append(fast_glcm_dissimilarity(img, 0.0, 1.0, 8, 5))
    result.append(fast_glcm_homogeneity(img, 0.0, 1.0, 8, 5))   # 同质性
    # glcm_ASM, _ = fast_glcm_ASM(img, 0.0, 1.0, 8, 5)
    # result.append(glcm_ASM)
    result.append(fast_glcm_entropy(img, 0.0, 1.0, 8, 5))   # 熵
    return result

def batch_glcm(imgs):
    batch_results = [integrate_glcm(img) for img in imgs]
    # 将列表中的numpy数组转换为张量，并堆叠形成一个新的张量
    # 首先堆叠每个特征的所有图像
    batch_results = [torch.tensor(np.stack(feature)) for feature in zip(*batch_results)]
    # 再交换轴以正确地排列维度
    batch_results = torch.stack(batch_results, dim=0)
    batch_results = batch_results.permute(1, 0, 2, 3)  # 从 (5, 20, W, H) 调整到 (20, 5, W, H)
    return batch_results

def visualize_all_glcm(imgs):
    for img in imgs:
        display_img(img, name='Original')
        glcm_mean = fast_glcm_mean(img, 0.0, 1.0, 8, 5)
        display_img(glcm_mean, name='Mean')
        glcm_std = fast_glcm_std(img, 0.0, 1.0, 8, 5)
        display_img(glcm_std, name='Std')
        glcm_contrast = fast_glcm_contrast(img, 0.0, 1.0, 8, 5)
        display_img(glcm_contrast, name='Contrast')
        glcm_dissimilarity = fast_glcm_dissimilarity(img, 0.0, 1.0, 8, 5)
        display_img(glcm_dissimilarity, name='Dissimilarity')
        glcm_homogeneity = fast_glcm_homogeneity(img, 0.0, 1.0, 8, 5)
        display_img(glcm_homogeneity, name='Homogeneity')
        glcm_ASM, glcm_energ = fast_glcm_ASM(img, 0.0, 1.0, 8, 5)
        display_img(glcm_ASM, name='ASM')
        display_img(glcm_energ, name='Energy')
        glcm_max = fast_glcm_max(img, 0.0, 1.0, 8, 5)
        display_img(glcm_max, name='Max')
        glcm_entropy = fast_glcm_entropy(img, 0.0, 1.0, 8, 5)
        display_img(glcm_entropy, name='Entropy')      
        


if __name__ == "__main__":
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    dataset = ImageDataset(train_or_test='test')
    dataloader = DataLoader(dataset, batch_size=20, shuffle=True, num_workers=4)
    for _, images, gt_values in tqdm(dataloader):
        gray_imgs = tensor_to_grayscale_list(images)
        # batch_result = batch_glcm(gray_imgs)
        # print(batch_result.shape)  # 应该输出 (20, 5, W, H)
        visualize_all_glcm(gray_imgs)
            
