#!/usr/bin/env python3
# coding: utf-8

import os
import wandb    
import torch
import time
import numpy as np
from torch.utils.data import DataLoader
# from dataset import ImageDataset
from tqdm import tqdm
import cv2

from torchvision.transforms import Grayscale, ToPILImage, ToTensor

def setup_wandb(args, name):
    """
    Sets up wandb for experiment tracking.

    Args:
        args (Namespace): Command line arguments.
        name (str): Name of the experiment.
    """
    os.environ["WANDB_API_KEY"] = '28cbf19c5cd0619337ae4fb844d56992a283b007'
    wandb.init(project='Camera-based Friction Coefficient Estimation', config=args, name=name) 
    
def get_class_name(one_hot_labels):
    """
    Converts one-hot encoded labels to class indices.

    Args:
        one_hot_labels (Tensor): One-hot encoded labels.

    Returns:
        Tensor: Class indices.
    """
    label_idices = []
    for one_hot_label in one_hot_labels:
        label_idx = torch.argmax(one_hot_label).item()
        label_idices.append(label_idx)
    return torch.tensor(np.array(label_idices))

def calculate_accuracy(tensor1, tensor2):
    """
    Calculates the percentage of matching elements between two tensors.

    Args:
        tensor1 (Tensor): First tensor.
        tensor2 (Tensor): Second tensor.

    Returns:
        float: Percentage of matching elements.
    """
    same_elements = torch.sum(tensor1 == tensor2).item()
    total_elements = tensor1.numel()
    percentage = same_elements / total_elements
    return percentage

def fast_glcm(img, vmin=0, vmax=255, levels=8, kernel_size=5, distance=1.0, angle=0.0):
    """
    Calculates the grey-level co-occurrence matrix (GLCM) for an image.

    Args:
        img (array): Input image.
        vmin (int): Minimum value of input image.
        vmax (int): Maximum value of input image.
        levels (int): Number of grey levels in GLCM.
        kernel_size (int): Patch size to calculate GLCM.
        distance (float): Pixel pair distance.
        angle (float): Pixel pair angle.

    Returns:
        array: GLCM for each pixel.
    """
    mi, ma = vmin, vmax
    ks = kernel_size
    h,w = img.shape

    bins = np.linspace(mi, ma+1, levels+1)
    gl1 = np.digitize(img, bins) - 1

    dx = distance * np.cos(np.deg2rad(angle))
    dy = distance * np.sin(np.deg2rad(-angle))
    mat = np.array([[1.0, 0.0, -dx], [0.0, 1.0, -dy]], dtype=np.float32)
    gl2 = cv2.warpAffine(gl1, mat, (w,h), flags=cv2.INTER_NEAREST, borderMode=cv2.BORDER_REPLICATE)

    glcm = np.zeros((levels, levels, h, w), dtype=np.uint8)
    for i in range(levels):
        for j in range(levels):
            mask = ((gl1 == i) & (gl2 == j))
            glcm[i, j, mask] = 1

    kernel = np.ones((ks, ks), dtype=np.uint8)
    for i in range(levels):
        for j in range(levels):
            glcm[i, j] = cv2.filter2D(glcm[i, j], -1, kernel)

    glcm = glcm.astype(np.float32)
    return glcm

def fast_glcm_mean(img, vmin=0, vmax=255, levels=8, ks=5, distance=1.0, angle=0.0):
    """
    Calculates the mean of the GLCM.

    Args:
        img (array): Input image.

    Returns:
        array: GLCM mean.
    """
    h,w = img.shape
    glcm = fast_glcm(img, vmin, vmax, levels, ks, distance, angle)
    mean = np.zeros((h,w), dtype=np.float32)
    for i in range(levels):
        for j in range(levels):
            mean += glcm[i,j] * i / (levels)**2
    return mean

def fast_glcm_std(img, vmin=0, vmax=255, levels=8, ks=5, distance=1.0, angle=0.0):
    """
    Calculates the standard deviation of the GLCM.

    Args:
        img (array): Input image.

    Returns:
        array: GLCM standard deviation.
    """
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
    """
    Calculates the contrast of the GLCM.

    Args:
        img (array): Input image.

    Returns:
        array: GLCM contrast.
    """
    h,w = img.shape
    glcm = fast_glcm(img, vmin, vmax, levels, ks, distance, angle)
    cont = np.zeros((h,w), dtype=np.float32)
    for i in range(levels):
        for j in range(levels):
            cont += glcm[i,j] * (i-j)**2
    return cont

def fast_glcm_dissimilarity(img, vmin=0, vmax=255, levels=8, ks=5, distance=1.0, angle=0.0):
    """
    Calculates the dissimilarity of the GLCM.

    Args:
        img (array): Input image.

    Returns:
        array: GLCM dissimilarity.
    """
    h,w = img.shape
    glcm = fast_glcm(img, vmin, vmax, levels, ks, distance, angle)
    diss = np.zeros((h,w), dtype=np.float32)
    for i in range(levels):
        for j in range(levels):
            diss += glcm[i,j] * np.abs(i-j)
    return diss

def fast_glcm_homogeneity(img, vmin=0, vmax=255, levels=8, ks=5, distance=1.0, angle=0.0):
    """
    Calculates the homogeneity of the GLCM.

    Args:
        img (array): Input image.

    Returns:
        array: GLCM homogeneity.
    """
    h,w = img.shape
    glcm = fast_glcm(img, vmin, vmax, levels, ks, distance, angle)
    homo = np.zeros((h,w), dtype=np.float32)
    for i in range(levels):
        for j in range(levels):
            homo += glcm[i,j] / (1.+(i-j)**2)
    return homo

def fast_glcm_ASM(img, vmin=0, vmax=255, levels=8, ks=5, distance=1.0, angle=0.0):
    """
    Calculates the angular second moment (ASM) and energy of the GLCM.

    Args:
        img (array): Input image.

    Returns:
        tuple: GLCM ASM and energy.
    """
    h,w = img.shape
    glcm = fast_glcm(img, vmin, vmax, levels, ks, distance, angle)
    asm = np.zeros((h,w), dtype=np.float32)
    for i in range(levels):
        for j in range(levels):
            asm += glcm[i,j]**2
    ene = np.sqrt(asm)
    return asm, ene

def fast_glcm_max(img, vmin=0, vmax=255, levels=8, ks=5, distance=1.0, angle=0.0):
    """
    Calculates the maximum value of the GLCM.

    Args:
        img (array): Input image.

    Returns:
        array: GLCM maximum value.
    """
    glcm = fast_glcm(img, vmin, vmax, levels, ks, distance, angle)
    max_ = np.max(glcm, axis=(0,1))
    return max_

def fast_glcm_entropy(img, vmin=0, vmax=255, levels=8, ks=5, distance=1.0, angle=0.0):
    """
    Calculates the entropy of the GLCM.

    Args:
        img (array): Input image.

    Returns:
        array: GLCM entropy.
    """
    glcm = fast_glcm(img, vmin, vmax, levels, ks, distance, angle)
    pnorm = glcm / np.sum(glcm, axis=(0,1)) + 1./ks**2
    ent = np.sum(-pnorm * np.log(pnorm), axis=(0,1))
    return ent

def display_img(img, name='?'):
    """
    Displays an image using OpenCV.

    Args:
        img (array): Input image.
        name (str): Window name.
    """
    img_normalized = cv2.normalize(img, None, alpha=0, beta=255, norm_type=cv2.NORM_MINMAX)
    img_normalized = img_normalized.astype(np.uint8)
    cv2.imshow(name, img_normalized)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
    
def tensor_to_grayscale(img_tensor):
    """
    Converts a tensor image to a grayscale numpy array.

    Args:
        img_tensor (Tensor): Input image tensor.

    Returns:
        array: Grayscale image.
    """
    to_pil = ToPILImage()
    to_tensor = ToTensor()
    pil_img = to_pil(img_tensor)
    pil_gray = pil_img.convert("L")
    gray_tensor = to_tensor(pil_gray)
    gray_np = gray_tensor.squeeze(0).numpy()
    return gray_np

def integrate_glcm(img):
    """
    Integrates multiple GLCM features.

    Args:
        img (array): Input image.

    Returns:
        list: List of GLCM features.
    """
    result = []
    result.append(fast_glcm_contrast(img, 0.0, 1.0, 8, 5))
    result.append(fast_glcm_homogeneity(img, 0.0, 1.0, 8, 5))
    result.append(fast_glcm_entropy(img, 0.0, 1.0, 8, 5))
    return result

def batch_glcm(img):
    """
    Calculates GLCM features for a batch of images.

    Args:
        img (Tensor): Batch of images.

    Returns:
        Tensor: Batch of GLCM features.
    """
    batch_results = integrate_glcm(img)
    batch_results = [torch.tensor(np.stack(feature)) for feature in zip(*batch_results)]
    batch_results = torch.stack(batch_results, dim=0).permute(1, 0, 2)
    return batch_results

def visualize_all_glcm(imgs):
    """
    Visualizes all GLCM features for a batch of images.

    Args:
        imgs (list): List of images.
    """
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
        
def record_time_once(last):
    """
    Records and prints the time difference from the last recorded time.

    Args:
        last (float): Last recorded time.

    Returns:
        float: Current time.
    """
    now = time.time()
    diff = now - last
    return now

# if __name__ == "__main__":
#     device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
#     dataset = ImageDataset(train_or_test='test')
#     dataloader = DataLoader(dataset, batch_size=20, shuffle=True, num_workers=4)
#     for _, images, gt_values in tqdm(dataloader):
#         gray_imgs = tensor_to_grayscale_list(images)
#         visualize_all_glcm(gray_imgs)
