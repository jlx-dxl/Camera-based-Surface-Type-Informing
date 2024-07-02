#!/usr/bin/env python3

import torch
from river import cluster
from PIL import Image
from torchvision import transforms
import cv2
import numpy as np
import os
import matplotlib.pyplot as plt
import time
from matplotlib.animation import FuncAnimation
from sklearn.decomposition import PCA
from mpl_toolkits.mplot3d import Axes3D

from model import Classifer18
from util import *

# Set the device to use GPU if available, otherwise use CPU
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def img_preprocess(img):
    """
    Preprocesses the input image for feature extraction.

    Parameters:
    img (numpy.ndarray): The input image to preprocess.

    Returns:
    tuple: A tuple containing the preprocessed image tensor and GLCM textures tensor.
    """
    transform = transforms.Compose([
        transforms.Resize((224, 224)),  # Optimal input size for ResNet
        transforms.ToTensor(),
    ])

    # Convert numpy.ndarray to PIL.Image if necessary
    if isinstance(img, np.ndarray):
        img = Image.fromarray(img)

    img = transform(img)

    # Convert the image tensor to grayscale
    gray_img = tensor_to_grayscale(img)

    # Compute GLCM textures from the grayscale image
    glcm_textures = batch_glcm(gray_img)

    return img.unsqueeze(0), glcm_textures.unsqueeze(0)

def feature_extract(backbone, resnet18, inputs_res, inputs_glcm):
    """
    Extracts features from input images using the backbone and ResNet18 models.

    Parameters:
    backbone (torch.nn.Module): The backbone model for feature extraction.
    resnet18 (torch.nn.Module): The ResNet18 model for feature extraction.
    inputs_res (torch.Tensor): The input image tensor.
    inputs_glcm (torch.Tensor): The GLCM textures tensor.

    Returns:
    numpy.ndarray: The extracted features.
    """
    inputs_res = inputs_res.to(device)
    inputs_glcm = inputs_glcm.to(device)

    features_res = resnet18(inputs_res).to(device)
    features_glcm = resnet18(inputs_glcm).to(device)

    input = torch.cat((features_res, features_glcm), dim=1).to(device)

    features = backbone(input)
    features = features.view(features.size(0), -1).detach().cpu().numpy()

    return features

def numpy_to_dict(array):
    """
    Converts a numpy array to a dictionary.

    Parameters:
    array (numpy.ndarray): The input numpy array.

    Returns:
    dict: The converted dictionary.
    """
    return {i: array[0, i] for i in range(array.shape[1])}

def process_video_frame(cap, backbone, resnet18, dbstream, all_data, visualize):
    """
    Processes a single frame from the video, extracts features, and updates the DBSTREAM clusterer.

    Parameters:
    cap (cv2.VideoCapture): The video capture object.
    backbone (torch.nn.Module): The backbone model for feature extraction.
    resnet18 (torch.nn.Module): The ResNet18 model for feature extraction.
    dbstream (cluster.DBSTREAM): The DBSTREAM clusterer.
    all_data (list): List to store all feature dictionaries.
    visualize (bool): Whether to visualize the frame and clustering result.

    Returns:
    bool: True if the frame was processed successfully, False otherwise.
    """
    ret, frame = cap.read()
    
    if not ret:
        return False

    if visualize:
        # 显示处理后的帧
        cv2.imshow('Video Frame', frame)

        # 等待按键，按 'q' 退出
        if cv2.waitKey(1) & 0xFF == ord('q'):
            return False
    
    inputs_res, inputs_glcm = img_preprocess(frame)
    feature = feature_extract(backbone, resnet18, inputs_res, inputs_glcm)
    feature_dict = numpy_to_dict(feature)

    dbstream.learn_one(feature_dict)
    all_data.append(feature_dict)

    labels = [dbstream.predict_one(data) for data in all_data]
    unique_labels = set(labels)
    
    # print(f"Current cluster number: {dbstream.n_clusters}, Total unique labels: {len(unique_labels)}")

    return True

def update_plot(frame, cap, backbone, resnet18, dbstream, all_data, ax, scatter, visualize):
    """
    Updates the 3D plot with new data from the video frames.

    Parameters:
    frame (int): The current frame number (unused).
    cap (cv2.VideoCapture): The video capture object.
    backbone (torch.nn.Module): The backbone model for feature extraction.
    resnet18 (torch.nn.Module): The ResNet18 model for feature extraction.
    dbstream (cluster.DBSTREAM): The DBSTREAM clusterer.
    all_data (list): List to store all feature dictionaries.
    ax (Axes3D): The 3D axis object for plotting.
    scatter (PathCollection): The scatter plot object.
    visualize (bool): Whether to visualize the frame and clustering result.

    Returns:
    PathCollection: The updated scatter plot object.
    """
    if not process_video_frame(cap, backbone, resnet18, dbstream, all_data, visualize):
        return scatter,

    if visualize:
        data_array = np.array([list(d.values()) for d in all_data])

        # Perform PCA only if there are enough data points
        if data_array.shape[0] >= 3:
            pca = PCA(n_components=3)
            data_3d = pca.fit_transform(data_array)
            labels = [dbstream.predict_one(data) for data in all_data]

            ax.clear()
            scatter = ax.scatter(data_3d[:, 0], data_3d[:, 1], data_3d[:, 2], c=labels, cmap='viridis', marker='.', s=60)

            # Add legend
            legend1 = ax.legend(*scatter.legend_elements(), title="Clusters")
            ax.add_artist(legend1)

            ax.set_title('DBSTREAM Clustering in 3D')
            ax.set_xlabel('PC 1')
            ax.set_ylabel('PC 2')
            ax.set_zlabel('PC 3')

    return scatter,

def static_plot(all_data, dbstream):
    """
    Generates a static 3D plot of the clustering result after processing all video frames.

    Parameters:
    all_data (list): List of all feature dictionaries.
    dbstream (cluster.DBSTREAM): The DBSTREAM clusterer.
    """
    data_array = np.array([list(d.values()) for d in all_data])

    if data_array.shape[0] >= 3:
        pca = PCA(n_components=3)
        data_3d = pca.fit_transform(data_array)
        labels = [dbstream.predict_one(data) for data in all_data]

        fig = plt.figure(figsize=(10, 7))
        ax = fig.add_subplot(111, projection='3d')
        scatter = ax.scatter(data_3d[:, 0], data_3d[:, 1], data_3d[:, 2], c=labels, cmap='viridis', marker='.', s=60)

        # Add legend
        legend1 = ax.legend(*scatter.legend_elements(), title="Clusters")
        ax.add_artist(legend1)

        ax.set_title('DBSTREAM Clustering in 3D')
        ax.set_xlabel('PC 1')
        ax.set_ylabel('PC 2')
        ax.set_zlabel('PC 3')

        plt.show()

def main(visualize=True):
    """
    Main function to run the video processing and clustering visualization.

    Parameters:
    visualize (bool): Whether to visualize the frame and clustering result.
    """
    video_path = 'data/test_video.mp4'

    model = Classifer18().to(device)
    model.load_state_dict(torch.load(os.path.join('model', 'train0630-Res18-1', 'best.pth')))
    backbone = torch.nn.Sequential(*(list(model.children())[:-1]))
    resnet18 = torch.jit.load("model/resnet18_traced.pt").to(device)
    backbone.eval()
    resnet18.eval()

    dbstream = cluster.DBSTREAM(
        clustering_threshold=10.0,
        fading_factor=0.99,
        cleanup_interval=1e8,
        intersection_factor=0.99,
        minimum_weight=0.01
    )

    cap = cv2.VideoCapture(video_path)
    all_data = []
    time_sum = 0
    count = 0
    
    now = time.time()

    if visualize:
        fig = plt.figure(figsize=(10, 7))
        ax = fig.add_subplot(111, projection='3d')
        scatter = ax.scatter([0], [0], [0], c=[0], cmap='viridis', marker='.')

        ani = FuncAnimation(fig, update_plot, fargs=(cap, backbone, resnet18, dbstream, all_data, ax, scatter, visualize),
                            interval=100, blit=False, cache_frame_data=False)

        plt.show()
    else:
        while True:
            if not process_video_frame(cap, backbone, resnet18, dbstream, all_data, visualize):
                break
            
            # timeing
            time_diff = time.time() - now
            now = time.time()
            time_sum += time_diff
            count += 1
            
        print(f"Average FPS: {count / time_sum:.2f}")
            
        # 在所有视频帧处理完成后生成静态图
        static_plot(all_data, dbstream)

    cap.release()
    if visualize:
        cv2.destroyAllWindows()

if __name__ == "__main__":
    main(visualize=False)
