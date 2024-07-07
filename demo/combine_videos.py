
import cv2
import numpy as np
from tqdm import tqdm

def extract_center_frame(frame, target_width, target_height, x_bias = 0):
    """
    Extract the center part of the frame with given target width and height.
    """
    height, width, _ = frame.shape
    start_x = max(0, width // 2 - target_width // 2)
    start_y = max(0, height // 2 - target_height // 2)
    end_x = start_x + target_width
    end_y = start_y + target_height

    return frame[start_y:end_y, start_x+x_bias:end_x+x_bias]

def create_combined_video(video1_path, video2_path, output_path):
    # Open the video files
    cap1 = cv2.VideoCapture(video1_path)
    cap2 = cv2.VideoCapture(video2_path)

    # Get properties of the videos
    fps1 = cap1.get(cv2.CAP_PROP_FPS)
    fps2 = cap2.get(cv2.CAP_PROP_FPS)
    fps = min(fps1, fps2)
    
    frame_count1 = int(cap1.get(cv2.CAP_PROP_FRAME_COUNT))
    frame_count2 = int(cap2.get(cv2.CAP_PROP_FRAME_COUNT))
    frame_count = min(frame_count1, frame_count2)
    
    # Define the codec and create VideoWriter object
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(output_path, fourcc, fps, (1280, 720))

    for _ in tqdm(range(frame_count)):
        ret1, frame1 = cap1.read()
        ret2, frame2 = cap2.read()
        
        if not ret1 or not ret2:
            break
        
        # Extract center parts
        center_frame1 = extract_center_frame(frame1, 640, 720)
        center_frame2 = extract_center_frame(frame2, 640, 700, x_bias=25)

        # Create a white canvas
        combined_frame = np.ones((720, 1280, 3), dtype=np.uint8) * 255
        
        # Place the center parts in the combined frame
        combined_frame[0:720, 0:640] = center_frame1
        combined_frame[10:710, 640:1280] = center_frame2
        
        # Write the frame to the output video
        out.write(combined_frame)

    # Release everything
    cap1.release()
    cap2.release()
    out.release()

# 示例用法
video1_path = 'test_video.mp4'
video2_path = 'Clustering_Result.mp4'
output_path = 'output_video.mp4'

create_combined_video(video1_path, video2_path, output_path)
