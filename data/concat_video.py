import cv2
import os
import ffmpeg
import shutil

# This script processes multiple video files by extracting clips from each and combining them into a single video.
# It uses OpenCV to capture video frames and ffmpeg to concatenate the video clips.
# The final integrated video is saved as 'integrated_video.mp4'.

# Define input folder and output file path
input_folder = 'videos'
output_file = 'integrated_video.mp4'
temp_folder = 'temp_clips'

# Create temporary folder to save video clips
if not os.path.exists(temp_folder):
    os.makedirs(temp_folder)

# List of video files
video_files = [f'{i}.mkv' for i in range(6)]

# Extract clips from each video and save
for i, video_file in enumerate(video_files):
    print(i)
    cap = cv2.VideoCapture(os.path.join(input_folder, video_file))
    
    # Check if video is successfully opened
    if not cap.isOpened():
        print(f"Error: Could not open video {video_file}.")
        continue
    
    # Get video frame rate
    fps = cap.get(cv2.CAP_PROP_FPS)
    start_time = 5  # Start time (seconds)
    duration = 5  # Clip duration (seconds)
    
    start_frame = int(start_time * fps)
    end_frame = int((start_time + duration) * fps)
    
    # Jump to start frame
    cap.set(cv2.CAP_PROP_POS_FRAMES, start_frame)
    
    # Read and save the clip
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(os.path.join(temp_folder, f'clip_{i}.mp4'), fourcc, fps, (int(cap.get(3)), int(cap.get(4))))
    
    for frame_num in range(start_frame, end_frame):
        ret, frame = cap.read()
        if not ret:
            break
        out.write(frame)
    
    cap.release()
    out.release()

# Use ffmpeg to concatenate video clips
input_files = [os.path.join(temp_folder, f'clip_{i}.mp4') for i in range(6)]
concat_file = os.path.join(temp_folder, 'concat.txt')

# Create ffmpeg input file list
with open(concat_file, 'w') as f:
    for file in input_files:
        f.write(f"file '{os.path.abspath(file)}'\n")

# Concatenate using ffmpeg
ffmpeg.input(concat_file, format='concat', safe=0).output(output_file, c='copy').run()

# Delete temporary folder and its contents
shutil.rmtree(temp_folder)

print(f"Output video saved as {output_file}")
