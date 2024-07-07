
import cv2
import os

def create_video_from_images(image_folder, output_video, duration):
    # 获取图片列表，并按文件名排序
    images = [img for img in os.listdir(image_folder) if img.endswith((".png", ".jpg", ".jpeg"))]
    images.sort()

    # 获取图片尺寸
    first_image_path = os.path.join(image_folder, images[0])
    first_image = cv2.imread(first_image_path)
    height, width, layers = first_image.shape

    # 计算帧率
    frame_count = len(images)
    fps = frame_count / duration
    
    print(f"height: {height}, width: {width}, fps: {fps}")

    # 创建视频写入对象
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')  # 指定视频编码器
    video = cv2.VideoWriter(output_video, fourcc, fps, (width, height))

    # 读取图片并写入视频
    for image in images:
        image_path = os.path.join(image_folder, image)
        frame = cv2.imread(image_path)
        video.write(frame)

    # 释放视频写入对象
    video.release()
    print(f"视频已保存为 {output_video}")

# 示例用法
image_folder = 'frames'  # 替换为你的图片文件夹路径
output_video = 'output_video.mp4'      # 输出视频文件名
duration = 22                          # 视频时长（单位：秒）

create_video_from_images(image_folder, output_video, duration)
