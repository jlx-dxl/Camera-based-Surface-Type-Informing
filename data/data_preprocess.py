import cv2
import os
import shutil
import random

skip = [1,3,2,2,6,2]

def convert_video_to_imgs():
# 定义输入视频文件和输出文件夹
    for i in range(6):
        video_file = 'videos/' + str(i) + '.mkv'
        output_folder = 'train/' + str(i) + '/'

        # 确保输出文件夹存在
        if not os.path.exists(output_folder):
            os.makedirs(output_folder)

        # 打开视频文件
        cap = cv2.VideoCapture(video_file)

        frame_count = 0
        saved_frame_count = 0

        # 检查视频是否成功打开
        if not cap.isOpened():
            print("Error: Could not open video file.")
        else:
            while True:
                # 读取视频帧
                ret, frame = cap.read()

                # 如果读取成功，ret 为 True，继续处理帧
                if ret:
                    # 仅每隔 k 帧读取一次帧
                    if frame_count % skip[i] == 0:
                        # 定义输出文件名模板
                        base_filename = os.path.join(output_folder, f"{saved_frame_count:06d}")

                        # 保存原始帧
                        original_file = f"{base_filename}_000.png"
                        cv2.imwrite(original_file, frame)
                        print(f"Saved frame {saved_frame_count:06d}_000 to {original_file}")

                        # 旋转90度
                        rotated_90 = cv2.rotate(frame, cv2.ROTATE_90_CLOCKWISE)
                        rotated_90_file = f"{base_filename}_090.png"
                        cv2.imwrite(rotated_90_file, rotated_90)
                        print(f"Saved frame {saved_frame_count:06d}_090 to {rotated_90_file}")

                        # 旋转180度
                        rotated_180 = cv2.rotate(frame, cv2.ROTATE_180)
                        rotated_180_file = f"{base_filename}_180.png"
                        cv2.imwrite(rotated_180_file, rotated_180)
                        print(f"Saved frame {saved_frame_count:06d}_180 to {rotated_180_file}")

                        # 旋转270度
                        rotated_270 = cv2.rotate(frame, cv2.ROTATE_90_COUNTERCLOCKWISE)
                        rotated_270_file = f"{base_filename}_270.png"
                        cv2.imwrite(rotated_270_file, rotated_270)
                        print(f"Saved frame {saved_frame_count:06d}_270 to {rotated_270_file}")

                        # 增加已保存帧计数
                        saved_frame_count += 1

                    # 增加总帧计数
                    frame_count += 1
                else:
                    # 如果没有更多帧可读取，退出循环
                    break

        # 释放视频捕获对象
        cap.release()

        print(f"All frames of Video {i:01d} have been saved.")


def move_random_files(source_dir, destination_dir, proportion=0.1):
    """
    从source_dir中随机选择一定比例的文件移动到destination_dir。
    
    参数：
    source_dir: 源文件夹路径
    destination_dir: 目标文件夹路径
    proportion: 移动文件的比例 (0到1之间的浮点数)
    """
    if not os.path.exists(destination_dir):
        os.makedirs(destination_dir)

    # 获取源文件夹中的所有文件
    files = [f for f in os.listdir(source_dir) if os.path.isfile(os.path.join(source_dir, f))]

    # 计算需要移动的文件数量
    num_files_to_move = int(len(files) * proportion)

    # 随机选择文件
    files_to_move = random.sample(files, num_files_to_move)

    # 移动文件
    for file in files_to_move:
        src_file_path = os.path.join(source_dir, file)
        dest_file_path = os.path.join(destination_dir, file)
        shutil.move(src_file_path, dest_file_path)
        print(f"Moved {file} to {destination_dir}")

    print(f"Moved {len(files_to_move)} files from {source_dir} to {destination_dir}")



if __name__ == "__main__":
    # convert_video_to_imgs()

    # 示例用法
    for i in range(6):
        train = 'train/' + str(i) + '/'
        dev = 'dev/' + str(i) + '/'
        test = 'test/' + str(i) + '/'

        move_random_files(train, dev)
        move_random_files(train, test)
