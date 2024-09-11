import os
import shutil
import random
import argparse


def split_and_move_images(src_folder, dst_folder, ratio):
    """
    从 src_folder 的子文件夹中按比例分割图片，并移动到 dst_folder 中相应的子文件夹。

    Parameters:
    src_folder (str): 源文件夹路径，包含多个子文件夹，每个子文件夹对应一个label。
    dst_folder (str): 目标文件夹路径，存储分割后的文件。
    ratio (float): 要分割并移动的文件比例（0 到 1 之间）。
    """
    if not os.path.exists(src_folder):
        print(f"源文件夹 {src_folder} 不存在")
        return

    if not os.path.exists(dst_folder):
        os.makedirs(dst_folder)
        print(f"创建目标文件夹 {dst_folder}")

    # 遍历源文件夹中的每一个子文件夹
    for label in os.listdir(src_folder):
        label_src_path = os.path.join(src_folder, label)

        # 仅处理文件夹
        if not os.path.isdir(label_src_path):
            continue

        # 获取子文件夹中的所有文件（假设图片文件格式为 jpg, png, jpeg）
        image_files = [f for f in os.listdir(label_src_path) if f.endswith(('.jpg', '.jpeg', '.png'))]

        if not image_files:
            print(f"子文件夹 {label_src_path} 中没有图片文件")
            continue

        # 计算要移动的文件数量
        num_files_to_move = int(len(image_files) * ratio)
        files_to_move = random.sample(image_files, num_files_to_move)

        # 创建目标文件夹中的相应子文件夹
        label_dst_path = os.path.join(dst_folder, label)
        if not os.path.exists(label_dst_path):
            os.makedirs(label_dst_path)
            print(f"创建目标子文件夹 {label_dst_path}")

        # 移动文件
        for file in files_to_move:
            src_file_path = os.path.join(label_src_path, file)
            dst_file_path = os.path.join(label_dst_path, file)
            shutil.move(src_file_path, dst_file_path)
            print(f"移动文件: {src_file_path} -> {dst_file_path}")

    print("文件移动完成")


if __name__ == "__main__":

    split_and_move_images(src_folder="./train", dst_folder="./test", ratio=0.1)
