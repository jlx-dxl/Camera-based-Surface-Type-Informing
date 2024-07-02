import os
import subprocess

# 定义视频文件路径和截取秒数列表
video_folder = 'videos'  # 替换为你的视频文件夹路径
output_folder = 'temp'
os.makedirs(output_folder, exist_ok=True)

# 定义截取秒数列表（例如：每个视频从不同秒数开始截取5秒）
start_times = [10.0, 20.0, 20.0, 30.0, 30.0, 30.0]  # 替换为你自己的开始时间列表
duration = 3.0  # 截取的持续时间

# 获取视频文件列表
video_files = [os.path.join(video_folder, f) for f in os.listdir(video_folder) if f.endswith('.mkv')]

# 确保视频文件数量和开始时间列表长度一致
assert len(video_files) == len(start_times), "视频文件数量和开始时间列表长度不一致"

# 使用 ffmpeg 截取片段并保存到 temp 文件夹
for i, (video_file, start_time) in enumerate(zip(video_files, start_times)):
    output_file = os.path.join(output_folder, f'clip_{i}.mkv')
    ffmpeg_cmd = f"ffmpeg -ss {start_time} -i \"{video_file}\" -t {duration} -c copy \"{output_file}\""
    subprocess.run(ffmpeg_cmd, shell=True)

print("视频片段已成功截取并保存到 temp 文件夹")

# 获取 temp 文件夹中的视频片段
clip_files = [os.path.join(output_folder, f) for f in sorted(os.listdir(output_folder)) if f.endswith('.mkv')]

# 创建一个文本文件，列出所有要拼接的视频文件
with open("file_list.txt", "w") as file_list:
    for clip in clip_files:
        file_list.write(f"file '{clip}'\n")

# 使用 ffmpeg 拼接视频
output_video = 'integrated_video.mkv'
ffmpeg_cmd = f"ffmpeg -f concat -safe 0 -i file_list.txt -c copy {output_video}"
subprocess.run(ffmpeg_cmd, shell=True)

# 清理临时文件
os.remove("file_list.txt")
print(f"视频片段已成功拼接并保存为 {output_video}")
