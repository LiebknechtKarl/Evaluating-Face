import cv2
import os

def process_video(input_video_path, output_video_path):
    # 打开输入视频文件
    cap = cv2.VideoCapture(input_video_path)

    # 检查视频是否成功打开
    if not cap.isOpened():
        print(f"无法打开视频文件: {input_video_path}")
        return

    # 获取视频的帧率和尺寸
    fps = cap.get(cv2.CAP_PROP_FPS)
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    # 定义裁剪区域
    left = 150
    right = 1150

    # 创建输出视频文件
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(output_video_path, fourcc, fps, (right - left, height))

    frame_count = 0
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        frame_count += 1
        print(f"处理第 {frame_count} 帧")

        # 裁剪帧
        cropped_frame = frame[:, left:right]

        # 写入输出视频
        out.write(cropped_frame)

    # 释放资源
    cap.release()
    out.release()

if __name__ == "__main__":
    input_video_path = 'data/pic_violin/li_rena.mp4'  # 输入视频文件路径
    output_video_path = 'data/pic_violin/cropped_li_rena.mp4'  # 输出视频文件路径

    process_video(input_video_path, output_video_path)