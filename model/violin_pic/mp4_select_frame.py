import cv2
import os,sys

def get_frame_by_number(video_path, frame_number):
    # 打开视频文件
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print(f"无法打开视频文件: {video_path}")
        return None

    # 获取总帧数
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    if frame_number < 0 or frame_number >= total_frames:
        print(f"帧号 {frame_number} 超出范围 (0 到 {total_frames - 1})")
        cap.release()
        return None

    # 设置要读取的帧号
    cap.set(cv2.CAP_PROP_POS_FRAMES, frame_number)

    # 读取指定帧
    ret, frame = cap.read()

    # 释放资源
    cap.release()

    if ret:
        return frame
    else:
        print(f"无法读取帧 {frame_number}")
        return None

# 示例用法
if __name__ == "__main__":
    video_path = 'data/pic_violin/li_rena.mp4'      # 替换为你的视频文件路径
    frame_number = 0                                # 替换为你想要的帧号

    frame = get_frame_by_number(video_path, frame_number)
    if frame is not None:
        # 显示获取的帧
        cv2.imshow('Frame', frame)
        cv2.waitKey(0)
        cv2.destroyAllWindows()
    else:
        print("获取帧失败")