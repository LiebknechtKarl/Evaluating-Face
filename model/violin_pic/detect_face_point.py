
import cv2
import dlib
import os
import sys

# 初始化Dlib的人脸检测器和关键点检测器
detector = dlib.get_frontal_face_detector()
predictor = dlib.shape_predictor(os.path.dirname(__file__) + "/shape_predictor_68_face_landmarks.dat")

def detect_and_visualize_face_landmarks(frame):
    # 转换为灰度图像
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    # 检测人脸
    faces = detector(gray)

    # 遍历检测到的人脸
    for face in faces:
        # 获取关键点
        landmarks = predictor(gray, face)
        for n in range(0, 68):
            x = landmarks.part(n).x
            y = landmarks.part(n).y
            # 绘制关键点
            cv2.circle(frame, (x, y), 1, (0, 0, 255), -1)

    return frame

if __name__ == "__main__":
    video_path = 'data/pic_violin/cropped_li_rena.mp4'              # 替换为你的视频文件路径

    # 打开视频文件
    cap = cv2.VideoCapture(video_path)

    # 检查视频是否成功打开
    if not cap.isOpened():
        print(f"无法打开视频文件: {video_path}")
        sys.exit(1)

    # 获取视频的帧率和总帧数
    fps = cap.get(cv2.CAP_PROP_FPS)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    print(f"视频帧率: {fps} fps")
    print(f"总帧数: {total_frames}")

    frame_count = 0
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        frame_count += 1
        print(f"处理第 {frame_count} 帧")

        # 检测和可视化人脸关键点
        frame_with_landmarks = detect_and_visualize_face_landmarks(frame)

        # 显示当前帧
        cv2.imshow('Video Frame', frame_with_landmarks)

        # 按 'q' 键退出
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    # 释放资源
    cap.release()
    cv2.destroyAllWindows()


















