import cv2
import dlib
import os
import sys
import numpy as np
import matplotlib.pyplot as plt
Face_Length = 190
Face_Width = 120
plt.style.use('_mpl-gallery')

def face_measure(face_landmark_single):

    # 获取每列
    col1 = face_landmark_single[:, 0]
    col2 = face_landmark_single[:, 1]

    # 放缩到 0-180
    col1_scaled = Face_Length * (col1 - np.min(col1)) / (np.max(col1) - np.min(col1))

    # 放缩到 0-140
    col2_scaled = Face_Width * (col2 - np.min(col2)) / (np.max(col2) - np.min(col2))

    # 合并回来
    face_landmark_single_scaled = np.stack([col1_scaled, col2_scaled], axis=1)
    return face_landmark_single_scaled


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

def detect_face_landmarks(frame):
    # 转换为灰度图像
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    # 初始化Dlib的人脸检测器和关键点检测器
    detector = dlib.get_frontal_face_detector()
    predictor = dlib.shape_predictor(os.path.dirname(__file__) + "/shape_predictor_68_face_landmarks.dat")

    # 检测人脸
    faces = detector(gray)

    face_landmark_list = []

    # 遍历检测到的人脸
    for face in faces:
        # 获取关键点
        landmarks = predictor(gray, face)
        landmarks_points = []
        for n in range(0, 68):
            x = landmarks.part(n).x
            y = landmarks.part(n).y
            landmarks_points.append((x, y))
            # 绘制关键点
            cv2.circle(frame, (x, y), 1, (0, 0, 255), -1)

        # 打印关键点坐标
        print("人脸关键点坐标：")
        for i, (x, y) in enumerate(landmarks_points, 1):
            print(f"关键点 {i}: ({x}, {y})")

        face_landmark_list.append(landmarks_points)

    return frame, face_landmark_list

def double_face_distance(landmarks_list):
    assert len(landmarks_list) >= 2

    face_alpha =  np.array(landmarks_list[0], dtype=np.float32)
    face_beta =  np.array(landmarks_list[1], dtype=np.float32)
    face_alpha_ = face_measure(face_alpha)
    face_beta_ = face_measure(face_beta)
    print(face_alpha)

    distances = np.linalg.norm(face_alpha_ - face_beta_, axis=1)  # shape: (68,)
    return distances

def violin_plot(distances_) :

    all_data = distances_
    fig, axs = plt.subplots(nrows=1, ncols=2, figsize=(2, 1))

    axs[0].violinplot(all_data, widths=2,
                    showmeans=False,
                    showmedians=False,
                    showextrema = False
                    )    
    axs[0].set_title('Violin plot')

    axs[1].boxplot(all_data)
    axs[1].set_title('Box plot')

    for ax in axs:
        ax.yaxis.grid(True)
        ax.set_xlabel('Four separate samples')
        ax.set_ylabel('Observed values')
    plt.show()


def multi_violin_plot(distances_) :

    all_data = distances_
    fig, axs = plt.subplots(nrows=1, ncols=2, figsize=(9, 4))

    axs[0].violinplot(all_data,
                    showmeans=False,
                    showmedians=True)
    axs[0].set_title('Violin plot')

    axs[1].boxplot(all_data)
    axs[1].set_title('Box plot')

    for ax in axs:
        ax.yaxis.grid(True)
        ax.set_xlabel('Four separate samples')
        ax.set_ylabel('Observed values')
    plt.show()


if __name__ == "__main__":
    video_path ='data/pic_violin/li_rena.mp4'  # 替换为你的视频文件路径
    frame_number = 0  # 替换为你想要的帧号
    frame = get_frame_by_number(video_path, frame_number)
    if frame is not None:
        frame_with_landmarks, landmarks_list = detect_face_landmarks(frame)

        distances_ = double_face_distance(landmarks_list)

        eye_distances = distances_[36:49]
        brow_distances = distances_[17:27]
        nose_distances = distances_[27:36]
        cheek_distances = distances_[0:17]
        
        mouth_distances = distances_[61:]
        jaw_distances = distances_[48:61]

        distances_whole_face = [eye_distances, brow_distances, nose_distances, cheek_distances, mouth_distances  , jaw_distances ]

        # all
        print(distances_whole_face)
        multi_violin_plot(distances_whole_face)

    else:
        print("获取帧失败")