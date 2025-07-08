import cv2
import dlib
import os
import numpy as np
# import dtw
from dtaidistance import dtw
# 初始化Dlib的人脸检测器和关键点检测器
detector = dlib.get_frontal_face_detector()
predictor = dlib.shape_predictor(os.path.dirname(__file__) + "/shape_predictor_68_face_landmarks.dat")

def normalize_landmarks(landmarks, height_range=(0, 180), width_range=(0, 140)):
    normalized_landmarks = []
    for (x, y) in landmarks:
        # 归一化x坐标到宽度范围
        normalized_x = np.interp(x, [0, 1920], width_range)
        # 归一化y坐标到高度范围
        normalized_y = np.interp(y, [0, 1080], height_range)
        normalized_landmarks.append((normalized_x, normalized_y))
    return normalized_landmarks


def normalize_each_face(landmarks_points, height_range=(0, 180), width_range=(0, 140)):
    # max(np.array(landmarks_points, dtype=np.float32)[:,0])        np.float32(1016.0)
    # min(np.array(landmarks_points, dtype=np.float32)[:,0])        np.float32(791.0)
    # max(np.array(landmarks_points, dtype=np.float32)[:,0]) - min(np.array(landmarks_points, dtype=np.float32)[:,0])  
    h_ =  height_range[1] - height_range[0]
    h0 = max(np.array(landmarks_points, dtype=np.float32)[:,0]) - min(np.array(landmarks_points, dtype=np.float32)[:,0])

    # max(np.array(landmarks_points, dtype=np.float32)[:,1])        np.float32(582.0)
    # min(np.array(landmarks_points, dtype=np.float32)[:,1])        np.float32(376.0)
    # max(np.array(landmarks_points, dtype=np.float32)[:,1]) - min(np.array(landmarks_points, dtype=np.float32)[:,1])   

    normalized_landmarks = []
    for (x, y) in landmarks_points:
        # 归一化x坐标到宽度范围
        normalized_x = np.interp(x, [min(np.array(landmarks_points, dtype=np.float32)[:,0]) , max(np.array(landmarks_points, dtype=np.float32)[:,0]) ], width_range)
        # 归一化y坐标到高度范围
        normalized_y = np.interp(y, [min(np.array(landmarks_points, dtype=np.float32)[:,1]) , max(np.array(landmarks_points, dtype=np.float32)[:,1]) ], height_range)

        normalized_landmarks.append((normalized_x, normalized_y))


    # width = 
    return normalized_landmarks


def Space_similarity(face1,face2) :
    if len(face1) != len(face2):
        return None
    # 计算每个关键点的欧几里得距离
    distances = []
    for p1, p2 in zip(face1, face2):
        distance = np.linalg.norm(np.array(p1) - np.array(p2))
        distances.append(distance)
    # 计算平均距离作为空间相似性度量（距离越小，相似性越高）
    average_distance = np.mean(distances)
    return average_distance    


def Dtw_similarity(face1_landmarks, face2_landmarks):
    # 计算每个关键点的时间相似性
    dtw_distances = []
    for i in range(68):
        # 提取单个关键点的 x 和 y 坐标序列
        # seq1_x = face1_landmarks[:, i, 0].flatten().astype(np.int32)  # x 坐标
        # seq2_x = face2_landmarks[:, i, 0].flatten().astype(np.int32)  # x 坐标
        # seq1_y = face1_landmarks[:, i, 1].flatten().astype(np.int32)  # y 坐标
        # seq2_y = face2_landmarks[:, i, 1].flatten().astype(np.int32)  # y 坐标
        

        seq1_x = face1_landmarks[:, i, 0].flatten().astype(np.float64)
        seq2_x = face2_landmarks[:, i, 0].flatten().astype(np.float64)
        seq1_y = face1_landmarks[:, i, 1].flatten().astype(np.float64)
        seq2_y = face2_landmarks[:, i, 1].flatten().astype(np.float64)


        # 确保序列长度一致
        min_len = min(len(seq1_x), len(seq2_x))
        seq1_x = seq1_x[:min_len]
        seq2_x = seq2_x[:min_len]
        seq1_y = seq1_y[:min_len]
        seq2_y = seq2_y[:min_len]
        
        # 计算 DTW 距离
        distance_x = dtw.distance_fast(seq1_x, seq2_x, use_pruning=True)
        distance_y = dtw.distance_fast(seq1_y, seq2_y, use_pruning=True)
        
        dtw_distances.append((distance_x + distance_y) / 2)
    
    # 计算平均 DTW 距离作为时间相似性度量
    average_dtw_distance = np.mean(dtw_distances)

    assert face1_landmarks.shape[0] == face2_landmarks.shape[0]
    return average_dtw_distance/face1_landmarks.shape[0]


def Movement_smoothness(face_landmarks):
    """
    输入: face_landmarks.shape = (T帧数, 68关键点, 2)
    输出: float，越小表示越平滑
    """
    face_landmarks = np.array(face_landmarks, dtype=np.float32)  # 确保类型正确

    # 计算每个点每一维的速度
    velocity = np.diff(face_landmarks, axis=0)  # shape (T-1, 68, 2)

    # 计算加速度
    acceleration = np.diff(velocity, axis=0)    # shape (T-2, 68, 2)

    # 计算每个加速度向量的模长（L2范数）
    acc_magnitude = np.linalg.norm(acceleration, axis=2)  # shape (T-2, 68)

    # 平均加速度模长作为平滑度指标（越小越平滑）
    smoothness_score = np.mean(acc_magnitude)

    return smoothness_score

def process_video(input_video_path, output_matrix1_path, output_matrix2_path):
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

    # 初始化两个矩阵来存储人脸关键点坐标
    face1_landmarks_matrix = []
    face2_landmarks_matrix = []

    frame_count = 0
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        frame_count += 1
        print(f"处理第 {frame_count} 帧")

        # 转换为灰度图像
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        # 检测人脸
        faces = detector(gray)

        # 确保检测到两个人脸
        if len(faces) != 2:
            print(f"第 {frame_count} 帧检测到的人脸数量不是2，跳过此帧")
            continue
        
        # if frame_count >10 :
        #     break

        # 遍历检测到的人脸
        landmarks_list = []
        for face in faces:
            # 获取关键点
            landmarks = predictor(gray, face)
            landmarks_points = []
            for n in range(0, 68):
                x = landmarks.part(n).x
                y = landmarks.part(n).y
                landmarks_points.append((x, y))
        # np.array(landmarks_points, dtype=np.float32)
            landmarks_points = normalize_each_face(landmarks_points)


            # max(np.array(landmarks_points, dtype=np.float32)[:,0])        np.float32(140.0)
            # max(np.array(landmarks_points, dtype=np.float32)[:,1])        np.float32(180.0)

            landmarks_list.append(landmarks_points)

        # # 归一化关键点坐标
        # normalized_landmarks_list = [normalize_landmarks(landmarks, (0, 180), (0, 140)) for landmarks in landmarks_list]
        normalized_landmarks_list = landmarks_list

        # 将归一化后的关键点坐标添加到矩阵
        face1_landmarks_matrix.append(normalized_landmarks_list[0])
        face2_landmarks_matrix.append(normalized_landmarks_list[1])

    # 释放资源
    cap.release()

    # 将矩阵保存到文件
    np.save(output_matrix1_path, face1_landmarks_matrix)
    np.save(output_matrix2_path, face2_landmarks_matrix)

    # np.array(face1_landmarks_matrix, dtype=np.float32).shape
    # (208, 68, 2)
    face_1 = np.array(face1_landmarks_matrix, dtype=np.float32)
    face_2 = np.array(face2_landmarks_matrix, dtype=np.float32)

    print(face_1)

    ss_distance = Space_similarity(face_1, face_2)

    print('Space-similarity（空间相似性）' , ss_distance)


    # dtw_distance = Dtw_similarity(face_1, face_2)
    dtw_distance = Dtw_similarity(face_1, face_2)

    print('动态时间规整（DTW）' , dtw_distance)

    smoothness1 = Movement_smoothness(face_1)
    smoothness2 = Movement_smoothness(face_2)

    print("face_1 平滑度评分：", smoothness1)
    print("face_2 平滑度评分：", smoothness2)


if __name__ == "__main__":
    # input_video_path = os.path.dirname(__file__) + '/cropped_li_rena.mp4'  # 输入视频文件路径
    input_video_path = 'data/pic_violin/cropped_li_rena.mp4'  # 输入视频文件路径

    output_matrix1_path = os.path.dirname(__file__) + '/face1_landmarks.npy'  # 第一张人脸关键点矩阵保存路径
    output_matrix2_path = os.path.dirname(__file__) + '/face2_landmarks.npy'  # 第二张人脸关键点矩阵保存路径

    process_video(input_video_path, output_matrix1_path, output_matrix2_path)


