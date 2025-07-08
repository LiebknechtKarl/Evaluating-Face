





import cv2
import os
import mediapipe as mp

import numpy as np
from skimage.metrics import structural_similarity as ssim
from PIL import Image
import numpy as np

from PIL import Image

mp_face_detection = mp.solutions.face_detection
face_detection = mp_face_detection.FaceDetection(model_selection=1, min_detection_confidence=0.5)

# def face2point(image_path) :
#     image = cv2.imread(image_path)
# 初始化 MediaPipe 人脸检测器
def face2point(image) :
    # image = cv2.imread(image_path)
    # 初始化 Mediapipe 的人脸检测和面部关键点提取模块
    mp_face_mesh = mp.solutions.face_mesh
    mp_drawing = mp.solutions.drawing_utils
    mp_drawing_styles = mp.solutions.drawing_styles

    # # 将图片转换为RGB格式，因为Mediapipe使用的是RGB格式
    image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    # image_rgb = cut_pic(image)

    # 获取图片尺寸
    image_height, image_width, _ = image.shape

    # 创建一个 FaceMesh 对象，允许检测多张人脸
    with mp_face_mesh.FaceMesh(
        static_image_mode=True,  # 对静态图像进行检测
        max_num_faces=2,         # 允许检测最多两张人脸
        refine_landmarks=True,   # 细化嘴唇、眼睛、瞳孔区域的关键点
        # min_detection_confidence=0.5  # 最小检测置信度
        min_detection_confidence=0.5 # 最小检测置信度

    ) as face_mesh:

        # 对图片进行面部关键点检测
        results = face_mesh.process(image_rgb)

        # 创建一个字典来存储每张脸的关键点
        # faces_dict = {}
        faces_dict = []


        if results.multi_face_landmarks:
            print('检测面部')
            for face_idx, face_landmarks in enumerate(results.multi_face_landmarks):
                # 创建一个列表存储每张脸的 x 和 y 坐标
                landmarks_list = []

                print(f'-------------第 {face_idx+1}/{len(results.multi_face_landmarks)} 张脸')

                # 遍历每个关键点
                for i, landmark in enumerate(face_landmarks.landmark):
                    x_px = int(landmark.x * image_width)  # 转换为像素值
                    y_px = int(landmark.y * image_height) # 转换为像素值
                    # x_px = landmark.x  # 此处是归一化
                    # y_px = landmark.y  # 此处是归一化
                    # landmarks_list.append((x_px, y_px))   # 将 x 和 y 坐标存储为元组
                    landmarks_list.append([x_px, y_px])   # 将 x 和 y 坐标存储为列表


                # 将关键点列表存储在字典中，以当前脸的编号作为键
                # faces_dict[f'Face_{face_idx+1}'] = landmarks_list
                # faces_dict[f'{face_idx+1}'] = landmarks_list
                faces_dict.append(landmarks_list)

        return np.array(faces_dict)



def point_normalization(faces_dict) :   # 归一化

    for i in range(faces_dict.shape[0]):

        # 提取所有的 x 坐标
        # x_coords = faces_dict[:, :, 0]
        x_coords = faces_dict[i, :, 0]
        y_coords = faces_dict[i, :, 1]



        # 在 x 坐标上进行归一化
        x_min = np.min(x_coords)
        x_max = np.max(x_coords)
        y_min = np.min(y_coords)
        y_max = np.max(y_coords)        

        # 归一化公式
        x_normalized = (x_coords - x_min) / (x_max - x_min )
        y_normalized = (y_coords - y_min) / (y_max - y_min )
        # x_normalized = (x_coords - x_min) 
        # y_normalized = (y_coords - y_min) 

        # # 将归一化后的 x 坐标赋回原始数组中
        # faces_dict[i, :, 0] = x_normalized
        # faces_dict[i, :, 1] = y_normalized

        # 将两个数组沿最后一个维度进行堆叠，形成 (478, 2) 的数组
        faces_dict = np.stack((x_normalized, y_normalized), axis=-1)

        # print(faces_dict)
    return faces_dict

def face_norm(image_path) :

    # image_path = os.path.dirname(__file__) + '/pic_cu/1_h.jpg'
    image = cv2.imread(image_path)
    face478point = face2point(image)
    face_norm = point_normalization(face478point)

    return face_norm
def compute_lmd(predicted_points, true_points):
    # 假设 predicted_points 和 true_points 都是 n x 2 的数组，每行是一个标志点的 (x, y) 坐标
    distances = np.sqrt(np.sum((predicted_points - true_points)**2, axis=1))
    # np.sqrt(np.sum((np.array([[1,1]]),np.array([[2,3]]))**2, axis=1))

    # predicted_points
    # array([[0.80232558, 0.71717172],
    #     [0.87209302, 0.57575758],
    #     [0.8255814 , 0.61616162],

    # true_points
    # array([[0.71875   , 0.68571429],
    #     [0.765625  , 0.57142857],
    #     [0.734375  , 0.61428571],
    #     [0.703125  , 0.44285714],

    # (0.80232558 - 0.71875)**2  +  (0.71717172- 0.68571429)**2   =  0.007974447474541298
    lmd = np.mean(distances)
    # lmd = np.sum(distances)

    return lmd

def main() :
    # pass
    # image1 = Image.open('pic_cu/1_h.jpg')

    # image_path = os.path.dirname(os.path.dirname(__file__)) + 'pic_cu/1_h.jpg'

    file_list =  [['/pic_cu/1_h.jpg','/pic_cu/1_r.jpg'] , 
                  ['/pic_cu/2_h.jpg','/pic_cu/2_r.jpg'] , 
                  ['/pic_cu/3_h.jpg','/pic_cu/3_r.jpg'] ,
                  ['/pic_cu/4_h.jpg','/pic_cu/4_r.jpg'] ,
                  ['/pic_cu/5_h.jpg','/pic_cu/5_r.jpg'] 

                  ] 

    lmd_loss = []
    # for [i,j] in [['/pic_cu/1_h.jpg','/pic_cu/1_r.jpg'] , ['/pic_cu/1_h.jpg','/pic_cu/1_r.jpg'] , ['/pic_cu/1_h.jpg','/pic_cu/1_r.jpg'] ] :
    for [i,j] in file_list :

        # image_path1_h = os.path.dirname(__file__) + '/pic_cu/1_h.jpg'
        # image_path1_h = os.path.dirname(__file__) + i
        image_path1_h = os.path.dirname(os.path.dirname(__file__)) + '/data' + i


        face_norm1_h = face_norm(image_path1_h)
        # print(face_norm1_h)

        # image_path1_r = os.path.dirname(__file__) + '/pic_cu/1_r.jpg'
        image_path1_r =  os.path.dirname(os.path.dirname(__file__)) + '/data' + j

        face_norm1_r = face_norm(image_path1_r)
        # print(face_norm1_r)

        lmd1 = compute_lmd(face_norm1_h,face_norm1_r)
        lmd_loss.append(lmd1)

    # for k in rang
    k = 0 
    for [i,j] in file_list :
        print(i,j, lmd_loss[k])
        k = k +1


if __name__ == "__main__":
    main()









