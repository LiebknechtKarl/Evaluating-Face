


import cv2
import os
import mediapipe as mp

import numpy as np


# def face2point(image_path) :
#     image = cv2.imread(image_path)
# 初始化 MediaPipe 人脸检测器
mp_face_detection = mp.solutions.face_detection
face_detection = mp_face_detection.FaceDetection(model_selection=1, min_detection_confidence=0.5)

def cut_pic(image):
        # 进行人脸检测
    rgb_image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    results = face_detection.process(rgb_image)

    # 如果检测到人脸
    if results.detections:
        for detection in results.detections:
            # 获取人脸的边界框
            bboxC = detection.location_data.relative_bounding_box
            ih, iw, _ = image.shape
            x1, y1 = int(bboxC.xmin * iw), int(bboxC.ymin * ih)
            x2, y2 = int((bboxC.xmin + bboxC.width) * iw), int((bboxC.ymin + bboxC.height) * ih)

            # 裁剪包含人脸的部分
            cropped_face = image[y1:y2, x1:x2]
    return cropped_face



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

                # np.array(landmarks_list)
                #                                      真值     差值
                # max(np.array(landmarks_list)[:,0])   414                1063
                # min(np.array(landmarks_list)[:,0])   198     216        852        211         1.0236966824644549
                # max(np.array(landmarks_list)[:,1])    336               354
                # min(np.array(landmarks_list)[:,1])    87      249       123        231         1.077922077922078
                # 1.0236966824644549            1.077922077922078


        # # 输出每张脸的关键点坐标
        # for face_id, landmarks in faces_dict.items():
        #     print(f'{face_id}: {landmarks}')

        return faces_dict

def video2point(video_path,frame_l = 20) :

    # 加载视频文件
    # video_path = 'path_to_your_video.mp4'  # 替换为你的视频路径
    # video_path = os.path.dirname(os.path.dirname(__file__)) + '/1.mp4'

    cap = cv2.VideoCapture(video_path)

    # 检查视频是否成功打开
    if not cap.isOpened():
        print("无法打开视频文件")
        exit()

    # 设置要提取的帧数（例如第100帧）
    # frame_number = 104

    print('总共帧数:',cap.get(cv2.CAP_PROP_FRAME_COUNT))        # 104    0,1,2,3,4 .... 103

    video_data = []

    # for frame_number in range(int(cap.get(cv2.CAP_PROP_FRAME_COUNT))) :
    # for frame_number in range(25) :
    # for frame_number in range(20) :
    # for frame_number in range(15) :
    for frame_number in range(frame_l) :
    # for frame_number in range(80) :





        # 第 frame_number 张脸
        cap.set(cv2.CAP_PROP_POS_FRAMES, frame_number)

        # 读取该帧
        ret, frame = cap.read()

        # data_1 = face2point(os.path.dirname(os.path.dirname(__file__)) + '/2.png')
        data_pic = face2point(frame)
        # print(frame_number)

        video_data.append(data_pic)

        print('第',frame_number,'帧')

    return video_data
    # print(data_1)

def video_Gs(video_path1,video_path2,b = 1) :

    data1 = np.array(video2point(video_path1))
    data2 = np.array(video2point(video_path2))

    # data1.shape   (5, 1, 478, 2)
    # 5帧  1人     478点    xy

    point_loss_list = []

    for point_i in range(data1.shape[2]) :

        # xh0 = 
        # 第0帧人  point_i点 
        xh0 = data1[0,0,point_i,0]        # 0.53679472, 0.52348125, 0.53761703])
        yh0 = data1[0,0,point_i,1]        # 0.53679472, 0.52348125, 0.53761703])

        xr0 = data2[0,0,point_i,0]        # 0.53679472, 0.52348125, 0.53761703])
        yr0 = data2[0,0,point_i,1]        # 0.53679472, 0.52348125, 0.53761703])


        # xh = data1[1:,0,point_i,0]
        # yh = data1[1:,0,point_i,1]

        # xr = data2[1:,0,point_i,0]
        # yr = data2[1:,0,point_i,1]

        xh_ = data1[1:,0,point_i,0] - xh0
        yh_ = data1[1:,0,point_i,1] - yh0

        xr_ = data2[1:,0,point_i,0] - xr0
        yr_ = data2[1:,0,point_i,1] - yr0




        dh_list = np.sqrt(np.square(xh_) + np.square(yh_))
        dr_list = np.sqrt(np.square(xr_) + np.square(yr_))


        F_list =np.exp( -np.square(dh_list - dr_list)/b)

        point_loss_list.append(sum(F_list)/len(F_list))

    # print(point_loss_list)

    # print('最终Gs', sum(point_loss_list)/len(point_loss_list))

    return sum(point_loss_list)/len(point_loss_list)

def video_Gs_doubleface(video_path1,frame_l,b = 1) :

    # data1 = np.array(video2point(video_path1))

    # data2 = np.array(video2point(video_path1))

    data = np.array(video2point(video_path1,frame_l))
    np.save("data.npy",data)



    if data.shape[1] < 2 :
        print('没有检测到两张脸')
        return 
    data1 = data[:,:1,:,:]
    data2 = data[:,1:2,:,:]


    # data1.shape   (5, 1, 478, 2)
    # 5帧  1人     478点    xy

    point_loss_list = []


    # np.where(np.array(point_loss_list) == 1.0)[0]
    # array([  0,   1,   3,   6,   7,   8,   9,  10,  11,  12,  13,  14,  15,
    #         21,  23,  25,  26,  27,  30,  31,  33,  34,  35,  36,  37,  38,
    #         39,  40,  41,  42,  43,  46,  49,  51,  53,  54,  56,  57,  58,
    #         59,  60,  61,  62,  63,  67,  68,  73,  74,  75,  77,  78,  80,
    #         81,  82,  84,  86,  88,  89,  90,  91,  93,  95,  96,  97,  98,
    #         99, 103, 104, 106, 108, 109, 110, 111, 115, 116, 117, 120, 121,
    #     123, 124, 125, 126, 127, 128, 129, 130, 132, 134, 136, 137, 139,
    #     141, 143, 145, 146, 147, 151, 153, 154, 155, 156, 159, 160, 161,
    #     162, 163, 164, 165, 167, 172, 173, 177, 178, 179, 180, 183, 185,
    #     187, 188, 189, 190, 191, 192, 193, 194, 200, 201, 203, 206, 207,
    #     210, 213, 214, 219, 220, 221, 224, 226, 227, 228, 230, 232, 233,
    #     234, 241, 243, 244, 245, 246, 247, 248, 249, 250, 251, 253, 256,
    #     257, 258, 259, 260, 261, 264, 265, 266, 267, 268, 269, 270, 271,
    #     272, 273, 274, 275, 277, 278, 279, 280, 281, 282, 284, 285, 286,
    #     287, 289, 290, 291, 292, 294, 296, 297, 300, 301, 302, 303, 304,
    #     306, 307, 308, 310, 311, 312, 314, 315, 316, 317, 318, 319, 320,
    #     321, 323, 324, 325, 327, 328, 329, 331, 333, 334, 335, 339, 341,
    #     342, 343, 344, 345, 346, 350, 351, 353, 355, 358, 360, 362, 363,
    #     364, 366, 368, 371, 372, 375, 376, 381, 382, 383, 385, 387, 389,
    #     395, 397, 398, 399, 400, 403, 404, 405, 406, 407, 408, 409, 411,
    #     413, 415, 416, 417, 418, 419, 420, 425, 426, 427, 431, 432, 433,
    #     434, 435, 436, 438, 439, 441, 442, 443, 444, 446, 447, 448, 450,
    #     453, 454, 455, 456, 457, 458, 459, 461, 463, 464, 465, 468, 470,
    #     471, 472, 473, 474, 476, 477])


    point_indices = [  1,   3,   6,   7,   8,   9,  11,  15,  21,  23,  25,  27,  28,
        30,  31,  34,  36,  40,  42,  43,  46,  49,  51,  53,  54,  57,
        59,  60,  67,  68,  75,  77,  84,  85,  87,  88,  90,  91,  95,
        96,  97,  98,  99, 103, 105, 106, 109, 110, 115, 117, 120, 121,
       123, 125, 126, 128, 129, 132, 133, 134, 137, 141, 145, 146, 147,
       149, 151, 152, 153, 154, 155, 156, 159, 161, 162, 164, 165, 167,
       168, 170, 172, 173, 177, 179, 180, 183, 185, 187, 188, 192, 193,
       194, 200, 201, 203, 206, 207, 208, 210, 214, 215, 219, 220, 221,
       222, 224, 227, 228, 230, 232, 233, 241, 243, 244, 245, 246, 247,
       248, 249, 250, 253, 256, 257, 258, 261, 264, 266, 269, 270, 272,
       274, 275, 278, 279, 281, 282, 285, 286, 289, 290, 291, 292, 293,
       294, 296, 298, 299, 306, 307, 314, 315, 316, 318, 319, 320, 321,
       322, 323, 327, 328, 331, 332, 335, 339, 341, 343, 344, 346, 350,
       351, 352, 353, 355, 358, 360, 361, 362, 363, 365, 366, 367, 371,
       381, 382, 385, 398, 399, 400, 403, 404, 405, 406, 407, 408, 409,
       418, 419, 420, 423, 431, 432, 433, 434, 435, 436, 438, 439, 441,
       442, 443, 444, 448, 450, 453, 455, 456, 457, 458, 459, 461, 463,
       464, 465, 468, 472, 474, 476]
    # for point_i in range(int(data1.shape[2]/2)) :
    # for point_i in range(int(data1.shape[2])) :
    # for point_i in range(10) :
    for point_i in point_indices :


        # xh0 = 
        # 第0帧人  point_i点 
        xh0 = data1[0,0,point_i,0]        # 0.53679472, 0.52348125, 0.53761703])
        yh0 = data1[0,0,point_i,1]        # 0.53679472, 0.52348125, 0.53761703])

        xr0 = data2[0,0,point_i,0]        # 0.53679472, 0.52348125, 0.53761703])
        yr0 = data2[0,0,point_i,1]        # 0.53679472, 0.52348125, 0.53761703])

        # xh0,yh0,   xr0,yr0
        # xh,yh,    xr,yr
        # data1[1:,0,point_i,:] , data2[1:,0,point_i,:]
        # data1[1:,0,point_i,:]



        # xh = data1[1:,0,point_i,0]
        # yh = data1[1:,0,point_i,1]

        # xr = data2[1:,0,point_i,0]
        # yr = data2[1:,0,point_i,1]


        # xh_ = data1[1:,0,point_i,0] - xh0
        # yh_ = data1[1:,0,point_i,1] - yh0

        # xr_ = data2[1:,0,point_i,0] - xr0
        # yr_ = data2[1:,0,point_i,1] - yr0

        # b0 = -0.5
        b0 = 0

        print(b0)
        xh_ = data1[1:,0,point_i,0] - xh0 + b0
        yh_ = data1[1:,0,point_i,1] - yh0+ b0

        xr_ = data2[1:,0,point_i,0] - xr0+ b0
        yr_ = data2[1:,0,point_i,1] - yr0+ b0
        # 1.0236966824644549            1.077922077922078    # 下面两行乘以比例
        # xr_ = xr_ * 1.0236966824644549
        # yr_ = yr_ * 1.077922077922078






        # xh_,yh_,            xr_,yr_

        dh_list = np.sqrt(np.square(xh_) + np.square(yh_))
        dr_list = np.sqrt(np.square(xr_) + np.square(yr_))


        F_list =np.exp( -np.square(dh_list - dr_list)/b)

        # point_loss_list.append(sum(F_list)/len(F_list))
        point_loss_list.append(np.average(F_list))


    # print(point_loss_list)

    # print('最终Gs', sum(point_loss_list)/len(point_loss_list))

    print(point_loss_list,'逐点loss')
    print(np.where(np.array(point_loss_list) >= 0.8)[0])
    print(len(np.where(np.array(point_loss_list) >= 0.8)[0]))


    # return sum(point_loss_list)/len(point_loss_list)
    return np.average(point_loss_list)




def main() :
    # 加载图片
    # image_path = os.path.dirname(os.path.dirname(__file__)) + '/3.png'
    image_path = 'data/model_pic/1.jpg'
    image = cv2.imread(image_path)

    data = face2point(image)

    # video_path = os.path.dirname(os.path.dirname(__file__)) + '/1.mp4'
    # video_path = os.path.dirname(os.path.dirname(__file__)) + '/2.webm'
    video_path = 'data/model_pic/2.webm'

    video_data = video2point(video_path)

    # print(video_data)
    video_data_array = np.array(video_data)
    # print(video_data_array)
if __name__ == "__main__":
    main()

