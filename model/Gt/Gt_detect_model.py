import cv2
import os
import mediapipe as mp
import numpy as np
import sys,os
sys.path.append(os.path.dirname(os.path.dirname(__file__)))
from Gs.face_detect_model import video2point, face2point

def video_Gt_doubleface(video_path1,frame_l,b = 1) :

    data = np.array(video2point(video_path1,frame_l))
    np.save("data_gt.npy",data)

    if data.shape[1] < 2 :
        print('没有检测到两张脸')
        return 
    data1 = data[:,:1,:,:]
    data2 = data[:,1:2,:,:]

    # data1.shape   (5, 1, 478, 2)
    # 5帧  1人     478点    xy
    point_loss_list = []

    for point_i in range(data1.shape[2]) :

        # 第0帧人  point_i点 
        xh0 = data1[0,0,point_i,0]                          # 0.53679472, 0.52348125, 0.53761703])
        yh0 = data1[0,0,point_i,1]                          # 0.53679472, 0.52348125, 0.53761703])

        xr0 = data2[0,0,point_i,0]                          # 0.53679472, 0.52348125, 0.53761703])
        yr0 = data2[0,0,point_i,1]                          # 0.53679472, 0.52348125, 0.53761703])

        # xh0,yh0,   xr0,yr0
        # xh,yh,    xr,yr

        b0 = 1
        print(b0)
        xh_ = data1[1:,0,point_i,0] - xh0 + b0
        yh_ = data1[1:,0,point_i,1] - yh0+ b0

        xr_ = data2[1:,0,point_i,0] - xr0+ b0
        yr_ = data2[1:,0,point_i,1] - yr0+ b0
        # 1.0236966824644549            1.077922077922078    # 下面两行乘以比例
        # xr_ = xr_ * 1.0236966824644549
        # yr_ = yr_ * 1.077922077922078

        # xh_,yh_,            xr_,yr_
        dh_list = np.sqrt(np.square(xh_) + np.square(yh_))   # dh_list.shape     (9,)
        dr_list = np.sqrt(np.square(xr_) + np.square(yr_))

        diff_dh_list = np.diff(dh_list)
        diff_dr_list = np.diff(dr_list)

        F_list =np.exp( -np.square(diff_dh_list - diff_dr_list)/b)
        point_loss_list.append(np.average(F_list))

    return np.average(point_loss_list)

if __name__ == "__main__":
    # 加载图片
    image_path = 'data/model_pic/1.jpg'
    image = cv2.imread(image_path)

    data = face2point(image)

    # video_path = os.path.dirname(os.path.dirname(__file__)) + '/1.mp4'
    video_path ='data/model_pic/2.webm'
    video_data = video2point(video_path)

    video_data_array = np.array(video_data)
    print(video_data_array)

