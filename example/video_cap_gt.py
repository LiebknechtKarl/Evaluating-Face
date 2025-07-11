import cv2, os
import sys,os
import numpy as np
sys.path.append(os.path.dirname(os.path.dirname(__file__)))
sys.path.append(os.path.dirname(__file__))
from model.Gt.Gt_detect_model import video_Gt_doubleface

if __name__ == "__main__":
    b = 1
    video_path1 = 'data/model_pic/10.mp4'
    frame_l = 58
    loss = video_Gt_doubleface(video_path1,frame_l,b = 0.5)
    print(loss)


