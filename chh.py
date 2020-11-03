import dlib
import numpy as np
import cv2
import os
import pandas as pd
import time

test_img_path = ["D:/data/a.jpg"]
imgs = [cv2.imread(test_img_path[0])]

# Dlib 正向人脸检测器 / Use frontal face detector of Dlib
detector = dlib.get_frontal_face_detector()

# Dlib 人脸 landmark 特征点检测器 / Get face landmarks
predictor = dlib.shape_predictor('data/data_dlib/shape_predictor_68_face_landmarks.dat')

# Dlib Resnet 人脸识别模型，提取 128D 的特征矢量 / Use Dlib resnet50 model to get 128D face descriptor
face_reco_model = dlib.face_recognition_model_v1("data/data_dlib/dlib_face_recognition_resnet_model_v1.dat")

# cap = cv2.VideoCapture(0)
# fps = cap.get(cv2.CAP_PROP_FPS)
# ret, frame = cap.read()



faces = detector(imgs[0], 0)

dict={'a':1, 'b':2}
print(dict)
faces1 = faces
faces_= {}
for k,d  in enumerate(faces1):
    rect1 = dlib.rectangle(d.bottom(),  d.top(), d.right(), d.left())
    height = (d.bottom() - d.top())
    width = (d.right() - d.left())
    hh = int(height / 2)
    ww = int(width / 2)
    # d.bottom() = 0
    # d.top() = 1080
    # d.left() = 0
    # d.right() = 1440

shape = predictor(imgs[0],faces[0])
a = shape.num_parts
b = shape.rect
face_descriptor = face_reco_model.compute_face_descriptor(imgs[0], shape)
print(face_descriptor)
print("@")
print(shape)
print("@@@@@@")

