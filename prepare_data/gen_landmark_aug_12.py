# coding: utf-8
import os
import time
import math
from os.path import join, exists
import cv2
import numpy as np
from BBox_utils import getDataFromTxt,processImage,shuffle_in_unison_scary,BBox
from Landmark_utils import show_landmark,rotate,flip
import random
import tensorflow as tf
import sys
import numpy.random as npr
dstdir = "12/train_PNet_landmark_aug"
OUTPUT = '12'
if not exists(OUTPUT): os.mkdir(OUTPUT)
if not exists(dstdir): os.mkdir(dstdir)
assert(exists(dstdir) and exists(OUTPUT))

def IoU(box, boxes):
    box_area = (box[2] - box[0] + 1) * (box[3] - box[1] + 1)
    area = (boxes[:, 2] - boxes[:, 0] + 1) * (boxes[:, 3] - boxes[:, 1] + 1)
    xx1 = np.maximum(box[0], boxes[:, 0])
    yy1 = np.maximum(box[1], boxes[:, 1])
    xx2 = np.minimum(box[2], boxes[:, 2])
    yy2 = np.minimum(box[3], boxes[:, 3])
     # compute the width and height of the bounding box
    w = np.maximum(0, xx2 - xx1 + 1)
    h = np.maximum(0, yy2 - yy1 + 1)
    inter = w * h
    ovr = inter*1.0 / (box_area + area - inter)
    return ovr

def GenerateData(ftxt, output,net,argument=False):
    if net == "PNet":
        size = 12
    elif net == "RNet":
        size = 24
    elif net == "ONet":
        size = 48
    else:
        print('Net type error')
        return
    image_id = 0
    f = open(join(OUTPUT,"landmark_%s_aug.txt" %(size)), 'w')
    #dstdir = "train_landmark_few"
   
    data = getDataFromTxt(ftxt)  # ftxt == trainImageList.txt，从文本里面拿到每张图片人脸的3份数据（文件名，人脸框，特征点坐标 ）
    idx = 0
    for (imgPath, bbox, landmarkGt) in data:      #遍历每张原始照片，image_path bbox landmark(5*2)
        F_imgs = []
        F_landmarks = []        
        img = cv2.imread(imgPath)
        assert(img is not None)
        img_h, img_w, img_c = img.shape
        gt_box = np.array([bbox.left, bbox.top, bbox.right, bbox.bottom])  # 标签中人脸的区域
        f_face = img[bbox.top: bbox.bottom+1, bbox.left: bbox.right+1]  # 拿到标签中人脸的矩形框
        f_face = cv2.resize(f_face, (size, size))   # 将人脸resize成12 * 12大小
        landmark = np.zeros((5, 2))
        #normalize
        for index, one in enumerate(landmarkGt):  # 遍历人脸5个特征点的label,index是数组下标。
            rv = ((one[0]-gt_box[0])/(gt_box[2]-gt_box[0]), (one[1]-gt_box[1])/(gt_box[3]-gt_box[1]))  # rv是一个（x, y），标识1个特征点坐标，做归一化处理，相对于label框框的比例
            landmark[index] = rv
        
        F_imgs.append(f_face)  # 将1个人脸像素数据加入F_img的list中
        F_landmarks.append(landmark.reshape(10))  # 将1个人脸特征点数据加入F_landmarks的list中
        landmark = np.zeros((5, 2))        
        if argument:
            idx = idx + 1
            if idx % 100 == 0:
                print(idx, "images done")
            x1, y1, x2, y2 = gt_box  # 点1（x1，y1）位于左上角，点2（x2，y2）位于右下角
            gt_w = x2 - x1 + 1    # gt's width
            gt_h = y2 - y1 + 1    # gt's height
            if max(gt_w, gt_h) < 40 or x1 < 0 or y1 < 0:  # 过小的图片 && 非法图片忽略
                continue
            #random shift
            for i in range(10):
                bbox_size = npr.randint(int(min(gt_w, gt_h) * 0.8), np.ceil(1.25 * max(gt_w, gt_h)))
                delta_x = npr.randint(-gt_w * 0.2, gt_w * 0.2)
                delta_y = npr.randint(-gt_h * 0.2, gt_h * 0.2)
                nx1 = max(x1+gt_w/2-bbox_size/2+delta_x,0)  # gt_w/2-bbox_size/2 == 因为box尺寸变化产生的偏移
                ny1 = max(y1+gt_h/2-bbox_size/2+delta_y,0)
                
                nx2 = nx1 + bbox_size
                ny2 = ny1 + bbox_size
                if nx2 > img_w or ny2 > img_h:
                    continue
                crop_box = np.array([nx1, ny1, nx2, ny2])
                cropped_im = img[int(ny1):int(ny2)+1, int(nx1):int(nx2)+1, :]
                resized_im = cv2.resize(cropped_im, (size, size))
                #cal iou
                iou = IoU(crop_box, np.expand_dims(gt_box, 0))

                if iou > 0.65:
                    F_imgs.append(resized_im)  # 将12*12的crop框框也加入F_imgs 的list中
                    for index, one in enumerate(landmarkGt):   # normalize归一化处理
                        rv = ((one[0]-nx1)/bbox_size, (one[1]-ny1)/bbox_size)
                        landmark[index] = rv
                    F_landmarks.append(landmark.reshape(10))
                    landmark = np.zeros((5, 2))
                    landmark_ = F_landmarks[-1].reshape(-1,2)  # 将最新加入的list[-1]的len为10的一维数组，reshape成(-1,2)
                    bbox = BBox([nx1, ny1, nx2, ny2])

                    # 1 mirror   # 翻转，以轴为参照
                    if random.choice([0,1]) > 0:  #
                        face_flipped, landmark_flipped = flip(resized_im, landmark_)  # 截取的像素数据 && 特征点数据做翻转
                        face_flipped = cv2.resize(face_flipped, (size, size))
                        #c*h*w
                        F_imgs.append(face_flipped)  # 12 * 12的图片加入F_imgs队列。
                        F_landmarks.append(landmark_flipped.reshape(10))
                     # 逆时针旋转 以中心点为参照, 旋转α角度
                    if random.choice([0, 1]) > 0:
                        face_rotated_by_alpha, landmark_rotated = rotate(img, bbox, \
                                                                         bbox.reprojectLandmark(landmark_), 5)#逆时针旋转
                    # 2  顺时针旋转
                        landmark_rotated = bbox.projectLandmark(landmark_rotated)  # 根据landmark的坐标做重新生成landmark_rotated？
                        face_rotated_by_alpha = cv2.resize(face_rotated_by_alpha, (size, size))
                        F_imgs.append(face_rotated_by_alpha)
                        F_landmarks.append(landmark_rotated.reshape(10))

                     # 3  顺时针旋转后再翻转
                        face_flipped, landmark_flipped = flip(face_rotated_by_alpha, landmark_rotated)
                        face_flipped = cv2.resize(face_flipped, (size, size))
                        F_imgs.append(face_flipped)
                        F_landmarks.append(landmark_flipped.reshape(10))                
                    

                    if random.choice([0,1]) > 0:
                    # 4  逆时针旋转
                        face_rotated_by_alpha, landmark_rotated = rotate(img, bbox, \
                                                                         bbox.reprojectLandmark(landmark_), -5)#顺时针旋转
                        landmark_rotated = bbox.projectLandmark(landmark_rotated)
                        face_rotated_by_alpha = cv2.resize(face_rotated_by_alpha, (size, size))
                        F_imgs.append(face_rotated_by_alpha)
                        F_landmarks.append(landmark_rotated.reshape(10))
                    # 5  逆时针旋转
                        face_flipped, landmark_flipped = flip(face_rotated_by_alpha, landmark_rotated)
                        face_flipped = cv2.resize(face_flipped, (size, size))
                        F_imgs.append(face_flipped)
                        F_landmarks.append(landmark_flipped.reshape(10)) 
                    
            F_imgs, F_landmarks = np.asarray(F_imgs), np.asarray(F_landmarks) # 将list转换为数组，维度保持不变。
            #print F_imgs.shape            # print F_landmarks.shape
            for i in range(len(F_imgs)):   # 一张pic里面，数据增强后形成所有 12 * 12的图片
                print(image_id)                                      # 剔除非法特征点？哪里来的非法特征点？
                if np.sum(np.where(F_landmarks[i] <= 0, 1, 0)) > 0:  # 将F_landmarks中所有 <= 0的数字变为1，>0 的数变为0
                    continue
                if np.sum(np.where(F_landmarks[i] >= 1, 1, 0)) > 0:
                    continue

                cv2.imwrite(join(dstdir,"%d.jpg" %(image_id)), F_imgs[i])  # dstdir=12/train_PNet_landmark_aug"
                landmarks = map(str, list(F_landmarks[i]))
                f.write(join(dstdir, "%d.jpg" %(image_id))+" -2 "+" ".join(landmarks)+"\n")  # 将特征点坐标，写入landmark_12_aug的txt文件。 12/train_PNet_landmark_aug
                image_id = image_id + 1

    f.close()
    return F_imgs,F_landmarks

if __name__ == '__main__':
    # train data
    net = "PNet"
    #train_txt = "train.txt"
    train_txt = "trainImageList.txt"
    imgs,landmarks = GenerateData(train_txt, OUTPUT,net,argument=True)
    
   
