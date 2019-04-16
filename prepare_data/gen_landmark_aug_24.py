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
dstdir = "24/train_RNet_landmark_aug"
OUTPUT = '24'
if not exists(OUTPUT): os.mkdir(OUTPUT)
if not exists(dstdir): os.mkdir(dstdir)
assert(exists(dstdir) and exists(OUTPUT))


def IoU(box, boxes):
    """Compute IoU between detect box and gt boxes

    Parameters:
    ----------
    box: numpy array , shape (5, ): x1, y1, x2, y2, score
        input box
    boxes: numpy array, shape (n, 4): x1, y1, x2, y2
        input ground truth boxes

    Returns:
    -------
    ovr: numpy.array, shape (n, )
        IoU
    """
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
    f = open(join(OUTPUT,"landmark_%s_aug.txt" %(size)),'w')
    data = getDataFromTxt(ftxt)   # ftxt == prepare_data/trainImageList.txt， 存储有lfw_5590内保存的landmark信息
    idx = 0

    for (imgPath, bbox, landmarkGt) in data:  # 一条data记录：lfw_5590\Aaron_Eckhart_0001.jpg 84 161 92 169 106.25 107.75 146.75 112.25 125.25 142.75 105.25 157.75 139.75 161.75
        F_imgs = []      #print imgPath
        F_landmarks = []        
        img = cv2.imread(imgPath)
        assert(img is not None)
        img_h, img_w, img_c = img.shape

        gt_box = np.array([bbox.left, bbox.top, bbox.right, bbox.bottom])
        f_face_raw = img[bbox.top:bbox.bottom + 1, bbox.left:bbox.right + 1]
        f_face = cv2.resize(f_face_raw, (size, size))
        landmark = np.zeros((5, 2))

        for index, one in enumerate(landmarkGt):          # 归1化处理
            rv = ((one[0]-gt_box[0])/(gt_box[2]-gt_box[0]), (one[1]-gt_box[1])/(gt_box[3]-gt_box[1]))
            landmark[index] = rv
        
        F_imgs.append(f_face)
        F_landmarks.append(landmark.reshape(10))
        landmark = np.zeros((5, 2))        
        if argument:
            idx = idx + 1
            if idx % 100 == 0:
                print(idx, "images done")
            x1, y1, x2, y2 = gt_box
            gt_w = x2 - x1 + 1     # gt's width
            gt_h = y2 - y1 + 1     # gt's height

            if max(gt_w, gt_h) < 40 or x1 < 0 or y1 < 0:
                continue

            for i in range(10):  # random shift
                bbox_size = npr.randint(int(min(gt_w, gt_h) * 0.8), np.ceil(1.25 * max(gt_w, gt_h)))
                delta_x = npr.randint(-gt_w * 0.2, gt_w * 0.2)
                delta_y = npr.randint(-gt_h * 0.2, gt_h * 0.2)
                nx1 = max(x1+gt_w/2-bbox_size/2+delta_x, 0)
                ny1 = max(y1+gt_h/2-bbox_size/2+delta_y, 0)
                nx2 = nx1 + bbox_size
                ny2 = ny1 + bbox_size
                if nx2 > img_w or ny2 > img_h:
                    continue
                crop_box = np.array([nx1,ny1,nx2,ny2])
                cropped_im = img[int(ny1):int(ny2)+1,int(nx1):int(nx2)+1,:]
                resized_im = cv2.resize(cropped_im, (size, size))

                iou = IoU(crop_box, np.expand_dims(gt_box,0))     #cal iou
                if iou > 0.65:
                    F_imgs.append(resized_im)
                    #normalize
                    for index, one in enumerate(landmarkGt):
                        rv = ((one[0]-nx1)/bbox_size, (one[1]-ny1)/bbox_size)
                        landmark[index] = rv
                    F_landmarks.append(landmark.reshape(10))
                    landmark = np.zeros((5, 2))
                    landmark_ = F_landmarks[-1].reshape(-1,2)
                    bbox = BBox([nx1,ny1,nx2,ny2])                    

                    #mirror                    
                    if random.choice([0,1]) > 0:
                        face_flipped, landmark_flipped = flip(resized_im, landmark_)
                        face_flipped = cv2.resize(face_flipped, (size, size))
                        #c*h*w
                        F_imgs.append(face_flipped)
                        F_landmarks.append(landmark_flipped.reshape(10))
                    #rotate
                    if random.choice([0,1]) > 0:
                        face_rotated_by_alpha, landmark_rotated = rotate(img, bbox, \
                                                                         bbox.reprojectLandmark(landmark_), 5)#逆时针旋转
                        #landmark_offset
                        landmark_rotated = bbox.projectLandmark(landmark_rotated)
                        face_rotated_by_alpha = cv2.resize(face_rotated_by_alpha, (size, size))
                        F_imgs.append(face_rotated_by_alpha)
                        F_landmarks.append(landmark_rotated.reshape(10))
                
                        #flip
                        face_flipped, landmark_flipped = flip(face_rotated_by_alpha, landmark_rotated)
                        face_flipped = cv2.resize(face_flipped, (size, size))
                        F_imgs.append(face_flipped)
                        F_landmarks.append(landmark_flipped.reshape(10))                
                    
                    #inverse clockwise rotation
                    if random.choice([0, 1]) > 0:
                        face_rotated_by_alpha, landmark_rotated = rotate(img, bbox, \
                                                                         bbox.reprojectLandmark(landmark_), -5)#顺时针旋转
                        landmark_rotated = bbox.projectLandmark(landmark_rotated)
                        face_rotated_by_alpha = cv2.resize(face_rotated_by_alpha, (size, size))
                        F_imgs.append(face_rotated_by_alpha)
                        F_landmarks.append(landmark_rotated.reshape(10))
                
                        face_flipped, landmark_flipped = flip(face_rotated_by_alpha, landmark_rotated)
                        face_flipped = cv2.resize(face_flipped, (size, size))
                        F_imgs.append(face_flipped)
                        F_landmarks.append(landmark_flipped.reshape(10)) 
                    
            F_imgs, F_landmarks = np.asarray(F_imgs), np.asarray(F_landmarks)

            for i in range(len(F_imgs)):
                print(image_id)
                if np.sum(np.where(F_landmarks[i] <= 0, 1, 0)) > 0:
                    continue
                if np.sum(np.where(F_landmarks[i] >= 1, 1, 0)) > 0:
                    continue

                cv2.imwrite(join(dstdir,"%d.jpg" %(image_id)), F_imgs[i])
                landmarks = map(str,list(F_landmarks[i]))
                f.write(join(dstdir,"%d.jpg" %(image_id))+" -2 "+" ".join(landmarks)+"\n")
                image_id = image_id + 1
    f.close()
    return F_imgs, F_landmarks


if __name__ == '__main__':
    net = "RNet"     # train data
    train_txt = "trainImageList.txt"      # prepare_data/trainImageList.txt存储有lfw_5590内保存的landmark信息
    imgs,landmarks = GenerateData(train_txt, OUTPUT,net,argument=True)
    
   
