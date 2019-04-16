#coding:utf-8
import sys
#生成pnet的对齐数据
import numpy as np
import cv2
import os
import numpy.random as npr
from utils import IoU
anno_file = "wider_face_train.txt"
im_dir = "WIDER_train/images"
pos_save_dir = "12/positive"
part_save_dir = "12/part"
neg_save_dir = '12/negative'
save_dir = "./12"
if not os.path.exists(save_dir):
    os.mkdir(save_dir)
if not os.path.exists(pos_save_dir):
    os.mkdir(pos_save_dir)
if not os.path.exists(part_save_dir):
    os.mkdir(part_save_dir)
if not os.path.exists(neg_save_dir):
    os.mkdir(neg_save_dir)

f1 = open(os.path.join(save_dir, 'pos_12.txt'), 'w')
f2 = open(os.path.join(save_dir, 'neg_12.txt'), 'w')
f3 = open(os.path.join(save_dir, 'part_12.txt'), 'w')
with open(anno_file, 'r') as f:  # anno_file = "wider_face_train.txt"，存储的人脸区域数据
    annotations = f.readlines()  # 每一行保存一个图片中，一个人脸的矩形框参数
num = len(annotations)
print("%d pics in total" % num)
p_idx = 0 # positive
n_idx = 0 # negative
d_idx = 0 # dont care
idx = 0
box_idx = 0
for annotation in annotations:
    annotation = annotation.strip().split(' ')  # strip()
    #image path
    im_path = annotation[0]
    #boxed change to float type
    bbox = map(float, annotation[1:])
    #gt
    boxes = np.array(list(bbox), dtype=np.float32).reshape(-1, 4)
    #load image
    img = cv2.imread(os.path.join(im_dir, im_path + '.jpg'))  # 'WIDER_train/images       0--Parade/0_Parade_marchingband_1_849                   .jpg'
    idx += 1  # 记录已读图片的张数
    if idx % 100 == 0:
        print(idx, "i"
                   "mages done")
        
    height, width, channel = img.shape

    neg_num = 0
    #1---->50
    while neg_num < 50:  # 在1个图片中找50个negative的矩形框
        #neg_num's size [40,min(width, height) / 2],min_size:40 
        size = npr.randint(12, min(width, height) / 2)  # 随机产生一个矩形框，边长是12 ~ 矩形短边长的一半
        #top_left
        nx = npr.randint(0, width - size)
        ny = npr.randint(0, height - size)
        #random crop
        crop_box = np.array([nx, ny, nx + size, ny + size])
        #cal iou  ,intersection-over-union
        Iou = IoU(crop_box, boxes)  # boxex中存储label中标注好的人脸矩形框坐标
        
        cropped_im = img[ny : ny + size, nx : nx + size, :]  # 剪裁图像的三维尺寸 ，矩形框的最后一行表示rgb的3个通道全采集
        resized_im = cv2.resize(cropped_im, (12, 12), interpolation=cv2.INTER_LINEAR)  # 把图片保存成12 * 12 大小的

        if np.max(Iou) < 0.3:
            # Iou with all gts must below 0.3
            save_file = os.path.join(neg_save_dir, "%s.jpg"%n_idx)
            f2.write("12/negative/%s.jpg"%n_idx + ' 0\n')
            cv2.imwrite(save_file, resized_im)
            n_idx += 1
            neg_num += 1
    #as for 正 part样本
    for box in boxes:
        # box (x_left, y_top, x_right, y_bottom)
        x1, y1, x2, y2 = box
        w = x2 - x1 + 1   #gt's width   标签中的矩形框宽
        h = y2 - y1 + 1   #gt's height  标签中的矩形框宽

        # ignore small faces， in case the ground truth boxes of small faces are not accurate
        if max(w, h) < 40 or x1 < 0 or y1 < 0:
            continue
        for i in range(5):  #
            size = npr.randint(12, min(width, height) / 2)  # 初始化一个随机的矩形框

            delta_x = npr.randint(max(-size, -x1), w)  # delta_x and delta_y are offsets of (x1, y1)
            delta_y = npr.randint(max(-size, -y1), h)
            nx1 = int(max(0, x1 + delta_x))  # x1 + delta_x可能会比width大， 下一步剔除这些点
            ny1 = int(max(0, y1 + delta_y))
            if nx1 + size > width or ny1 + size > height:
                continue
            crop_box = np.array([nx1, ny1, nx1 + size, ny1 + size])  # 以标注的点为基础，随机偏移切割形成新的点
            Iou = IoU(crop_box, boxes)
    
            cropped_im = img[ny1: ny1 + size, nx1: nx1 + size, :]
            resized_im = cv2.resize(cropped_im, (12, 12), interpolation=cv2.INTER_LINEAR)
    
            if np.max(Iou) < 0.3:

                save_file = os.path.join(neg_save_dir, "%s.jpg" % n_idx)   # Iou with all gts must below 0.3
                f2.write("12/negative/%s.jpg" % n_idx + ' 0\n')
                cv2.imwrite(save_file, resized_im)
                n_idx += 1        
	# generate positive examples and part faces
        for i in range(20):
            size = npr.randint(int(min(w, h) * 0.8), np.ceil(1.25 * max(w, h)))  #np.ceil(1.2) == 2 ,函数返回比输入值大的，最小整数， pos and part face size [minsize*0.8,maxsize*1.25]
            delta_x = npr.randint(-w * 0.2, w * 0.2)   # delta here is the offset of box center
            delta_y = npr.randint(-h * 0.2, h * 0.2)

            nx1 = int(max(x1 + w / 2 + delta_x - size / 2, 0))  # show this way: nx1 = max(x1+w/2-size/2+delta_x)
            ny1 = int(max(y1 + h / 2 + delta_y - size / 2, 0))  # show this way: ny1 = max(y1+h/2-size/2+delta_y)
            nx2 = nx1 + size
            ny2 = ny1 + size

            if nx2 > width or ny2 > height:
                continue 
            crop_box = np.array([nx1, ny1, nx2, ny2])
            #yu gt de offset
            offset_x1 = (x1 - nx1) / float(size)  # crop框偏移比例，相比于crop框的size
            offset_y1 = (y1 - ny1) / float(size)
            offset_x2 = (x2 - nx2) / float(size)
            offset_y2 = (y2 - ny2) / float(size)
            #crop
            cropped_im = img[ny1 : ny2, nx1 : nx2, :]
            #resize
            resized_im = cv2.resize(cropped_im, (12, 12), interpolation=cv2.INTER_LINEAR)

            box_ = box.reshape(1, -1)  # box为boxex里面的每一个正样本矩形框的参数
            if IoU(crop_box, box_) >= 0.65:
                save_file = os.path.join(pos_save_dir, "%s.jpg"%p_idx)
                f1.write("12/positive/%s.jpg"%p_idx + ' 1 %.2f %.2f %.2f %.2f\n'%(offset_x1, offset_y1, offset_x2, offset_y2)) # 往文件里面写offset
                cv2.imwrite(save_file, resized_im)
                p_idx += 1
            elif IoU(crop_box, box_) >= 0.4:
                save_file = os.path.join(part_save_dir, "%s.jpg"%d_idx)
                f3.write("12/part/%s.jpg"%d_idx + ' -1 %.2f %.2f %.2f %.2f\n'%(offset_x1, offset_y1, offset_x2, offset_y2))
                cv2.imwrite(save_file, resized_im)
                d_idx += 1
        box_idx += 1
        print("%s images done, pos: %s part: %s neg: %s"%(idx, p_idx, d_idx, n_idx))
f1.close()
f2.close()
f3.close()
