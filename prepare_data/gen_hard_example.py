#coding:utf-8
import sys
#sys.path.append("../")
sys.path.insert(0,'..')
import numpy as np
import argparse
import os
import pickle as pickle
import cv2
from train_models.mtcnn_model import P_Net,R_Net,O_Net
from train_models.MTCNN_config import config
from loader import TestLoader
from Detection.detector import Detector
from Detection.fcn_detector import FcnDetector
from Detection.MtcnnDetector import MtcnnDetector
from utils import *
from data_utils import *

def save_hard_example(net, data, save_path):   # load ground truth from annotation file. format of each line: image/path [x1,y1,x2,y2] for each gt_box in this image
    im_idx_list = data['images']        # data保存WIDER_train里面所有的image，读pic的文件名，和box的ground truth值，
    gt_boxes_list = data['bboxes']      # data['images']==all image pathes
                                        # data['bboxes'] =all image bboxes（4列）

    num_of_images = len(im_idx_list)  # 12880
    print("processing %d images in total" % num_of_images)

    neg_label_file = "%d/neg_%d.txt" % (net, image_size)     # neg_label_file == '24/neg_24.txt'
    neg_file = open(neg_label_file, 'w')                     # 打开保存pos， neg， part的文件夹
    pos_label_file = "%d/pos_%d.txt" % (net, image_size)
    pos_file = open(pos_label_file, 'w')
    part_label_file = "%d/part_%d.txt" % (net, image_size)
    part_file = open(part_label_file, 'w')

    det_boxes = pickle.load(open(os.path.join(save_path, 'detections.pkl'), 'rb')) # 从pkl里面读取检测框信息（5列）
    print (len(det_boxes))      # 测试用的10
    print (num_of_images)       # 12880
 #   assert len(det_boxes) == num_of_images # len(det_boxes) == num_of_images应该为TRUE，如果不是，则返回错误。"incorrect detections or ground truths"

    n_idx = 0  # index of neg, pos and part face, used as their image names
    p_idx = 0
    d_idx = 0
    image_done = 0
    #im_idx_list image index(list)
    #det_boxes detect result(list)
    #gt_boxes_list gt(list)
    for im_idx, dets, gts in zip(im_idx_list, det_boxes, gt_boxes_list): # im_idx保存1张图片名， dets保存1张图片过P_NET生成的800+框框
        gts = np.array(gts, dtype=np.float32).reshape(-1, 4)             # gts保存txt里面的存储的label
        if image_done % 100 == 0:
            print("%d images done" % image_done)
        image_done += 1

        if dets.shape[0] == 0:
            continue
        img = cv2.imread(im_idx)

        dets = convert_to_square(dets)       # 通过偏移左上角，把框框变成正方形
        dets[:, 0:4] = np.round(dets[:, 0:4])  # 取出detections的前4维信息
        neg_num = 0
        for box in dets: # 遍历detections的前4维信息
            x_left, y_top, x_right, y_bottom, _ = box.astype(int)
            width = x_right - x_left + 1
            height = y_bottom - y_top + 1

            if width < 20 or x_left < 0 or y_top < 0 or x_right > img.shape[1] - 1 or y_bottom > img.shape[0] - 1: # 去掉过小，或者超出边界的框框
                continue

            Iou = IoU(box, gts)      # compute intersection over union(IoU) between current box and all gt boxes
            cropped_im = img[y_top:y_bottom + 1, x_left:x_right + 1, :]  # 拿到检测框
            resized_im = cv2.resize(cropped_im, (image_size, image_size), interpolation=cv2.INTER_LINEAR)

            if np.max(Iou) < 0.3 and neg_num < 60:   # save negative images and write label，Iou with all gts must below 0.3
                save_file = get_path(neg_dir, "%s.jpg" % n_idx)  # save the examples，'24/negative/0.jpg'
                neg_file.write(save_file + ' 0\n')
                cv2.imwrite(save_file, resized_im)
                n_idx += 1
                neg_num += 1
            else:
                idx = np.argmax(Iou)  # find gt_box with the highest iou,找正例
                assigned_gt = gts[idx]
                x1, y1, x2, y2 = assigned_gt

                offset_x1 = (x1 - x_left) / float(width)  # compute bbox reg label计算偏移，x1是label值
                offset_y1 = (y1 - y_top) / float(height)
                offset_x2 = (x2 - x_right) / float(width)
                offset_y2 = (y2 - y_bottom) / float(height)

                if np.max(Iou) >= 0.65:  # save positive and part-face images and write labels
                    save_file = get_path(pos_dir, "%s.jpg" % p_idx)
                    pos_file.write(save_file + ' 1 %.2f %.2f %.2f %.2f\n' % (offset_x1, offset_y1, offset_x2, offset_y2))
                    cv2.imwrite(save_file, resized_im)
                    p_idx += 1

                elif np.max(Iou) >= 0.4:
                    save_file = os.path.join(part_dir, "%s.jpg" % d_idx)
                    part_file.write(save_file + ' -1 %.2f %.2f %.2f %.2f\n' % (
                        offset_x1, offset_y1, offset_x2, offset_y2))
                    cv2.imwrite(save_file, resized_im)
                    d_idx += 1
    neg_file.close()
    part_file.close()
    pos_file.close()


def t_net(prefix, epoch,  batch_size, test_mode="PNet",  thresh=[0.6, 0.6, 0.7], min_face_size=25, # prefix保存模型文件路径
          stride=2, slide_window=False, shuffle=False, vis=False):
    detectors = [None, None, None]
    print("Test model: ", test_mode)
    #PNet-echo
    model_path = ['%s-%s' % (x, y) for x, y in zip(prefix, epoch)]  # model_path == <class 'list'>: ['../data/MTCNN_model/PNet_landmark/PNet-16', '../data/MTCNN_model/RNet_landmark/RNet-6', '../data/MTCNN_model/ONet/ONet-22']

    if slide_window:
        PNet = Detector(P_Net, 12, batch_size[0], model_path[0])
    else:
        PNet = FcnDetector(P_Net, model_path[0])  # '../data/MTCNN_model/PNet_landmark/PNet-16'， 生成全连接detection的检测对象P_Net
    detectors[0] = PNet  # 将PNet对象放入检测器的第1个位

    # load rnet model
    if test_mode in ["RNet", "ONet"]:
        print("==================================", test_mode)
        RNet = Detector(R_Net, 24, batch_size[1], model_path[1])
        detectors[1] = RNet

    # load onet model
    if test_mode == "ONet":
        print("==================================", test_mode)
        ONet = Detector(O_Net, 48, batch_size[2], model_path[2])
        detectors[2] = ONet
        
    basedir = '.'
    filename = './wider_face_train_bbx_gt.txt'  # 获取检测框的ground truth值
    data = read_annotation(basedir, filename)  # 读pic的文件名，和box的ground truth值，data['images']==all image pathes
                                                                                    #  data['bboxes'] =all image bboxes
    mtcnn_detector = MtcnnDetector(detectors=detectors, min_face_size=min_face_size, stride=stride, threshold=thresh, slide_window=slide_window)
    print("==================================")

    test_data = TestLoader(data['images'])  # 生成输入图片的管理对象test_data， # 注意是在“test”模式下，  imdb = IMDB("wider", image_set, root_path, dataset_path, 'test')，  gt_imdb = imdb.gt_imdb()
    detections, _ = mtcnn_detector.detect_face(test_data)  # 从test_data输入人脸pixel,返回detection和cls

    save_net = 'RNet'
    if test_mode == "PNet":
        save_net = "RNet"
    elif test_mode == "RNet":
        save_net = "ONet"
    #save detect result
    save_path = os.path.join(data_dir, save_net)  # save_path == “24 / Rnet”
    print(save_path)
    if not os.path.exists(save_path):
        os.mkdir(save_path)

    save_file = os.path.join(save_path, "detections.pkl")
    with open(save_file, 'wb') as f:  # save_file == detections.pkl
        pickle.dump(detections, f,1)  # 将detection结果写入文件
    print("%s测试完成开始OHEM" % image_size)
    save_hard_example(image_size, data, save_path)     # data，读pic的文件名，和box的ground truth值，
                                                       # data['images']==all image pathes
                                                       # data['bboxes'] =all image bboxes（4列）


def parse_args():
    parser = argparse.ArgumentParser(description='Test mtcnn', formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('--test_mode', dest='test_mode', help='test net type, can be pnet, rnet or onet', default='PNet', type=str)  # 添加参数，默认是PNET

    parser.add_argument('--prefix', dest='prefix', help='prefix of model name', nargs="+",
                        default=['../data/MTCNN_model/PNet_landmark/PNet', '../data/MTCNN_model/RNet_landmark/RNet', '../data/MTCNN_model/ONet/ONet'],
                        type=str)
    parser.add_argument('--epoch', dest='epoch', help='epoch number of model to load', nargs="+",  # default=[18, 14, 22], type=int)
                        default=[16, 6, 22], type=int)
    parser.add_argument('--batch_size', dest='batch_size', help='list of batch size used in prediction', nargs="+", default=[2048, 256, 16], type=int)
    parser.add_argument('--thresh', dest='thresh', help='list of thresh for pnet, rnet, onet', nargs="+", default=[0.4, 0.05, 0.7], type=float)
    parser.add_argument('--min_face', dest='min_face', help='minimum face size for detection', default=24, type=int)
    parser.add_argument('--stride', dest='stride', help='stride of sliding window', default=2, type=int)
    parser.add_argument('--sw', dest='slide_window', help='use sliding window in pnet', action='store_true')
    parser.add_argument('--shuffle', dest='shuffle', help='shuffle data on visualization', action='store_true')   # parser.add_argument('--gpu', dest='gpu_id', help='GPU device to train with', default=0, type=int)
    args = parser.parse_args()    # parser.add_argument('--vis', dest='vis', help='turn on visualization', action='store_true')
    return args


if __name__ == '__main__':
    net = 'RNet'
    if net == "RNet":
        image_size = 24
    if net == "ONet":
        image_size = 48

    base_dir = '../prepare_data/WIDER_train'
    data_dir = '%s' % str(image_size)  # data_dir = '24'
    
    neg_dir = get_path(data_dir, 'negative')
    pos_dir = get_path(data_dir, 'positive')
    part_dir = get_path(data_dir, 'part')

    for dir_path in [neg_dir, pos_dir, part_dir]:      # create dictionary shuffle
        if not os.path.exists(dir_path):
            os.makedirs(dir_path)

    args = parse_args()

    print('Called with argument:')
    print(args)
    t_net(args.prefix,# model param's file
          args.epoch, # final epoches
          args.batch_size, # test batch_size
          args.test_mode,# test which model       # 传入要测试的网络
          args.thresh, # cls threshold
          args.min_face, # min_face
          args.stride,# stride
          args.slide_window, 
          args.shuffle, 
          vis=False)
