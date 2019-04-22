import cv2
import time
import numpy as np
import sys
sys.path.append("../")
from train_models.MTCNN_config import config
from nms import py_nms


class MtcnnDetector(object):
    def __init__(self, detectors, min_face_size=25,
                 stride=2, threshold=[0.6, 0.7, 0.7],  # 框框偏移比例0.7, 就舍去这个框框
                 scale_factor=0.79,
                 slide_window=False):

        self.pnet_detector = detectors[0]
        self.rnet_detector = detectors[1]
        self.onet_detector = detectors[2]

        self.min_face_size = min_face_size
        self.stride = stride
        self.thresh = threshold
        self.scale_factor = scale_factor
        self.slide_window = slide_window

    def convert_to_square(self, bbox):
        """
            convert bbox to square
        Parameters:
        ----------
            bbox: numpy array , shape n x 5
                input bbox
        Returns:
        -------
            square bbox
        """
        square_bbox = bbox.copy()

        h = bbox[:, 3] - bbox[:, 1] + 1
        w = bbox[:, 2] - bbox[:, 0] + 1
        max_side = np.maximum(h, w)
        square_bbox[:, 0] = bbox[:, 0] + w * 0.5 - max_side * 0.5
        square_bbox[:, 1] = bbox[:, 1] + h * 0.5 - max_side * 0.5
        square_bbox[:, 2] = square_bbox[:, 0] + max_side - 1
        square_bbox[:, 3] = square_bbox[:, 1] + max_side - 1
        return square_bbox

    def calibrate_box(self, bbox, reg):
        """
            calibrate bboxes
        Parameters:
        ----------
            bbox: numpy array, shape n x 5
                input bboxes
            reg:  numpy array, shape n x 4
                bboxes adjustment
        Returns:
        -------
            bboxes after refinement
        """

        bbox_c = bbox.copy()
        w = bbox[:, 2] - bbox[:, 0] + 1
        w = np.expand_dims(w, 1)
        h = bbox[:, 3] - bbox[:, 1] + 1
        h = np.expand_dims(h, 1)
        reg_m = np.hstack([w, h, w, h])
        aug = reg_m * reg
        bbox_c[:, 0:4] = bbox_c[:, 0:4] + aug
        return bbox_c

    def generate_bbox(self, cls_map, bbox, scale, threshold):  # cls_map的shape是341*251
        stride = 2        #stride = 4
        cellsize = 12     #cellsize = 25
        t_index = np.where(cls_map > threshold)  # [0,1]>0.4的标记索引，返回341*251这样的tuple中，大于0.4的元素索引
                                                 # 总共有784个可能是人脸的区域。t_index对应的索引，即
        # find nothing
        if t_index[0].size == 0:
            return np.array([])
        dx1, dy1, dx2, dy2 = [bbox[t_index[0], t_index[1], i] for i in range(4)]  # 根据候选的人脸区域，拿出左上，和右下点相对于ground_truth左上，和右下的偏移距离 / crop框框的size

        bbox = np.array([dx1, dy1, dx2, dy2])
        score = cls_map[t_index[0], t_index[1]]  # t_index[0]是所有可能是人脸的框框的0维坐标，score根据坐标，把预测结果提取出来，shape为784 * 1
          # 当前框的坐标，相当于label，后面根据bbox的偏移值，计算出正确的框应该在图上哪个位置
        boundingbox = np.vstack([np.round((stride * t_index[1]) / scale),  # stride是pool的步长， index[1]即为最后1道特征图上的坐标， index[1] * stride为原始图片缩放后，起点的x坐标
                                 np.round((stride * t_index[0]) / scale),  # boundingbox前4列是每个box的索引 + 偏移？第5列是人脸预测可能性，后4列是可能的坐标
                                 np.round((stride * t_index[1] + cellsize) / scale),  # 核上的1个点感受野是12
                                 np.round((stride * t_index[0] + cellsize) / scale),
                                 score,
                                 bbox])  # bbox是前项传播的计算结果。

        return boundingbox.T

    def processed_image(self, img, scale):   # pre-process images
        height, width, channels = img.shape
        new_height = int(height * scale)   # resized new height
        new_width = int(width * scale)   # resized new width
        new_dim = (new_width, new_height)
        img_resized = cv2.resize(img, new_dim, interpolation=cv2.INTER_LINEAR)  # resized image
        img_resized = (img_resized - 127.5) / 128  # pixel数值做归1化处理
        return img_resized

    def pad(self, bboxes, w, h):  # w, h是原始图片的宽， 高，bboxes是上一层网络计算出来的框框绝对坐标
        detc_w, detc_h = bboxes[:, 2] - bboxes[:, 0] + 1, bboxes[:, 3] - bboxes[:, 1] + 1  # 检测出来的框框们的宽tmpw，和高tmph
        num_box = bboxes.shape[0]

        dx, dy = np.zeros((num_box,)), np.zeros((num_box,))  #
        edx, edy = detc_w.copy() - 1, detc_h.copy() - 1

        x, y, ex, ey = bboxes[:, 0], bboxes[:, 1], bboxes[:, 2], bboxes[:, 3]  # 上一层网络计算出来的框框绝对坐标

        tmp_index = np.where(ex > w - 1)
        edx[tmp_index] = detc_w[tmp_index] + w - 2 - ex[tmp_index]
        ex[tmp_index] = w - 1

        tmp_index = np.where(ey > h - 1)
        edy[tmp_index] = detc_h[tmp_index] + h - 2 - ey[tmp_index]
        ey[tmp_index] = h - 1

        tmp_index = np.where(x < 0)
        dx[tmp_index] = 0 - x[tmp_index]
        x[tmp_index] = 0

        tmp_index = np.where(y < 0)
        dy[tmp_index] = 0 - y[tmp_index]
        y[tmp_index] = 0

        return_list = [dy, edy, dx, edx, y, ey, x, ex, detc_w, detc_h]
        return_list = [item.astype(np.int32) for item in return_list]

        return return_list
    
    def detect_pnet(self, im):  # 用pnet计算1张图的detection
        net_size = 12
        current_scale = float(net_size) / self.min_face_size  # find initial scale
        im_resized = self.processed_image(im, current_scale)   # 将img按照scale缩放，并将pixel做归1化处理
        current_height, current_width, _ = im_resized.shape

        all_boxes = list()
        cnt = 0
        while min(current_height, current_width) > net_size: # 将输入图不停的做0.79^i的缩放，直到最小尺寸 > 12
            cnt += 1
            print(cnt)
            cls_cls_map, bbox = self.pnet_detector.predict(im_resized)  # 用P_Net计算输入图片的cls和box结果
            boxes = self.generate_bbox(cls_cls_map[:, :,1], bbox, current_scale, self.thresh[0])  # 将 > thresh对应的位置的预测detection框框计算后， 提取出来，cls_cls_map的shape是341*251*2， reg的shape是341*251*2，
                                                                                                  # 根据预测bbox的offset，将预测的人脸框坐标提取出来。
            # 缩放单元
            current_scale *= self.scale_factor  # 缩放因子，0.79^i
            im_resized = self.processed_image(im, current_scale)  # 将image(1385, 1024, 3)，按照0.79^i的比例缩放
            current_height, current_width, _ = im_resized.shape  # 将输入图不停的做0.79的缩放

            if boxes.size == 0:
                continue                                # boxes（781 * 9）
            keep = py_nms(boxes[:, :5], 0.5, 'Union')   # 抽0维全部（781多行），1维前5列元素  0.5是与当前候选框重叠率之间的threshold，如果小于这个值，认为是另一个物体的分类框。选取概率最大的crop，舍去与最大概率重叠率高的crop，筛选出来239个框框
            boxes = boxes[keep]                         # keep为框框在781 * 9 里面的索引
            all_boxes.append(boxes)                     # 将1种尺寸的图片候选框集框加入集合，all_boxes[i], 循环16轮，但是第10轮以后，就没有新的box产生了， boundingbox前4列是每个box的crop框框， 第5列是人脸预测可能性，后4列是预测的offset
        if len(all_boxes) == 0:
            return None, None, None

        all_boxes = np.vstack(all_boxes)  # 收集所有的869个框框

        keep = py_nms(all_boxes[:, 0:5], 0.7, 'Union')  # 对这869个框框再进行nms筛选，对第1个阶段的框框进行merge
        all_boxes = all_boxes[keep]
        boxes = all_boxes[:, :5]  # 取box的前5维数据，这是截图框框的坐标

        bbw = all_boxes[:, 2] - all_boxes[:, 0] + 1  # 找到框框的宽
        bbh = all_boxes[:, 3] - all_boxes[:, 1] + 1  # 找到框框的高

        boxes_c = np.vstack([all_boxes[:, 0] + all_boxes[:, 5] * bbw,  # refine the boxes，boxes_c存放calibrate后，预测的detection区域
                             all_boxes[:, 1] + all_boxes[:, 6] * bbh,
                             all_boxes[:, 2] + all_boxes[:, 7] * bbw,
                             all_boxes[:, 3] + all_boxes[:, 8] * bbh,
                             all_boxes[:, 4]])                         # boxes_c存放预测区域分类
        boxes_c = boxes_c.T  # 数组转置(869, 5)变成(5, 869)

        return boxes, boxes_c, None

    def detect_rnet(self, im, dets):  # dets为上一层网络传入的detections
        h, w, c = im.shape
        dets = self.convert_to_square(dets)
        dets[:, 0:4] = np.round(dets[:, 0:4])

        [dy, edy, dx, edx, y, ey, x, ex, det_w, det_h] = self.pad(dets, w, h)  #
        num_boxes = dets.shape[0]
        cropped_ims = np.zeros((num_boxes, 24, 24, 3), dtype=np.float32)  # p_net生成的18个框框，
        for i in range(num_boxes):
            tmp = np.zeros((det_h[i], det_w[i], 3), dtype=np.uint8)  # 生成检测窗口
            tmp[dy[i]:edy[i] + 1, dx[i]:edx[i] + 1, :] = im[y[i]:ey[i] + 1, x[i]:ex[i] + 1, :]  # 将im中y框框中的像素点，复制到tmp图片的dy框框
            cropped_ims[i, :, :, :] = (cv2.resize(tmp, (24, 24))-127.5) / 128  # 将18张图片resize到24 * 24，然后做归1化处理

        cls_scores, reg, _ = self.rnet_detector.predict(cropped_ims)  # 用R_net预测cls，和detection
        cls_scores = cls_scores[:,1]
        keep_inds = np.where(cls_scores > self.thresh[1])[0]  # 喂入的18个图片是人脸区域的可能性 > thresh[1] == 0.6的索引号取出来
        if len(keep_inds) > 0:
            boxes = dets[keep_inds]  # 上一层网络的预测区域
            boxes[:, 4] = cls_scores[keep_inds]
            reg = reg[keep_inds]  # 本层网络的预测区域
        else:
            return None, None, None
        keep = py_nms(boxes, 0.6)  # 对上一层detection，经过本次计算，筛选后的框框，做非极大值抑制，阈值0.6
        boxes = boxes[keep]  # 提取框框
        boxes_c = self.calibrate_box(boxes, reg[keep])  # 进行偏移计算。
        return boxes, boxes_c, None

    def detect_onet(self, im, dets):
        h, w, c = im.shape
        dets = self.convert_to_square(dets)
        dets[:, 0:4] = np.round(dets[:, 0:4])
        [dy, edy, dx, edx, y, ey, x, ex, tmpw, tmph] = self.pad(dets, w, h)
        num_boxes = dets.shape[0]
        cropped_ims = np.zeros((num_boxes, 48, 48, 3), dtype=np.float32)
        for i in range(num_boxes):
            tmp = np.zeros((tmph[i], tmpw[i], 3), dtype=np.uint8)
            tmp[dy[i]:edy[i] + 1, dx[i]:edx[i] + 1, :] = im[y[i]:ey[i] + 1, x[i]:ex[i] + 1, :]
            cropped_ims[i, :, :, :] = (cv2.resize(tmp, (48, 48))-127.5) / 128
            
        cls_scores, reg, landmark = self.onet_detector.predict(cropped_ims)  # 计算cls， dec ，和landmark
        #prob belongs to face
        cls_scores = cls_scores[:,1]        
        keep_inds = np.where(cls_scores > self.thresh[2])[0]        # 0.7
        if len(keep_inds) > 0:
            boxes = dets[keep_inds]    #pickout filtered box
            boxes[:, 4] = cls_scores[keep_inds]
            reg = reg[keep_inds]
            landmark = landmark[keep_inds]
        else:
            return None, None, None

        w = boxes[:,2] - boxes[:,0] + 1   #width
        h = boxes[:,3] - boxes[:,1] + 1   #height
        landmark[:,0::2] = (np.tile(w,(5,1)) * landmark[:,0::2].T + np.tile(boxes[:,0],(5,1)) - 1).T
        landmark[:,1::2] = (np.tile(h,(5,1)) * landmark[:,1::2].T + np.tile(boxes[:,1],(5,1)) - 1).T        
        boxes_c = self.calibrate_box(boxes, reg)

        boxes = boxes[py_nms(boxes, 0.6, "Minimum")]
        keep = py_nms(boxes_c, 0.6, "Minimum")
        boxes_c = boxes_c[keep]
        landmark = landmark[keep]
        return boxes, boxes_c,landmark

    #use for video
    def detect(self, img):
        """Detect face over image
        """
        boxes = None
        t = time.time()
    
        # pnet
        t1 = 0
        if self.pnet_detector:
            boxes, boxes_c,_ = self.detect_pnet(img)
            if boxes_c is None:
                return np.array([]),np.array([])
            t1 = time.time() - t
            t = time.time()
    
        # rnet
        t2 = 0
        if self.rnet_detector:
            boxes, boxes_c,_ = self.detect_rnet(img, boxes_c)
            if boxes_c is None:
                return np.array([]),np.array([])
    
            t2 = time.time() - t
            t = time.time()
    
        # onet
        t3 = 0
        if self.onet_detector:
            boxes, boxes_c,landmark = self.detect_onet(img, boxes_c)
            if boxes_c is None:
                return np.array([]),np.array([])
    
            t3 = time.time() - t
            t = time.time()
            print(
                "time cost " + '{:.3f}'.format(t1 + t2 + t3) + '  pnet {:.3f}  rnet {:.3f}  onet {:.3f}'.format(t1, t2, t3))
        return boxes_c, landmark

    def detect_face(self, test_data):  # 用3层网络检测数据
        all_boxes = []  # save each image's bboxes
        landmarks = []
        batch_idx = 0
        sum_time = 0
        for databatch in test_data:  # databatch(image returned), 从test_data对象中拿出一个batch（此时batch.size == 1）的测试pic
            if batch_idx % 1 == 0:
                print("%d images done" % batch_idx)
            im = databatch  # databatch里面存储1个img的像素信息

            t1 = 0  # pnet
            if self.pnet_detector:
                t = time.time()
                boxes, boxes_c, landmark = self.detect_pnet(im)  # boxes是原始的截图框， boxes_c是calibrate offset后的框框
                t1 = time.time() - t
                sum_time += t1
                if boxes_c is None:  #  第1次进循环，创建空的数组
                    print("boxes_c is None...")
                    all_boxes.append(np.array([]))
                    landmarks.append(np.array([]))     # pay attention
                    batch_idx += 1
                    continue

            t2 = 0   # rnet
            if self.rnet_detector:  # 第一张图片预测的detections传入下一层网络。
                t = time.time()
                boxes, boxes_c, landmark = self.detect_rnet(im, boxes_c)  # 用P_net计算boxes, boxes_c，和landmark
                t2 = time.time() - t
                sum_time += t2
                if boxes_c is None:
                    all_boxes.append(np.array([]))
                    landmarks.append(np.array([]))
                    batch_idx += 1
                    continue

            t3 = 0   # onet
            if self.onet_detector:
                t = time.time()
                boxes, boxes_c, landmark = self.detect_onet(im, boxes_c)  # 用R_net计算boxes, boxes_c，和landmark
                t3 = time.time() - t
                sum_time += t3
                if boxes_c is None:
                    all_boxes.append(np.array([]))
                    landmarks.append(np.array([]))                    
                    batch_idx += 1
                    continue
                print(
                    "time cost " + '{:.3f}'.format(sum_time) + '  pnet {:.3f}  rnet {:.3f}  onet {:.3f}'.format(t1, t2,t3))
                                                                                                                    
                                                                                                                   
            all_boxes.append(boxes_c)  #
            landmarks.append(landmark)
            batch_idx += 1
        return all_boxes,landmarks      # num_of_data * 9,num_of_data * 10

