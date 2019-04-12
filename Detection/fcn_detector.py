import numpy as np
import tensorflow as tf
import sys
sys.path.append("../")
from train_models.MTCNN_config import config


class FcnDetector(object):
    #net_factory: which net
    #model_path: where the params'file is
    def __init__(self, net_factory, model_path):
        graph = tf.Graph()   # create a graph
        with graph.as_default():
            self.image_op = tf.placeholder(tf.float32, name='input_image')  # define tensor and op in graph(-1,1)
            self.width_op = tf.placeholder(tf.int32, name='image_width')  # 定义图像宽度
            self.height_op = tf.placeholder(tf.int32, name='image_height')  # 定义图像高
            image_reshape = tf.reshape(self.image_op, [1, self.height_op, self.width_op, 3])

    
            self.cls_prob, self.bbox_pred, _ = net_factory(image_reshape, training=False)  # 用传入的网络（P/R/O）计算分类，和detection结果
            # 根据传入的图片，生成预测，当输入是12 *12 *3时，输出bbox_pred是1 * 1 * 4, 当输入是692 * 512 * 3时，输出341 * 251 * 4

            #allow 采用GPU训练
            self.sess = tf.Session(config=tf.ConfigProto(allow_soft_placement=True, gpu_options=tf.GPUOptions(allow_growth=True)))
            saver = tf.train.Saver()
            #check whether the dictionary is valid
            model_dict = '/'.join(model_path.split('/')[:-1])
            ckpt = tf.train.get_checkpoint_state(model_dict)
            print(model_path)
            readstate = ckpt != "" and ckpt.model_checkpoint_path != ""  # ckpt.model_checkpoint_path == '../data/MTCNN_model/PNet_landmark\\PNet-2'
            assert  readstate != 0, "the params dictionary is not valid"
            print("restore models' param")
            saver.restore(self.sess, model_path)  # 从模型中恢复出计算landmark的参数

    def predict(self, databatch):  # 根据传入的1个pic，计算图中划定的框框，和各个框框的分类
        height, width, _ = databatch.shape
        # print(height, width)
        cls_prob, bbox_pred = self.sess.run([self.cls_prob, self.bbox_pred], feed_dict={self.image_op: databatch,
                                                                                        self.width_op: width,
                                                                                        self.height_op: height})
        return cls_prob, bbox_pred
