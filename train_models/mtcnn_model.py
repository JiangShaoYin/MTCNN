#coding:utf-8
import tensorflow as tf
from tensorflow.contrib import slim
import numpy as np
num_keep_radio = 0.7
#define prelu
def prelu(inputs):
    alphas = tf.get_variable("alphas", shape=inputs.get_shape()[-1], dtype=tf.float32, initializer=tf.constant_initializer(0.25))
    pos = tf.nn.relu(inputs)
    neg = alphas * (inputs-abs(inputs))*0.5
    return pos + neg

def dense_to_one_hot(labels_dense,num_classes):
    num_labels = labels_dense.shape[0]
    index_offset = np.arange(num_labels)*num_classes
    #num_sample*num_classes
    labels_one_hot = np.zeros((num_labels,num_classes))
    labels_one_hot.flat[index_offset + labels_dense.ravel()] = 1
    return labels_one_hot
#cls_prob:batch*2
#label:batch

def cls_ohem(cls_prob, label):  # online hard example mining
    zeros = tf.zeros_like(label)  # 产生与label同维度的全0矩阵zeros
    #label=-1 --> label=0net_factory

    # 将label中 < 0 的数，换成0，> 0 的数，保持不变
    label_filter_invalid = tf.where(tf.less(label, 0), zeros, label)
    # cls_prob的shape是(4608, 2)，返回tensor元素的个数4608*2=9216
    num_cls_prob = tf.size(cls_prob)
    # 将cls_prob矩阵转置成4608*2=9216行，1列的数据
    cls_prob_reshape = tf.reshape(cls_prob, [num_cls_prob, -1])
    label_int = tf.cast(label_filter_invalid, tf.int32)          # 标签的整数转换

    num_row = tf.to_int32(cls_prob.get_shape()[0])  #cls_prob的shape是(4608, 2)， 获取0维的大小即4608

    row = tf.range(num_row) * 2  # 产生[0,2.....9612] 的4806维张量
    indices_ = row + label_int   # 同维度的每个元素加上一个对应的label_int值，如果label是0就取第1位上的数据，如果label是1就取第2位上的数据。对应onehot编码
    # 计算预测值
    label_prob = tf.squeeze(tf.gather(cls_prob_reshape, indices_))   # cls_prob_reshape == (9216, 1), indices = 4608 一维数组
                                                                     # label_prob经过squeeze成为了1维数组
    loss = -tf.log(label_prob+1e-10)  #

    zeros = tf.zeros_like(label_prob, dtype=tf.float32)
    ones = tf.ones_like(label_prob,dtype=tf.float32)
    valid_inds = tf.where(label < zeros, zeros, ones)   # valid_inds = label的二值化处理，<0 置为0，>0 置为1

    num_valid = tf.reduce_sum(valid_inds)  # 元素个数
    keep_num = tf.cast(num_valid * num_keep_radio,dtype=tf.int32)  # 只保留最大的前70%的数据
    #set 0 to invalid sample
    loss = loss * valid_inds  # loss = -tf.log(label_prob+1e-10)，逐位相乘，保持维度不变，loss值记录计算结果， 与label结果的相似度，值越大，判断结果与label距离越远
    loss, _ = tf.nn.top_k(loss, k=keep_num)  # 取出最大的前k个loss
    return tf.reduce_mean(loss)


def bbox_ohem_smooth_L1_loss(bbox_pred,bbox_target,label):
    sigma = tf.constant(1.0)
    threshold = 1.0/(sigma**2)
    zeros_index = tf.zeros_like(label, dtype=tf.float32)
    valid_inds = tf.where(label!=zeros_index,tf.ones_like(label,dtype=tf.float32),zeros_index)
    abs_error = tf.abs(bbox_pred-bbox_target)
    loss_smaller = 0.5*((abs_error*sigma)**2)
    loss_larger = abs_error-0.5/(sigma**2)
    smooth_loss = tf.reduce_sum(tf.where(abs_error<threshold,loss_smaller,loss_larger),axis=1)
    keep_num = tf.cast(tf.reduce_sum(valid_inds)*num_keep_radio,dtype=tf.int32)
    smooth_loss = smooth_loss*valid_inds
    _, k_index = tf.nn.top_k(smooth_loss, k=keep_num)
    smooth_loss_picked = tf.gather(smooth_loss, k_index)
    return tf.reduce_mean(smooth_loss_picked)

def bbox_ohem_orginal(bbox_pred,bbox_target,label):
    zeros_index = tf.zeros_like(label, dtype=tf.float32)
    #pay attention :there is a bug!!!!
    valid_inds = tf.where(label!=zeros_index,tf.ones_like(label,dtype=tf.float32),zeros_index)
    #(batch,)
    square_error = tf.reduce_sum(tf.square(bbox_pred-bbox_target),axis=1)
    #keep_num scalar
    keep_num = tf.cast(tf.reduce_sum(valid_inds)*num_keep_radio,dtype=tf.int32)
    #keep valid index square_error
    square_error = square_error*valid_inds
    _, k_index = tf.nn.top_k(square_error, k=keep_num)
    square_error = tf.gather(square_error, k_index)
    return tf.reduce_mean(square_error)

#label=1 or label=-1 then do regression
def bbox_ohem(bbox_pred, bbox_target, label):  # bbox经过shuffle，是4维度的，做数据集的时候，把负例的框框，全写成了0
    zeros_index = tf.zeros_like(label, dtype=tf.float32)
    ones_index = tf.ones_like(label, dtype=tf.float32)

    # valid_inds是个1维4608个成员的tensor，label中的pos和part变为1， neg变为0，
    valid_inds = tf.where(tf.equal(tf.abs(label), 1), ones_index, zeros_index)
    # (batch,)batchsize
    square_error = tf.square(bbox_pred-bbox_target)  # 与标签差值的平方，为什么计算结果shape为(4806,2)
    square_error = tf.reduce_sum(square_error, axis=1)  # 对1维求和，计算后square_error = (4806, )
    # num_valid == 数据集中pos和part图片的数量
    num_valid = tf.reduce_sum(valid_inds)
    keep_num = tf.cast(num_valid, dtype=tf.int32)  # 将num_valid转为int型

    #keep valid index square_error
    square_error = square_error * valid_inds           # (4806, ) * (4806, )
    _, k_index = tf.nn.top_k(square_error, k=keep_num) # 将pos和part的框框损失提出来
    square_error = tf.gather(square_error, k_index)    # 将损失写入新的square_error
    return tf.reduce_mean(square_error)                # 求损失的平均值

def landmark_ohem(landmark_pred,landmark_target,label):
    #keep label =-2  then do landmark detection
    ones = tf.ones_like(label,dtype=tf.float32)
    zeros = tf.zeros_like(label,dtype=tf.float32)
    valid_inds = tf.where(tf.equal(label,-2),ones,zeros)
    square_error = tf.square(landmark_pred-landmark_target)
    square_error = tf.reduce_sum(square_error,axis=1)
    num_valid = tf.reduce_sum(valid_inds)
    #keep_num = tf.cast(num_valid*num_keep_radio,dtype=tf.int32)
    keep_num = tf.cast(num_valid, dtype=tf.int32)
    square_error = square_error*valid_inds
    _, k_index = tf.nn.top_k(square_error, k=keep_num)
    square_error = tf.gather(square_error, k_index)
    return tf.reduce_mean(square_error)
    
def cal_accuracy(cls_prob,label):
    pred = tf.argmax(cls_prob,axis=1)
    label_int = tf.cast(label,tf.int64)
    cond = tf.where(tf.greater_equal(label_int,0))
    picked = tf.squeeze(cond)
    label_picked = tf.gather(label_int,picked)
    pred_picked = tf.gather(pred,picked)
    accuracy_op = tf.reduce_mean(tf.cast(tf.equal(label_picked,pred_picked),tf.float32))
    return accuracy_op
#construct Pnet
#label:batch

def P_Net(inputs,label=None,bbox_target=None,landmark_target=None,training=True):  # P_Net的前向传播

    with slim.arg_scope([slim.conv2d],      # 设置默认参数
                        activation_fn=prelu,
                        weights_initializer=slim.xavier_initializer(),
                        biases_initializer=tf.zeros_initializer(),
                        weights_regularizer=slim.l2_regularizer(0.0005), 
                        padding='valid'):
            #   print(inputs.get_shape())
        net1 = slim.conv2d(inputs, 10, 3, stride=1,scope='conv1')                             # 第1层卷积网络
            #   print(net.get_shape())  # 池化核 2 * 2， 步长 2
        net2 = slim.max_pool2d(net1, kernel_size=[2,2], stride=2, scope='pool1', padding='SAME') # 第1层池化
            #   print(net.get_shape())
        net3 = slim.conv2d(net2,num_outputs=16,kernel_size=[3,3],stride=1,scope='conv2')     # 第2层卷积网络
            #   print(net.get_shape())
        net4 = slim.conv2d(net3,num_outputs=32,kernel_size=[3,3],stride=1,scope='conv3')     # 第3层卷积网络
            #    print(net.get_shape())

        #  conv4_1 = 4608 * 1 * 1 * 2                                                                                   # 第4-1层卷积网络
        conv4_1 = slim.conv2d(net4, num_outputs=2, kernel_size=[1, 1], stride=1, scope='conv4_1', activation_fn = tf.nn.softmax)

        bbox_pred = slim.conv2d(net4, num_outputs=4, kernel_size=[1, 1], stride=1, scope='conv4_2', activation_fn=None)
            #print(bbox_pred.get_shape())
        # batch*H*W*10

        landmark_pred = slim.conv2d(net4,num_outputs=10, kernel_size=[1, 1], stride=1,scope='conv4_3', activation_fn=None)

        if training:
            # batch*2
            cls_prob = tf.squeeze(conv4_1, [1, 2], name='cls_prob')
            cls_loss = cls_ohem(cls_prob,label)  # 根据预测结果，计算loss，label 1正例，0负例，-1是part
            # batch
            bbox_pred = tf.squeeze(bbox_pred, [1, 2], name='bbox_pred')
            bbox_loss = bbox_ohem(bbox_pred, bbox_target, label)
            # batch*10
            landmark_pred = tf.squeeze(landmark_pred, [1, 2], name="landmark_pred")
            landmark_loss = landmark_ohem(landmark_pred, landmark_target, label)

            accuracy = cal_accuracy(cls_prob, label)
            L2_loss = tf.add_n(slim.losses.get_regularization_losses())
            return cls_loss,bbox_loss,landmark_loss,L2_loss,accuracy 
        #test
        else:
            #when test,batch_size = 1
            cls_pro_test = tf.squeeze(conv4_1, axis=0)
            bbox_pred_test = tf.squeeze(bbox_pred,axis=0)
            landmark_pred_test = tf.squeeze(landmark_pred,axis=0)
            return cls_pro_test,bbox_pred_test,landmark_pred_test




def R_Net(inputs,label=None,bbox_target=None,landmark_target=None,training=True):
    with slim.arg_scope([slim.conv2d],
                        activation_fn = prelu,
                        weights_initializer=slim.xavier_initializer(),
                        biases_initializer=tf.zeros_initializer(),
                        weights_regularizer=slim.l2_regularizer(0.0005),                        
                        padding='valid'):
        print(inputs.get_shape())
        net = slim.conv2d(inputs, num_outputs=28, kernel_size=[3,3], stride=1, scope="conv1")
        print(net.get_shape())
        net = slim.max_pool2d(net, kernel_size=[3, 3], stride=2, scope="pool1", padding='SAME')
        print(net.get_shape())
        net = slim.conv2d(net,num_outputs=48,kernel_size=[3,3],stride=1,scope="conv2")
        print(net.get_shape())
        net = slim.max_pool2d(net,kernel_size=[3,3],stride=2,scope="pool2")
        print(net.get_shape())
        net = slim.conv2d(net,num_outputs=64,kernel_size=[2,2],stride=1,scope="conv3")
        print(net.get_shape())
        fc_flatten = slim.flatten(net)
        print(fc_flatten.get_shape())
        fc1 = slim.fully_connected(fc_flatten, num_outputs=128,scope="fc1", activation_fn=prelu)
        print(fc1.get_shape())
        #batch*2
        cls_prob = slim.fully_connected(fc1,num_outputs=2,scope="cls_fc",activation_fn=tf.nn.softmax)
        print(cls_prob.get_shape())
        #batch*4
        bbox_pred = slim.fully_connected(fc1,num_outputs=4,scope="bbox_fc",activation_fn=None)
        print(bbox_pred.get_shape())
        #batch*10
        landmark_pred = slim.fully_connected(fc1,num_outputs=10,scope="landmark_fc",activation_fn=None)
        print(landmark_pred.get_shape())
        #train
        if training:
            cls_loss = cls_ohem(cls_prob,label)
            bbox_loss = bbox_ohem(bbox_pred,bbox_target,label)
            accuracy = cal_accuracy(cls_prob,label)
            landmark_loss = landmark_ohem(landmark_pred,landmark_target,label)
            L2_loss = tf.add_n(slim.losses.get_regularization_losses())
            return cls_loss,bbox_loss,landmark_loss,L2_loss,accuracy
        else:
            return cls_prob,bbox_pred,landmark_pred
    
def O_Net(inputs,label=None,bbox_target=None,landmark_target=None,training=True):
    with slim.arg_scope([slim.conv2d],
                        activation_fn = prelu,
                        weights_initializer=slim.xavier_initializer(),
                        biases_initializer=tf.zeros_initializer(),
                        weights_regularizer=slim.l2_regularizer(0.0005),                        
                        padding='valid'):
        print(inputs.get_shape())
        net = slim.conv2d(inputs, num_outputs=32, kernel_size=[3,3], stride=1, scope="conv1")
        print(net.get_shape())
        net = slim.max_pool2d(net, kernel_size=[3, 3], stride=2, scope="pool1", padding='SAME')
        print(net.get_shape())
        net = slim.conv2d(net,num_outputs=64,kernel_size=[3,3],stride=1,scope="conv2")
        print(net.get_shape())
        net = slim.max_pool2d(net, kernel_size=[3, 3], stride=2, scope="pool2")
        print(net.get_shape())
        net = slim.conv2d(net,num_outputs=64,kernel_size=[3,3],stride=1,scope="conv3")
        print(net.get_shape())
        net = slim.max_pool2d(net, kernel_size=[2, 2], stride=2, scope="pool3", padding='SAME')
        print(net.get_shape())
        net = slim.conv2d(net,num_outputs=128,kernel_size=[2,2],stride=1,scope="conv4")
        print(net.get_shape())
        fc_flatten = slim.flatten(net)
        print(fc_flatten.get_shape())
        fc1 = slim.fully_connected(fc_flatten, num_outputs=256,scope="fc1", activation_fn=prelu)
        print(fc1.get_shape())
        #batch*2
        cls_prob = slim.fully_connected(fc1,num_outputs=2,scope="cls_fc",activation_fn=tf.nn.softmax)
        print(cls_prob.get_shape())
        #batch*4
        bbox_pred = slim.fully_connected(fc1,num_outputs=4,scope="bbox_fc",activation_fn=None)
        print(bbox_pred.get_shape())
        #batch*10
        landmark_pred = slim.fully_connected(fc1,num_outputs=10,scope="landmark_fc",activation_fn=None)
        print(landmark_pred.get_shape())
        #train
        if training:
            cls_loss = cls_ohem(cls_prob,label)
            bbox_loss = bbox_ohem(bbox_pred,bbox_target,label)
            accuracy = cal_accuracy(cls_prob,label)
            landmark_loss = landmark_ohem(landmark_pred, landmark_target,label)
            L2_loss = tf.add_n(slim.losses.get_regularization_losses())
            return cls_loss,bbox_loss,landmark_loss,L2_loss,accuracy
        else:
            return cls_prob,bbox_pred,landmark_pred
            
        
        
                                                                  
