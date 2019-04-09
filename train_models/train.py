#coding:utf-8
import tensorflow as tf
import numpy as np
import os
from datetime import datetime
import sys
sys.path.append("../prepare_data")
print(sys.path)
from read_tfrecord_v2 import read_multi_tfrecords,read_single_tfrecord
from MTCNN_config import config
from mtcnn_model import P_Net
import random
import numpy.random as npr
import cv2
def train_model(base_lr, loss, data_num, global_):

    lr_factor = 0.1
    global_step = tf.Variable(0, trainable=False)
    lr_op = tf.train.natural_exp_decay(learning_rate=0.01,global_step=global_,
                                       decay_steps=1000,decay_rate=0.5,staircase=True)
    train_op = tf.train.AdamOptimizer(lr_op).minimize(loss, global_step)
    return train_op, lr_op

def random_flip_images(image_batch,label_batch,landmark_batch):
    if random.choice([0, 1]) > 0:
        num_images = image_batch.shape[0]
        fliplandmarkindexes = np.where(label_batch==-2)[0]
        flipposindexes = np.where(label_batch==1)[0]
        #only flip
        flipindexes = np.concatenate((fliplandmarkindexes,flipposindexes))
        #random flip    
        for i in flipindexes:
            cv2.flip(image_batch[i],1,image_batch[i])        
        
        #pay attention: flip landmark    
        for i in fliplandmarkindexes:
            landmark_ = landmark_batch[i].reshape((-1,2))
            landmark_ = np.asarray([(1-x, y) for (x, y) in landmark_])
            landmark_[[0, 1]] = landmark_[[1, 0]]#left eye<->right eye
            landmark_[[3, 4]] = landmark_[[4, 3]]#left mouth<->right mouth        
            landmark_batch[i] = landmark_.ravel()
    return image_batch,landmark_batch

def train(net_factory, prefix, end_epoch, base_dir, display=200, base_lr=0.01):
    net = prefix.split('/')[-1]

    label_file = os.path.join(base_dir,'train_%s_landmark.txt' % net)
    print(label_file)     # label_file = os.path.join(base_dir,'landmark_12_few.txt')
    f = open(label_file, 'r')
    num = len(f.readlines())  # 142w个 数据
    print("Total datasets is: ", num)
    print(prefix)  # prefix == '../data/MTCNN_model/PNet_landmark/PNet'

    #PNet use this method to get data
    if net == 'PNet':
        # dataset_dir = '../prepare_data/imglists/PNet\\train_PNet_landmark.tfrecord_shuffle'
        dataset_dir = os.path.join(base_dir,'train_%s_landmark.tfrecord_shuffle' % net)
        print(dataset_dir)
        # 一个batch == 4608， 从数据集中读取1个batch的pixel和label
        image_batch, label_batch, bbox_batch,landmark_batch = read_single_tfrecord(dataset_dir, config.BATCH_SIZE, net)
        
    #RNet use 3 tfrecords to get data    
    else:
        pos_dir = os.path.join(base_dir,'pos_landmark.tfrecord_shuffle')
        part_dir = os.path.join(base_dir,'part_landmark.tfrecord_shuffle')
        neg_dir = os.path.join(base_dir,'neg_landmark.tfrecord_shuffle')
        landmark_dir = os.path.join(base_dir,'landmark_landmark.tfrecord_shuffle')
        dataset_dirs = [pos_dir,part_dir,neg_dir,landmark_dir]
        pos_radio = 1.0/6;part_radio = 1.0/6;landmark_radio=1.0/6;neg_radio=3.0/6
        pos_batch_size = int(np.ceil(config.BATCH_SIZE*pos_radio))
        assert pos_batch_size != 0,"Batch Size Error "
        part_batch_size = int(np.ceil(config.BATCH_SIZE*part_radio))
        assert part_batch_size != 0,"Batch Size Error "        
        neg_batch_size = int(np.ceil(config.BATCH_SIZE*neg_radio))
        assert neg_batch_size != 0,"Batch Size Error "
        landmark_batch_size = int(np.ceil(config.BATCH_SIZE*landmark_radio))
        assert landmark_batch_size != 0,"Batch Size Error "
        batch_sizes = [pos_batch_size,part_batch_size,neg_batch_size,landmark_batch_size]
        image_batch, label_batch, bbox_batch,landmark_batch = read_multi_tfrecords(dataset_dirs,batch_sizes, net)        
        
    #landmark_dir    
    if net == 'PNet':
        image_size = 12
        radio_cls_loss = 1.0; radio_bbox_loss = 0.5; radio_landmark_loss = 0.5;  # cls_prob == classify probability
    elif net == 'RNet':
        image_size = 24
        radio_cls_loss = 1.0;radio_bbox_loss = 0.5;radio_landmark_loss = 0.5;
    else:
        radio_cls_loss = 1.0;radio_bbox_loss = 0.5;radio_landmark_loss = 1.0;
        image_size = 48
    
    #define placeholder
    input_image = tf.placeholder(tf.float32, shape=[config.BATCH_SIZE, image_size, image_size, 3], name='input_image')
    label = tf.placeholder(tf.float32, shape=[config.BATCH_SIZE], name='label')
    bbox_target = tf.placeholder(tf.float32, shape=[config.BATCH_SIZE, 4], name='bbox_target')
    landmark_target = tf.placeholder(tf.float32,shape=[config.BATCH_SIZE,10],name='landmark_target')

    global_ = tf.Variable(tf.constant(0), trainable=False)

    #class,regression
    cls_loss_op, bbox_loss_op, landmark_loss_op, L2_loss_op, accuracy_op = net_factory(input_image, label, bbox_target,landmark_target,training=True)
    #train,update learning rate(3 loss)
    train_op, lr_op = train_model(base_lr, radio_cls_loss * cls_loss_op + radio_bbox_loss * bbox_loss_op + radio_landmark_loss * landmark_loss_op + L2_loss_op, num,global_)
    # init
    init = tf.global_variables_initializer()
    sess = tf.Session()
    #save model
    saver = tf.train.Saver(max_to_keep=0)
    sess.run(init)
    #visualize some variables
    tf.summary.scalar("cls_loss",cls_loss_op)#cls_loss
    tf.summary.scalar("bbox_loss",bbox_loss_op)#bbox_loss
    tf.summary.scalar("landmark_loss",landmark_loss_op)#landmark_loss
    tf.summary.scalar("cls_accuracy",accuracy_op)#cls_acc
    tf.summary.scalar("learning rate", lr_op)  # learning rate
    summary_op = tf.summary.merge_all()
    logs_dir = "../logs/%s" %(net)
    if os.path.exists(logs_dir) == False:
        os.mkdir(logs_dir)
    writer = tf.summary.FileWriter(logs_dir,sess.graph)
    #begin 
    coord = tf.train.Coordinator()
    #begin enqueue thread
    threads = tf.train.start_queue_runners(sess=sess, coord=coord)
    i = 0
    #total steps
    MAX_STEP = int(num / config.BATCH_SIZE + 1) * end_epoch
    epoch = 0
    sess.graph.finalize()    
    try:
        for step in range(MAX_STEP):
            i = i + 1
            if coord.should_stop():
                break
            image_batch_array, label_batch_array, bbox_batch_array,landmark_batch_array = sess.run([image_batch, label_batch, bbox_batch,landmark_batch])
            #random flip
            image_batch_array,landmark_batch_array = random_flip_images(image_batch_array,label_batch_array,landmark_batch_array)

            _,_,summary = sess.run([train_op, lr_op ,summary_op], feed_dict={input_image: image_batch_array, label: label_batch_array, bbox_target: bbox_batch_array,landmark_target:landmark_batch_array,global_:step})
            
            if (step+1) % display == 0:
                #acc = accuracy(cls_pred, labels_batch)
            #    cls_loss, bbox_loss,landmark_loss,L2_loss,lr,acc = sess.run([cls_loss_op, bbox_loss_op,landmark_loss_op,L2_loss_op,lr_op,accuracy_op],
            #                                                 feed_dict={input_image: image_batch_array, label: label_batch_array, bbox_target: bbox_batch_array, landmark_target: landmark_batch_array,global_:step})

                cls_loss = sess.run(cls_loss_op,
                                     feed_dict={input_image: image_batch_array, label: label_batch_array, bbox_target: bbox_batch_array, landmark_target: landmark_batch_array,global_:step})

                print("%s : Step: %d, accuracy: %3f, cls loss: %4f, bbox loss: %4f, landmark loss: %4f,L2 loss: %4f, lr: %4f" % (
                datetime.now(), step+1, acc, cls_loss, bbox_loss, landmark_loss, L2_loss, lr))
            #save every two epochs
            if i * config.BATCH_SIZE > num*2:
                epoch = epoch + 1
                i = 0
                saver.save(sess, prefix, global_step=epoch*2)
            if (step+1) % display == 0:
                writer.add_summary(summary,global_step=step)
    except tf.errors.OutOfRangeError:
        print("完成！！！")
    finally:
        coord.request_stop()
        writer.close()
    coord.join(threads)
    sess.close()
