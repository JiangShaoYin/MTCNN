#coding:utf-8
from mtcnn_model import P_Net
from train import train

def train_PNet(base_dir, prefix, end_epoch, display, lr):
    net_factory = P_Net
    train(net_factory, prefix, end_epoch, base_dir, display=display, base_lr=lr)

if __name__ == '__main__':
    base_dir = '../prepare_data/imglists/PNet' # 数据读取文件

    model_name = 'MTCNN'
    model_path = '../data/%s_model/PNet_landmark/PNet' % model_name  # 模型输出文件
            
    prefix = model_path
    end_epoch = 30  # 结束
    display = 1
    lr = 0.01
    train_PNet(base_dir, prefix, end_epoch, display, lr)
