import numpy as np
import numpy.random as npr
import os

data_dir = '.'
#anno_file = os.path.join(data_dir, "anno.txt")

size = 12

if size == 12:
    net = "PNet"
elif size == 24:
    net = "RNet"
elif size == 48:
    net = "ONet"

with open(os.path.join(data_dir, '%s/pos_%s.txt' % (size, size)), 'r') as f:  # 19万个pos照片
    pos = f.readlines()

with open(os.path.join(data_dir, '%s/neg_%s.txt' % (size, size)), 'r') as f:  # 80万个负例照片
    neg = f.readlines()

with open(os.path.join(data_dir, '%s/part_%s.txt' % (size, size)), 'r') as f:  # 54万个part照片
    part = f.readlines()

with open(os.path.join(data_dir,'%s/landmark_%s_aug.txt' %(size,size)), 'r') as f:
    landmark = f.readlines()
    
dir_path = os.path.join(data_dir, 'imglists')  # data_dir == '.'
if not os.path.exists(dir_path):
    os.makedirs(dir_path)  # 创建imglists文件夹
if not os.path.exists(os.path.join(dir_path, "%s" %(net))):
    os.makedirs(os.path.join(dir_path, "%s" %(net)))  # 创建PNet文件夹
with open(os.path.join(dir_path, "%s" %(net),"train_%s_landmark.txt" % (net)), "w") as f:#打开imglists/PNet/train_PNet_landmark.txt文件
    nums = [len(neg), len(pos), len(part)]
    ratio = [3, 1, 1]

    base_num = 250000
    print(len(neg), len(pos), len(part), base_num)
    if len(neg) > base_num * 3:
        neg_keep = npr.choice(len(neg), size=base_num * 3, replace=True)  # len(neg)样本可以有重复
    else:
        neg_keep = npr.choice(len(neg), size=len(neg), replace=True)
    pos_keep = npr.choice(len(pos), size=base_num, replace=True)  # npr.choice(192100, 250000, replace=True)
    part_keep = npr.choice(len(part), size=base_num, replace=True)
    print(len(neg_keep), len(pos_keep), len(part_keep))

    for i in pos_keep:  # 从250000个0~250000的随机数中顺序抽取个,将pos_12.txt里面的第i行写入imglists/PNet/train_PNet_landmark.txt文件
        f.write(pos[i])
    for i in neg_keep:
        f.write(neg[i])
    for i in part_keep:
        f.write(part[i])
    for item in landmark:
        f.write(item)
