import cv2
import numpy as np
import Augmentor
import os
import shutil
import random

# seed_value = 1105   # 设定随机数种子
# np.random.seed(seed_value)
# random.seed(seed_value)
# os.environ['PYTHONHASHSEED'] = str(seed_value)  # 为了禁止hash随机化，使得实验可复现。

data_in = np.load('./data/Data_in.npy')
data_out = np.load('./data/Data_out.npy')

p = Augmentor.DataPipeline(np.stack((data_in, data_out), axis=1))

p.rotate(probability=1.0, max_left_rotation=25, max_right_rotation=25)
p.flip_left_right(probability=0.5)
p.flip_top_bottom(probability=0.5)
p.skew(probability=0.2)
p.random_distortion(probability=0.2, grid_height=16, grid_width=16, magnitude=1)
p.crop_random(probability=0.5, percentage_area=0.8)
p.resize(probability=1, width=224, height=224)

g = p.generator(batch_size=256)

img_tot = np.array(next(g))
img_aug = img_tot[:,0,:,:]
gt_aug = img_tot[:,1,:,:]
# Remember to do Normalization
# print(np.mean(img_aug))
# ind = 16
# a = np.hstack((img_aug[ind], gt_aug[ind]))
# cv2.imshow('img01', a)
# cv2.waitKey(0)
# print(np.array(img_aug).shape)