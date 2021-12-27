import cv2
import numpy as np
import Augmentor
import os
import shutil
import random

class data_generator():
    def __init__(self, batch_size = 256, seed=None):
        super(data_generator, self).__init__()

        data_in = np.load('./data/Data_in.npy')
        data_out = np.load('./data/Data_out.npy')

        if seed:
            np.random.seed(seed)
            random.seed(seed)
            os.environ['PYTHONHASHSEED'] = str(seed)  # 为了禁止hash随机化，使得实验可复现。

        self.p = Augmentor.DataPipeline(np.stack((data_in, data_out), axis=1))

        self.p.rotate(probability=1.0, max_left_rotation=25, max_right_rotation=25)
        self.p.flip_left_right(probability=0.5)
        self.p.flip_top_bottom(probability=0.5)
        self.p.skew(probability=0.2)
        self.p.random_distortion(probability=0.2, grid_height=16, grid_width=16, magnitude=1)
        self.p.crop_random(probability=0.5, percentage_area=0.8)
        self.p.resize(probability=1, width=224, height=224)

        self.g = self.p.generator(batch_size)

    def gen(self):
        img_tot = np.array(next(self.g))
        img_aug = img_tot[:,0,:,:]
        gt_aug = img_tot[:,1,:,:]

        return img_aug, gt_aug
