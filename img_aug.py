import cv2
import numpy as np
import Augmentor
import os
import shutil
import random
import torch

class data_generator():
    def __init__(self, batch_size = 128, seed=None):
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
        # self.p.skew(probability=0.2)
        # self.p.random_distortion(probability=0.2, grid_height=16, grid_width=16, magnitude=1)
        # self.p.crop_random(probability=0.5, percentage_area=0.6)
        # self.p.resize(probability=1, width=224, height=224)

        self.g = self.p.generator(batch_size)

    def gen(self):
        img_tot = np.array(next(self.g))
        img_aug = np.expand_dims(img_tot[:,0,:,:], axis=1)/255
        gt_aug = np.expand_dims(img_tot[:,1,:,:], axis=1)/255
        img_aug = torch.from_numpy(img_aug).repeat(1, 3, 1, 1).float().detach().requires_grad_(True)
        gt_aug = torch.from_numpy(gt_aug).float().detach().requires_grad_(True)

        return img_aug, gt_aug

if __name__=='__main__':
    g = data_generator()
    img, tar = g.gen()
    print(img.shape, tar.shape) # 256, 1, 128, 128
    # print(tar[0][0][0][0:10])
    ind = 0
    a = np.hstack((img.detach().numpy()[ind][0]*255, tar.detach().numpy()[ind][0]*255))
    # cv2.imshow('data', img.detach().numpy()[ind][0]*255)
    cv2.imwrite('testimage.png', a)
    # cv2.waitKey(0)