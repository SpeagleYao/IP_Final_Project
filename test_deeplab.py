from torchvision.models.segmentation import deeplabv3_resnet50 as dlrs50
from img_aug import data_generator
# from network import *
from models import *
from loss import *
import numpy as np
import cv2
import torch

model = UNetModel(out_features=1)
model.load_state_dict(torch.load('./pth/Unet_2.pth'))
model.eval()
# data_in = np.load('./data/img_val.npy')
# data_out = np.load('./data/tar_val.npy')
# ind = 0
# ori = data_in[ind]
# tar = data_out[ind]
# img = torch.from_numpy(ori/255).float().reshape(1, 1, 224, 224)
criterion = DiceLoss()
g_val = data_generator('./data/img_val.npy', './data/tar_val.npy', 10, train=False)
img, tar = g_val.gen()
out = model(img)
loss_val = criterion(1-out, 1-tar)
print("Loss_val:{0}".format(format(loss_val, ".4f")))
# print(img.shape)

# model.eval()
# out = model(img)
# print(out.shape)
out = torch.where(out>=0.5, 1, 0)
out = out.numpy().reshape(224, 224)*255
tar = np.load('./data/tar_val.npy')
a = np.hstack((tar[0], out[0]))
cv2.imwrite('testimage.png', a)
