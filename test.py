from img_aug import data_generator
from models import *
from loss import *
import numpy as np
import cv2
import torch

model = CENet_My()
model.load_state_dict(torch.load('./pth/CENet_My.pth'))

model.eval()
criterion = DiceLoss()
g_val = data_generator('./data/img_val.npy', './data/tar_val.npy', 10, train=False)
img, tar = g_val.gen()
out = model(img)
loss_val = criterion(1-out, 1-tar)
print("Loss_val:{0}".format(format(loss_val, ".4f")))

out = torch.where(out>=0.5, 1, 0)
out = out.numpy().reshape(10, 224, 224)*255
tar = tar.detach().numpy().reshape(10, 224, 224)*255
for i in range(out.shape[0]):
    a = np.hstack((tar[i], out[i]))
    cv2.imwrite('./prdimg/prdimg'+str(i)+'.png', a)
