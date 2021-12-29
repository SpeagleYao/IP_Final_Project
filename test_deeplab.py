from torchvision.models.segmentation import deeplabv3_resnet50 as dlrs50
from network import *

import numpy as np
import cv2
import torch

model = deeplabv3_resnet50(num_classes=1)

data_in = np.load('./data/Data_in.npy')
model.load_state_dict(torch.load('./pth/model_1.pth'))
ori = data_in[0]
img = torch.from_numpy(ori/255).float().reshape(1, 1, 224, 224).repeat(1, 3, 1, 1)
print(img.shape)

model.eval()
out = model(img)
# print(out.shape)
out = torch.where(out>=0.5, 1, 0)
out = out.numpy().reshape(224, 224)*255
a = np.hstack((ori, out))
cv2.imwrite('testimage.png', a)
