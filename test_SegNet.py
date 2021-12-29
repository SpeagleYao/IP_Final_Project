import cv2
import numpy as np
from numpy.core.numeric import outer
import torch
import torch.nn as nn
import torch.nn.functional as F
from models import *

model = SegNet(1, 1)
model.load_state_dict(torch.load('./pth/SegNet_2.pth'))
data_in = np.load('./data/Data_in.npy')
ori = data_in[0]
img = torch.from_numpy(ori/255).float().reshape(1, 1, 224, 224)
model.eval()
out = model(img)
out = torch.where(out>=0.5, 1, 0)
out = out.numpy().reshape(224, 224)*255
a = np.hstack((ori, out))
cv2.imwrite('testimage.png', a)