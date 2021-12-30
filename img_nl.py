import cv2
import numpy as np
import os
import torch
from models.CENET import CENet_My
from tqdm import tqdm

srcimg = None
path = "./IMAGES_NL/"  #待读取的文件夹
path_list = os.listdir(path)
path_list.sort() #对读取的路径进行排序
for filename in path_list:
    filepath = os.path.join(path, filename)
    image = cv2.imread(filepath, cv2.IMREAD_GRAYSCALE)
    dim = (224, 224)
    newimg = cv2.resize(image, dim).reshape(1, 1, 224, 224)
    if filename == path_list[0]:
        srcimg = newimg
    else:
        srcimg = np.append(srcimg, newimg, axis=0)

model = CENet_My()
model.load_state_dict(torch.load('./pth/CENet_My.pth'))
model.eval()
for i in tqdm(range(srcimg.shape[0])):
    img = srcimg[i][0]
    data = torch.from_numpy(img.reshape(1, 1, 224, 224)).float()/255
    out = model(data)
    out = torch.where(out>=0.5, 1, 0)
    out = out.numpy().reshape(224, 224)*255
    a = np.hstack((img, out))
    cv2.imwrite('./nlimg/nlimg'+str(i)+'.png', a)