import cv2
import numpy as np
import os

srcimg = None
path = "./IMAGES_GT/"  #待读取的文件夹
path_list = os.listdir(path)
path_list.sort() #对读取的路径进行排序
for filename in path_list:
    filepath = os.path.join(path, filename)
    image = cv2.imread(filepath, cv2.IMREAD_GRAYSCALE)
    dim = (224, 224)
    newimg = cv2.resize(image, dim).reshape(1, 224, 224)
    if filename == path_list[0]:
        srcimg = newimg
    else:
        srcimg = np.append(srcimg, newimg, axis=0)
print(srcimg.shape)
srcgt = None
path = "./GROUND_TRUTH/"  #待读取的文件夹
path_list = os.listdir(path)
path_list.sort() #对读取的路径进行排序
for filename in path_list:
    filepath = os.path.join(path, filename)
    gt = cv2.imread(filepath, cv2.IMREAD_GRAYSCALE)
    dim = (224, 224)
    newgt = cv2.resize(gt, dim).reshape(1, 224, 224)
    if filename == path_list[0]:
        srcgt = newgt
    else:
        srcgt = np.append(srcgt, newgt, axis=0)
print(srcgt.shape)
np.save("Data_in.npy", srcimg)
np.save("Data_out.npy", srcgt)