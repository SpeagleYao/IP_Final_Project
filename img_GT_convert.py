import cv2
import numpy as np
import os

srcimg = None
path = "./IMAGES_GT/"  #待读取的文件夹
path_list = os.listdir(path)
path_list.sort() #对读取的路径进行排序
for filename in path_list:
    # if filename == path_list[0]: continue
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
# filepath1 = os.path.join(path, path_list[1])
# image1 = cv2.imread(filepath1, cv2.IMREAD_GRAYSCALE)
# filepath2 = os.path.join(path, path_list[2])
# image2 = cv2.imread(filepath2, cv2.IMREAD_GRAYSCALE)
# dim = (224,224)
# newimg1 = cv2.resize(image1, dim).reshape(1, 224, 224)
# newimg2 = cv2.resize(image2, dim).reshape(1, 224, 224)
# newimg = np.append(newimg1, newimg2, axis=0)
# newimg = np.append(newimg, newimg2, axis=0)
    # try:
    #     image = cv2.imread(filepath, cv2.IMREAD_GRAYSCALE)
    #     dim = (224,224,1)
    #     newimg = cv2.resize(image,dim)
    #     if srcimg == None:
    #         srcimg = newimg
    #     else:
    #         srcimg = np.stack((srcimg, newimg), dim=2)
    # except:
    #     print(filepath)
        # os.remove(filepath)
# print(newimg.shape)

# srcimg = None
# dir = "./IMAGES_GT/"
# for root, dirs, files in os.walk(dir):
#     for file in files:
#         filepath = os.path.join(root, file)
        # try:
        #     image = cv2.imread(filepath, cv2.IMREAD_GRAYSCALE)
        #     dim = (224,224,1)
        #     newimg = cv2.resize(image,dim)
        #     if srcimg == None:
        #         srcimg = newimg
        #     else:
        #         srcimg = np.stack((srcimg, newimg), dim=2)
        # except:
        #     print(filepath)
        #     # os.remove(filepath)
# print(newimg.shape)

# srcimg = cv2.imread('./IMAGES_GT/Im_GT_AIGLE_RN_F09aor.png', cv2.IMREAD_GRAYSCALE)
# srcimg = cv2.imread('./IMAGES_GT/Im_GT_ESAR_30a.jpg', cv2.IMREAD_GRAYSCALE)
# gtimg  = cv2.imread('./GROUND_TRUTH/GT_AIGLE_RN_C18a.png', cv2.IMREAD_GRAYSCALE)
# srcimg = cv2.resize(srcimg, (224, 224))
# cv2.imshow('histogram', hist)
# gtimg  = cv2.resize(gtimg, (224, 224))
# print(srcimg.shape)
# print(gtimg.shape)
# imgs = np.hstack([srcimg, gtimg])
# cv2.imshow('image', imgs)
# cv2.waitKey(0)
