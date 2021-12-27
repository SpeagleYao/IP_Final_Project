from torch import optim
from models import SegNet
from loss import DiceLoss
from img_aug import data_generator
import torch
import torch.nn as nn
import torch.nn.functional as F
import argparse
from tqdm import tqdm
import time
# import os

parser = argparse.ArgumentParser()
parser.add_argument("--epoch", type=int, default=50, help="训练迭代次数")
parser.add_argument("--batch_size", type=int, default=10, help="批训练大小")
parser.add_argument("--learning_rate", type=float, default=0.1, help="学习率大小")
parser.add_argument("--momentum", type=float, default=0.9)
# parser.add_argument("--category_weight", type=float, default=[0.7502381287857225, 1.4990483912788268], help="损失函数中类别的权重")
# parser.add_argument("--train_txt", type=str, default="train.txt", help="训练的图片和标签的路径")
# parser.add_argument("--pre_training_weight", type=str, default="vgg16_bn-6c64b313.pth", help="编码器预训练权重路径")
parser.add_argument("--weights", type=str, default="./pth/", help="训练好的权重保存路径")
opt = parser.parse_args()

def train(SegNet):

    # SegNet = SegNet.cuda()
    # SegNet.load_weights(PRE_TRAINING)

    # train_loader = Data.DataLoader(train_data, batch_size=BATCH_SIZE, shuffle=True)

    g = data_generator(BATCH_SIZE)

    optimizer = torch.optim.SGD(SegNet.parameters(), lr=LR, momentum=MOMENTUM)

    # loss_func = nn.CrossEntropyLoss(weight=torch.from_numpy(np.array(CATE_WEIGHT)).float()).cuda()
    criterion = DiceLoss()

    SegNet.train()
    for epoch in tqdm(range(EPOCH)):
        img, tar = g.gen()
        # img = img.cuda()
        # tar = tar.cuda()
        out = SegNet(img)
        loss = criterion(1-out, 1-tar)
        # loss = loss.cuda()
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        tqdm.write("Epoch:{0} || Loss:{1}".format(epoch, format(loss, ".4f")))

        # for step, (b_x, b_y) in enumerate(train_loader):
        #     b_x = b_x.cuda()
        #     b_y = b_y.cuda()
        #     b_y = b_y.view(BATCH_SIZE, 224, 224)
        #     output = SegNet(b_x)
        #     loss = loss_func(output, b_y.long())
        #     loss = loss.cuda()
        #     optimizer.zero_grad()
        #     loss.backward()
        #     optimizer.step()
        #     if step % 1 == 0:
        #         print("Epoch:{0} || Step:{1} || Loss:{2}".format(epoch, step, format(loss, ".4f")))

    torch.save(SegNet.state_dict(), WEIGHTS + "SegNet_" + str(time.time()) + ".pth")

# print(opt)

# CLASS_NUM = opt.class_num
EPOCH = opt.epoch
BATCH_SIZE = opt.batch_size
LR = opt.learning_rate
MOMENTUM = opt.momentum
# CATE_WEIGHT = opt.category_weight
# TXT_PATH = opt.train_txt
# PRE_TRAINING = opt.pre_training_weight
WEIGHTS = opt.weights


# train_data = MyDataset(txt_path=TXT_PATH)

SegNet = SegNet(1, 1)
train(SegNet)