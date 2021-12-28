from torch import optim
from models import SegNet
from loss import *
from img_aug import data_generator
import torch
import torch.nn as nn
import torch.nn.functional as F
import argparse
from tqdm import tqdm
import time
# from torch.cuda.amp import autocast
# from torch.cuda.amp import GradScaler
# import os

parser = argparse.ArgumentParser()
parser.add_argument("--epoch", type=int, default=150, help="训练迭代次数")
parser.add_argument("--batch_size", type=int, default=32, help="批训练大小")
parser.add_argument("--learning_rate", type=float, default=1e-3, help="学习率大小")
parser.add_argument("--momentum", type=float, default=0.9)
parser.add_argument("--weights", type=str, default="./pth/", help="训练好的权重保存路径")
opt = parser.parse_args()

def train(SegNet):

    SegNet = SegNet.cuda()
    # SegNet.load_weights(PRE_TRAINING)

    g = data_generator(BATCH_SIZE)

    optimizer = torch.optim.SGD(SegNet.parameters(), lr=LR, momentum=MOMENTUM)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=140, gamma=0.1)
    # scaler = GradScaler()
    # loss_func = nn.CrossEntropyLoss(weight=torch.from_numpy(np.array(CATE_WEIGHT)).float()).cuda()
    # criterion = DiceLoss().cuda()
    criterion = DiceBCELoss().cuda()
    # criterion = nn.CrossEntropyLoss().cuda()

    SegNet.train()
    for epoch in tqdm(range(EPOCH)):
        img, tar = g.gen()
        img = img.cuda()
        tar = tar.cuda()

        optimizer.zero_grad()
        # with autocast():
        out = SegNet(img)
        loss = criterion(1-out, 1-tar)
        loss.backward()
        optimizer.step()
        scheduler.step()
        # scaler.scale(loss).backward()
        # scaler.step(optimizer)
        # scaler.update()

        if epoch % 5 == 0:
            tqdm.write("Epoch:{0} || Loss:{1}".format(epoch, format(loss, ".4f")))

    torch.save(SegNet.state_dict(), WEIGHTS + "SegNet_" + str(time.time()) + ".pth")

EPOCH = opt.epoch
BATCH_SIZE = opt.batch_size
LR = opt.learning_rate
MOMENTUM = opt.momentum
WEIGHTS = opt.weights

SegNet = SegNet(1, 1)
train(SegNet)