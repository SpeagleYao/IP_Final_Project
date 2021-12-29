from torch import optim
from models import *
from loss import *
from img_aug import data_generator
import torch
import torch.nn as nn
import torch.nn.functional as F
import argparse
from tqdm import tqdm
import time
# from network import *
# from torch.cuda.amp import autocast
# from torch.cuda.amp import GradScaler
# import os

parser = argparse.ArgumentParser()
parser.add_argument("--epoch", type=int, default=500, help="训练迭代次数")
parser.add_argument("--batch_size", type=int, default=16, help="批训练大小")
parser.add_argument("--learning_rate", type=float, default=1e-1, help="学习率大小")
parser.add_argument("--momentum", type=float, default=0.9)
parser.add_argument("--weights", type=str, default="./pth/", help="训练好的权重保存路径")
opt = parser.parse_args()

def train(model):

    model = model.cuda()
    # SegNet.load_weights(PRE_TRAINING)

    g_train = data_generator('./data/img_train.npy', './data/tar_train.npy', BATCH_SIZE, train=True)
    g_val = data_generator('./data/img_val.npy', './data/tar_val.npy', 10, train=False)

    optimizer = torch.optim.SGD(model.parameters(), lr=LR, momentum=MOMENTUM)
    # scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=300, gamma=0.1)
    # scaler = GradScaler()
    # loss_func = nn.CrossEntropyLoss(weight=torch.from_numpy(np.array(CATE_WEIGHT)).float()).cuda()
    criterion = DiceLoss().cuda()
    # criterion = DiceBCELoss().cuda()
    # criterion = FocalLoss(gamma=2, alpha=0.75).cuda()
    # criterion = nn.CrossEntropyLoss().cuda()
    
    for epoch in tqdm(range(EPOCH)):
        model.train()
        img, tar = g_train.gen()
        img = img.cuda()
        tar = tar.cuda()

        optimizer.zero_grad()
        # with autocast():
        out = model(img)
        loss = criterion(1-out, 1-tar)
        loss.backward()
        optimizer.step()

        if epoch % 10 == 0:
            model.eval()
            img, tar = g_val.gen()
            img = img.cuda()
            tar = tar.cuda()
            out = model(img)
            loss_val = criterion(1-out, 1-tar)
            tqdm.write("Epoch:{0} || Loss_train:{1}".format(epoch, format(loss, ".4f")))
            tqdm.write("Epoch:{0} || Loss_val:{1}".format(epoch, format(loss_val, ".4f")))

    torch.save(model.state_dict(), "./pth/Unet_" + str(2) + ".pth")

EPOCH = opt.epoch
BATCH_SIZE = opt.batch_size
LR = opt.learning_rate
MOMENTUM = opt.momentum
WEIGHTS = opt.weights

# model = deeplabv3plus_mobilenet(num_classes=1, pretrained_backbone=False)
model = UNetModel(out_features=1)
train(model)