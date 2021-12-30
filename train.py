from models import *
from loss import *
from img_aug import data_generator
import torch
import torch.nn.functional as F
import argparse
from tqdm import tqdm
from torch.cuda.amp import autocast
from torch.cuda.amp import GradScaler

from models.CENET import CENet50
# import os

parser = argparse.ArgumentParser()
parser.add_argument("--epoch", type=int, default=500, help="训练迭代次数")
parser.add_argument("--batch_size", type=int, default=2, help="批训练大小")
parser.add_argument("--learning_rate", type=float, default=1e-1, help="学习率大小")
parser.add_argument("--momentum", type=float, default=0.9)
parser.add_argument("--weights", type=str, default="./pth/", help="训练好的权重保存路径")
opt = parser.parse_args()

def train(model):

    model = model.cuda()
    best_model = None
    best_val = 1.0
    # SegNet.load_weights(PRE_TRAINING)

    g_train = data_generator('./data/img_train.npy', './data/tar_train.npy', BATCH_SIZE, train=True)
    g_val = data_generator('./data/img_val.npy', './data/tar_val.npy', 10, train=False)

    optimizer = torch.optim.SGD(model.parameters(), lr=LR, momentum=MOMENTUM)
    # optimizer = torch.optim.Adam(model.parameters(), lr=LR)
    scaler = GradScaler()
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=50, verbose=True)
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
        with autocast():
            out = model(img)
            loss = criterion(1-out, 1-tar)
        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()

        model.eval()
        img, tar = g_val.gen()
        img = img.cuda()
        tar = tar.cuda()
        out = model(img)
        loss_val = criterion(1-out, 1-tar)
        if loss_val < best_val:
            best_val = loss_val
            best_model = model
        if epoch > 120:
            scheduler.step(loss_val)

        # loss.backward()
        # optimizer.step()

        if epoch % 10 == 0:
            tqdm.write("Epoch:{0} || Loss_train:{1}".format(epoch, format(loss, ".4f")))
            tqdm.write("Epoch:{0} || Loss_val:{1}".format(epoch, format(loss_val, ".4f")))

    best_model.eval()
    img, tar = g_val.gen()
    img = img.cuda()
    tar = tar.cuda()
    out = best_model(img)
    loss_val = criterion(1-out, 1-tar)
    tqdm.write("Final Epoch || Loss_val:{0}".format(loss_val, ".4f"))
    torch.save(best_model.state_dict(), "./pth/CENet50.pth")

EPOCH = opt.epoch
BATCH_SIZE = opt.batch_size
LR = opt.learning_rate
MOMENTUM = opt.momentum
WEIGHTS = opt.weights

# model = deeplabv3plus_mobilenet(num_classes=1, pretrained_backbone=False)
# model = CE_Net_()
model = CENet50()
train(model)