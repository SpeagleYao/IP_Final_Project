from models import *
from loss import *
from img_aug import data_generator
import torch
import argparse
from tqdm import tqdm
from torch.cuda.amp import autocast
from torch.cuda.amp import GradScaler

parser = argparse.ArgumentParser()
parser.add_argument("--epoch", type=int, default=1000, help="训练迭代次数")
parser.add_argument("--batch_size", type=int, default=32, help="批训练大小")
parser.add_argument("--learning_rate", type=float, default=1e-1, help="学习率大小")
args = parser.parse_args()

def train(model):

    model = model.cuda()
    best_model = None
    best_epoch = 0
    best_val = 1.0

    g_train = data_generator('./data/img_train.npy', './data/tar_train.npy', args.batch_size, train=True)
    g_val = data_generator('./data/img_val.npy', './data/tar_val.npy', 10, train=False)
 
    optimizer = torch.optim.SGD(model.parameters(), lr=args.learning_rate, momentum=0.9)
    scaler = GradScaler()
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.25, patience=80, verbose=True)
    criterion = DiceLoss().cuda()
    
    for epoch in tqdm(range(args.epoch)):
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
        if loss_val + loss < best_val and epoch > 300:
            best_val = loss_val + loss
            best_model = model
            best_epoch = epoch
        if epoch > 120:
            scheduler.step(loss_val)

        if epoch % 10 == 0:
            tqdm.write("Epoch:{0} || Loss_train:{1}".format(epoch, format(loss, ".4f")))
            tqdm.write("Epoch:{0} || Loss_val:{1}".format(epoch, format(loss_val, ".4f")))

        if (optimizer.state_dict()['param_groups'][0]['lr'] < 1e-3): break

    best_model.eval()
    img, tar = g_val.gen()
    img = img.cuda()
    tar = tar.cuda()
    out = best_model(img)
    loss_val = criterion(1-out, 1-tar)
    tqdm.write("Best Epoch:{0} || Loss_val:{1}".format(best_epoch, format(loss_val, ".4f")))
    torch.save(best_model.state_dict(), "./pth/CENet_My.pth")

if __name__ == '__main__':
    model = CENet_My()
    train(model)